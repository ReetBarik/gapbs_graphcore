// Generic C++ headers 
#include <algorithm>
#include <iostream>
#include <vector>

// Graphcore Poplar headers
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/Loop.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/Cast.hpp>
#include <popops/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/AllTrue.hpp>
#include <popsparse/MatMul.hpp>
#include <popsparse/MatMulParams.hpp>
#include <popsparse/SparsePartitioner.hpp>
#include <popsparse/SparseStorageFormats.hpp>
#include <popsparse/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

// GAP Benchmark Suite's headers
#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

// To avoid confusion between Poplar Graph and GAP's input Graph
typedef CSRGraph<NodeID> Network;

/*
GAP Benchmark Suite on Graphcore
Kernel: PageRank (PR)
Author: Reet Barik

Will return pagerank scores for all vertices once total change < epsilon
*/


using namespace std;
using namespace poplar;
using namespace poplar::program;
using namespace popops;


typedef float ScoreT;
const float kDamp = 0.85;

/* Sequential SpMV implementation of Pagerank */
pvector<ScoreT> PageRankPull(const Network &g, int max_iters,
                             double epsilon = 0) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> r(g.num_nodes(), init_score);
  vector<vector<float>> L(g.num_nodes(), vector<float>(g.num_nodes(), 0));

  for (int64_t i = 0; i < g.num_nodes(); i++) {
    for(auto j : g.in_neigh(i)) {
      L[i][j] = 1.0f / g.out_degree(j);
    }
  }

  for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
    pvector<ScoreT> temp(g.num_nodes(), 0);
    for(int64_t i = 0; i < g.num_nodes(); i++) {
      temp[i] = 0;
      for (int64_t j = 0; j < g.num_nodes(); j++){
        temp[i] += (r[j]* L[i][j]);
      }
      temp[i] = kDamp * temp[i] + base_score;
      error += fabs(temp[i] - r[i]);
    }

    printf(" %2d    %lf\n", iter, error);

    for(int64_t i = 0; i < g.num_nodes(); i++) {
      r[i] = temp[i];
    }
    if (error < epsilon)
      break;
  }
  return r;
}



pvector<ScoreT> PageRankPull_ipu(const Network &g, int max_iters,
                             double epsilon = 0.0) {

  std::cout << "Max Iters:" << max_iters << "\n";
  unsigned numRows = g.num_nodes();
  unsigned numCols = g.num_nodes();

  pvector<ScoreT> r(g.num_nodes());
  float base_score = (1.0f - kDamp) / g.num_nodes();

    // Create the DeviceManager which is used to discover devices
  auto manager = DeviceManager::createDeviceManager();

  // Attempt to attach to a single IPU:
  auto devices = manager.getDevices(poplar::TargetType::IPU, 4);
  std::cout << "Trying to attach to IPU\n";
  auto it = std::find_if(devices.begin(), devices.end(), [](Device &device) { return device.attach(); });

  if (it == devices.end()) {
    std::cerr << "Error attaching to device\n";
    return r; // EXIT_FAILURE
  }

  auto device = std::move(*it);
  std::cout << "Attached to IPU " << device.getId() << std::endl;

  auto target = device.getTarget();

  poplar::Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  poprand::addCodelets(graph);
  std::cout << "Constructing compute graph and control program\n";

  // Create host buffers for the inputs and outputs and fill the inputs
  // with sample data.
  auto hMatrix = std::vector<float>(numRows * numCols, 0);
  auto hInput = std::vector<float>(numCols, 1.0f / g.num_nodes());
  auto hDamp = std::vector<float>(numCols, kDamp);
  auto hBase = std::vector<float>(numCols, base_score);
  auto hOutput = std::vector<float>(numRows);

  for (unsigned i = 0; i < numRows; i++) {
    for(auto j : g.in_neigh(i)) {
      hMatrix[i * numCols + j] = 1.0f / g.out_degree(j);
    }
  }

  // Set up data streams to copy data in and out of graph
  auto inStreamV = graph.addHostToDeviceFIFO("inputVector", FLOAT, numCols);
  auto inStreamM = graph.addHostToDeviceFIFO("inputMatrix", FLOAT, numCols * numRows);
  auto inStreamD = graph.addHostToDeviceFIFO("dampVector", FLOAT, numCols);
  auto inStreamB = graph.addHostToDeviceFIFO("baseVector", FLOAT, numCols);
  auto outStream = graph.addDeviceToHostFIFO("out", FLOAT, numRows);

  // IPUModel ipuModel;
  // Device device = ipuModel.createDevice();
  // Target target = device.getTarget();

  std::cout << "Creating environment (compiling vertex programs)\n";


  // auto outStreamM = graph.addDeviceToHostFIFO("outputMatrix", FLOAT, numCols * numRows);

  /*  To Print the input matrix
  // for (size_t i = 0; i < numRows; i++) {
  //   for (size_t j = 0; j < numCols; j++) {
  //     std::cout << hMatrix[i * numCols + j] << " ";
  //   }
  //   std::cout << endl;
  // }
  */

  // Create a control program that is a sequence of steps
  auto mainProg = Sequence();
  
  Sequence mul;
  
  // Prepare LHS and RHS
  Tensor matrix = poplin::createMatMulInputLHS(graph, FLOAT, {numRows, numCols}, {numRows, 1}, "A");
  Tensor score = poplin::createMatMulInputRHS(graph, FLOAT, {numRows, numCols}, {numRows, 1}, "B");

  // matrix = poprand::uniform(graph, NULL, 0, matrix, FLOAT, 0., 1., mul, "randA");
  // score = poprand::uniform(graph, NULL, 0, score, FLOAT, 0., 1., mul, "randB");

  // Copy data from stream to input tensors
  mainProg.add(Copy(inStreamV, score)); 
  mainProg.add(Copy(inStreamM, matrix));
  
  // Do the matrix multiplication
  Tensor output = poplin::matMul(graph, matrix, score, mul, FLOAT, "C");

  popops::mulInPlace(graph, output, kDamp, mul, "Damp");
  popops::addInPlace(graph, output, base_score, mul, "Base");

  // Print the tensors 
  mul.add(PrintTensor("matrix", matrix));
  mul.add(PrintTensor("score", score));
  mul.add(PrintTensor("output", output));

  // Elementwise subtraction between score and output
  Tensor error = popops::sub(graph, score, output, mul, "Sub");
  mul.add(PrintTensor("error", error));

  // copy output to input (score) of next iteration
  mul.add(Copy(output, score));

  // fabs(error) to get absolute error
  popops::absInPlace(graph, error, mul, "Abs");
  mul.add(PrintTensor("abserror", error));

  // auto condProg = Sequence();
  // reduce to get total error
  Tensor e = popops::reduce(graph, error, FLOAT, {0}, popops::Operation::ADD, mul, "Red");
  
  Tensor eps = graph.addConstant<float>(FLOAT, {1}, {(float)epsilon});
  graph.setTileMapping(eps, graph.getTileMapping(e));

  Tensor i = graph.addConstant<int>(INT, {1}, {0});
  graph.setTileMapping(i, graph.getTileMapping(e));

  Tensor max = graph.addConstant<int>(INT, {1}, {max_iters});
  graph.setTileMapping(max, graph.getTileMapping(e));

  Tensor iters = graph.addVariable(INT, {1}, "Iterations");
  graph.setTileMapping(iters, graph.getTileMapping(e));
  
  mainProg.add(Copy(i, iters));
  
  popops::addInPlace(graph, iters, 1, mul, "IterationIncrement");
  Tensor finalIteration = lteq(graph, iters, max, mul, "FinalIter");

  
  // bool of gteq between 0 and error
  Tensor neZero = gteq(graph, e, eps, mul);

  mul.add(PrintTensor("totalerror", e));
  mul.add(PrintTensor("nezero", neZero));
  mul.add(PrintTensor("finalIteration", finalIteration));
  mul.add(PrintTensor("iteration", iters));
  mul.add(PrintTensor("maxiteration", max));


  Tensor doNotTerminate = popops::logicalAnd(graph, neZero, finalIteration, mul, "Terminate");
  mul.add(PrintTensor("terminate", doNotTerminate));
  auto predicate = allTrue(graph, doNotTerminate, mul, "all_true");
  mul.add(PrintTensor("condition", predicate));

  // Repeat based on predicate
  mainProg.add(RepeatWhileTrue(mul, predicate, mul));

  // // Repeat max_iters number of times
  // mainProg.add(Repeat(max_iters, mul));

  // run mainProg and copy output to outstream
  auto prog = Sequence({mainProg, Copy(output, outStream)});  //, Copy(matrix, outStreamM)});
                        

  // Create an engine from the compute graph and control program.
  Engine engine(graph, prog);
  engine.load(device);
  engine.connectStream("inputVector", hInput.data());
  engine.connectStream("inputMatrix", hMatrix.data());
  engine.connectStream("out", hOutput.data());
  // engine.connectStream("outputMatrix", hMatrix.data());

  // Execute the program
  std::cout << "Running graph program to multiply matrix by vector\n";
  engine.run();

  for (size_t i = 0; i < g.num_nodes(); i++) {
    r[i] = (ScoreT)hOutput[i];
    // std::cout << r[i] << ' ';
  }

  return r;

}


void PrintTopScores(const Network &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Network &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}


int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Network g = b.MakeGraph();


  // pvector<ScoreT> s = PageRankPull_ipu(g, cli.max_iters(), cli.tolerance());
  

  auto PRBound = [&cli] (const Network &g) {
    return PageRankPull_ipu(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli] (const Network &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}

