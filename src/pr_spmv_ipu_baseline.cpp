// Generic C++ headers 
#include <algorithm>
#include <iostream>
#include <vector>

// Graphcore Poplar headers
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poputil/TileMapping.hpp>

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
    for (int64_t j = 0; j < g.num_nodes(); j++){
      L[i][j] = 1.0f / g.out_degree(j);
    }
  }

  for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
    pvector<ScoreT> temp(g.num_nodes(), 0);
    for(int64_t i = 0; i < g.num_nodes(); i++) {
      temp[i] = 0;
      for(auto j : g.in_neigh(i)) {
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


// This function returns a device side program that will multiply
// the data in the 2-d tensor 'matrix' with the 1-d vector held
// in the 'in' tensor. When the program executes
// the result is placed in the 'out' 1-d tensor.
Program buildPrProgram(Graph &graph, Tensor matrix, Tensor in,
                             Tensor out) {
  // Create a compute set to hold the vertices to perform the calculation
  ComputeSet mulCS = graph.addComputeSet("mulCS");

  // The compute set holds a vertex for every output value. Each vertex
  // takes a row of the matrix as input and the whole input vector and
  // performs a dot-product placing the result in an element of the
  // output vector.
  auto numRows = matrix.dim(0);

  for (int iter=0; iter < max_iters; iter++) {
    auto error = 0;
    for (unsigned i = 0; i < numRows; ++i)
        auto temp = 
    for (unsigned i = 0; i < numRows; ++i) {
        auto v = graph.addVertex(mulCS,             // Put the vertex in the
                                                    // 'mulCS' compute set.
                                "DotProductVertex", // Create a vertex of this
                                                    // type.
                                {{"a", matrix[i]},  // Connect input 'a' of the
                                                    // vertex to a row of the
                                                    // matrix.
                                {"b", in},         // Connect input 'b' of the
                                                    // vertex to whole
                                                    // input vector.
                                {"out", out[i]}}); // Connect the output 'out'
                                                    // of the vertex to a single
                                                    // element of the output
                                                    // vector.
        graph.setTileMapping(v, i);
    }
  }
  // The returned program just executes the 'mulCS' compute set that is,
  // executes every vertex calculation in parallel.
  return Execute(mulCS);
}


pvector<ScoreT> PageRankPull_ipu(const Network &g, int max_iters,
                             double epsilon = 0) {

  unsigned numRows = g.num_nodes();
  unsigned numCols = g.num_nodes();

  pvector<ScoreT> r(g.num_nodes());

  // Create the DeviceManager which is used to discover devices
  auto manager = DeviceManager::createDeviceManager();

  // Attempt to attach to a single IPU:
  auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
  std::cout << "Trying to attach to IPU\n";
  auto it = std::find_if(devices.begin(), devices.end(), [](Device &device) { return device.attach(); });

  if (it == devices.end()) {
    std::cerr << "Error attaching to device\n";
    return 1; // EXIT_FAILURE
  }

  auto device = std::move(*it);
  std::cout << "Attached to IPU " << device.getId() << std::endl;

  auto target = device.getTarget();

  std::cout << "Creating environment (compiling vertex programs)\n";

  Graph graph(target);
  graph.addCodelets("matrix-mul-codelets_baseline.cpp");

  std::cout << "Constructing compute graph and control program\n";

  // Create tensors in the graph to hold the input/output data.
  Tensor matrix = graph.addVariable(FLOAT, {numRows, numCols}, "matrix");
  Tensor inputVector = graph.addVariable(FLOAT, {numCols}, "inputVector");
  Tensor outputVector = graph.addVariable(FLOAT, {numRows}, "outputVector");
  poputil::mapTensorLinearly(graph, matrix);
  poputil::mapTensorLinearly(graph, inputVector);
  poputil::mapTensorLinearly(graph, outputVector);

  // Create host buffers for the inputs and outputs and fill the inputs
  // with sample data.
  auto hMatrix = std::vector<float>(numRows * numCols);
  auto hInput = std::vector<float>(numCols);
  auto hOutput = std::vector<float>(numRows);

  for (unsigned j = 0; j < numCols; j++) {
    hInput[j] = 1.0f / g.num_nodes();
    for (unsigned i = 0; i < numRows; i++) {
      hMatrix[i * numCols + j] = 1.0f / g.out_degree(j);
    }
  }

  // Create a device program to multiply two tensors together.
  auto prKernel = buildPrProgram(graph, matrix, inputVector, outputVector);

  // Set up data streams to copy data in and out of graph
  auto inStreamV = graph.addHostToDeviceFIFO("inputVector", FLOAT, numCols);
  auto inStreamM = graph.addHostToDeviceFIFO("inputMatrix", FLOAT, numCols * numRows);
  auto outStream = graph.addDeviceToHostFIFO("out", FLOAT, numRows);

  // Create a program that copies data from the host buffers, multiplies
  // the result and copies the result back to the host.
  auto prog = Sequence({Copy(inStreamV, inputVector), Copy(inStreamM, matrix), prKernel, Copy(outputVector, outStream)});
                        

  // Create an engine from the compute graph and control program.
  Engine engine(graph, prog);
  engine.load(device);
  engine.connectStream("inputVector", hInput.data());
  engine.connectStream("inputMatrix", hMatrix.data());
  engine.connectStream("out", hOutput.data());

  // Execute the program
  std::cout << "Running graph program to multiply matrix by vector\n";
  engine.run();

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



  

  auto PRBound = [&cli] (const Network &g) {
    return PageRankPull(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli] (const Network &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}

