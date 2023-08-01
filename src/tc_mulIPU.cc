// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

// Encourage use of gcc's parallel algorithms (for sort for relabeling)
#ifdef _OPENMP
  #define _GLIBCXX_PARALLEL
#endif

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>
#include <chrono>


#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "half.h"

// Graphcore Poplar headers
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/CycleCount.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/Loop.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <poplin/TriangularSolve.hpp>
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


// To avoid confusion between Poplar Graph and GAP's input Graph
typedef CSRGraph<NodeID> Network;
#define THREADS 8
/*
GAP Benchmark Suite
Kernel: Triangle Counting (TC)
Author: Scott Beamer

Will count the number of triangles (cliques of size 3)

Requires input graph:
  - to be undirected
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

Other than symmetrizing, the rest of the requirements are done by SquishCSR
during graph building.

This implementation reduces the search space by counting each triangle only
once. A naive implementation will count the same triangle six times because
each of the three vertices (u, v, w) will count it in both ways. To count
a triangle only once, this implementation only counts a triangle if u > v > w.
Once the remaining unexamined neighbors identifiers get too big, it can break
out of the loop, but this requires that the neighbors to be sorted.

Another optimization this implementation has is to relabel the vertices by
degree. This is beneficial if the average degree is high enough and if the
degree distribution is sufficiently non-uniform. To decide whether or not
to relabel the graph, we use the heuristic in WorthRelabelling.
*/


using namespace std;
using namespace poplar;
using namespace poplar::program;
using namespace popops;

size_t OrderedCount(const Network &g) {
  size_t total = 0;
  #pragma omp parallel for reduction(+ : total) num_threads(THREADS) schedule(dynamic)
  for (NodeID u=0; u < g.num_nodes(); u++) {
    for (NodeID v : g.out_neigh(u)) {
      if (v > u)
        break;
      auto it = g.out_neigh(u).begin();
      for (NodeID w : g.out_neigh(v)) {
        if (w > v)
          break;
        while (*it < w)
          it++;
        if (w == *it)
          total++;
      }
    }
  }
  return total;
}


// heuristic to see if sufficently dense power-law graph
bool WorthRelabelling(const Network &g) {
  int64_t average_degree = g.num_edges() / g.num_nodes();
  if (average_degree < 10)
    return false;
  SourcePicker<Network> sp(g);
  int64_t num_samples = min(int64_t(1000), g.num_nodes());
  int64_t sample_total = 0;
  pvector<int64_t> samples(num_samples);
  for (int64_t trial=0; trial < num_samples; trial++) {
    samples[trial] = g.out_degree(sp.PickNext());
    sample_total += samples[trial];
  }
  sort(samples.begin(), samples.end());
  double sample_average = static_cast<double>(sample_total) / num_samples;
  double sample_median = samples[num_samples/2];
  return sample_average / 1.3 > sample_median;
}


// uses heuristic to see if worth relabeling
size_t Hybrid(const Network &g) {
  // if (WorthRelabelling(g))
  //   return OrderedCount(Builder::RelabelByDegree(g));
  // else
    return OrderedCount(g);
}

size_t Hybrid_LA(const Network &g) {
  
  vector<vector<int>> A(g.num_nodes(), vector<int>(g.num_nodes(), 0));
  vector<vector<int>> L(g.num_nodes(), vector<int>(g.num_nodes(), 0));
  vector<vector<int>> U(g.num_nodes(), vector<int>(g.num_nodes(), 0));
  vector<vector<int>> B(g.num_nodes(), vector<int>(g.num_nodes(), 0));
  int sum = 0;

  for (size_t i = 0; i < g.num_nodes(); i++) {
    for (auto j : g.out_neigh(i)) {
      A[i][j] = 1;
      if (i <= j)
        U[i][j] = 1;
      else
        L[i][j] = 1;
    }
  }

  for (size_t i = 0; i < g.num_nodes(); i++) {
    for (size_t j = 0; j < g.num_nodes(); j++) {
      for (size_t k = 0; k < g.num_nodes(); k++) {
        B[i][j] += L[i][k] * U[k][j];
      }
    }
  }

  // for (size_t i = 0; i < g.num_nodes(); i++) {
  //   for (size_t j = 0; j < g.num_nodes(); j++) {
  //     std::cout << A[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  // for (size_t i = 0; i < g.num_nodes(); i++) {
  //   for (size_t j = 0; j < g.num_nodes(); j++) {
  //     std::cout << L[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  // for (size_t i = 0; i < g.num_nodes(); i++) {
  //   for (size_t j = 0; j < g.num_nodes(); j++) {
  //     std::cout << U[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  // for (size_t i = 0; i < g.num_nodes(); i++) {
  //   for (size_t j = 0; j < g.num_nodes(); j++) {
  //     std::cout << B[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  for (size_t i = 0; i < g.num_nodes(); i++) {
    for (size_t j = 0; j < g.num_nodes(); j++) {
      sum += A[i][j] * B[i][j];
      // std::cout << A[i][j] * B[i][j] << " ";
    }
    // std::cout << std::endl;
  }
  // std::cout << std::endl;

  return sum / 2;
       
}

size_t Hybrid_IPU(const Network &g) {

  unsigned numRows = g.num_nodes();
  unsigned numCols = g.num_nodes();
  auto hMatrix = std::vector<float>(numRows * numCols, 0);
  auto res = std::vector<float>(1, -1);

  // Create the DeviceManager which is used to discover devices
  auto manager = DeviceManager::createDeviceManager();

  // Attempt to attach to a single IPU:
  auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
  std::cout << "Trying to attach to IPU\n";
  auto it = std::find_if(devices.begin(), devices.end(), [](Device &device) { return device.attach(); });

  if (it == devices.end()) {
    std::cerr << "Error attaching to device\n";
    return res[0]; // EXIT_FAILURE
  }


  auto device = std::move(*it);
  std::cout << "Attached to IPU " << device.getId() << std::endl;

  auto target = device.getTarget();

  poplar::Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  poprand::addCodelets(graph);
  std::cout << "Constructing compute graph and control program\n";

  auto LH = std::vector<float>(numRows * numCols, 0);

  for (unsigned i = 0; i < numRows; i++) {
    for (auto j : g.out_neigh(i)) {
      hMatrix[i * numCols + j] = 1.0;
      if(i <= j) LH[i * numCols + j] = 1.0;
    }
  }

  auto inStreamA = graph.addHostToDeviceFIFO("inputMatrix", FLOAT, numCols * numRows);
  auto outStream = graph.addDeviceToHostFIFO("out", FLOAT, 1);

  std::cout << "Creating environment (compiling vertex programs)\n";

  auto mainProg = Sequence();
  
  Sequence mul;
  
  Tensor A = poplin::createMatMulInputLHS(graph, FLOAT, {numRows, numCols}, {numRows, numCols}, "A");

  // Prepare LHS and RHS
  Tensor L = poplin::createMatMulInputLHS(graph, FLOAT, {numRows, numCols}, {numRows, numCols}, "L");
  Tensor U = poplin::createMatMulInputRHS(graph, FLOAT, {numRows, numCols}, {numRows, numCols}, "U");

  mainProg.add(Copy(inStreamA, A));
  mainProg.add(Copy(A, L));
  mainProg.add(Copy(A, U));
  L = poplin::triangularMask(graph, A, true, false, mul);
  U = poplin::triangularMask(graph, A, false, false, mul);


  Tensor B = poplin::matMul(graph, L, U, mul, FLOAT, "B");

  popops::mulInPlace(graph, A, B, mul, "ElementWiseMul");

  Tensor result = popops::reduce(graph, A, FLOAT, {0,1}, popops::Operation::ADD, mul, "Reduction");

  mainProg.add(Sequence({mul, Copy(result, outStream)}));

  // Create an engine from the compute graph and control program.
  Engine engine(graph, mainProg);
  engine.load(device);
  engine.connectStream("inputMatrix", hMatrix.data());
  engine.connectStream("out", res.data());

  // Execute the program
  std::cout << "Running graph program to multiply matrix by vector\n";
  engine.run();

  return ((int)res[0]) / 2;

}

// Utility code to acquire a real IPU device
poplar::Device getIpuHwDevice(std::size_t numIpus) {
  auto dm = poplar::DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(poplar::TargetType::IPU, numIpus);
  auto it =
      std::find_if(hwDevices.begin(), hwDevices.end(),
                   [](poplar::Device &device) { return device.attach(); });
  if (it != hwDevices.end()) {
    return std::move(*it);
  }
  throw std::runtime_error("No IPU hardware available.");
}

size_t Hybrid_multiple_IPU(const Network &g) {

  // Tiles per IPU to use
  long unsigned int numTiles = 1471;
  const unsigned NUM_IPUS = 16;
  unsigned int numBlocks = NUM_IPUS;
  long unsigned int N = g.num_nodes();
  long unsigned int blocksize = N / (int)std::sqrt(NUM_IPUS);
  long unsigned int nrows = (int)std::sqrt(NUM_IPUS);

  Device device = getIpuHwDevice(NUM_IPUS);
  Target target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  poprand::addCodelets(graph);

  auto mainProg = Sequence();
  Sequence mul, eleMul, reduction;
  Sequence copyIn, copyOut;

  auto hMatrix = std::vector<float>(N * N, 0);
  auto hLMatrix = std::vector<float>(N * N, 0);
  auto hUMatrix = std::vector<float>(N * N, 0);
  auto result = std::vector<float>(1, -1);

  long unsigned int m = blocksize, n = blocksize, k = blocksize;

  auto hostBlockA = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));
  auto hostBlock1 = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));
  auto hostBlock2 = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));

  auto zero = std::vector<std::vector<float>>(blocksize, std::vector<float>(blocksize, 0));

  std::vector<Tensor> matA;
  std::vector<Tensor> mat1;
  std::vector<Tensor> mat2;
  std::vector<Tensor> matB(NUM_IPUS);
  std::vector<Tensor> res;
  
  std::vector<DataStream> inStreamMA;
  std::vector<DataStream> inStreamM1;
  std::vector<DataStream> inStreamM2;

  // Initialize Adjacency and the Upper and Lower triangle
  for (int64_t i = 0; i < g.num_nodes(); i++) {
    for (auto j : g.out_neigh(i)) {
      hMatrix[i * N + j] = 1;
      if (i <= j)
        hUMatrix[i * N + j] = 1;
      else
        hLMatrix[i * N + j] = 1;
    }
  }

  //  To Print the input matrix
    // for (size_t i = 0; i < N; i++) {
    //     for (size_t j = 0; j < N; j++) {
    //         std::cout << hUMatrix[i * N + j] << " ";
    //     }
    //     std::cout << endl;
    // }

  // Copy them to the host tiles
  for (int64_t b = 0; b < numBlocks; b++) {
      int64_t idx = 0; 
      // std::cout << b << std::endl;
      for (int64_t i = ((int)(b / (N / blocksize))) * blocksize; i < ((int)(b / (N / blocksize))) * blocksize + blocksize; i++) {
          for (int64_t j = ((int)(b % (N / blocksize))) * blocksize; j < (((int)(b % (N / blocksize))) * blocksize) + blocksize; j++) {
              hostBlockA[b][idx] = hMatrix[i * N + j];
              hostBlock1[b][idx] = hLMatrix[i * N + j];
              hostBlock2[b][idx] = hUMatrix[i * N + j];
              idx++;
              // std::cout << i << " " << j << std::endl;
          }
      } 
  }

  
  // for (size_t i = 0; i < hostBlock2[1].size(); i++)
  //   std::cout << hostBlock2[1][i] << " " << std::endl;

  for (int64_t i = 0; i < NUM_IPUS; i++) {        
    auto g = graph.createVirtualGraph(numTiles);

    // One stream per IPU for LHS, RHS, Output  
    inStreamMA.push_back(graph.addHostToDeviceFIFO("AdjMatrix_" + std::to_string(i), FLOAT, m * n));
    inStreamM1.push_back(graph.addHostToDeviceFIFO("inputMatrix1_" + std::to_string(i), FLOAT, m * n));
    inStreamM2.push_back(graph.addHostToDeviceFIFO("inputMatrix2_" + std::to_string(i), FLOAT, n * k));
    

    // Create LHS and RHS
    mat1.push_back(poplin::createMatMulInputLHS(g, FLOAT, {m, n}, {n, k}, "M1_" + std::to_string(i)));
    mat2.push_back(poplin::createMatMulInputRHS(g, FLOAT, {m, n}, {n, k}, "M2_" + std::to_string(i)));

    matA.push_back(poplin::createMatMulInputLHS(g, FLOAT, {m, n}, {n, k}, "MA_" + std::to_string(i)));
    matB[i] = poplin::createMatMulInputLHS(g, FLOAT, {m, n}, {n, k}, "MB_" + std::to_string(i));

  
    Tensor z = graph.addConstant<float>(FLOAT, {m, n}, 0.0f);
    g.setTileMapping(z, g.getTileMapping(matB[i]));

    mul.add(Copy(z,matB[i]));
  }

  
  auto outStream = graph.addDeviceToHostFIFO("out", FLOAT, 1);

  for (int64_t i = 0; i < NUM_IPUS; i++) {
      // Copy tiles from stream
      copyIn.add(Copy(inStreamMA[i], matA[i]));
      copyIn.add(Copy(inStreamM1[i], mat1[i]));
      copyIn.add(Copy(inStreamM2[i], mat2[i]));
  }
  

  mainProg.add(copyIn);

  for (int64_t i = 0; i < nrows; i++) {
    for (int64_t j = 0; j < nrows; j++) {
      for (int64_t k = 0; k < nrows; k++) {
          popops::addInPlace(graph, matB[i * nrows + j], poplin::matMul(graph, mat1[i * nrows + k], mat2[k * nrows + j], mul, FLOAT), mul, "ElementWiseMul");
      }
    }
  }

  // for (int64_t i = 0; i < NUM_IPUS; i++) {
  //   mul.add(PrintTensor("B" + std::to_string(i), matB[i]));
  // }
  
  for (int64_t i = 0; i < NUM_IPUS; i++) {
    // Do the elementwise multiplication
    popops::mulInPlace(graph, matA[i], matB[i], eleMul, "ElementWiseMul");
    // mul.add(PrintTensor("M1_" + std::to_string(i), matB[i]));
    res.push_back(popops::reduce(graph, matA[i], FLOAT, {0,1}, popops::Operation::ADD, eleMul, "Reduction"));
  }

  // for (int64_t i = 0; i < NUM_IPUS; i++) {
  //   res.push_back(popops::reduce(graph, matA[i], FLOAT, {0,1}, popops::Operation::ADD, reduction, "Reduction"));
  // }
  
  for (int64_t i = 0; i < NUM_IPUS - 1; i++) {
    popops::addInPlace(graph, res[0], res[i + 1], eleMul, "FinalReduction");
  }

  auto cycleMul = cycleCount(graph, mul, 0, SyncType::INTERNAL, "Matmul");
  graph.createHostRead("cycleMul", cycleMul);

  auto cycleEleMul = cycleCount(graph, eleMul, 0, SyncType::INTERNAL, "ElementWiseMul");
  graph.createHostRead("cycleEleMul", cycleEleMul);

  // auto cycleRed = cycleCount(graph, reduction, 0, SyncType::INTERNAL, "Reduction");
  // graph.createHostRead("cycleRed", cycleRed);

  mainProg.add(mul);
  mainProg.add(eleMul);
  // mainProg.add(reduction);

  copyOut.add(Copy(res[0], outStream));
  mainProg.add(copyOut);

  Engine engine(graph, mainProg);
    engine.load(device);


  for (int64_t i = 0; i < NUM_IPUS; i++) {
    engine.connectStream("AdjMatrix_" + std::to_string(i), hostBlockA[i].data());
    engine.connectStream("inputMatrix1_" + std::to_string(i), hostBlock1[i].data());
    engine.connectStream("inputMatrix2_" + std::to_string(i), hostBlock2[i].data());
  }


  engine.connectStream("out", result.data());

  engine.run();   

  std::uint64_t cyclesBuffer, cyclesBufferMul, cyclesBufferEleMul; //, cyclesBufferRed;
  engine.readTensor("cycleMul", &cyclesBufferMul, &cyclesBufferMul + 1);
  engine.readTensor("cycleEleMul", &cyclesBufferEleMul, &cyclesBufferEleMul + 1);
  // engine.readTensor("cycleRed", &cycleRed, &cycleRed + 1);
  constexpr double freqGHz = 1.85;
  double tFlopsMul = (2 * g.num_nodes() * g.num_nodes() * g.num_nodes() * freqGHz * 1.0e9 / cyclesBufferMul);
  double tFlopsEleMul = (2 * g.num_nodes() * g.num_nodes() * freqGHz * 1.0e9 / cyclesBufferEleMul);
  // double tFlopsRed = (((g.num_nodes() * g.num_nodes()) - 1) * freqGHz * 1.0e9 / cyclesBufferRed);
  double tFlops = tFlopsMul + tFlopsEleMul; // + tFlopsRed;
  cyclesBuffer = cyclesBufferMul + cyclesBufferEleMul; // + cyclesBufferRed;
  std::cerr << "matmul cycles: " << cyclesBufferMul << std::endl;
  std::cerr << "elemul cycles: " << cyclesBufferEleMul << std::endl;
  // std::cerr << "TFlops/sec @" << freqGHz << "GHz = " << tFlops << std::endl;

  return ((int)result[0]) / 2;

}


void PrintTriangleStats(const Network &g, size_t total_triangles) {
  cout << total_triangles << " triangles" << endl;
}


// Compares with simple serial implementation that uses std::set_intersection
bool TCVerifier(const Network &g, size_t test_total) {
  size_t total = 0;
  vector<NodeID> intersection;
  intersection.reserve(g.num_nodes());
  for (NodeID u : g.vertices()) {
    for (NodeID v : g.out_neigh(u)) {
      auto new_end = set_intersection(g.out_neigh(u).begin(),
                                      g.out_neigh(u).end(),
                                      g.out_neigh(v).begin(),
                                      g.out_neigh(v).end(),
                                      intersection.begin());
      intersection.resize(new_end - intersection.begin());
      total += intersection.size();
    }
  }
  total = total / 6;  // each triangle was counted 6 times
  if (total != test_total)
    cout << total << " != " << test_total << endl;
  return total == test_total;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "triangle count");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Network g = b.MakeGraph();
  if (g.directed()) {
    cout << "Input graph is directed but tc requires undirected" << endl;
    return -2;
  }
  // auto start = std::chrono::high_resolution_clock::now();
  // size_t r = Hybrid(g);
  // auto end = std::chrono::high_resolution_clock::now();

  // // auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // auto t = end - start;
  
  // long int ops = 2 * (g.num_nodes() * g.num_nodes() * g.num_nodes() + g.num_nodes() * g.num_nodes());
  // std::cout << t.count() / 1000 << " seconds" << std::endl; 
  // std::cout << ops << " OPs" << std::endl;
  // std::cout << (float)ops / (t.count() / 1000) << " FLOPS/sec" << std::endl;
  BenchmarkKernel(cli, g, Hybrid_multiple_IPU, PrintTriangleStats, TCVerifier);
  return 0;
}
