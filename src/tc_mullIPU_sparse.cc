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
#include <popsparse/SparseTensor.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>


// To avoid confusion between Poplar Graph and GAP's input Graph
typedef CSRGraph<NodeID> Network;

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

typedef struct csr{
  std::vector<float> nnz;
  int numRows, numCols;
  std::vector<size_t> colIndices;
  std::vector<size_t> rowindices;
}csr;

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
  const unsigned NUM_IPUS = 4;
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
  Sequence mul;
  Sequence copyIn, copyOut;

  auto hMatrix = std::vector<float>(N * N, 0);
  auto hLMatrix = std::vector<float>(N * N, 0);
  auto hUMatrix = std::vector<float>(N * N, 0);
  auto result = std::vector<float>(1, -1);

  long unsigned int m = blocksize, n = blocksize, k = blocksize;

  auto hostBlockA = std::vector<csr>(numBlocks);
  auto hostBlock1 = std::vector<csr>(numBlocks);
  auto hostBlock2 = std::vector<csr>(numBlocks);

  auto zero = std::vector<std::vector<float>>(blocksize, std::vector<float>(blocksize, 0));

  std::vector<SparseTensor> matA;
  std::vector<SparseTensor> mat1;
  std::vector<SparseTensor> mat2;
  std::vector<Tensor> matB(NUM_IPUS);
  std::vector<Tensor> res;

  std::vector<popsparse::CSRMatrix> csrA;
  std::vector<popsparse::CSRMatrix> csr1;
  std::vector<popsparse::CSRMatrix> csr2;
  
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
      
      hostBlockA[b].numRows = blocksize;
      hostBlockA[b].numCols = blocksize;

      hostBlock1[b].numRows = blocksize;
      hostBlock1[b].numCols = blocksize;

      hostBlock2[b].numRows = blocksize;
      hostBlock2[b].numCols = blocksize;
      // std::cout << b << std::endl;
      for (int64_t i = ((int)(b / (N / blocksize))) * blocksize, size_t x = 0; i < ((int)(b / (N / blocksize))) * blocksize + blocksize; x++, i++) {
          for (int64_t j = ((int)(b % (N / blocksize))) * blocksize, size_t y = 0;; j < (((int)(b % (N / blocksize))) * blocksize) + blocksize; y++, j++) {
              
              if(hMatrix[i * N + j]) {
                hostBlockA[b].rowindices.push_back(x);
                hostBlockA[b].colIndices.push_back(y);
                hostBlockA[b].nnz.push_back(1.0f);
              }

              if(hUMatrix[i * N + j]) {
                hostBlock2[b].rowindices.push_back(x);
                hostBlock2[b].colIndices.push_back(y);
                hostBlock2[b].nnz.push_back(1.0f);
              }

              if(hLMatrix[i * N + j]) {
                hostBlock1[b].rowindices.push_back(x);
                hostBlock1[b].colIndices.push_back(y);
                hostBlock1[b].nnz.push_back(1.0f);
              }
              // std::cout << i << " " << j << std::endl;
          }
      }

      csrA.push_back(popsparse::CSRMatrix(hostBlockA[b].nnz, hostBlockA[b].colIndices, hostBlockA[b].rowindices, {1,1}));
      csr1.push_back(popsparse::CSRMatrix(hostBlock1[b].nnz, hostBlock1[b].colIndices, hostBlock1[b].rowindices, {1,1}));
      csr2.push_back(popsparse::CSRMatrix(hostBlockA[b].nnz, hostBlock2[b].colIndices, hostBlock2[b].hostBlock2, {1,1}));
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
    mat1.push_back(popsparse::createSparseDenseMatMulLHS(g, FLOAT, popsparse::createForSparseDense(1, m, n, k), "M1_" + std::to_string(i)));
    mat2.push_back(popsparse::createSparseDenseMatMulLHS(g, FLOAT, popsparse::createForSparseDense(1, m, n, k), "M2_" + std::to_string(i)));

    matA.push_back(popsparse::createSparseDenseMatMulLHS(g, FLOAT, popsparse::createForSparseDense(1, m, n, k), "MA_" + std::to_string(i)));
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

  for (int64_t i = 0; i < NUM_IPUS; i++) {
    mul.add(PrintTensor("B" + std::to_string(i), matB[i]));
  }
  
  for (int64_t i = 0; i < NUM_IPUS; i++) {
    // Do the elementwise multiplication
    popops::mulInPlace(graph, matA[i], matB[i], mul, "ElementWiseMul");
    // mul.add(PrintTensor("M1_" + std::to_string(i), matB[i]));
    res.push_back(popops::reduce(graph, matA[i], FLOAT, {0,1}, popops::Operation::ADD, mul, "Reduction"));
  }
  
  for (int64_t i = 0; i < NUM_IPUS - 1; i++) {
    popops::addInPlace(graph, res[0], res[i + 1], mul, "FinalReduction");
  }

  auto cyclesComp = cycleCount(graph, mul, 0, SyncType::INTERNAL, "Computation");
  graph.createHostRead("cyclesComp", cyclesComp);

  mainProg.add(mul);

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

  std::uint64_t cyclesBuffer;
  engine.readTensor("cyclesComp", &cyclesBuffer, &cyclesBuffer + 1);
  constexpr double freqGHz = 1.85;
  double tFlops =
      2 * g.num_edges() * g.num_nodes() * freqGHz * 1.0e9 / cyclesBuffer;
  std::cerr << "Total cycles: " << cyclesBuffer << std::endl;
  std::cerr << "TFlops/sec @" << freqGHz << "GHz = " << tFlops << std::endl;

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
  BenchmarkKernel(cli, g, Hybrid_multiple_IPU, PrintTriangleStats, TCVerifier);
  return 0;
}
