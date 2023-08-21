// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

// g++ --std=c++17 -O3 -Wall -fopenmp -w tc_final.cc -lpoplar -lpopops -lpoplin -lpoputil -o tc

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
#include <poplar/OptionFlags.hpp>
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
#include <poputil/VertexTemplates.hpp>



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

size_t countTriangles(const Network &g) {

  unsigned NUM_IPUS = 4;
  
  size_t tc = 0;

  long unsigned int N = g.num_nodes();
  long unsigned int blocksize = N / 8;
  long unsigned int blocksPerRow = N / blocksize;
  long unsigned int numBlocks = blocksPerRow * blocksPerRow;
  unsigned blocksPerIPU = numBlocks / NUM_IPUS;
  auto hMatrix = std::vector<float>(N * N, 0);
  auto hLMatrix = std::vector<float>(N * N, 0);
  auto hUMatrix = std::vector<float>(N * N, 0);
  
  auto hostBlockA = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));
  auto hostBlockL = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));
  auto hostBlockU = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));

  std::cout << "Matrix: " << N << std::endl;
  std::cout << "IPUs: " << NUM_IPUS << std::endl;
  std::cout << "blocksize: " << blocksize << std::endl;
  std::cout << "numblocks: " << numBlocks << std::endl;

  // Initialize Adjacency from the network
  for (int64_t i = 0; i < N; i++) {
    for (auto j : g.out_neigh(i)) {
      hMatrix[i * N + j] = 1;
      if (i < j)
        hUMatrix[i * N + j] = 1;
      else
        hLMatrix[i * N + j] = 1;
    }
  }

  // Tile up the Adjacency into 'numBlocks'-many blocks of size blocksize X blocksize
  for (int64_t b = 0; b < numBlocks; b++) {
    
    int64_t idx = 0; 
    for (int64_t i = ((int)(b / (N / blocksize))) * blocksize; i < ((int)(b / (N / blocksize))) * blocksize + blocksize; i++) {
        for (int64_t j = ((int)(b % (N / blocksize))) * blocksize; j < (((int)(b % (N / blocksize))) * blocksize) + blocksize; j++) {
            
            hostBlockA[b][idx] = hMatrix[i * N + j];
            hostBlockL[b][idx] = hLMatrix[i * N + j];
            hostBlockU[b][idx] = hUMatrix[i * N + j];
            idx++;
            
        }
    } 
  }

  Device device = getIpuHwDevice(NUM_IPUS);
  Target mainTarget = device.getTarget();

  std::vector<std::vector<Tensor>> matLHS(numBlocks);
  std::vector<std::vector<Tensor>> matRHS(numBlocks);
  std::vector<Tensor> matB(numBlocks);
  std::vector<Tensor> matA(numBlocks);
  std::vector<Tensor> reduction;

  std::vector<std::vector<DataStream>> inStreamLHS(numBlocks);
  std::vector<std::vector<DataStream>> inStreamRHS(numBlocks);
  std::vector<DataStream> inStreamA(numBlocks);
  std::vector<poplar::Graph> vg(numBlocks);

  // #pragma omp parallel for reduction(+:tc)
  for (int64_t ipu = 0; ipu < NUM_IPUS; ipu++) {

    auto result = std::vector<float>(1, -1);
    
    
    poplar::Graph graph(mainTarget);//.createIPUTarget(1, "IPU-POD64"));
    popops::addCodelets(graph);
    poplin::addCodelets(graph);

    

    const unsigned tilesPerBlock = (int)(1472 * NUM_IPUS / numBlocks);
    
    auto mainProg = Sequence();
    Sequence mul, eleMul;
    Sequence copyIn, copyOut;

    // placeholder for LHS, RHS, and Adjacency blocks
    bool copyAdj = false;
    
    // #pragma omp parallel 
    for (int64_t i = 0; i < blocksPerRow; i++) {
      
      for (int64_t j = 0; j < blocksPerRow; j++) {
        copyAdj = false;
        // Virtual graph for every block
        // TODO::weighted number of tiles per virtual graphs based on memory requirement calculated at runtime
        unsigned startTile = (i * blocksPerRow + j) * tilesPerBlock;
        unsigned endTile = startTile + tilesPerBlock;
        std::cout << "Block " << i * blocksPerRow + j << " start " << startTile << " end " << endTile << std::endl;
        
        
        for (int64_t k = 0; k < blocksPerRow; k++) {
          if ((i >= k) && (k <= j) && ((int)((i * blocksPerRow + j) / blocksPerIPU) == ipu)) {
            vg[i * blocksPerRow + j] = graph.createVirtualGraph(startTile, endTile);
            copyAdj = true;
            //LHS
            inStreamLHS[i * blocksPerRow + j].push_back(vg[i * blocksPerRow + j].addHostToDeviceFIFO("LHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k), FLOAT, blocksize * blocksize));
            matLHS[i * blocksPerRow + j].push_back(poplin::createMatMulInputLHS(vg[i * blocksPerRow + j], FLOAT, {blocksize, blocksize}, {blocksize, blocksize}, "LHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k)));
            // poputil::mapTensorLinearlyWithOffset(vg[i * blocksPerRow + j], matLHS[i * blocksPerRow + j].back(), startTile);

            //RHS
            inStreamRHS[i * blocksPerRow + j].push_back(vg[i * blocksPerRow + j].addHostToDeviceFIFO("RHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k), FLOAT, blocksize * blocksize));
            matRHS[i * blocksPerRow + j].push_back(poplin::createMatMulInputRHS(vg[i * blocksPerRow + j], FLOAT, {blocksize, blocksize}, {blocksize, blocksize}, "RHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k)));
            // poputil::mapTensorLinearlyWithOffset(vg[i * blocksPerRow + j], matRHS[i * blocksPerRow + j].back(), startTile);

            //matmul result
            matB[i * blocksPerRow + j] = poplin::createMatMulInputLHS(vg[i * blocksPerRow + j], FLOAT, {blocksize, blocksize}, {blocksize, blocksize}, "B_" + std::to_string(i * blocksPerRow + j));
            // poputil::mapTensorLinearlyWithOffset(vg[i * blocksPerRow + j], matB[i * blocksPerRow + j], startTile);
            // Initialize the matmul results to zero
            Tensor z = vg[i * blocksPerRow + j].addConstant<float>(FLOAT, {blocksize, blocksize}, 0.0f);
            vg[i * blocksPerRow + j].setTileMapping(z, vg[i * blocksPerRow + j].getTileMapping(matB[i * blocksPerRow + j]));
            copyIn.add(Copy(z, matB[i * blocksPerRow + j]));
          }
        }
        
        // only copy the adjacency where needed
        if (copyAdj) {
          
          inStreamA[i * blocksPerRow + j] = vg[i * blocksPerRow + j].addHostToDeviceFIFO("A_" + std::to_string(i * blocksPerRow + j), FLOAT, blocksize * blocksize);
          matA[i * blocksPerRow + j] = poplin::createMatMulInputLHS(vg[i * blocksPerRow + j], FLOAT, {blocksize, blocksize}, {blocksize, blocksize}, "A_" + std::to_string(i * blocksPerRow + j));
          // poputil::mapTensorLinearlyWithOffset(vg[i * blocksPerRow + j], matA[i * blocksPerRow + j], startTile);
        }
        
      }
    }
    
    

    auto outStream = graph.addDeviceToHostFIFO("out", FLOAT, 1);
    std::cout << "Boo3" << std::endl;
    // Copy in the tile data from host to IPU through streams
    for (int64_t i = 0; i < blocksPerRow; i++) {
      for (int64_t j = 0; j < blocksPerRow; j++) {
        copyAdj = false;

        if ((int)((i * blocksPerRow + j) / blocksPerIPU) == ipu) {
          
          for (int64_t k = 0; k < blocksPerRow; k++) {
            if ((i >= k) && (k <= j) && ((int)((i * blocksPerRow + j) / blocksPerIPU) == ipu)) {
              copyAdj = true;
              copyIn.add(Copy(inStreamLHS[i * blocksPerRow + j][k], matLHS[i * blocksPerRow + j][k])); 
              copyIn.add(Copy(inStreamRHS[i * blocksPerRow + j][k], matRHS[i * blocksPerRow + j][k]));
            }
          }

        }
        

        if (copyAdj) {
          copyIn.add(Copy(inStreamA[i * blocksPerRow + j], matA[i * blocksPerRow + j]));
        }

      }
    }

    std::cout << "Boo4" << std::endl;
    mainProg.add(copyIn);

    for (int64_t i = 0; i < numBlocks; i++) {
      
      if((int)(i / blocksPerIPU) == ipu)  {
        // matmul
        for (int64_t j = 0; j < matLHS[i].size(); j++) {

          popops::addInPlace(vg[i], matB[i], poplin::matMul(vg[i], matLHS[i][j], matRHS[i][j], mul, FLOAT), mul, "Matmul");
          
        }    
      }
    }
    std::cout << "Boo5" << std::endl;
    mainProg.add(mul);

    for (int64_t i = 0; i < blocksPerRow; i++) {

      for (int64_t j = 0; j < blocksPerRow; j++) {

        copyAdj = false;
        for (int64_t k = 0; k < blocksPerRow; k++) {

          if ((i >= k) && (k <= j) && ((int)((i * blocksPerRow + j) / blocksPerIPU) == ipu)) {
            copyAdj = true;
          }

        }

        if (copyAdj) {
          
          popops::mulInPlace(vg[i * blocksPerRow + j], matA[i * blocksPerRow + j], matB[i * blocksPerRow + j], eleMul, "ElementWiseMul");
          reduction.push_back(popops::reduce(vg[i * blocksPerRow + j], matA[i * blocksPerRow + j], FLOAT, {0,1}, popops::Operation::ADD, eleMul, "Reduction"));
        }
        
      }    

    }
    std::cout << "Boo6" << std::endl;
    for (int64_t i = 1; i < reduction.size(); i++) {
      popops::addInPlace(graph, reduction[0], reduction[i], eleMul, "FinalReduction");
    }
    std::cout << "Boo7" << std::endl;
    mainProg.add(eleMul);

    copyOut.add(Copy(reduction[0], outStream));
    mainProg.add(copyOut);

    Engine engine(graph, mainProg);
    engine.load(device);
    
    // Connect the input streams
    for (int64_t i = 0; i < blocksPerRow; i++) {
      for (int64_t j = 0; j < blocksPerRow; j++) {
        
        copyAdj = false;
        for (int64_t k = 0; k < blocksPerRow; k++) {
          if ((i >= k) && (k <= j) && ((int)((i * blocksPerRow + j) / blocksPerIPU) == ipu)) {

            copyAdj = true;
            engine.connectStream("LHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k), hostBlockL[i * blocksPerRow + k].data());
            engine.connectStream("RHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k), hostBlockU[k * blocksPerRow + j].data());

          }          

        }

        if (copyAdj) {
          engine.connectStream("A_" + std::to_string(i * blocksPerRow + j), hostBlockA[i * blocksPerRow + j].data());
        }

      }
    }
    
    // Connect the output stream
    engine.connectStream("out", result.data());
    engine.run(); 

    tc += (int)result[0] / 2;

  }

  return tc;

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
  BenchmarkKernel(cli, g, countTriangles, PrintTriangleStats, TCVerifier);
  return 0;
}