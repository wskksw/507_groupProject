#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <time.h>

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 5) // 32

Graph *initGraph(int numVertices, GraphDensity density)
{
  Graph *graph = (Graph *)malloc(sizeof(Graph));
  graph->numVertices = numVertices;
  graph->numEdges = 0; // This will be calculated in generateEdges

  // Allocate memory for arrays
  graph->adjacencyList = (int *)malloc(numVertices * numVertices * sizeof(int));
  graph->edgesOffset = (int *)malloc(numVertices * sizeof(int));
  graph->edgesSize = (int *)malloc(numVertices * sizeof(int));

  // Initialize arrays
  for (int i = 0; i < numVertices; i++)
  {
    graph->edgesOffset[i] = i * numVertices;
    graph->edgesSize[i] = 0;
  }

  // Generate edges based on density
  generateEdges(graph, density);

  return graph;
}
void generateEdges(Graph *graph, GraphDensity density)
{
  srand(time(NULL)); // Seed the random number generator
  int maxAdditionalEdges;

  // First, create a chain to ensure connectivity
  for (int i = 0; i < graph->numVertices - 1; i++)
  {
    // Connect vertex i to i + 1
    graph->adjacencyList[graph->edgesOffset[i] + graph->edgesSize[i]] = i + 1;
    graph->edgesSize[i]++;
    graph->numEdges++;
  }

  // Determine the maximum number of additional edges based on density
  switch (density)
  {
  case Sparse:
    maxAdditionalEdges = graph->numVertices;
    break;
  case Medium:
    maxAdditionalEdges = graph->numVertices * 2;
    break;
  case Dense:
    maxAdditionalEdges = graph->numVertices * (graph->numVertices - 1) / 2;
    break;
  }

  // Add additional edges
  for (int i = 0; i < maxAdditionalEdges; i++)
  {
    int v = rand() % graph->numVertices;
    int w = rand() % graph->numVertices;

    // Avoid self-loops and duplicate edges
    if (v != w && graph->edgesSize[v] < graph->numVertices - 1)
    {
      // Add edge from v to w
      graph->adjacencyList[graph->edgesOffset[v] + graph->edgesSize[v]] = w;
      graph->edgesSize[v]++;

      // Increment the total edge count
      graph->numEdges++;
    }
  }
}

// Function to free the graph from memory
void freeGraph(Graph *graph)
{
  free(graph->adjacencyList);
  free(graph->edgesOffset);
  free(graph->edgesSize);
  free(graph);
}

__global__ void initializeDeviceArray(int n, int *d_arr, int value, int start_index)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == start_index)
  {
    d_arr[start_index] = 0;
  }
  else if (tid < n)
  {
    d_arr[tid] = value;
  }
}

__global__ void printDeviceArray(int *d_arr, int n)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    printf("d_arr[%i] = %i \n", tid, d_arr[tid]);
  }
}

__global__ void computeNextQueue(int *adjacencyList, int *edgesOffset, int *edgesSize, int *visited,
                                 int queueSize, int *currentQueue, int *nextQueueSize, int *nextQueue, int level)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < queueSize)
  {
    int current = currentQueue[tid];
    for (int i = edgesOffset[current]; i < edgesOffset[current] + edgesSize[current]; ++i)
    {
      int v = adjacencyList[i];
      if (oldVisited == INT_MAX) // Check if the node was previously unvisited
      {
        int oldVisited = atomicExch(&visited[v], 1); // Atomically set visited status to 1
        int position = atomicAdd(nextQueueSize, 1);  // Atomically add to next queue
        nextQueue[position] = v;
      }
    }
  }
}
void bfsGPU(int start, Graph *G, int *visited)
{
  const int n_blocks = (G->numVertices + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

  // Initialization of GPU variables
  int *d_adjacencyList, *d_edgesOffset, *d_edgesSize, *d_firstQueue, *d_secondQueue, *d_nextQueueSize, *d_visited;

  // Allocation on device
  const int size = G->numVertices * sizeof(int);
  const int adjacencySize = G->numVertices * G->numVertices * sizeof(int); // Assuming the adjacency list is a full matrix
  cudaMalloc((void **)&d_adjacencyList, adjacencySize);
  cudaMalloc((void **)&d_edgesOffset, size);
  cudaMalloc((void **)&d_edgesSize, size);
  cudaMalloc((void **)&d_firstQueue, size);
  cudaMalloc((void **)&d_secondQueue, size);
  cudaMalloc((void **)&d_visited, size);
  cudaMalloc((void **)&d_nextQueueSize, sizeof(int));

  // Copy inputs to device
  cudaMemcpy(d_adjacencyList, G->adjacencyList, adjacencySize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgesOffset, G->edgesOffset, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgesSize, G->edgesSize, size, cudaMemcpyHostToDevice);

  visited[start] = 1;
  cudaMemcpy(d_visited, visited, size, cudaMemcpyHostToDevice);

  int currentQueueSize = 1, level = 0;

  while (currentQueueSize > 0)
  {
    int *d_currentQueue, *d_nextQueue;
    if (level % 2 == 0)
    {
      d_currentQueue = d_firstQueue;
      d_nextQueue = d_secondQueue;
    }
    else
    {
      d_currentQueue = d_secondQueue;
      d_nextQueue = d_firstQueue;
    }

    computeNextQueue<<<n_blocks, N_THREADS_PER_BLOCK>>>(d_adjacencyList, d_edgesOffset, d_edgesSize, d_visited,
                                                        currentQueueSize, d_currentQueue, d_nextQueueSize, d_nextQueue, level);
    cudaDeviceSynchronize();
    ++level;
    cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
    int resetQueueSize = 0;
    cudaMemcpy(d_nextQueueSize, &resetQueueSize, sizeof(int), cudaMemcpyHostToDevice);
  }

  cudaMemcpy(visited, d_visited, size, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_adjacencyList);
  cudaFree(d_edgesOffset);
  cudaFree(d_edgesSize);
  cudaFree(d_firstQueue);
  cudaFree(d_secondQueue);
  cudaFree(d_visited);
  cudaFree(d_nextQueueSize);
}
int main()
{
  int numVertices = 25; // Example number of vertices
  Graph *myGraph = initGraph(numVertices, Dense);

  printf("Graph's Adjacency List:\n");
  printGraph(myGraph);

  // Allocate memory for BFS visitedand visited arrays
  int *visited = (int *)malloc(numVertices * sizeof(int));
  for (int i = 0; i < G->numVertices; ++i)
  {
    visited[i] = INT_MAX;
  }
  printf("Visited array before CUDA BFS\n");
  printArray(visited);
  clock_t startParallel, endParallel;
  startParallel = clock();
  bfsGPU(0, myGraph, visited);
  endParallel = clock();
  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
  printf("Cuda BFS took %f seconds.\n", timeTakenParallel);
  printf("Visited array after CUDA BFS\n");
  printArray(visited);
}
