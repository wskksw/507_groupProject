#include "graph.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

// Note Cuda test are separated cause it's ran on Colab.
void test_Cuda_Sparse_10Nodes()
{
  int numVertices = 10; // Number of vertices
  Graph *myGraph = initGraph(numVertices, Sparse);

  int *distance = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  bfsGPU(0, myGraph, distance);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  // Clean up
  free(distance);
  freeGraph(myGraph);

  printf("Cuda implementations for sparse 10 node graph\n");
}

void test_Cuda_Sparse_1000Nodes()
{
  int numVertices = 1000; // Number of vertices
  Graph *myGraph = initGraph(numVertices, Sparse);

  int *distance = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  bfsGPU(0, myGraph, distance);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  // Clean up
  free(distance);
  freeGraph(myGraph);

  printf("Cuda implementations for sparse 1000 node graph \n");
}

void test_Cuda_Medium_10Nodes()
{
  int numVertices = 10; // Number of vertices
  Graph *myGraph = initGraph(numVertices, Medium);

  int *distance = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  bfsGPU(0, myGraph, distance);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  // Clean up
  free(distance);
  freeGraph(myGraph);

  printf("Cuda implementations for medium 10 node graph\n");
}

void test_Cuda_Medium_1000Nodes()
{
  int numVertices = 1000; // Number of vertices
  Graph *myGraph = initGraph(numVertices, Medium);

  int *distance = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  bfsGPU(0, myGraph, distance);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  // Clean up
  free(distance);
  freeGraph(myGraph);

  printf("Cuda implementations for medium 1000 node graph\n");
}

void test_Cuda_Dense_10Nodes()
{
  int numVertices = 10; // Number of vertices
  Graph *myGraph = initGraph(numVertices, Dense);

  int *distance = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  bfsGPU(0, myGraph, distance);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  // Clean up
  free(distance);
  freeGraph(myGraph);

  printf("Cuda implementations for dense 10 node graph\n");
}

void test_Cuda_Dense_1000Nodes()
{
  int numVertices = 1000; // Number of vertices
  Graph *myGraph = initGraph(numVertices, Dense);

  int *distance = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  bfsGPU(0, myGraph, distance);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distance[i] != INT_MAX);
  }

  // Clean up
  free(distance);
  freeGraph(myGraph);

  printf("Cuda implementations for dense 1000 node graph\n");
}
int main()
{
  printf("Running BFS implementation tests...\n");

  test_Cuda_Dense_10Nodes();
  test_Cuda_Dense_1000Nodes();
  test_Cuda_Medium_10Nodes();
  test_Cuda_Medium_1000Nodes();
  test_Cuda_Sparse_10Nodes();
  test_Cuda_Sparse_1000Nodes();

  printf("All tests passed successfully.\n");
  return 0;
}
