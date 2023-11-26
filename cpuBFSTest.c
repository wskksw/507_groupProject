#include "graph.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

int testCount = 0;
void ompBFS(int start, Graph *G, int *distance);
void bfsCPU(int start, Graph *G, int *distance, int *visited);

void test_Sparse_10Nodes();
void test_Medium_1000Nodes();
void test_Sparse_1000Nodes();
void test_Medium_10Nodes();
void test_Dense_10Nodes();
void test_Dense_1000Nodes();

int main()
{
  printf("Running BFS implementation tests...\n");

  test_Sparse_10Nodes();
  test_Medium_1000Nodes();
  test_Sparse_1000Nodes();
  test_Medium_10Nodes();
  test_Dense_10Nodes();
  test_Dense_1000Nodes();

  printf("%d/%d tests passed successfully.\n", testCount, testCount);
  return 0;
}

void test_Sparse_10Nodes()
{
  int numVertices = 10;
  Graph *myGraph = initGraph(numVertices, Sparse);

  int *distanceSerial = (int *)malloc(numVertices * sizeof(int));
  int *visited = (int *)malloc(numVertices * sizeof(int));
  int *distanceParallel = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    distanceSerial[i] = INT_MAX;
    distanceParallel[i] = INT_MAX;
  }

  // Perform standard BFS
  bfsCPU(0, myGraph, distanceSerial, visited);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceSerial[i] != INT_MAX);
  }
  testCount++;
  // Perform OpenMP BFS
  ompBFS(0, myGraph, distanceParallel);

  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceParallel[i] != INT_MAX);
  }
  testCount++;
  free(distanceSerial);
  free(visited);
  free(distanceParallel);
  freeGraph(myGraph);

  printf("test_bfsImplementations passed.\n");
}

void test_Medium_1000Nodes()
{
  int numVertices = 1000;
  Graph *myGraph = initGraph(numVertices, Medium);

  int *distanceSerial = (int *)malloc(numVertices * sizeof(int));
  int *visited = (int *)malloc(numVertices * sizeof(int));
  int *distanceParallel = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    distanceSerial[i] = INT_MAX;
    distanceParallel[i] = INT_MAX;
  }

  // Perform standard BFS
  bfsCPU(0, myGraph, distanceSerial, visited);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceSerial[i] != INT_MAX);
  }
  testCount++;
  // Perform OpenMP BFS
  ompBFS(0, myGraph, distanceParallel);

  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceParallel[i] != INT_MAX);
  }
  testCount++;
  free(distanceSerial);
  free(visited);
  free(distanceParallel);
  freeGraph(myGraph);

  printf("OMP and serial BFS on medium 1000 node graph passed\n");
}

void test_Medium_10Nodes()
{
  int numVertices = 10;
  Graph *myGraph = initGraph(numVertices, Medium);

  int *distanceSerial = (int *)malloc(numVertices * sizeof(int));
  int *visited = (int *)malloc(numVertices * sizeof(int));
  int *distanceParallel = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    distanceSerial[i] = INT_MAX;
    distanceParallel[i] = INT_MAX;
  }

  // Perform standard BFS
  bfsCPU(0, myGraph, distanceSerial, visited);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceSerial[i] != INT_MAX);
  }
  testCount++;
  // Perform OpenMP BFS
  ompBFS(0, myGraph, distanceParallel);

  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceParallel[i] != INT_MAX);
  }
  testCount++;
  free(distanceSerial);
  free(visited);
  free(distanceParallel);
  freeGraph(myGraph);

  printf("OMP and serial BFS on medium 10 node graph passed\n");
}

void test_Sparse_1000Nodes()
{
  int numVertices = 1000;
  Graph *myGraph = initGraph(numVertices, Sparse);

  int *distanceSerial = (int *)malloc(numVertices * sizeof(int));
  int *visited = (int *)malloc(numVertices * sizeof(int));
  int *distanceParallel = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    distanceSerial[i] = INT_MAX;
    distanceParallel[i] = INT_MAX;
  }

  // Perform standard BFS
  bfsCPU(0, myGraph, distanceSerial, visited);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceSerial[i] != INT_MAX);
  }
  testCount++;
  // Perform OpenMP BFS
  ompBFS(0, myGraph, distanceParallel);

  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceParallel[i] != INT_MAX);
  }
  testCount++;
  free(distanceSerial);
  free(visited);
  free(distanceParallel);
  freeGraph(myGraph);

  printf("OMP and serial BFS on sparse 1000 node graph passed\n");
}

void test_Dense_10Nodes()
{
  int numVertices = 10;
  Graph *myGraph = initGraph(numVertices, Dense);

  int *distanceSerial = (int *)malloc(numVertices * sizeof(int));
  int *visited = (int *)malloc(numVertices * sizeof(int));
  int *distanceParallel = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    distanceSerial[i] = INT_MAX;
    distanceParallel[i] = INT_MAX;
  }

  // Perform standard BFS
  bfsCPU(0, myGraph, distanceSerial, visited);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceSerial[i] != INT_MAX);
  }
  testCount++;
  // Perform OpenMP BFS
  ompBFS(0, myGraph, distanceParallel);

  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceParallel[i] != INT_MAX);
  }
  testCount++;
  free(distanceSerial);
  free(visited);
  free(distanceParallel);
  freeGraph(myGraph);

  printf("OMP and serial BFS on dense 10 node graph passed\n");
}

void test_Dense_1000Nodes()
{
  int numVertices = 1000;
  Graph *myGraph = initGraph(numVertices, Dense);

  int *distanceSerial = (int *)malloc(numVertices * sizeof(int));
  int *visited = (int *)malloc(numVertices * sizeof(int));
  int *distanceParallel = (int *)malloc(numVertices * sizeof(int));

  for (int i = 0; i < numVertices; i++)
  {
    distanceSerial[i] = INT_MAX;
    distanceParallel[i] = INT_MAX;
  }

  // Perform standard BFS
  bfsCPU(0, myGraph, distanceSerial, visited);

  // Check that all nodes have been visited
  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceSerial[i] != INT_MAX);
  }
  testCount++;
  // Perform OpenMP BFS
  ompBFS(0, myGraph, distanceParallel);

  for (int i = 0; i < numVertices; i++)
  {
    assert(distanceParallel[i] != INT_MAX);
  }
  testCount++;
  free(distanceSerial);
  free(visited);
  free(distanceParallel);
  freeGraph(myGraph);

  printf("OMP and serial BFS on dense 1000 node graph passed\n");
}