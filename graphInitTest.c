#include "graph.h"
#include <assert.h>
#include <stdio.h>

/*
Results:
./testInit
Running graph initialization tests...
Graph with 10 vertices and 19 edges
Graph with 1000 vertices and 1998 edges
test_initGraph_Sparse_1000Nodes passed.
Graph with 10 vertices and 29 edges
test_initGraph_Medium_10Nodes passed.
Graph with 1000 vertices and 2997 edges
test_initGraph_Medium_1000Nodes passed.
Graph with 10 vertices and 49 edges
test_initGraph_Dense_10Nodes passed.
Graph with 1000 vertices and 500017 edges
test_initGraph_Dense_1000Nodes passed.
6/6 tests passed successfully.*/
void test_initGraph_Sparse_10Nodes();
void test_initGraph_Sparse_1000Nodes();
void test_initGraph_Medium_10Nodes();
void test_initGraph_Medium_1000Nodes();
void test_initGraph_Dense_10Nodes();
void test_initGraph_Dense_1000Nodes();
int testCount = 0;

int main()
{
  printf("Running graph initialization tests...\n");

  test_initGraph_Sparse_10Nodes();
  test_initGraph_Sparse_1000Nodes();
  test_initGraph_Medium_10Nodes();
  test_initGraph_Medium_1000Nodes();
  test_initGraph_Dense_10Nodes();
  test_initGraph_Dense_1000Nodes();

  printf("%d/%d tests passed successfully.\n", testCount, testCount);
  return 0;
}

// Test function definitions

void test_initGraph_Sparse_10Nodes()
{
  Graph *graph = initGraph(10, Sparse);

  assert(graph != NULL);            // Graph should be successfully created
  assert(graph->numVertices == 10); // Should have 10 vertices
  assert(graph->numEdges <= 20);    // Sparse, so edges should be low
  printf("Graph with %d vertices and %d edges\n", graph->numVertices, graph->numEdges);
  freeGraph(graph);
  testCount++;
}

void test_initGraph_Sparse_1000Nodes()
{
  Graph *graph = initGraph(1000, Sparse);

  assert(graph != NULL);
  assert(graph->numVertices == 1000);
  assert(graph->numEdges <= 2000);
  printf("Graph with %d vertices and %d edges\n", graph->numVertices, graph->numEdges);
  freeGraph(graph);
  printf("test_initGraph_Sparse_1000Nodes passed.\n");
  testCount++;
}

void test_initGraph_Medium_10Nodes()
{
  Graph *graph = initGraph(10, Medium);

  assert(graph != NULL);
  assert(graph->numVertices == 10);
  assert(graph->numEdges <= 40);
  printf("Graph with %d vertices and %d edges\n", graph->numVertices, graph->numEdges);
  freeGraph(graph);
  printf("test_initGraph_Medium_10Nodes passed.\n");
  testCount++;
}

void test_initGraph_Medium_1000Nodes()
{
  Graph *graph = initGraph(1000, Medium);

  assert(graph != NULL);
  assert(graph->numVertices == 1000);
  assert(graph->numEdges <= 4000);
  printf("Graph with %d vertices and %d edges\n", graph->numVertices, graph->numEdges);
  freeGraph(graph);
  printf("test_initGraph_Medium_1000Nodes passed.\n");
  testCount++;
}

void test_initGraph_Dense_10Nodes()
{
  Graph *graph = initGraph(10, Dense);

  assert(graph != NULL);
  assert(graph->numVertices == 10);
  assert(graph->numEdges >= 45); // Dense, close to fully connected
  printf("Graph with %d vertices and %d edges\n", graph->numVertices, graph->numEdges);
  freeGraph(graph);
  printf("test_initGraph_Dense_10Nodes passed.\n");
  testCount++;
}

void test_initGraph_Dense_1000Nodes()
{
  Graph *graph = initGraph(1000, Dense);

  assert(graph != NULL);
  assert(graph->numVertices == 1000);
  assert(graph->numEdges >= 499500); // Dense, close to fully connected
  printf("Graph with %d vertices and %d edges\n", graph->numVertices, graph->numEdges);
  freeGraph(graph);
  printf("test_initGraph_Dense_1000Nodes passed.\n");
  testCount++;
}
