#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
// Function to print the graph
void printGraph(Graph *graph)
{
  printf("Graph with %d vertices and %d edges\n", graph->numVertices, graph->numEdges);

  for (int i = 0; i < graph->numVertices; i++)
  {
    printf("Vertex %d: ", i);
    for (int j = 0; j < graph->edgesSize[i]; j++)
    {
      printf("%d ", graph->adjacencyList[graph->edgesOffset[i] + j]);
    }
    printf("\n");
  }
}

void bfsCPU(int start, Graph *G, int *visited);
void ompBFS(int start, Graph *G, int *visited);

int main()
{
  int numVertices = 10; // Example number of vertices
  Graph *myGraph = initGraph(numVertices, Sparse);
  printf("Graph with %d vertices and %d edges\n", myGraph->numVertices, myGraph->numEdges);
  printf("Graph's Adjacency List:\n");
  printGraph(myGraph);

  // Allocate memory for BFS visited and visited arrays
  int *visited = (int *)malloc(numVertices * sizeof(int));
  // Timing variables
  clock_t startSerial, endSerial, startParallel, endParallel;

  for (int i = 0; i < myGraph->numVertices; i++)
  {
    visited[i] = INT_MAX;
  }
  // Perform standard BFS
  startSerial = clock();
  bfsCPU(0, myGraph, visited);
  endSerial = clock();
  double timeTakenSerial = (double)(endSerial - startSerial) / CLOCKS_PER_SEC;
  printf("Standard BFS took %f seconds.\n", timeTakenSerial);
  printf("Visiteds from vertex 0:\n");
  for (int i = 0; i < numVertices; i++)
  {
    printf("Vertex %d: %d\n", i, visited[i]);
  }

  // Reset visited and visited for ompBFS
  for (int i = 0; i < numVertices; i++)
  {
    visited[i] = INT_MAX;
  }

  // Perform OpenMP BFS
  startParallel = clock();
  ompBFS(0, myGraph, visited);
  endParallel = clock();
  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
  printf("OpenMP BFS took %f seconds.\n", timeTakenParallel);

  // Print visiteds from vertex 0(from ompBFS)
  printf("Visiteds from vertex 0:\n");
  for (int i = 0; i < numVertices; i++)
  {
    printf("Vertex %d: %d\n", i, visited[i]);
  }

  // Free allocated resources
  free(visited);
  freeGraph(myGraph);

  return 0;
}
