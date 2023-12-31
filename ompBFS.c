#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include "graph.h"

void ompBFS(int start, Graph *G, int *visited)
{
  int numThreads;
#pragma omp parallel
  {
#pragma omp single
    numThreads = omp_get_num_threads();
  }

  int **localNextFrontiers = (int **)malloc(numThreads * sizeof(int *));
  int *localNextFrontierSizes = (int *)calloc(numThreads, sizeof(int));
  for (int i = 0; i < numThreads; i++)
  {
    localNextFrontiers[i] = (int *)malloc(G->numVertices * sizeof(int));
  }

  int *currentFrontier = (int *)malloc(G->numVertices * sizeof(int));
  int currentFrontierSize = 1;
  currentFrontier[0] = start;
  visited[start] = 1;

  while (currentFrontierSize > 0)
  {
#pragma omp parallel
    {
      // Each thread processes a chunk of the current frontier
      int threadNum = omp_get_thread_num();
      int chunkSize = currentFrontierSize / numThreads;
      int start = threadNum * chunkSize;
      int end = (threadNum == numThreads - 1) ? currentFrontierSize : start + chunkSize; // last thread takes any remainder

      int *localNextFrontier = localNextFrontiers[threadNum];
      int localNextFrontierSize = 0;

      for (int i = start; i < end; i++)
      {
        int current = currentFrontier[i];
        for (int j = G->edgesOffset[current]; j < G->edgesOffset[current] + G->edgesSize[current]; ++j)
        {
          int v = G->adjacencyList[j];
          {
            if (visited[v] == INT_MAX)
#pragma omp critical
            {
              visited[v] = 1;
              localNextFrontier[localNextFrontierSize++] = v;
            }
          }
        }
      }

      localNextFrontierSizes[threadNum] = localNextFrontierSize;
    }

    // Merging local buffers into the global next frontier
    int totalSize = 0;
    for (int i = 0; i < numThreads; i++)
    {
      totalSize += localNextFrontierSizes[i];
    }

    int *nextFrontier = (int *)malloc(totalSize * sizeof(int));
    int nextFrontierSize = 0;

    for (int i = 0; i < numThreads; i++)
    {
      for (int j = 0; j < localNextFrontierSizes[i]; j++)
      {
        nextFrontier[nextFrontierSize++] = localNextFrontiers[i][j];
      }
      localNextFrontierSizes[i] = 0; // Reset for next iteration
    }

    free(currentFrontier);
    currentFrontier = nextFrontier;
    currentFrontierSize = nextFrontierSize;
  }

  free(currentFrontier);
  for (int i = 0; i < numThreads; i++)
  {
    free(localNextFrontiers[i]);
  }
  free(localNextFrontiers);
  free(localNextFrontierSizes);
}

void bfsCPU(int start, Graph *G, int *visited)
{
  visited[start] = 1;

  int *queue = (int *)malloc(G->numVertices * sizeof(int)); // Queue
  int front = 0, rear = 0;                                  // Queue front and rear

  // Enqueue start vertex
  queue[rear++] = start;

  while (front < rear)
  {
    int current = queue[front++];
    for (int i = G->edgesOffset[current]; i < G->edgesOffset[current] + G->edgesSize[current]; ++i)
    {
      int v = G->adjacencyList[i];
      if (visited[v] == INT_MAX)
      {
        visited[v] = 1;
        queue[rear++] = v;
      }
    }
  }

  free(queue);
}