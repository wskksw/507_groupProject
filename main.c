#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test.h"
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

int bfsCPU(int start, Graph *G, int *distance, int *visited)
{
  for (int i = 0; i < G->numVertices; i++)
  {
    distance[i] = INT_MAX;
    visited[i] = 0;
  }
  distance[start] = 0;

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
      if (distance[v] == INT_MAX)
      {
        distance[v] = distance[current] + 1;
        queue[rear++] = v;
      }
    }
  }

  free(queue);
  return distance;
}
int ompBFS(int start, Graph *G, int *distance); // Make sure this is declared
//
//int main()
//{
////  int *distanceCPU, *distanceOpenMP;
//  int numVertices = 20000; // Example number of vertices
//  Graph *myGraph = initGraph(numVertices, Dense);
//  printf("Graph with %d vertices and %d edges\n", myGraph->numVertices, myGraph->numEdges);
//  // printf("Graph's Adjacency List:\n");
//  // printGraph(myGraph);
//
//  // Allocate memory for BFS distance and visited arrays
//  int *distance = (int *)malloc(numVertices * sizeof(int));
//  int *distanceCPU = (int *)malloc(numVertices * sizeof(int));
//  int *distanceOpenMP = (int *)malloc(numVertices * sizeof(int));
//  int *visited = (int *)malloc(numVertices * sizeof(int));
//
//  // Timing variables
//  clock_t startSerial, endSerial, startParallel, endParallel;
//
//  // Perform standard BFS
//  startSerial = clock();
//  distanceCPU = bfsCPU(0, myGraph, distance, visited);
//  endSerial = clock();
//  double timeTakenSerial = (double)(endSerial - startSerial) / CLOCKS_PER_SEC;
//  printf("Standard BFS took %f seconds.\n", timeTakenSerial);
//
//
//
//
//  // printf("Distances from vertex 0:\n");
//  // for (int i = 0; i < numVertices; i++)
//  // {
//  //   printf("Vertex %d: %d\n", i, distance[i]);
//  // }
//
//  // Reset distance and visited for ompBFS
//  for (int i = 0; i < numVertices; i++)
//  {
//    distance[i] = INT_MAX;
//    visited[i] = 0;
//  }
//
////   Perform OpenMP BFS
//  startParallel = clock();
//  distanceOpenMP = ompBFS(0, myGraph, distance);
//  endParallel = clock();
//  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
//  printf("OpenMP BFS took %f seconds.\n", timeTakenParallel);

  // Print distances from vertex 0 (from ompBFS)
  // printf("Distances from vertex 0:\n");
  // for (int i = 0; i < numVertices; i++)
  // {
  //   printf("Vertex %d: %d\n", i, distance[i]);
  // }

//    printf("Distances from vertex %d:\n", start);
//  	int count=0;
//    for (int i = 0; i < myGraph->numVertices; ++i) {
//    	if(distanceCPU[i]!=distanceOpenMP[i]){
//    		printf("Error at Vertex %d:\n", i);
//    		count+=1;
//    	}
//    	else{
//    		printf("Vertex %d: %d, %d\n", i, distanceCPU[i], distanceOpenMP[i]);
//    	}
//    }
//    if(count==0){
//    	printf("Perfect \n");
//    }
//
//
//
//  // Free allocated resources
//  free(distance);
//  free(visited);
//  freeGraph(myGraph);
//
//  return 0;
//}


//Tests

static char *  test1 (){

	  printf("Running test1 for Sparse Graph \n");
	  int n=0;
	  int verticeArr[] = {10, 50, 100, 500, 1000, 5000, 10000, 20000};
	  int numVertices = 20000; // Example number of vertices
	  while(n<8){
		  Graph *myGraph = initGraph(verticeArr[n], Sparse);
		  printf("Graph with %d vertices and %d edges\n", myGraph->numVertices, myGraph->numEdges);
		  // printf("Graph's Adjacency List:\n");
		  // printGraph(myGraph);

		  // Allocate memory for BFS distance and visited arrays
		  int *distance = (int *)malloc(numVertices * sizeof(int));
		  int *distanceCPU = (int *)malloc(numVertices * sizeof(int));
		  int *distanceOpenMP = (int *)malloc(numVertices * sizeof(int));
		  int *visited = (int *)malloc(numVertices * sizeof(int));
		  // Timing variables
		  clock_t startSerial, endSerial, startParallel, endParallel;

		  // Perform standard BFS
		  startSerial = clock();
		  distanceCPU = bfsCPU(0, myGraph, distance, visited);
		  endSerial = clock();
		  double timeTakenSerial = (double)(endSerial - startSerial) / CLOCKS_PER_SEC;
		  printf("Standard BFS took %f seconds.\n", timeTakenSerial);
		  // printf("Distances from vertex 0:\n");
		  // for (int i = 0; i < numVertices; i++)
		  // {
		  //   printf("Vertex %d: %d\n", i, distance[i]);
		  // }

		  // Reset distance and visited for ompBFS
		  for (int i = 0; i < numVertices; i++)
		  {
			distance[i] = INT_MAX;
			visited[i] = 0;
		  }

		  // Perform OpenMP BFS
		  startParallel = clock();
		  distanceOpenMP = ompBFS(0, myGraph, distance);
		  endParallel = clock();
		  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
		  printf("OpenMP BFS took %f seconds.\n", timeTakenParallel);
//
//		  // Print distances from vertex 0 (from ompBFS)
//		  // printf("Distances from vertex 0:\n");
//		  // for (int i = 0; i < numVertices; i++)
//		  // {
//		  //   printf("Vertex %d: %d\n", i, distance[i]);
		  for (int i = 0; i < myGraph->numVertices; ++i) {
		 				if(distanceCPU[i]!=distanceOpenMP[i]){
		   					printf("Error at Vertex %d:\n", i);
		   //					count+=1;
		 					return 1;
		 				}
		 }

		  // Free allocated resources
		  free(distance);
		  free(distanceCPU);
		  free(distanceOpenMP);
		  free(visited);
		  freeGraph(myGraph);
		  n+=1;
	  }
	  return 0;

}



static char *  test2 (){

	  printf("Running test2 for Medium Graph \n");
	  int n=0;
	  int verticeArr[] = {10, 50, 100, 500, 1000, 5000, 10000, 20000};
	  int numVertices = 20000; // Example number of vertices
	  while(n<8){
		  Graph *myGraph = initGraph(verticeArr[n], Medium);
		  printf("Graph with %d vertices and %d edges\n", myGraph->numVertices, myGraph->numEdges);
		  // printf("Graph's Adjacency List:\n");
		  // printGraph(myGraph);

		  // Allocate memory for BFS distance and visited arrays
		  int *distance = (int *)malloc(numVertices * sizeof(int));
		  int *distanceCPU = (int *)malloc(numVertices * sizeof(int));
		  int *distanceOpenMP = (int *)malloc(numVertices * sizeof(int));
		  int *visited = (int *)malloc(numVertices * sizeof(int));
		  // Timing variables
		  clock_t startSerial, endSerial, startParallel, endParallel;

		  // Perform standard BFS
		  startSerial = clock();
		  distanceCPU=bfsCPU(0, myGraph, distance, visited);
		  endSerial = clock();
		  double timeTakenSerial = (double)(endSerial - startSerial) / CLOCKS_PER_SEC;
		  printf("Standard BFS took %f seconds.\n", timeTakenSerial);
		  // printf("Distances from vertex 0:\n");
		  // for (int i = 0; i < numVertices; i++)
		  // {
		  //   printf("Vertex %d: %d\n", i, distance[i]);
		  // }

		  // Reset distance and visited for ompBFS
		  for (int i = 0; i < numVertices; i++)
		  {
			distance[i] = INT_MAX;
			visited[i] = 0;
		  }

		  // Perform OpenMP BFS
		  startParallel = clock();
		  distanceOpenMP=ompBFS(0, myGraph, distance);
		  endParallel = clock();
		  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
		  printf("OpenMP BFS took %f seconds.\n", timeTakenParallel);

		  for (int i = 0; i < myGraph->numVertices; ++i) {
		 				if(distanceCPU[i]!=distanceOpenMP[i]){
		   					printf("Error at Vertex %d:\n", i);
		 					return 1;
		 				}
		 }

		  // Free allocated resources
		  free(distance);
		  free(distanceCPU);
		  free(distanceOpenMP);
		  free(visited);
		  freeGraph(myGraph);
		  n+=1;
	  }
	  return 0;

}


static char *  test3 (){

	  printf("Running test3 for Dense Graph \n");
	  int n=0;
	  int verticeArr[] = {10, 50, 100, 500, 1000, 5000, 10000, 20000};
	  int numVertices = 20000; // Example number of vertices
	  while(n<8){
		  Graph *myGraph = initGraph(verticeArr[n], Dense);
		  printf("Graph with %d vertices and %d edges\n", myGraph->numVertices, myGraph->numEdges);
		  // printf("Graph's Adjacency List:\n");
		  // printGraph(myGraph);

		  // Allocate memory for BFS distance and visited arrays
		  int *distance = (int *)malloc(numVertices * sizeof(int));
		  int *distanceCPU = (int *)malloc(numVertices * sizeof(int));
		  int *distanceOpenMP = (int *)malloc(numVertices * sizeof(int));
		  int *visited = (int *)malloc(numVertices * sizeof(int));
		  // Timing variables
		  clock_t startSerial, endSerial, startParallel, endParallel;

		  // Perform standard BFS
		  startSerial = clock();
		  distanceCPU=bfsCPU(0, myGraph, distance, visited);
		  endSerial = clock();
		  double timeTakenSerial = (double)(endSerial - startSerial) / CLOCKS_PER_SEC;
		  printf("Standard BFS took %f seconds.\n", timeTakenSerial);
		  // printf("Distances from vertex 0:\n");
		  // for (int i = 0; i < numVertices; i++)
		  // {
		  //   printf("Vertex %d: %d\n", i, distance[i]);
		  // }

		  // Reset distance and visited for ompBFS
		  for (int i = 0; i < numVertices; i++)
		  {
			distance[i] = INT_MAX;
			visited[i] = 0;
		  }

		  // Perform OpenMP BFS
		  startParallel = clock();
		  distanceOpenMP=ompBFS(0, myGraph, distance);
		  endParallel = clock();
		  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
		  printf("OpenMP BFS took %f seconds.\n", timeTakenParallel);

		  for (int i = 0; i < myGraph->numVertices; ++i) {
				if(distanceCPU[i]!=distanceOpenMP[i]){
  					printf("Error at Vertex %d:\n", i);
					return 1;
				}
		  }

		  // Free allocated resources
		  free(distance);
		  free(distanceCPU);
		  free(distanceOpenMP);
		  free(visited);
		  freeGraph(myGraph);
		  n+=1;
	  }
	  return 0;

}

static char *  test4 (){

	  printf("Running test4 for Dense Graph and check distances \n");
	  int n=0;
//	  int verticeArr[] = {10, 50, 100, 500, 1000, 5000, 10000, 20000};
	  int numVertices = 20000; // Example number of vertices
	  Graph *myGraph = initGraph(numVertices, Sparse);
	  printf("Graph with %d vertices and %d edges\n", myGraph->numVertices, myGraph->numEdges);
	  // printf("Graph's Adjacency List:\n");
	  // printGraph(myGraph);

	  // Allocate memory for BFS distance and visited arrays
	  int *distance = (int *)malloc(numVertices * sizeof(int));
	  int *distanceCPU = (int *)malloc(numVertices * sizeof(int));
	  int *distanceOpenMP = (int *)malloc(numVertices * sizeof(int));
	  int *visited = (int *)malloc(numVertices * sizeof(int));
	  // Timing variables
	  clock_t startSerial, endSerial, startParallel, endParallel;

	  // Perform standard BFS
	  startSerial = clock();
	  distanceCPU = bfsCPU(0, myGraph, distance, visited);
	  endSerial = clock();
	  double timeTakenSerial = (double)(endSerial - startSerial) / CLOCKS_PER_SEC;
	  printf("Standard BFS took %f seconds.\n", timeTakenSerial);
	  // printf("Distances from vertex 0:\n");
	  // for (int i = 0; i < numVertices; i++)
	  // {
	  //   printf("Vertex %d: %d\n", i, distance[i]);
	  // }

	  // Reset distance and visited for ompBFS
	  for (int i = 0; i < numVertices; i++)
	  {
		distance[i] = INT_MAX;
		visited[i] = 0;
	  }

	  // Perform OpenMP BFS
	  startParallel = clock();
	  distanceOpenMP = ompBFS(0, myGraph, distance);
	  endParallel = clock();
	  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
	  printf("OpenMP BFS took %f seconds.\n", timeTakenParallel);
//
//		  // Print distances from vertex 0 (from ompBFS)
//		  // printf("Distances from vertex 0:\n");
//		  // for (int i = 0; i < numVertices; i++)
//		  // {
//		  //   printf("Vertex %d: %d\n", i, distance[i]);
//		  // }
		int count=0;
		int i=0;

		while (i < myGraph->numVertices) {
			if(distanceCPU[i]!=distanceOpenMP[i]){
				printf("Error at Vertex %d:\n", i);
				count+=1;
				return 1;
			}
			else{
				printf("Vertex %d: %d, %d\n", i, distanceCPU[i], distanceOpenMP[i]);
				i++;
			}
		}
		if(count==0){
			printf("Perfect \n");
		}

  // Free allocated resources
  free(distance);
  free(distanceCPU);
  free(distanceOpenMP);
  free(visited);
  freeGraph(myGraph);
  n+=1;

  return 0;


}


static char * all_tests() {
	run_test(test1);
	printf("\n");
	printf("\n");
	run_test(test2);
	printf("\n");
	printf("\n");
	run_test(test3);
	printf("\n");
	printf("\n");
	run_test(test4);
	return 0;
 }

 int main (int argc, char *argv[]) {


  char *result = all_tests();
	if (result != 0) {
	   printf("%s\n", result);
	}
	else {
	   printf("ALL TESTS PASSED\n");
	}
	printf("Tests run: %d\n", tests_run);
    return 0;

}

