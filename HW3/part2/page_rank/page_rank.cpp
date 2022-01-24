#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i){
      solution[i] = equal_prob;
  }
  bool converged = false;
  double* new_sol = (double*)malloc(sizeof(double) * numNodes);

  while(!converged){
      double sum = 0;
      #pragma omp parallel for reduction (+:sum)
      for(int j = 0; j < numNodes; ++j){
          if(outgoing_size(g, j) == 0){
              sum += damping * solution[j] / numNodes;
          }
      }

      double global_diff = 0;
      #pragma omp parallel for
      for(int i = 0; i < numNodes; ++i){
          new_sol[i] = 0;
          const Vertex* start = incoming_begin(g, i);
          const Vertex* end = incoming_end(g, i);
          for (const Vertex* v=start; v!=end; v++){
              new_sol[i] += (solution[*v] / outgoing_size(g, *v));
          }
          new_sol[i] = (damping * new_sol[i]) + (1.0-damping) / numNodes + sum;
      }
      #pragma omp parallel for reduction(+:global_diff)
      for(int i = 0; i < numNodes; ++i){
          global_diff += abs(new_sol[i] - solution[i]);
          solution[i] = new_sol[i];
      }
      converged = (global_diff < convergence);
  }
  delete new_sol;
}
