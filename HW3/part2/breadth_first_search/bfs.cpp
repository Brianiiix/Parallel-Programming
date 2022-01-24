#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    int local_count = 0;

    #pragma omp parallel firstprivate(local_count)
    {
        int* local_vertices = (int *)malloc(sizeof(int) * g->num_nodes);
        #pragma omp for
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)? g->num_edges: g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                {
                    //int idx = __sync_add_and_fetch(&new_frontier->count, 1);
                    //int idx = local_count++;
                    //new_frontier->vertices[idx-1] = outgoing;
                    local_vertices[local_count++] = outgoing;
                }
            }
        }

        #pragma omp critical
        {
            memcpy(new_frontier->vertices + new_frontier->count, local_vertices, sizeof(int) * local_count);
            new_frontier->count += local_count;
        }
        free(local_vertices);
    }
    //new_frontier->count = local_count-1;
}

// Implement top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int depth)
{
    int local_count = 0;
    int* local_vertices;

    #pragma omp parallel firstprivate(local_count) private(local_vertices)
    {
        local_vertices = (int *)malloc(sizeof(int) * g->num_nodes);
        #pragma omp for
        for (int i = 0; i < g->num_nodes; i++)
        {
            if (distances[i] == NOT_VISITED_MARKER)
            {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes - 1)? g->num_edges: g->incoming_starts[i + 1];

                for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
                {
                    int incoming = g->incoming_edges[neighbor];

                    if (distances[incoming] == depth)
                    {
                        distances[i] = distances[incoming] + 1;
                        local_vertices[local_count++] = i;
                        break;
                    }
                }
            }
        }

        #pragma omp critical
        {
            memcpy(new_frontier->vertices + new_frontier->count, local_vertices, sizeof(int) * local_count);
            new_frontier->count += local_count;
        }
        free(local_vertices);
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;
    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, depth++);

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;
    int state = 0;
    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);

        /*
        if (state == 0){
            int mf = 0;
            #pragma omp parallel for reduction(+:mf)
            for (int i = 0; i < frontier->count; i++){
                mf += outgoing_size(graph, frontier->vertices[i]);
            }
            int mu = 0;
            #pragma omp parallel for reduction(+:mu)
            for (int i = 0; i < graph->num_nodes; i++){
                if (sol->distances[i] == NOT_VISITED_MARKER){
                    mu ++;
                }
            }
            if (mf > mu / 14){
                bottom_up_step(graph, frontier, new_frontier, sol->distances, depth++);
                state = 1;
            }
            else{
                top_down_step(graph, frontier, new_frontier, sol->distances);
                depth++;
            }
        }
        else{
            if (frontier->count > graph->num_nodes / 24){
                top_down_step(graph, frontier, new_frontier, sol->distances);
                state = 0;
                depth++;
            }
            else{
                bottom_up_step(graph, frontier, new_frontier, sol->distances, depth++);
            }
        }*/
        if (frontier->count > graph->num_nodes / 24){
            bottom_up_step(graph, frontier, new_frontier, sol->distances, depth++);
        }
        else{
            top_down_step(graph, frontier, new_frontier, sol->distances);
            depth++;
        }
        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
