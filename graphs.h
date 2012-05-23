#include <iostream>
#include <vector>
#include <utility>

using namespace std;

struct graph
{
    vector< vector<pair <int,int> > > adj_mat;
    vector<int*> flag;
    unsigned int count;
    int order;
    int max_vertex_order;
};

void GenerateAllGraphs( vector<graph>& clusters, unsigned int initial_order, unsigned int final_order, int lattice_type);

void PrintGraphs( vector<graph>& clusters, unsigned int initial_order, unsigned int final_order);

void GenerateNewGraphs( graph& old_graphs, graph& new_graphs, int lattice_type );

void FindCanonicalGraphs( int* graphs, int graph_order);
