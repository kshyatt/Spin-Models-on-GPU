#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include "nauty.h"

using namespace std;

struct cluster
{
    vector< vector<pair <int,int> > > adj_mat;
    vector<int*> flag;
    unsigned int count;
    int order;
    int max_vertex_order;
};

void GenerateAllGraphs( vector<cluster>& clusters, unsigned int initial_order, unsigned int final_order, int lattice_type);

bool GraphSort( pair<int,int> i, pair<int,int> j);

void PrintGraphs( vector<cluster>& clusters, unsigned int initial_order, unsigned int final_order);

void GenerateNewGraphs( cluster& old_graphs, cluster& new_graphs, int lattice_type );

void FindCanonicalGraphs( vector<cluster>& cluster, int graph_order);
