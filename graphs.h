#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
extern "C" {
#include "nauty/nauty.h"
}

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

void FindCanonicalGraphs( vector<cluster>& clusters, int graph_order);

void CheckEmbeddableSquare( vector<cluster>& clusters, int graph_order);

std::string getFileContents(const std::string& filename);
