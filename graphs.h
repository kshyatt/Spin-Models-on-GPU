
struct graph
{
    int** adj_mat;
    int* flag;
    unsigned int count;
};

__host__ void GenerateAllGraphs(unsigned int initial_order, unsigned int final_order, int max_vertex_order, int lattice_type, graph**& graphs);

__global__ void GenerateNewGraphs( int* old_graphs, int old_graph_order, int* new_graphs, int* new_graphs_flags, int max_vertex_order, int lattice_type );

__global__ void FindCanonicalGraphs( int* graphs, int graph_order);
