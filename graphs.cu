#include "graphs.h"

__host__ void GenerateAllGraphs(unsigned int initial_order, unsigned int final_order, int max_vertex_order, int lattice_type, graph*& graphs)
{
    for( int i = initial_order; i < final_order, i++)
    {
        cudaMalloc( (void**) &graphs[i], ); //put a formula for maximum number of distinct graphs here
        
        GenerateNewGraphs( graphs[i].adj_mat, i, graphs[i].count, graphs[i+1].adj_mat, graphs[i + 1].flags, max_vertex_order, lattice_type);
        
        thrust::device_ptr<int> flag_ptr(graphs[i+1].flags);
        graphs[i+1].count = thrust::reduce(flag_ptr, flag_ptr + );
        
        thrust::sort_by_key(flag_ptr, flag_ptr + , graphs[i].adj_mat);

    }

}

__global__ void GenerateNewGraphs( int* old_graphs, int old_graph_order, unsigned int num_old_graphs, int* new_graphs, unsigned int* new_graphs_flags, int max_vertex_order, int lattice_type)
{
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = threadIdx.x % old_graph_order; 
    int2 vertex_order;
    int matrix_size = 0;
    int row_start = 0;
    unsigned int block_start = matrix_size * (threadIdx.x / old_graph_order);

    for (int i = 0; i < old_graph_order; i++)
    {
        matrix_size += i;
        row_start += row > i ? (old_graph_order - i) : 0;
    }

    __shared__ int s_old_graphs[1024];
    __shared__ int s_new_graphs[1024];

    if (gid < num_old_graphs * old_graph_order)
    {
        for( int i = 0; i < old_graph_order - row; i++)
        {
            s_old_graphs[ block_start + row_start + i ] = old_graphs[ (gid / old_graph_order) * matrix_size + row_start + i ];
        }

        vertex_order.x = s_old_graphs[ block_start + row_start ];
    
    }
    __syncthreads();

    int new_matrix_size = matrix_size + old_graph_order + 1;
    int new_matrix_start = (gid / old_graph_order)*new_matrix_size*(matrix_size - old_graph_order);
    int new_row_start = 0;

    if (gid < num_old_graphs * old_graph_order)
    {
        for( int i = 1; i < old_graph_order - row; i++)
        {
            for( int k = 0; k < old_graph_order; k++)
            {
                new_row_start += (i > k) ? k : 0;
            }
            
            vertex_order.y = s_old_graphs[block_start + new_row_start];

            for ( int j = 0; j < new_matrix_size; j++)
            {
                new_graphs[ new_matrix_start + i*new_matrix_size + j ] = ( j == row_start + i) ? 1 : s_old_graphs[block_start + j];
            }
            
            vertex_order.x += 1;
            vertex_order.y += 1;
            new_graphs_flags[ (gid/ ( old_graphs_order + 1 ) )*(matrix_size - old_graph_order) ] = (vertex_order.x <= max_vertex_order + 1) && (vertex_order.y <= max_vertex_order + 1);
        }
    }
}

__global__ void FindCanonicalGraphs( int* graphs, int graph_order )
{

}
