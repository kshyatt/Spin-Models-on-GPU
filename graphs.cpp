#include "graphs.h"

int main()
{
    vector<graph> clusters;
    clusters.resize(3);

    clusters[0].adj_mat.resize(1);
    clusters[0].adj_mat[0].resize(1);
    clusters[0].adj_mat[0][0] = make_pair(0,0);
    clusters[0].count = 1;
    clusters[0].max_vertex_order = 4;
    clusters[0].order = 1;
    
    clusters[1].adj_mat.resize(1);
    clusters[1].adj_mat[0].resize(1);
    clusters[1].adj_mat[0][0] = make_pair(0,1);
    clusters[1].count = 1;
    clusters[1].max_vertex_order = 4;
    clusters[1].order = 2;
    
    clusters[2].adj_mat.resize(2);
    clusters[2].adj_mat[0].resize(2);
    clusters[2].adj_mat[1].resize(2);
    clusters[2].adj_mat[0][0] = make_pair(0,1);
    clusters[2].adj_mat[0][1] = make_pair(0,2);
    clusters[2].adj_mat[1][0] = make_pair(0,1);
    clusters[2].adj_mat[1][1] = make_pair(1,2);
    clusters[2].count = 2;
    clusters[2].max_vertex_order = 4;
    clusters[2].order = 3;
    //PrintGraphs( clusters, 1, 1);
    GenerateAllGraphs( clusters, 3, 5, 0);
    cout<<clusters[3].count<<endl;
    PrintGraphs( clusters, 1, 4);
    return 0;
}

void PrintGraphs( vector<graph>& clusters, unsigned int initial_order, unsigned int final_order)
{
    for(unsigned int i = initial_order - 1; i <= final_order; i++)
    {
        cout<<"Current order: "<<i + 1<<" ------------"<<endl;
        for (unsigned int j = 0; j < clusters[i].count; j++)
        {
            for (unsigned int k = 0; k < clusters[i].adj_mat[j].size(); k++)
            {
                cout<<"( "<<clusters[i].adj_mat[j][k].first<<", "<<clusters[i].adj_mat[j][k].second<<")"<<endl;
            }
            cout<<endl;
        }
    }
}

void GenerateAllGraphs( vector<graph>& clusters, unsigned int initial_order, unsigned int final_order, int lattice_type )
{
    //cout<< clusters[0].adj_mat[0][0].second<<endl;
    for (unsigned int i = initial_order; i <= final_order; i++)
    {
        clusters.resize(clusters.size() + 1);
        GenerateNewGraphs(clusters.at(i-1), clusters.at(i), lattice_type);

    }
}

void GenerateNewGraphs(graph & old_graphs, graph & new_graphs, int lattice_type )
{
    //int matrix_size = 0;
    new_graphs.count = 0;
    new_graphs.order = old_graphs.order + 1;

    /*for (int i = 0; i < old_graphs.order; i++)
    {
        matrix_size += i;
    }*/

    //unsigned int total_new = old_graphs.count * matrix_size;

    //new_graphs.adj_mat.resize(total_new);

    for(unsigned int i = 0; i < old_graphs.count; i++) //look at each old graph
    {
        std::vector< std::pair<int,int> >& old_list = old_graphs.adj_mat[i];
        for( int j = 0; j < old_graphs.order; j++) //look at each row
        {
            int current_row_start = 0;
            while( old_list[current_row_start].first < j)
            {
                current_row_start++;
            }
            
            int current_vertex_order = 0;
            for(unsigned int l = 0; l < old_list.size(); l++ )
            {
                if (old_list[l].first == j || old_list[l].second == j)
                {
                    current_vertex_order++;
                }
            }
            
            if( current_vertex_order < old_graphs.max_vertex_order)
            {
                for (int k = j + 1; k < old_graphs.order + 1; k++) //look at each edge
                { 

                    int next_row_start = 0;
                    while( old_list[next_row_start].first < k)
                    {
                        next_row_start++;
                    }
                    int insert_index = 0;
                    bool dupe_flag = false;
                    int next_vertex_order = 0;
                    for(unsigned int l = 0; l < old_list.size(); l++ )
                    {
                        if (old_list[l].first == k || old_list[l].second == k)
                        {
                            next_vertex_order++;
                        }
                        if ((old_list[l].first == j && old_list[l].second < k) || (old_list[l].first == j && old_list[l+1].first > j))
                        {
                            insert_index = l+1;
                        }
                        if (old_list[l].first == j && old_list[l].second == k)
                        {
                            dupe_flag = true;
                        }
                    }
                
                    if (next_vertex_order < old_graphs.max_vertex_order && !dupe_flag)
                    {
                        new_graphs.count++;
                        //new_graphs.adj_mat.resize(new_graphs.count);
                        //new_graphs.adj_mat.at(new_graphs.count - 1) = old_graphs.adj_mat.at(i);
                        new_graphs.adj_mat.push_back(old_list);
                        pair<int,int> temp_pair = make_pair(j,k);
                        new_graphs.adj_mat[new_graphs.count - 1].insert( new_graphs.adj_mat[new_graphs.count - 1].begin() + insert_index , temp_pair);
                    }
                }
            }
        }
    }
}
