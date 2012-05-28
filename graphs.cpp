#include "graphs.h"

int main()
{
    vector<cluster> clusters;
    clusters.resize(2);

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
    
    /*clusters[2].adj_mat.resize(2);
    clusters[2].adj_mat[0].resize(2);
    clusters[2].adj_mat[1].resize(2);
    clusters[2].adj_mat[0][0] = make_pair(0,1);
    clusters[2].adj_mat[0][1] = make_pair(0,2);
    clusters[2].adj_mat[1][0] = make_pair(0,1);
    clusters[2].adj_mat[1][1] = make_pair(1,2);
    clusters[2].count = 2;
    clusters[2].max_vertex_order = 4;
    clusters[2].order = 3;
    */
    //PrintGraphs( clusters, 1, 1);
    GenerateAllGraphs( clusters, 2, 5, 0);
    //cout<<clusters[3].count<<endl;
    //PrintGraphs( clusters, 1, 5);
    return 0;
}

bool GraphSort( pair<int, int> i , pair<int, int> j)
{
    return (i.first == j.first) ? (i.second < j.second) : (i.first < j.first);
}

void PrintGraphs( vector<cluster>& clusters, unsigned int initial_order, unsigned int final_order)
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

void GenerateAllGraphs( vector<cluster>& clusters, unsigned int initial_order, unsigned int final_order, int lattice_type )
{
    //cout<< clusters[0].adj_mat[0][0].second<<endl;
    for (unsigned int i = initial_order; i <= final_order; i++)
    {
        clusters.resize(clusters.size() + 1);
        GenerateNewGraphs(clusters.at(i-1), clusters.at(i), lattice_type);
        cout<<"Automorphisms for order "<<i<<endl;
        if( i > 2 )FindCanonicalGraphs(clusters, i);

    }
}

void GenerateNewGraphs(cluster & old_graphs, cluster & new_graphs, int lattice_type )
{
    //int matrix_size = 0;
    new_graphs.count = 0;
    new_graphs.order = old_graphs.order + 1;
    new_graphs.max_vertex_order = old_graphs.max_vertex_order;
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
                    int insert_index = 1;
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
                        
                        new_graphs.adj_mat[new_graphs.count - 1].insert( new_graphs.adj_mat[new_graphs.count - 1].begin() + insert_index, temp_pair);
                        
                        std::sort( new_graphs.adj_mat[new_graphs.count - 1].begin(),  new_graphs.adj_mat[new_graphs.count - 1].end(), GraphSort);
                    }
                }
            }
        }
    }
}

void FindCanonicalGraphs( vector<cluster>& clusters, int graph_order)
{
    cluster cluster_container = clusters[graph_order - 1];
    unsigned int truecount = 0;
    for(unsigned int i = 0; i < cluster_container.count; i++)
    {
        cout<<"Cluster "<<i<<" of order "<<graph_order<<endl;
        DYNALLSTAT(graph, current, graph_size);
        DYNALLSTAT(graph, canonical, canonical_size);
        DYNALLSTAT(int, label, label_size);
        DYNALLSTAT(int, orbit, orbit_size);
        DYNALLSTAT(int, ptn, patt_size);
        DYNALLSTAT(setword, workspace, work_size);

        static DEFAULTOPTIONS_GRAPH(ops);

        int m = 1;
        nauty_check(WORDSIZE, m, graph_order, NAUTYVERSIONID); 

        set *gv;

        DYNALLOC2(graph, current, graph_size, m, graph_order, "malloc");
        DYNALLOC2(graph, canonical, canonical_size, m, graph_order, "malloc");
        DYNALLOC1(setword, workspace, work_size, 5*m, "malloc");
        DYNALLOC1(int, label, label_size, graph_order, "malloc");
        DYNALLOC1(int, orbit, orbit_size, graph_order, "malloc");
        DYNALLOC1(int, ptn, patt_size, graph_order, "malloc");

        FILE* autos;
        char* mode = "w";
        autos = fopen("automorphism", mode);

        //unsigned int* current = new unsigned int[graph_order];
        for( unsigned int j = 0; j < graph_order; j++ )
        {
            
            gv = GRAPHROW(current, j, m);
            EMPTYSET(gv, m);

            for( unsigned int count = 0; count < cluster_container.adj_mat[i].size(); count++)
            {
                if( cluster_container.adj_mat[i][count].first == j)
                {
                    ADDELEMENT(gv, cluster_container.adj_mat[i][count].second);
                }
                if( cluster_container.adj_mat[i][count].second == j)
                {
                    ADDELEMENT(gv, cluster_container.adj_mat[i][count].first);
                }
            }
        }
        //optionblk ops;
        ops.getcanon = TRUE;
        ops.defaultptn = TRUE;  
        ops.writeautoms = TRUE; 
        ops.writemarkers = FALSE;
        ops.cartesian = TRUE;
        ops.outfile = autos;
        statsblk stat;
        //setword* workspace = new setword[5*graph_order];
        //graph* canonical = new graph[graph_order*MAXM];

        nauty(current, label, ptn, NULL, orbit, &ops, &stat, workspace, 5*m, m, graph_order, canonical);
        fclose(autos);

        //---------Eliminate isomorphic graphs--------//

        std::string automorph = getFileContents("automorphism");
        stringstream iss (automorph);
        char buff[2*graph_order];
        int group[graph_order];
        for(int charcount = 0; charcount < automorph.length(); charcount += 2*graph_order + 1)
        {
            iss.getline(buff, 2*graph_order); 
            for(int j = 0; j < graph_order; j++)
            { 
                group[j] = buff[2*j];
            }
            vector< pair<int,int> > isograph = cluster_container.adj_mat[i];
            for(int j = 0; j < cluster_container.adj_mat[i].size(); j+=2)
            {
                isograph.insert(isograph.begin() + j, make_pair(isograph[j].second, isograph[j].first));
            }
            
            for(unsigned int j = 0; j<isograph.size(); j++)
            {
                isograph.at(j).first = group[isograph.at(j).first];
                isograph.at(j).second = group[isograph.at(j).second];
                if ( isograph.at(j).first > isograph.at(j).second )
                {
                    isograph.erase(isograph.begin() + j);
                }
            }
            sort(isograph.begin(), isograph.end());
            
            vector< vector< pair<int,int> > >::iterator dup_index = find(cluster_container.adj_mat.begin(), cluster_container.adj_mat.end(), isograph);
            if ( dup_index < cluster_container.adj_mat.end())
            {
                cluster_container.adj_mat.erase(dup_index);
            }
            truecount++;
        }
    }
    cout<<truecount<<endl;   
}

std::string getFileContents(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    return std::string(std::istreambuf_iterator<char>(file),
                        std::istreambuf_iterator<char>());
}
