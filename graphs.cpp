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
    GenerateAllGraphs( clusters, 2, 4, 0);
    //cout<<clusters[3].count<<endl;
    PrintGraphs( clusters, 1, 4);
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
        if( i > 2 )
        {
            FindCanonicalGraphs(clusters, i);
            CheckEmbeddableSquare(clusters, i);
        }

    }
}

void GenerateNewGraphs(cluster & oldGraphs, cluster & newGraphs, int lattice_type )
{
    
    newGraphs.count = 0;
    newGraphs.order = oldGraphs.order + 1;
    newGraphs.max_vertex_order = oldGraphs.max_vertex_order;

    for(unsigned int i = 0; i < oldGraphs.count; i++) //look at each old graph
    {
        vector< pair<int,int> >& oldList = oldGraphs.adj_mat[i];
        
        for( int currentSite = 0; currentSite < oldGraphs.order; currentSite++) //look at each row
        {
            int currentRowStart = 0;

            while( oldList[currentRowStart].first < currentSite)
            {
                currentRowStart++;
            }
            
            int currentVertexOrder = 0;
            
            for(unsigned int l = 0; l < oldList.size(); l++ )
            {
                if (oldList[l].first == currentSite || oldList[l].second == currentSite)
                {
                    currentVertexOrder++;
                }
            }
            
            if ( currentVertexOrder < oldGraphs.max_vertex_order)
            {
                for (int nextSite = currentSite + 1; nextSite < oldGraphs.order + 1; nextSite++) //look at each edge
                { 

                    int nextRowStart = 0;
                    
                    while( oldList[ nextRowStart ].first < nextSite) //find out where the pairs for the next site start
                    {
                        nextRowStart++;
                    }

                    int insertIndex = 1; 
                    bool dupeFlag = false; //check to make sure edge between currentSite and nextSite doesn't already exist
                    int nextVertexOrder = 0;

                    for( unsigned int currentPair = 0; currentPair < oldList.size(); currentPair++ )
                    {
                        if ( oldList[currentPair].first == nextSite || oldList[currentPair].second == nextSite )
                        {
                            nextVertexOrder++;
                        }
                        
                        if ( ( oldList[currentPair].first == currentSite && oldList[currentPair].second < nextSite ) || 
                             ( oldList[currentPair].first == currentSite && oldList[ currentPair + 1 ].first > currentSite )
                           )
                        {
                            insertIndex = currentPair + 1;
                        }

                        if ( oldList[currentPair].first == currentSite && oldList[currentPair].second == nextSite )
                        {
                            dupeFlag = true;
                        }
                    }
                
                    if (nextVertexOrder < oldGraphs.max_vertex_order && !dupeFlag)
                    {
                        newGraphs.count++;
                        newGraphs.adj_mat.push_back(oldList);
                        
                        pair<int,int> tempPair = make_pair(currentSite,nextSite);
                        
                        newGraphs.adj_mat[newGraphs.count - 1].insert( newGraphs.adj_mat[newGraphs.count - 1].begin() + insertIndex, temp_pair);
                        
                        std::sort( newGraphs.adj_mat[newGraphs.count - 1].begin(),  newGraphs.adj_mat[newGraphs.count - 1].end(), GraphSort);
                        currentVertexOrder++;
                    }
                }
            }
        }
    }
}

void FindCanonicalGraphs( vector<cluster>& clusters, int graph_order)
{
    cluster clusterContainer = clusters[graph_order - 1];
    unsigned int trueCount = 0;
    
    for(unsigned int i = 0; i < clusterContainer.count; i++)
    {
        //Allocate space to store graph information for nauty
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
        for( unsigned int currentVertex = 0; currentVertex < graph_order; currentVertex++ )
        {
            
            gv = GRAPHROW(current, currentVertex, m);
            EMPTYSET(gv, m);

            for( unsigned int count = 0; count < clusterContainer.adj_mat[i].size(); count++)
            {
                if( clusterContainer.adj_mat[i][count].first == currentVertex)
                {
                    ADDELEMENT(gv, clusterContainer.adj_mat[i][count].second);
                }
                if( clusterContainer.adj_mat[i][count].second == currentVertex)
                {
                    ADDELEMENT(gv, clusterContainer.adj_mat[i][count].first);
                }
            }
        }
        
        //Define options for nauty - get the automorphisms, and write them out to a file 
        ops.getcanon = TRUE;
        ops.defaultptn = TRUE;  
        ops.writeautoms = TRUE; 
        ops.writemarkers = FALSE;
        ops.cartesian = TRUE;
        ops.outfile = autos;
        statsblk stat;

        nauty(current, label, ptn, NULL, orbit, &ops, &stat, workspace, 5*m, m, graph_order, canonical);
        fclose(autos);

        //---------Eliminate isomorphic graphs--------//

        std::string automorph = getFileContents("automorphism");

        stringstream iss (automorph);
        
        char buff[2*graph_order];
        int group[graph_order];
        
        for(int charCount = 0; charCount < automorph.length(); charCount += 2*graph_order + 1)
        {
            iss.getline(buff, 2*graph_order); 
            for(int j = 0; j < graph_order; j++)
            { 
                group[j] = buff[2*j];
            }
            
            vector< pair<int,int> > isograph = clusterContainer.adj_mat[i];
            
            for(int oldSite = 0; oldSite < clusterContainer.adj_mat[i].size(); oldSite+=2)
            {
                isograph.insert( isograph.begin() + oldSite, make_pair( isograph[oldSite].second, isograph[oldSite].first ) );
            }
            
            for(unsigned int currentSite = 0; currentSite < isograph.size(); currentSite++)
            {
                isograph.at(currentSite).first = group[ isograph.at(currentSite).first ];
                isograph.at(currentSite).second = group[ isograph.at(currentSite).second ];
                if ( isograph.at(currentSite).first > isograph.at(currentSite).second )
                {
                    isograph.erase(isograph.begin() + currentSite);
                }
            }
            sort(isograph.begin(), isograph.end());
            
            vector< vector< pair<int,int> > >::iterator dupIndex = find(clusterContainer.adj_mat.begin(), clusterContainer.adj_mat.end(), isograph);
            if ( dupIndex < clusterContainer.adj_mat.end())
            {
                clusterContainer.adj_mat.erase(dupIndex);
            }
            trueCount++;
        }
    }
}

void CheckEmbeddableSquare( vector<cluster>& clusters, int order )
{
    for( unsigned int i = 0; i < clusters[order].count; i++)
    {
        int* colour = (int*)malloc(order*sizeof(int));
        colour[0] = 1;
        int checkIndex = 0;
        bool embeddable = true;
        for(int currentSite = 0; currentSite < order; currentSite++)
        {
        
            vector< pair<int,int> >& currentGraph = clusters[order].adj_mat[i];
            while( checkIndex < clusters[order].adj_mat[i].size() && clusters[order].adj_mat[i][checkIndex].first == currentSite )
            {
                if ( colour[currentGraph[checkIndex].second] == 0)
                {
                    colour[currentGraph[checkIndex].second] = colour[currentSite] == 1 ? 2 : 1;
                    checkIndex++;
                }

                else
                {
                    if (colour[currentGraph[checkIndex].second] == colour[currentSite] )
                    {
                        embeddable = false;
                    }
                    checkIndex++;

                }

            }
        }
        free(colour);
        if (!embeddable)
        {
            clusters[order].adj_mat.erase(clusters[order].adj_mat.begin() + i);
            clusters[order].count--;
            i--;
        }
    }
}

std::string getFileContents(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    return std::string(std::istreambuf_iterator<char>(file),
                        std::istreambuf_iterator<char>());
}
