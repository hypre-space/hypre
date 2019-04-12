// Helper functions to setup amgdd composite grids

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"   

#ifdef __cplusplus

#include <vector>
#include <map>

using namespace std;

extern "C"
{

#endif

HYPRE_Int
SetupNearestProcessorNeighborsNew( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int *communication_cost );

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int *communication_cost );

#ifdef __cplusplus
}

HYPRE_Int 
FindNeighborProcessorsNew(hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, 
   map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > &send_proc_dofs, 
   map<HYPRE_Int, vector<HYPRE_Int> > starting_dofs, 
   HYPRE_Int level, HYPRE_Int *communication_cost);

HYPRE_Int
FindNeighborProcessors( hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, HYPRE_Int ***add_flag,
   HYPRE_Int **num_starting_nodes, HYPRE_Int ***starting_nodes,
   HYPRE_Int **search_proc_marker,
   HYPRE_Int **num_request_nodes, HYPRE_Int ***request_nodes,
   HYPRE_Int *num_send_procs, HYPRE_Int **send_procs, 
   HYPRE_Int num_neighboring_procs,
   HYPRE_Int *send_proc_array_size,
   HYPRE_Int max_starting_nodes_size
   , HYPRE_Int level, HYPRE_Int iteration, HYPRE_Int *communication_cost );

HYPRE_Int
RecursivelyFindNeighborNodesNew(HYPRE_Int dof_index, HYPRE_Int distance, hypre_ParCompGrid *compGrid, 
   map<HYPRE_Int, HYPRE_Int> &send_dofs, 
   map<HYPRE_Int, HYPRE_Int> &request_dofs );

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, 
   HYPRE_Int *request_nodes, HYPRE_Int *num_request_nodes
   , HYPRE_Int level, HYPRE_Int iteration, HYPRE_Int proc );




HYPRE_Int
SetupNearestProcessorNeighborsNew( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int *communication_cost )
{
   HYPRE_Int               i,j,cnt;
   HYPRE_Int               num_nodes = hypre_ParCSRMatrixNumRows(A);
   hypre_ParCSRCommPkg     *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               start,finish;

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Get the default (distance 1) number of send and recv procs
   HYPRE_Int      num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
   HYPRE_Int      num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);

   // If num_sends and num_recvs are zero, then simply note that in compGridCommPkg and we are done
   if (num_sends == 0 && num_recvs == 0)
   {
      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] = 0;
      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level] = 0;
      hypre_ParCompGridCommPkgNumPartitions(compGridCommPkg)[level] = 0;
   }
   else
   {
      // Initialize add_flag (this is how we will track nodes to send to each proc until routine finishes)
      map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > send_proc_dofs; // send_proc_dofs[send_proc] = send_dofs, send_dofs[dof_index] = distance value
      map<HYPRE_Int, vector<HYPRE_Int> > starting_dofs; // starting_dofs[send_proc] = vector of starting dofs for searching through stencil
      for (i = 0; i < num_sends; i++)
      {
         send_proc_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)]; // initialize the send procs as the keys in the outer map
         starting_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)];
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         for (j = start; j < finish; j++)
         {
            send_proc_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)][hypre_ParCSRCommPkgSendMapElmt(commPkg,j)] = padding[level] + num_ghost_layers;
            starting_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)].push_back(hypre_ParCSRCommPkgSendMapElmt(commPkg,j));
         }
      }

      // Iteratively communicate with longer and longer distance neighbors to grow the communication stencils
      for (i = 0; i < padding[level] + num_ghost_layers - 1; i++)
      {
         FindNeighborProcessorsNew(compGrid, A, send_proc_dofs, starting_dofs, level, communication_cost);
      }
   }
   // !!! Finish

   return 0;
}

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int *communication_cost )
{
   HYPRE_Int               i,j,cnt;
   HYPRE_Int               num_nodes = hypre_ParCSRMatrixNumRows(A);
   hypre_ParCSRCommPkg     *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               start,finish;

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Get the default (distance 1) number of send and recv procs
   HYPRE_Int      num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
   HYPRE_Int      num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);

   // If num_sends and num_recvs are zero, then simply note that in compGridCommPkg and we are done
   if (num_sends == 0 && num_recvs == 0)
   {
      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] = 0;
      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level] = 0;
      hypre_ParCompGridCommPkgNumPartitions(compGridCommPkg)[level] = 0;
   }
   else
   {
      // Initialize add_flag (this is how we will track nodes to send to each proc until routine finishes)
      // Note: several allocations occur below for arrays that are meant to store objects/info for each proc that we end up sending to
      // This number is unknown a priori, so we start with double the number of initial send procs and reallocate as necessary inside FindNeighborProcessors()
      HYPRE_Int      send_proc_array_size = 2*num_sends;
      HYPRE_Int      *send_procs = hypre_CTAlloc( HYPRE_Int, send_proc_array_size, HYPRE_MEMORY_HOST);
      HYPRE_Int      **add_flag = hypre_CTAlloc( HYPRE_Int*, send_proc_array_size, HYPRE_MEMORY_HOST);
      HYPRE_Int      *search_proc_marker = hypre_CTAlloc(HYPRE_Int, send_proc_array_size, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         send_procs[i] = hypre_ParCSRCommPkgSendProc(commPkg,i);
         add_flag[i] = hypre_CTAlloc( HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST); // !!! Kind of a costly allocation, I guess... on the finest grid, this allocates num_sends*(num dofs on the fine grid) in all... don't currently know a better way of doing this though
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         for (j = start; j < finish; j++) add_flag[i][ hypre_ParCSRCommPkgSendMapElmt(commPkg,j) ] = padding[level] + num_ghost_layers; // must be set to padding + numGhostLayers (note that the starting nodes are already distance 1 from their neighbors on the adjacent processor)
      }


      // Setup initial num_starting_nodes and starting_nodes (these are the starting nodes when searching for long distance neighbors) !!! I don't think I actually have a good upper bound on sizes here... how to properly allocate/reallocate these? !!!
      HYPRE_Int *num_starting_nodes = hypre_CTAlloc( HYPRE_Int, send_proc_array_size, HYPRE_MEMORY_HOST );
      HYPRE_Int **starting_nodes = hypre_CTAlloc( HYPRE_Int*, send_proc_array_size, HYPRE_MEMORY_HOST );
      HYPRE_Int max_starting_nodes_size = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         search_proc_marker[i] = 1;
         num_starting_nodes[i] = finish - start;
         max_starting_nodes_size += num_starting_nodes[i];
      }
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         starting_nodes[i] = hypre_CTAlloc( HYPRE_Int, max_starting_nodes_size, HYPRE_MEMORY_HOST );
         for (j = 0; j < num_starting_nodes[i]; j++)
         {
            starting_nodes[i][j] = hypre_ParCSRCommPkgSendMapElmt(commPkg, j + start );
         }
      }

      // Find my own send nodes and communicate with neighbors to find off-processor long-range connections
      HYPRE_Int *num_request_nodes = hypre_CTAlloc(HYPRE_Int, send_proc_array_size, HYPRE_MEMORY_HOST);
      HYPRE_Int **request_nodes = hypre_CTAlloc(HYPRE_Int*, send_proc_array_size, HYPRE_MEMORY_HOST);

      for (i = 0; i < padding[level] + num_ghost_layers - 1; i++)
      {
         FindNeighborProcessors(compGrid, A, &(add_flag), 
            &(num_starting_nodes), &(starting_nodes), 
            &(search_proc_marker),
            &(num_request_nodes), &(request_nodes),
            &num_sends, &(send_procs), 
            hypre_ParCSRCommPkgNumSends(commPkg),
            &(send_proc_array_size),
            max_starting_nodes_size
            , level, i, communication_cost); // Note that num_sends may change here
      }

      // Use add_flag to generate relevant info for CompGridCommPkg
      cnt = 0;
      HYPRE_Int *send_map_starts = hypre_CTAlloc( HYPRE_Int, num_sends+1, HYPRE_MEMORY_HOST );
      for (i = 0; i < num_sends; i++)
      {
         send_map_starts[i] = cnt;
         for (j = 0; j < num_nodes; j++) if (add_flag[i][j] > 0) cnt++;
      }
      send_map_starts[num_sends] = cnt;
      HYPRE_Int *send_map_elmts = hypre_CTAlloc(HYPRE_Int, cnt, HYPRE_MEMORY_HOST);
      HYPRE_Int *ghost_marker = hypre_CTAlloc(HYPRE_Int, cnt, HYPRE_MEMORY_HOST);
      cnt = 0;
      for (i = 0; i < num_sends; i++) for (j = 0; j < num_nodes; j++) 
      {
         if (add_flag[i][j] > 0) 
         {
            send_map_elmts[cnt] = j;
            if (add_flag[i][j] > num_ghost_layers) ghost_marker[cnt] = 0;
            else ghost_marker[cnt] = 1;
            cnt++;
         }
      }
      send_procs = hypre_TReAlloc(send_procs, HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
      HYPRE_Int *recv_procs = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
      HYPRE_Int *partitions = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
      HYPRE_Int *proc_partitions = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         recv_procs[i] = send_procs[i];
         partitions[i] = send_procs[i];
         proc_partitions[i] = i;
      }

      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] = num_sends;
      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level] = num_sends;
      hypre_ParCompGridCommPkgNumPartitions(compGridCommPkg)[level] = num_sends;
      hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level] = send_procs;
      hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level] = recv_procs;
      hypre_ParCompGridCommPkgPartitions(compGridCommPkg)[level] = partitions;
      hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level] = proc_partitions;
      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level] = send_map_starts;
      hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level] = send_map_elmts;
      hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level] = ghost_marker;

      // Clean up memory
      for (i = 0; i < num_sends; i++)
      {
         hypre_TFree(add_flag[i], HYPRE_MEMORY_HOST);
         hypre_TFree(starting_nodes[i], HYPRE_MEMORY_HOST);
         hypre_TFree(request_nodes[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
      hypre_TFree(starting_nodes, HYPRE_MEMORY_HOST);
      hypre_TFree(search_proc_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(request_nodes, HYPRE_MEMORY_HOST);
      hypre_TFree(num_starting_nodes, HYPRE_MEMORY_HOST);
      hypre_TFree(num_request_nodes, HYPRE_MEMORY_HOST);
   }

   return 0;
}

HYPRE_Int 
FindNeighborProcessorsNew(hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, 
   map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > &send_proc_dofs, 
   map<HYPRE_Int, vector<HYPRE_Int> > starting_dofs, 
   HYPRE_Int level, HYPRE_Int *communication_cost)
{
   HYPRE_Int j;
   
   // Nodes to request from other processors
   map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > request_dofs;

   // Recursively search through the operator stencil to find longer distance neighboring dofs
   // Loop over destination processors
   for (auto it = starting_dofs.begin(); it != starting_dofs.end(); ++it)
   {
      HYPRE_Int destination_proc = it->first;
      // Loop over starting nodes for this proc
      for (j = 0; j < it->second.size(); j++)
      {
         HYPRE_Int dof_index = it->second[j];
         HYPRE_Int distance = send_proc_dofs[destination_proc][dof_index];
         RecursivelyFindNeighborNodesNew(dof_index, distance, compGrid, send_proc_dofs[destination_proc], request_dofs[destination_proc]);
      }
   }

   // !!! Finish

   return 0;
}



HYPRE_Int
FindNeighborProcessors( hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, HYPRE_Int ***add_flag,
   HYPRE_Int **num_starting_nodes, HYPRE_Int ***starting_nodes,
   HYPRE_Int **search_proc_marker,
   HYPRE_Int **num_request_nodes, HYPRE_Int ***request_nodes,
   HYPRE_Int *num_send_procs, HYPRE_Int **send_procs, 
   HYPRE_Int num_neighboring_procs,
   HYPRE_Int *send_proc_array_size,
   HYPRE_Int max_starting_nodes_size
   , HYPRE_Int level, HYPRE_Int iteration, HYPRE_Int *communication_cost )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int      i,j,k,cnt;

   // Update add_flag by recursively adding neighbors
   for (i = 0; i < (*num_send_procs); i++)
   {
      if ((*search_proc_marker)[i])
      {
         (*num_request_nodes)[i] = 0;
         if (!(*request_nodes)[i]) (*request_nodes)[i] = hypre_CTAlloc( HYPRE_Int, 2*hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST );

         for (j = 0; j < (*num_starting_nodes)[i]; j++)
         {
            if ((*add_flag)[i][ (*starting_nodes)[i][j] ] - 1 > 0)
               RecursivelyFindNeighborNodes( (*starting_nodes)[i][j], (*add_flag)[i][ (*starting_nodes)[i][j] ] - 1, compGrid, (*add_flag)[i], (*request_nodes)[i], &((*num_request_nodes)[i])
               , level, iteration, (*send_procs)[i] );
         }

         (*num_starting_nodes)[i] = 0;
      }
   }

   // Exchange message sizes
   HYPRE_Int send_size = 1;
   for (i = 0; i < (*num_send_procs); i++)
   {
      if ((*search_proc_marker)[i])
      {
         if ((*num_request_nodes)[i])
         {
            send_size += 2*(*num_request_nodes)[i] + 2;
         }
      }
   }
   HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int, num_neighboring_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, 4*num_neighboring_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Status *statuses = hypre_CTAlloc(hypre_MPI_Status, 4*num_neighboring_procs, HYPRE_MEMORY_HOST);
   HYPRE_Int request_cnt = 0;
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Irecv(&(recv_sizes[i]), 1, HYPRE_MPI_INT, (*send_procs)[i], 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      hypre_MPI_Isend(&send_size, 1, HYPRE_MPI_INT, (*send_procs)[i], 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*7 + 0]++;
         communication_cost[level*7 + 1] += sizeof(HYPRE_Int);
      }
   }

   // Wait on the recv sizes
   hypre_MPI_Waitall(2*num_neighboring_procs, requests, statuses);

   // Allocate recv buffers
   HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_neighboring_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_neighboring_procs; i++) recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_sizes[i], HYPRE_MEMORY_HOST);

   // Post the recvs
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Irecv(recv_buffers[i], recv_sizes[i], HYPRE_MPI_INT, (*send_procs)[i], 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   
   // Setup the send buffer and post the sends
   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int, send_size, HYPRE_MEMORY_HOST);
   cnt = 1;
   HYPRE_Int num_request_procs = 0;
   for (i = 0; i < (*num_send_procs); i++)
   {
      if ((*search_proc_marker)[i])
      {
         if ((*num_request_nodes)[i])
         {
            num_request_procs++;
            send_buffer[cnt++] = (*send_procs)[i];
            send_buffer[cnt++] = (*num_request_nodes)[i];
            for (j = 0; j < (*num_request_nodes)[i]; j++)
            {
               send_buffer[cnt++] = (*request_nodes)[i][2*j];
               send_buffer[cnt++] = (*request_nodes)[i][2*j+1];
            }
         }
      }
   }
   send_buffer[0] = num_request_procs;
   for (i = 0; i < num_neighboring_procs; i++)
   {
      hypre_MPI_Isend(send_buffer, send_size, HYPRE_MPI_INT, (*send_procs)[i], 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*7 + 0]++;
         communication_cost[level*7 + 1] += send_size*sizeof(HYPRE_Int);
      }
   }

   // Wait 
   hypre_MPI_Waitall(2*num_neighboring_procs, &(requests[2*num_neighboring_procs]), &(statuses[2*num_neighboring_procs]));

   // Reset search_proc_marker
   for (i = 0; i < (*num_send_procs); i++) (*search_proc_marker)[i] = 0;

   // Unpack recieved messages and update add_flag where appropriate
   for (i = 0; i < num_neighboring_procs; i++)
   {
      cnt = 0;
      HYPRE_Int num_incoming_procs = recv_buffers[i][cnt++];
      for (j = 0; j < num_incoming_procs; j++)
      {
         HYPRE_Int incoming_proc = recv_buffers[i][cnt++];
         HYPRE_Int num_incoming_nodes = recv_buffers[i][cnt++];

         // See whether we already have an add_flag array setup for this proc (and check to avoid info for our rank)
         if (incoming_proc != myid)
         {
            // Look for an add_flag already set up for this processor
            HYPRE_Int local_proc_index = -1;
            HYPRE_Int add_proc = 0;
            for (k = 0; k < (*num_send_procs); k++) if ((*send_procs)[k] == incoming_proc) local_proc_index = k;
            // If we didn't find one, this is a new proc that we may or may not need to account for
            if (local_proc_index < 0)
            {
               // Loop over incoming nodes and see whether we need to setup a new send_proc
               HYPRE_Int subCnt = cnt;
               for (k = 0; k < num_incoming_nodes; k++)
               {
                  HYPRE_Int local_index = recv_buffers[i][subCnt] - hypre_ParCompGridGlobalIndices(compGrid)[0]; 
                  subCnt += 2;
                  if (local_index >=0 && local_index < hypre_ParCSRMatrixNumRows(A))
                  {
                     add_proc = 1;
                     break;
                  }
               }
               if (add_proc)
               {
                  local_proc_index = (*num_send_procs);
                  (*num_send_procs)++;

                  // Note: we may need to grow several arrays here
                  if ((*num_send_procs) > (*send_proc_array_size))
                  {
                     (*send_proc_array_size) = 2*(*send_proc_array_size);
                     (*send_procs) = hypre_TReAlloc((*send_procs), HYPRE_Int, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*add_flag) = hypre_TReAlloc((*add_flag), HYPRE_Int*, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*search_proc_marker) = hypre_TReAlloc((*search_proc_marker), HYPRE_Int, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*num_starting_nodes) = hypre_TReAlloc((*num_starting_nodes), HYPRE_Int, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*starting_nodes) = hypre_TReAlloc((*starting_nodes), HYPRE_Int*, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*num_request_nodes) = hypre_TReAlloc((*num_request_nodes), HYPRE_Int, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     (*request_nodes) = hypre_TReAlloc((*request_nodes), HYPRE_Int*, (*send_proc_array_size), HYPRE_MEMORY_HOST);
                     for (k = local_proc_index; k < (*send_proc_array_size); k++)
                     {
                        (*send_procs)[k] = 0;
                        (*add_flag)[k] = NULL;
                        (*search_proc_marker)[k] = 0;
                        (*num_starting_nodes)[k] = 0;
                        (*starting_nodes)[k] = NULL;
                        (*num_request_nodes)[k] = 0;
                        (*request_nodes)[k] = NULL;
                     }
                  }

                  // Now
                  (*send_procs)[local_proc_index] = incoming_proc;
                  (*add_flag)[local_proc_index] = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A), HYPRE_MEMORY_HOST);
                  (*starting_nodes)[local_proc_index] = hypre_CTAlloc(HYPRE_Int, max_starting_nodes_size, HYPRE_MEMORY_HOST);
                  (*search_proc_marker)[local_proc_index] = 1;
               }
            }
            else add_proc = 1;
            // If we must account for the add_flag of this proc, process the incoming request nodes and update add_flag as appropriate
            if (add_proc)
            {
               for (k = 0; k < num_incoming_nodes; k++)
               {
                  HYPRE_Int local_index = recv_buffers[i][cnt++] - hypre_ParCompGridGlobalIndices(compGrid)[0]; 
                  HYPRE_Int incoming_flag = recv_buffers[i][cnt++];
                  if (local_index >=0 && local_index < hypre_ParCSRMatrixNumRows(A))
                  {
                     if (incoming_flag > (*add_flag)[local_proc_index][local_index])
                     {
                        (*add_flag)[local_proc_index][local_index] = incoming_flag;
                        (*starting_nodes)[local_proc_index][ (*num_starting_nodes)[local_proc_index]++ ] = local_index;
                        (*search_proc_marker)[local_proc_index] = 1;
                     }
                  }
               }
            }
            else
            {
               cnt += 2*num_incoming_nodes;
            }
         }
         else
         {
            cnt += 2*num_incoming_nodes;
         }
      }
   }
  
   // Clean up memory
   for (i = 0; i < num_neighboring_procs; i++) hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int
RecursivelyFindNeighborNodesNew(HYPRE_Int dof_index, HYPRE_Int distance, hypre_ParCompGrid *compGrid, 
   map<HYPRE_Int, HYPRE_Int> &send_dofs, 
   map<HYPRE_Int, HYPRE_Int> &request_dofs )
{
   HYPRE_Int         i,j;

   // Look at neighbors
   for (i = hypre_ParCompGridARowPtr(compGrid)[dof_index]; i < hypre_ParCompGridARowPtr(compGrid)[dof_index+1]; i++)
   {
      // Get the index of the neighbor
      HYPRE_Int neighbor_index = hypre_ParCompGridAColInd(compGrid)[i];

      // If the neighbor info is available on this proc
      if (neighbor_index >= 0)
      {
         // And if we still need to visit this index (note that send_dofs[neighbor_index] = distance means we have already added all distance-1 neighbors of index)
         
         // See whether this dof is in the send dofs
         auto neighbor_dof = send_dofs.find(neighbor_index);
         if (neighbor_dof == send_dofs.end())
         {
            // If neighbor dof isn't in the send dofs, add it with appropriate distance and recurse
            send_dofs[neighbor_index] = distance;
            if (distance-1 > 0) RecursivelyFindNeighborNodesNew(neighbor_index, distance-1, compGrid, send_dofs, request_dofs);
         }
         else if (neighbor_dof->second < distance)
         {
            // If neighbor dof is in the send dofs, but at smaller distance, also need to update distance and recurse
            send_dofs[neighbor_index] = distance;
            if (distance-1 > 0) RecursivelyFindNeighborNodesNew(neighbor_index, distance-1, compGrid, send_dofs, request_dofs);
         }
      }
      // otherwise note this as a request dof
      else
      {
         // Check whether we have already requested this node 
         HYPRE_Int global_index = hypre_ParCompGridAGlobalColInd(compGrid)[i];


         auto req_dof = request_dofs.find(global_index);
         if (req_dof == request_dofs.end())
         {
            // If this hasn't yet been requested, add it
            request_dofs[global_index] = distance;
         }
         else if (req_dof->second < distance)
         {
            // If reqest is already there, but at smaller distance, update the distance
            request_dofs[global_index] = distance;
         }
      }
   }

   return 0;
}

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, 
   HYPRE_Int *request_nodes, HYPRE_Int *num_request_nodes
   , HYPRE_Int level, HYPRE_Int iteration, HYPRE_Int proc )
{
   HYPRE_Int         i,j,index,coarse_grid_index;

   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Look at neighbors
   for (i = hypre_ParCompGridARowPtr(compGrid)[node]; i < hypre_ParCompGridARowPtr(compGrid)[node+1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_ParCompGridAColInd(compGrid)[i];

      // If the neighbor info is available on this proc
      if (index >= 0)
      {
         // And if we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[index] < m)
         {
            add_flag[index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) RecursivelyFindNeighborNodes(index, m-1, compGrid, add_flag, request_nodes, num_request_nodes
               , level, iteration, proc);
         }
      }
      // otherwise note this as a starting node to request from neighboring procs
      else
      {
         // Check whether we have already requested this node (!!! linear search, but should be over small set !!!)
         HYPRE_Int global_index = hypre_ParCompGridAGlobalColInd(compGrid)[i];
         HYPRE_Int add_requeset = 1;
         for (j = 0; j < (*num_request_nodes); j++)
         {
            if (request_nodes[2*j] == global_index)
            {
               add_requeset = 0;
               if (m > request_nodes[2*j+1])
               {
                  request_nodes[2*j+1] = m;
               }
            }
         }
         if (add_requeset)
         {
            request_nodes[2*(*num_request_nodes)] = global_index;
            request_nodes[2*(*num_request_nodes)+1] = m; 
            (*num_request_nodes)++;
         }
      }
   }

   return 0;
}

#endif