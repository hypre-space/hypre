// Helper functions to setup amgdd composite grids

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"   

#ifdef __cplusplus

#include <vector>
#include <map>
#include <set>

// !!! Debug
#include <chrono>

using namespace std;

extern "C"
{

#endif

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGrid *compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int *communication_cost );

HYPRE_Int
UnpackRecvBuffer( HYPRE_Int *recv_buffer, hypre_ParCompGrid **compGrid, 
      hypre_ParCompGridCommPkg *compGridCommPkg,
      HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
      HYPRE_Int ****recv_map, HYPRE_Int ***num_recv_nodes, 
      HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int transition_level,
      HYPRE_Int *nodes_added_on_level, HYPRE_Int buffer_number, HYPRE_Int *num_resizes );

#ifdef __cplusplus
}

HYPRE_Int
GetDofRecvProc(HYPRE_Int dof_index, HYPRE_Int neighbor_global_index, hypre_ParCSRMatrix *A)
{
   HYPRE_Int *colmap = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int *offdRowPtr = hypre_CSRMatrixI( hypre_ParCSRMatrixOffd(A) );
   HYPRE_Int offdColIndex = -1;

   // Get the appropriate column index in the offd part of A
   for (HYPRE_Int i = offdRowPtr[dof_index]; i < offdRowPtr[dof_index+1]; i++)
   {
      if (colmap[ hypre_CSRMatrixJ( hypre_ParCSRMatrixOffd(A) )[i] ] == neighbor_global_index)
      {
         offdColIndex = hypre_CSRMatrixJ( hypre_ParCSRMatrixOffd(A) )[i];
      }
   }

   // Use that column index to find which processor this dof is received from
   hypre_ParCSRCommPkg *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int recv_proc = -1;
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++)
   {
      if (offdColIndex >= hypre_ParCSRCommPkgRecvVecStart(commPkg,i) && offdColIndex < hypre_ParCSRCommPkgRecvVecStart(commPkg,i+1)) 
         recv_proc = hypre_ParCSRCommPkgRecvProc(commPkg,i);
   }

   return recv_proc;
}

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int dof_index, HYPRE_Int distance, hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A,
   map<HYPRE_Int, HYPRE_Int> &send_dofs, 
   map< HYPRE_Int, map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > > &request_proc_dofs, HYPRE_Int destination_proc )
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
            if (distance-1 > 0) RecursivelyFindNeighborNodes(neighbor_index, distance-1, compGrid, A, send_dofs, request_proc_dofs, destination_proc);
         }
         else if (neighbor_dof->second < distance)
         {
            // If neighbor dof is in the send dofs, but at smaller distance, also need to update distance and recurse
            send_dofs[neighbor_index] = distance;
            if (distance-1 > 0) RecursivelyFindNeighborNodes(neighbor_index, distance-1, compGrid, A, send_dofs, request_proc_dofs, destination_proc);
         }
      }
      // otherwise note this as a request dof
      else
      {
         HYPRE_Int neighbor_global_index = hypre_ParCompGridAGlobalColInd(compGrid)[i];

         HYPRE_Int recv_proc = GetDofRecvProc(dof_index, neighbor_global_index, A);

         // If request proc isn't the destination proc
         if (recv_proc != destination_proc)
         {
            // Check whether we have already requested this node 
            auto req_dof = request_proc_dofs[recv_proc][destination_proc].find(neighbor_global_index);
            if (req_dof == request_proc_dofs[recv_proc][destination_proc].end())
            {
               // If this hasn't yet been requested, add it
               request_proc_dofs[recv_proc][destination_proc][neighbor_global_index] = distance;
            }
            else if (req_dof->second < distance)
            {
               // If reqest is already there, but at smaller distance, update the distance
               request_proc_dofs[recv_proc][destination_proc][neighbor_global_index] = distance;
            }
         }
      }
   }

   return 0;
}

HYPRE_Int 
FindNeighborProcessors(hypre_ParCompGrid *compGrid, hypre_ParCSRMatrix *A, 
   map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > &send_proc_dofs, 
   map<HYPRE_Int, set<HYPRE_Int> > &starting_dofs, 
   set<HYPRE_Int> &recv_procs,
   HYPRE_Int level, HYPRE_Int *communication_cost)
{
   
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );


   // !!! Debug
   vector<chrono::duration<double>> timings;
   // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   auto start = chrono::system_clock::now();


   // Nodes to request from other processors. Note, requests are only issued to processors within distance 1, i.e. within the original communication stencil for A
   hypre_ParCSRCommPkg *commPkg = hypre_ParCSRMatrixCommPkg(A);
   map< HYPRE_Int, map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > > request_proc_dofs; // request_proc_dofs[proc to request from, i.e. recv_proc][destination_proc][dof global index][distance]
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++) request_proc_dofs[ hypre_ParCSRCommPkgRecvProc(commPkg,i) ];

   // Recursively search through the operator stencil to find longer distance neighboring dofs
   // Loop over destination processors
   for (auto dest_proc_it = starting_dofs.begin(); dest_proc_it != starting_dofs.end(); ++dest_proc_it)
   {
      HYPRE_Int destination_proc = dest_proc_it->first;
      // Loop over starting nodes for this proc
      for (auto dof_it = dest_proc_it->second.begin(); dof_it != dest_proc_it->second.end(); ++dof_it)
      {
         HYPRE_Int dof_index = *dof_it;
         HYPRE_Int distance = send_proc_dofs[destination_proc][dof_index];
         RecursivelyFindNeighborNodes(dof_index, distance-1, compGrid, A, send_proc_dofs[destination_proc], request_proc_dofs, destination_proc);
      }
   }
   // Clear the list of starting dofs
   starting_dofs.clear();


   auto end = chrono::system_clock::now();
   timings.push_back(end - start);




   //////////////////////////////////////////////////
   // Communicate newly connected longer-distance processors to send procs: sending to current long distance send_procs and receiving from current long distance recv_procs
   //////////////////////////////////////////////////

   // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   start = chrono::system_clock::now();


   // Get the sizes
   hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, send_proc_dofs.size() + recv_procs.size(), HYPRE_MEMORY_HOST);
   hypre_MPI_Status *statuses = hypre_CTAlloc(hypre_MPI_Status, send_proc_dofs.size() + recv_procs.size(), HYPRE_MEMORY_HOST);
   HYPRE_Int request_cnt = 0;

   HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int, recv_procs.size(), HYPRE_MEMORY_HOST);
   HYPRE_Int cnt = 0;
   for (auto recv_proc_it = recv_procs.begin(); recv_proc_it != recv_procs.end(); ++recv_proc_it)
   {
      hypre_MPI_Irecv(&(recv_sizes[cnt++]), 1, HYPRE_MPI_INT, *recv_proc_it, 6, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   HYPRE_Int *send_sizes = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
   cnt = 0;
   for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
   {
      for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it)
      {
         if (req_proc_it->second.find(send_proc_it->first) != req_proc_it->second.end()) send_sizes[cnt]++; 
      }
      hypre_MPI_Isend(&(send_sizes[cnt]), 1, HYPRE_MPI_INT, send_proc_it->first, 6, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*10 + 0]++;
         communication_cost[level*10 + 1] += sizeof(HYPRE_Int);
      }
      cnt++;
   }

   // Wait 
   hypre_MPI_Waitall(send_proc_dofs.size() + recv_procs.size(), requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request, send_proc_dofs.size() + recv_procs.size(), HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status, send_proc_dofs.size() + recv_procs.size(), HYPRE_MEMORY_HOST);
   request_cnt = 0;

   end = chrono::system_clock::now();
   timings.push_back(end - start);

   // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   start = chrono::system_clock::now();

   // Allocate and post the recvs
   HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, recv_procs.size(), HYPRE_MEMORY_HOST);
   cnt = 0;
   for (auto recv_proc_it = recv_procs.begin(); recv_proc_it != recv_procs.end(); ++recv_proc_it)
   {
      recv_buffers[cnt] = hypre_CTAlloc(HYPRE_Int, recv_sizes[cnt], HYPRE_MEMORY_HOST);
      hypre_MPI_Irecv(recv_buffers[cnt], recv_sizes[cnt], HYPRE_MPI_INT, *recv_proc_it, 7, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      cnt++;
   }
   // Setup and send the send buffers
   HYPRE_Int **send_buffers = hypre_CTAlloc(HYPRE_Int*, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
   cnt = 0;
   for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
   {
      send_buffers[cnt] = hypre_CTAlloc(HYPRE_Int, send_sizes[cnt], HYPRE_MEMORY_HOST);
      HYPRE_Int inner_cnt = 0;
      for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it)
      {
         if (req_proc_it->second.find(send_proc_it->first) != req_proc_it->second.end()) send_buffers[cnt][inner_cnt++] = req_proc_it->first; 
      }
      hypre_MPI_Isend(send_buffers[cnt], send_sizes[cnt], HYPRE_MPI_INT, send_proc_it->first, 7, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*10 + 0]++;
         communication_cost[level*10 + 1] += send_sizes[cnt]*sizeof(HYPRE_Int);
      }
      cnt++;
   }

   // Wait 
   hypre_MPI_Waitall(send_proc_dofs.size() + recv_procs.size(), requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);

   end = chrono::system_clock::now();
   timings.push_back(end - start);

   // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   start = chrono::system_clock::now();

   // Update recv_procs
   HYPRE_Int old_num_recv_procs = recv_procs.size();
   for (HYPRE_Int i = 0; i < old_num_recv_procs; i++)
   {
      for (HYPRE_Int j = 0; j < recv_sizes[i]; j++)
      {
         recv_procs.insert(recv_buffers[i][j]);
      }
   }

   // Clean up memory
   for (size_t i = 0; i < send_proc_dofs.size(); i++) hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   for (size_t i = 0; i < request_proc_dofs.size(); i++) hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_sizes, HYPRE_MEMORY_HOST);

   end = chrono::system_clock::now();
   timings.push_back(end - start);


   //////////////////////////////////////////////////
   // Communicate request dofs to processors that I recv from: sending to request_procs and receiving from distance 1 send procs
   //////////////////////////////////////////////////

   // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   start = chrono::system_clock::now();


   // Count up the send size: 1 + sum_{destination_procs}(2 + 2*num_requested_dofs)
   // send_buffer = [num destination procs, [request info for proc], [request info for proc], ... ]
   // [request info for proc] = [proc id, num requested dofs, [(dof index, distance), (dof index, distance), ...] ]

   // Exchange message sizes
   send_sizes = hypre_CTAlloc(HYPRE_Int, request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   recv_sizes = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgNumSends(commPkg), HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request, hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status, hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   request_cnt = 0;
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumSends(commPkg); i++)
   {
      hypre_MPI_Irecv(&(recv_sizes[i]), 1, HYPRE_MPI_INT, hypre_ParCSRCommPkgSendProc(commPkg,i), 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   cnt = 0;
   for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it)
   {
      send_sizes[cnt]++;
      for (auto dest_proc_it = req_proc_it->second.begin(); dest_proc_it != req_proc_it->second.end(); ++dest_proc_it)
      {
         send_sizes[cnt] += 2 + 2*dest_proc_it->second.size();
      }
      hypre_MPI_Isend(&(send_sizes[cnt]), 1, HYPRE_MPI_INT, req_proc_it->first, 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*10 + 0]++;
         communication_cost[level*10 + 1] += sizeof(HYPRE_Int);
      }
      cnt++;
   }

   // Wait on the recv sizes, then free and re-allocate the requests and statuses
   hypre_MPI_Waitall(hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request, hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status, hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   request_cnt = 0;

   // Allocate recv buffers and post the recvs
   recv_buffers = hypre_CTAlloc(HYPRE_Int*, hypre_ParCSRCommPkgNumSends(commPkg), HYPRE_MEMORY_HOST);
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumSends(commPkg); i++)
   {
      recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_sizes[i], HYPRE_MEMORY_HOST);
      hypre_MPI_Irecv(recv_buffers[i], recv_sizes[i], HYPRE_MPI_INT, hypre_ParCSRCommPkgSendProc(commPkg,i), 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   
   // Setup the send buffer and post the sends
   send_buffers = hypre_CTAlloc(HYPRE_Int*, request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   cnt = 0;
   for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it)
   {
      send_buffers[cnt] = hypre_CTAlloc(HYPRE_Int, send_sizes[cnt], HYPRE_MEMORY_HOST);
      HYPRE_Int inner_cnt = 0;
      send_buffers[cnt][inner_cnt++] = req_proc_it->second.size();
      for (auto dest_proc_it = req_proc_it->second.begin(); dest_proc_it != req_proc_it->second.end(); ++dest_proc_it)
      {
         send_buffers[cnt][inner_cnt++] = dest_proc_it->first;
         send_buffers[cnt][inner_cnt++] = dest_proc_it->second.size();
         for (auto dof_it = dest_proc_it->second.begin(); dof_it != dest_proc_it->second.end(); ++dof_it)
         {
            send_buffers[cnt][inner_cnt++] = dof_it->first;
            send_buffers[cnt][inner_cnt++] = dof_it->second;
         }
      }
      hypre_MPI_Isend(send_buffers[cnt], send_sizes[cnt], HYPRE_MPI_INT, req_proc_it->first, 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*10 + 0]++;
         communication_cost[level*10 + 1] += send_sizes[cnt]*sizeof(HYPRE_Int);
      }
      cnt++;
   }

   // Wait 
   hypre_MPI_Waitall(hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);

   // Update send_proc_dofs and starting_dofs 
   // Loop over send_proc's, i.e. the processors that we just received from 
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumSends(commPkg); i++)
   {
      cnt = 0;      
      HYPRE_Int num_destination_procs = recv_buffers[i][cnt++];
      for (HYPRE_Int destination_proc = 0; destination_proc < num_destination_procs; destination_proc++)
      {
         // Get destination proc id and the number of requested dofs
         HYPRE_Int proc_id = recv_buffers[i][cnt++];
         HYPRE_Int num_requested_dofs = recv_buffers[i][cnt++];

         // create new map for this destination proc if it doesn't already exist
         send_proc_dofs[proc_id];

         // Loop over the requested dofs for this destination proc
         for (HYPRE_Int j = 0; j < num_requested_dofs; j++)
         {
            // Get the local index for this dof on this processor
            HYPRE_Int req_dof_local_index = recv_buffers[i][cnt++] - hypre_ParCSRMatrixFirstRowIndex(A);

            // If we already have a this dof accounted for for this destination...
            if (send_proc_dofs[proc_id].find(req_dof_local_index) != send_proc_dofs[proc_id].end())
            {
               // ... but at a smaller distance, overwrite with new distance and add to starting_dofs
               if (send_proc_dofs[proc_id][req_dof_local_index] < recv_buffers[i][cnt])
               {
                  send_proc_dofs[proc_id][req_dof_local_index] = recv_buffers[i][cnt];
                  starting_dofs[proc_id].insert(req_dof_local_index);
               }
            } 
            // Otherwise, add this dof for this destination at this distance and add to starting_dofs
            else
            {
               send_proc_dofs[proc_id][req_dof_local_index] = recv_buffers[i][cnt];
               starting_dofs[proc_id].insert(req_dof_local_index);
            }
            cnt++;
         }
      }
   }

   // Clean up memory
   for (size_t i = 0; i < send_proc_dofs.size(); i++) hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   for (size_t i = 0; i < request_proc_dofs.size(); i++) hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_sizes, HYPRE_MEMORY_HOST);

   end = chrono::system_clock::now();
   timings.push_back(end - start);

   // !!! Debug
   // HYPRE_Int num_procs;
   // hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   // for (auto proc = 0; proc < num_procs; proc++)
   // {
   //    if (myid == proc) 
   //    {
   //       cout << "Rank " << myid << ": " << endl;
   //       for (size_t i = 0; i < timings.size(); i++) cout << timings[i].count() << ", ";
   //       cout << endl;
   //    }
   //    // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   // }

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
      hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level] = 0;
      
      // !!! Debug
      // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      HYPRE_Int num_procs;
      hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
      for (auto proc = 0; proc < num_procs; proc++)
      {
         // hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      }

   }
   else
   {
      // Initialize send_proc_dofs and the starting_dofs (this is how we will track nodes to send to each proc until routine finishes)
      map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > send_proc_dofs; // send_proc_dofs[send_proc] = send_dofs, send_dofs[dof_index] = distance value
      map<HYPRE_Int, set<HYPRE_Int> > starting_dofs; // starting_dofs[send_proc] = vector of starting dofs for searching through stencil
      for (i = 0; i < num_sends; i++)
      {
         send_proc_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)]; // initialize the send procs as the keys in the outer map
         starting_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)];
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         for (j = start; j < finish; j++)
         {
            send_proc_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)][hypre_ParCSRCommPkgSendMapElmt(commPkg,j)] = padding[level] + num_ghost_layers;
            starting_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)].insert(hypre_ParCSRCommPkgSendMapElmt(commPkg,j));
         }
      }

      //Initialize the recv_procs
      set<HYPRE_Int> recv_procs;
      for (i = 0; i < num_recvs; i++) recv_procs.insert( hypre_ParCSRCommPkgRecvProc(commPkg,i) );

      // Iteratively communicate with longer and longer distance neighbors to grow the communication stencils
      for (i = 0; i < padding[level] + num_ghost_layers - 1; i++)
      {
         FindNeighborProcessors(compGrid, A, send_proc_dofs, starting_dofs, recv_procs, level, communication_cost);
      }
   
      // Use send_proc_dofs and recv_procs to generate relevant info for CompGridCommPkg
      // Set the number of send and recv procs
      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] = send_proc_dofs.size();
      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level] = recv_procs.size();
      hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[level] = send_proc_dofs.size();
      // Setup the list of send procs, partitions, and proc partitions, and count up the total number of send elmts
      HYPRE_Int total_send_elmts = 0;
      hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
      cnt = 0;
      for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
      {
         hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][cnt] = send_proc_it->first;
         hypre_ParCompGridCommPkgSendPartitions(compGridCommPkg)[level][cnt] = send_proc_it->first;
         hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level][cnt] = cnt;
         total_send_elmts += send_proc_it->second.size();
         cnt++;
      }
      // Setup the list of recv procs
      hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, recv_procs.size(), HYPRE_MEMORY_HOST);
      cnt = 0;
      for (auto recv_proc_it = recv_procs.begin(); recv_proc_it != recv_procs.end(); ++recv_proc_it)
      {
         hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][cnt++] = *recv_proc_it;
      }
      // Setup the send map elmts, starts, and ghost marker
      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size() + 1, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, total_send_elmts, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, total_send_elmts, HYPRE_MEMORY_HOST);
      HYPRE_Int proc_cnt = 0;
      cnt = 0;
      for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
      {
         hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][proc_cnt++] = cnt;
         for (auto dof_it = send_proc_it->second.begin(); dof_it != send_proc_it->second.end(); ++dof_it)
         {
            hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level][cnt] = dof_it->first;
            if (dof_it->second <= num_ghost_layers) hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level][cnt] = 1;
            cnt++;
         }
      }
      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][send_proc_dofs.size()] = total_send_elmts;
   }

   return 0;
}

HYPRE_Int
UnpackRecvBuffer( HYPRE_Int *recv_buffer, hypre_ParCompGrid **compGrid, 
      hypre_ParCompGridCommPkg *compGridCommPkg,
      HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
      HYPRE_Int ****recv_map, HYPRE_Int ***num_recv_nodes, 
      HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int transition_level,
      HYPRE_Int *nodes_added_on_level, HYPRE_Int buffer_number, HYPRE_Int *num_resizes )
{
   // recv_buffer = [ num_psi_levels , [level] , [level] , ... ]
   // level = [ num send nodes, [global indices] , [coarse global indices] , [A row sizes] , [A col ind] ]

   // !!! Debug
   vector<chrono::duration<double>> timings(20);
   auto total_start = chrono::system_clock::now();

   HYPRE_Int            level, i, j, k;
   HYPRE_Int            num_psi_levels, row_size, level_start, add_node_cnt;
   // HYPRE_Int            *add_flag;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // initialize the counter
   HYPRE_Int            cnt = 0;

   // get the number of levels received
   num_psi_levels = recv_buffer[cnt++];

   // Init the recv_map_send_buffer_size !!! I think this can just be set a priori instead of counting it up in this function... !!!
   *recv_map_send_buffer_size = num_levels - current_level;

   // loop over coarser psi levels
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // get the number of nodes on this level
      num_recv_nodes[current_level][buffer_number][level] = recv_buffer[cnt++];
      level_start = cnt;
      *recv_map_send_buffer_size += num_recv_nodes[current_level][buffer_number][level];

      // !!! Debug
      auto start = chrono::system_clock::now();

      // Incoming nodes and existing (non-owned) nodes in the comp grid are both sorted by global index, so here we merge these lists together (getting rid of redundant nodes along the way)
      add_node_cnt = 0;
      HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
      HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
      HYPRE_Int num_nonowned_nodes = num_nodes - num_owned_nodes;
      HYPRE_Int dest = num_owned_nodes;
      HYPRE_Int compGrid_cnt = 0;
      HYPRE_Int incoming_cnt = 0;
      HYPRE_Int *compGrid_dest = hypre_CTAlloc(HYPRE_Int, num_nonowned_nodes, HYPRE_MEMORY_HOST);
      // NOTE: Don't free incoming_dest because we set that as recv_map and use it outside this function
      HYPRE_Int *incoming_dest = hypre_CTAlloc(HYPRE_Int, num_recv_nodes[current_level][buffer_number][level], HYPRE_MEMORY_HOST);

      // !!! Debug
      auto end = chrono::system_clock::now();
      timings[1] += end - start; // Allocation
      start = chrono::system_clock::now();

      while (incoming_cnt < num_recv_nodes[current_level][buffer_number][level] && compGrid_cnt < num_nonowned_nodes)
      {
         HYPRE_Int incoming_global_index = recv_buffer[cnt];
         HYPRE_Int incoming_is_real = 1;
         if (incoming_global_index < 0) 
         {
            incoming_global_index = -(incoming_global_index + 1);
            incoming_is_real = 0;
         }
         HYPRE_Int compGrid_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ compGrid_cnt + num_owned_nodes ];
         HYPRE_Int incoming_is_nonowned = 1;

         for (i = 0; i < hypre_ParCompGridNumOwnedBlocks(compGrid[level]); i++)
         {
            if (hypre_ParCompGridOwnedBlockStarts(compGrid[level])[i+1] - hypre_ParCompGridOwnedBlockStarts(compGrid[level])[i] > 0)
            {
               HYPRE_Int low_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[hypre_ParCompGridOwnedBlockStarts(compGrid[level])[i]];
               HYPRE_Int high_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[hypre_ParCompGridOwnedBlockStarts(compGrid[level])[i+1]-1];
               if (incoming_global_index >= low_global_index && incoming_global_index <= high_global_index)
               {
                  incoming_dest[incoming_cnt++] = -1;
                  cnt++;
                  incoming_is_nonowned = 0;
               }
            }
         }
         if (incoming_is_nonowned)
         {
            if (incoming_global_index == compGrid_global_index)
            {
               if (incoming_is_real && !hypre_ParCompGridRealDofMarker(compGrid[level])[ compGrid_cnt + num_owned_nodes ])
               {
                  hypre_ParCompGridRealDofMarker(compGrid[level])[ compGrid_cnt + num_owned_nodes ] = 1;
                  compGrid_dest[compGrid_cnt++] = dest;
                  incoming_dest[incoming_cnt++] = dest;
                  dest++;
                  cnt++;

               }
               else
               {
                  incoming_dest[incoming_cnt++] = -1;
                  cnt++;
               }
            }
            else if (incoming_global_index < compGrid_global_index)
            {
               incoming_dest[incoming_cnt++] = dest++;
               cnt++;
               add_node_cnt++;
               nodes_added_on_level[level] = 1;
            }
            else
            {
               compGrid_dest[compGrid_cnt++] = dest++;
            }
         }
      }
      while (incoming_cnt < num_recv_nodes[current_level][buffer_number][level])
      {
         HYPRE_Int incoming_global_index = recv_buffer[cnt];
         if (incoming_global_index < 0) 
         {
            incoming_global_index = -(incoming_global_index + 1);
         }
         HYPRE_Int incoming_is_nonowned = 1;
         for (i = 0; i < hypre_ParCompGridNumOwnedBlocks(compGrid[level]); i++)
         {
            if (hypre_ParCompGridOwnedBlockStarts(compGrid[level])[i+1] - hypre_ParCompGridOwnedBlockStarts(compGrid[level])[i] > 0)
            {
               HYPRE_Int low_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[hypre_ParCompGridOwnedBlockStarts(compGrid[level])[i]];
               HYPRE_Int high_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[hypre_ParCompGridOwnedBlockStarts(compGrid[level])[i+1]-1];
               if (incoming_global_index >= low_global_index && incoming_global_index <= high_global_index)
               {
                  incoming_dest[incoming_cnt++] = -1;
                  cnt++;
                  incoming_is_nonowned = 0;
               }
            }
         }
         if (incoming_is_nonowned)
         {
            incoming_dest[incoming_cnt++] = dest++;
            add_node_cnt++;
            nodes_added_on_level[level] = 1;
            cnt++;
         }
      }
      while (compGrid_cnt < num_nonowned_nodes)
      {
         compGrid_dest[compGrid_cnt++] = dest++;
      }

      // !!! Debug
      end = chrono::system_clock::now();
      timings[2] += end - start; // Merge
      start = chrono::system_clock::now();

      // Set recv_map[current_level] to incoming_dest
      recv_map[current_level][buffer_number][level] = incoming_dest;

      // if necessary, reallocate more space for compGrid
      if (add_node_cnt + num_nodes > hypre_ParCompGridMemSize(compGrid[level])) 
      {
         num_resizes[3*level]++;
         HYPRE_Int new_size = ceil(1.5*hypre_ParCompGridMemSize(compGrid[level]));
         if (new_size < add_node_cnt + num_nodes) new_size = add_node_cnt + num_nodes;
         hypre_ParCompGridResize(compGrid[level], new_size, level != num_levels-1, 0); // !!! Is there a better way to manage memory? !!!
         hypre_ParCompGridNumNodes(compGrid[level]) = add_node_cnt + num_nodes;
      }

      // Starting at the end of the list (to avoid overwriting info we want to access later), copy existing comp grid info to its new positions
      for (i = num_nonowned_nodes - 1; i >= 0; i--)
      {
         hypre_ParCompGridGlobalIndices(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridGlobalIndices(compGrid[level])[i + num_owned_nodes];
         hypre_ParCompGridRealDofMarker(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridRealDofMarker(compGrid[level])[i + num_owned_nodes];
      }
      
      if (level != transition_level-1)
      {
         for (i = num_nonowned_nodes - 1; i >= 0; i--)
            hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[i + num_owned_nodes];
         for (i = num_nonowned_nodes - 1; i >= 0; i--)
            hypre_ParCompGridCoarseLocalIndices(compGrid[level])[ compGrid_dest[i] ] = hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i + num_owned_nodes];
      }

      // !!! Debug
      end = chrono::system_clock::now();
      timings[3] += end - start; // Copy existing
      start = chrono::system_clock::now();

      // Fix up the send_flag and recv_map from previous levels
      for (i = current_level; i < num_levels; i++)
      {
         for (j = 0; j < hypre_ParCompGridCommPkgNumSendPartitions(compGridCommPkg)[i]; j++)
         {
            for (k = 0; k < num_send_nodes[i][j][level]; k++)
            {
               if (send_flag[i][j][level][k] >= num_owned_nodes)
                  send_flag[i][j][level][k] = compGrid_dest[ send_flag[i][j][level][k] - num_owned_nodes ];
               else if (send_flag[i][j][level][k] <= -(num_owned_nodes + 1))
                  send_flag[i][j][level][k] = -(compGrid_dest[ -(send_flag[i][j][level][k] + 1) - num_owned_nodes ] + 1);
            }
         }
      }
      for (i = current_level+1; i < num_levels; i++)
      {
         for (j = 0; j < hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[i]; j++)
         {
            for (k = 0; k < num_recv_nodes[i][j][level]; k++)
            {
               if (recv_map[i][j][level][k] >= num_owned_nodes)
                  recv_map[i][j][level][k] = compGrid_dest[ recv_map[i][j][level][k] - num_owned_nodes ];
               else if (recv_map[i][j][level][k] <= -(num_owned_nodes + 1))
                  recv_map[i][j][level][k] = -(compGrid_dest[ -(recv_map[i][j][level][k] + 1) - num_owned_nodes ] + 1);

            }
         }
      }
      for (i = 0; i < buffer_number; i++)
      {
         if (recv_map[current_level][i][level])
         {
            for (k = 0; k < num_recv_nodes[current_level][i][level]; k++)
            {
               if (recv_map[current_level][i][level][k] >= num_owned_nodes)
                  recv_map[current_level][i][level][k] = compGrid_dest[ recv_map[current_level][i][level][k] - num_owned_nodes ];
               else if (recv_map[current_level][i][level][k] <= -(num_owned_nodes + 1))
                  recv_map[current_level][i][level][k] = -(compGrid_dest[ -(recv_map[current_level][i][level][k] + 1 ) - num_owned_nodes ]  + 1);
            }
         }
      }

      // !!! Debug
      end = chrono::system_clock::now();
      timings[4] += end - start; // Fix up prev send/recv
      start = chrono::system_clock::now();

      // Now copy in the new nodes to their appropriate positions
      cnt = level_start;
      for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++) 
      {   
         if (incoming_dest[i] >= 0)
         {
            HYPRE_Int global_index = recv_buffer[cnt];
            if (global_index < 0) 
            {
               global_index = -(global_index + 1);
               hypre_ParCompGridRealDofMarker(compGrid[level])[ incoming_dest[i] ] = 0;
            }
            else hypre_ParCompGridRealDofMarker(compGrid[level])[ incoming_dest[i] ] = 1;
            hypre_ParCompGridGlobalIndices(compGrid[level])[ incoming_dest[i] ] = global_index;
         }
         cnt++;
      }
      if (level != transition_level-1)
      {
         for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++) 
         {   
            if (incoming_dest[i] >= 0)
            {
               hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[ incoming_dest[i] ] = recv_buffer[cnt];
            }
            cnt++;
         }
      }

      // Copy existing row sizes into correct positions in a new row size list
      HYPRE_Int *A_row_sizes = hypre_CTAlloc(HYPRE_Int, add_node_cnt + num_nodes - num_owned_nodes, HYPRE_MEMORY_HOST);

      for (i = num_nonowned_nodes - 1; i >= 0; i--)
      {
         A_row_sizes[ compGrid_dest[i] - num_owned_nodes ] = 
                                          hypre_ParCompGridARowPtr(compGrid[level])[i + num_owned_nodes + 1]
                                        - hypre_ParCompGridARowPtr(compGrid[level])[i + num_owned_nodes];
      }
      
      // !!! Debug
      end = chrono::system_clock::now();
      timings[5] += end - start; // More copies
      start = chrono::system_clock::now();

      // Place the incoming comp grid A row sizes in the appropriate places and count up number of nonzeros added to A
      HYPRE_Int added_A_nnz = 0;
      for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++)
      {
         if (incoming_dest[i] >= 0)
         {
            row_size = recv_buffer[cnt];
            added_A_nnz += row_size;
            A_row_sizes[ incoming_dest[i] - num_owned_nodes ] = row_size;
         }
         cnt++;
      }

      // Now that row sizes are in the right places, setup a new A row pointer appropriately
      HYPRE_Int *A_new_rowptr = hypre_CTAlloc(HYPRE_Int, add_node_cnt + num_nodes - num_owned_nodes + 1, HYPRE_MEMORY_HOST);
      A_new_rowptr[0] = hypre_ParCompGridARowPtr(compGrid[level])[num_owned_nodes];
      for (i = 0; i < add_node_cnt + num_nodes - num_owned_nodes; i++) 
         A_new_rowptr[i + 1] = A_new_rowptr[i] + A_row_sizes[i];

      // Check whether we need to reallocate space for A nonzero info
      if (A_new_rowptr[add_node_cnt + num_nodes - num_owned_nodes] > hypre_ParCompGridAMemSize(compGrid[level]))
      {
         num_resizes[3*level + 1]++;
         HYPRE_Int new_size = ceil(1.5*hypre_ParCompGridAMemSize(compGrid[level]));
         if (new_size < A_new_rowptr[add_node_cnt + num_nodes - num_owned_nodes]) new_size = A_new_rowptr[add_node_cnt + num_nodes - num_owned_nodes];
         hypre_ParCompGridResize(compGrid[level], new_size, level != num_levels-1, 1); // !!! Is there a better way to manage memory? !!!
      }

      // Move existing A col ind info
      for (i = num_nodes - 1; i >= num_owned_nodes; i--)
      {
         for (j = 0; j < hypre_ParCompGridARowPtr(compGrid[level])[ i+1 ] - hypre_ParCompGridARowPtr(compGrid[level])[ i ]; j++)
         {
            HYPRE_Int old_index = hypre_ParCompGridARowPtr(compGrid[level])[ i + 1 ] - 1 - j;
            HYPRE_Int new_index = A_new_rowptr[ compGrid_dest[i - num_owned_nodes] - num_owned_nodes + 1 ] - 1 - j;

            hypre_ParCompGridAColInd(compGrid[level])[ new_index ] = hypre_ParCompGridAColInd(compGrid[level])[ old_index ];
            hypre_ParCompGridAGlobalColInd(compGrid[level])[ new_index ] = hypre_ParCompGridAGlobalColInd(compGrid[level])[ old_index ];
         }
      }

      // Set new row ptr values
      for (i = num_owned_nodes; i < num_nodes + add_node_cnt + 1; i++) 
         hypre_ParCompGridARowPtr(compGrid[level])[i] = A_new_rowptr[i - num_owned_nodes];

      // Copy in new A global col ind info
      HYPRE_Int size_cnt = cnt - num_recv_nodes[current_level][buffer_number][level];
      for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++)
      {
         row_size = recv_buffer[size_cnt];

         if (incoming_dest[i] >= 0)
         {
            for (j = 0; j < row_size; j++)
            {
               HYPRE_Int index = hypre_ParCompGridARowPtr(compGrid[level])[ incoming_dest[i] ] + j;
               hypre_ParCompGridAGlobalColInd(compGrid[level])[ index ] = recv_buffer[cnt++];
            }
         }
         else
         {
            cnt += recv_buffer[size_cnt];
         }
         size_cnt++;
      }

      hypre_ParCompGridNumNodes(compGrid[level]) = num_nodes + add_node_cnt;

      // clean up memory
      hypre_TFree(A_row_sizes, HYPRE_MEMORY_HOST);
      hypre_TFree(A_new_rowptr, HYPRE_MEMORY_HOST);
      hypre_TFree(compGrid_dest, HYPRE_MEMORY_HOST);

      // !!! Debug
      end = chrono::system_clock::now();
      timings[6] += end - start; // Copy stuff for A
      
   }



   // !!! Debug

   auto total_end = chrono::system_clock::now();
   timings[0] = total_end - total_start;
   cout << "Rank " << myid << ", level " << current_level
                           << ": total " << timings[0].count() 
                           << ", allocate " << timings[1].count()
                           << ", merge " << timings[2].count()
                           << ", copy existing " << timings[3].count()
                           << ", Fix up prev send/recv " << timings[4].count()
                           << ", More copies " << timings[5].count()
                           << ", Copy stuff for A " << timings[6].count()
                           << endl;

   return 0;
}

#endif