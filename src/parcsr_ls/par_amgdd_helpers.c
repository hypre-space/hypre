/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"

HYPRE_Int
hypre_BoomerAMGDD_LocalToGlobalIndex( hypre_AMGDDCompGrid *compGrid,
                                      HYPRE_Int local_index )
{
   /* Local index starts with 0 at beginning of owned dofs and
      continues through  the nonowned (possible indices that are
      too large marking real overwriting ghost) */

   if (local_index < 0)
   {
      local_index = -(local_index + 1);
   }
   else if (local_index >= hypre_AMGDDCompGridNumOwnedNodes(compGrid) +
            hypre_AMGDDCompGridNumNonOwnedNodes(compGrid))
   {
      local_index -= hypre_AMGDDCompGridNumOwnedNodes(compGrid) +
                     hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);
   }

   if (local_index < hypre_AMGDDCompGridNumOwnedNodes(compGrid))
   {
      return local_index + hypre_AMGDDCompGridFirstGlobalIndex(compGrid);
   }
   else
   {
      return hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)[local_index -
                                                                            hypre_AMGDDCompGridNumOwnedNodes(compGrid)];
   }
}

HYPRE_Int
hypre_BoomerAMGDD_GetDofRecvProc( HYPRE_Int neighbor_local_index,
                                  hypre_ParCSRMatrix *A )
{
   // Use that column index to find which processor this dof is received from
   hypre_ParCSRCommPkg *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int recv_proc = -1;
   HYPRE_Int i;
   for (i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++)
   {
      if (neighbor_local_index >= hypre_ParCSRCommPkgRecvVecStart(commPkg, i) &&
          neighbor_local_index < hypre_ParCSRCommPkgRecvVecStart(commPkg, i + 1))
      {
         /* recv_proc = hypre_ParCSRCommPkgRecvProc(commPkg,i); */
         recv_proc = i;
         break;
      }
   }

   return recv_proc;
}

HYPRE_Int
hypre_BoomerAMGDD_RecursivelyFindNeighborNodes( HYPRE_Int            dof_index,
                                                HYPRE_Int            distance,
                                                hypre_ParCSRMatrix  *A,
                                                HYPRE_Int           *add_flag,
                                                HYPRE_Int           *add_flag_requests)
{
   hypre_CSRMatrix  *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix  *offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int         neighbor_index;
   HYPRE_Int         i;

   // Look at diag neighbors
   for (i = hypre_CSRMatrixI(diag)[dof_index]; i < hypre_CSRMatrixI(diag)[dof_index + 1]; i++)
   {
      // Get the index of the neighbor
      neighbor_index = hypre_CSRMatrixJ(diag)[i];

      // If the neighbor info is available on this proc
      // And if we still need to visit this index (note that send_dofs[neighbor_index] = distance means we have already added all distance-1 neighbors of index)

      // See whether this dof is in the send dofs
      if (add_flag[neighbor_index] < distance)
      {
         add_flag[neighbor_index] = distance;
         if (distance - 1 > 0)
         {
            hypre_BoomerAMGDD_RecursivelyFindNeighborNodes(neighbor_index, distance - 1, A, add_flag,
                                                           add_flag_requests);
         }
      }
   }

   // Look at offd neighbors
   for (i = hypre_CSRMatrixI(offd)[dof_index]; i < hypre_CSRMatrixI(offd)[dof_index + 1]; i++)
   {
      neighbor_index = hypre_CSRMatrixJ(offd)[i];

      if (add_flag_requests[neighbor_index] < distance)
      {
         add_flag_requests[neighbor_index] = distance;
      }
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_AddToSendAndRequestDofs( hypre_ParCSRMatrix     *A,
                                           HYPRE_Int              *add_flag,
                                           HYPRE_Int              *add_flag_requests,
                                           HYPRE_Int               level,
                                           HYPRE_Int               send_proc,
                                           hypre_AMGDDCommPkg     *compGridCommPkg,
                                           HYPRE_Int             **distances,
                                           hypre_UnorderedIntMap **send_dof_maps,
                                           HYPRE_Int              *send_dof_capacities,
                                           HYPRE_Int               num_csr_recv_procs,
                                           HYPRE_Int             **num_req_dofs,
                                           HYPRE_Int            ***req_dofs,
                                           HYPRE_Int            ***req_dof_dist )
{
   HYPRE_Int  *req_cnt;
   HYPRE_Int   neighbor_global_index;
   HYPRE_Int   recv_proc;

   HYPRE_Int   i, idx;

   for (i = 0; i < hypre_ParCSRMatrixNumRows(A); i++)
   {
      if (add_flag[i])
      {
         idx = hypre_UnorderedIntMapGet(send_dof_maps[send_proc],
                                        i); // Recall: key = local owned dof idx, data = idx into sendflag array
         if (idx == -1)
         {
            idx = hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][send_proc][level]++;
            // Check whether a resize of the send dofs (and related arrays) is necessary
            if (idx == send_dof_capacities[send_proc])
            {
               send_dof_capacities[send_proc] *= 2;
               hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][send_proc][level] = hypre_TReAlloc(
                                                                                         hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][send_proc][level],
                                                                                         HYPRE_Int, send_dof_capacities[send_proc], HYPRE_MEMORY_HOST);
               distances[send_proc] = hypre_TReAlloc(distances[send_proc], HYPRE_Int,
                                                     send_dof_capacities[send_proc], HYPRE_MEMORY_HOST);
            }
            hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][send_proc][level][idx] = i;
            distances[send_proc][idx] = add_flag[i];
            hypre_UnorderedIntMapPutIfAbsent(send_dof_maps[send_proc], i, idx);
         }
         else if (distances[send_proc][idx] < add_flag[i])
         {
            distances[send_proc][idx] = add_flag[i];
         }
      }
   }
   // Count request dofs and alloc/realloc req_dofs and req_dof_dist
   req_cnt = hypre_CTAlloc(HYPRE_Int, num_csr_recv_procs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_csr_recv_procs; i++)
   {
      req_cnt[i] = num_req_dofs[i][send_proc];
   }
   for (i = 0; i < hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)); i++)
   {
      if (add_flag_requests[i])
      {
         recv_proc = hypre_BoomerAMGDD_GetDofRecvProc(i, A);
         if (hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][recv_proc] != hypre_AMGDDCommPkgSendProcs(
                compGridCommPkg)[level][send_proc])
         {
            num_req_dofs[recv_proc][send_proc]++;
         }
      }
   }
   for (i = 0; i < num_csr_recv_procs; i++)
   {
      if (num_req_dofs[i][send_proc])
      {
         if (req_dofs[i][send_proc])
         {
            req_dofs[i][send_proc] = hypre_TReAlloc(req_dofs[i][send_proc], HYPRE_Int,
                                                    num_req_dofs[i][send_proc], HYPRE_MEMORY_HOST);
            req_dof_dist[i][send_proc] = hypre_TReAlloc(req_dof_dist[i][send_proc], HYPRE_Int,
                                                        num_req_dofs[i][send_proc], HYPRE_MEMORY_HOST);
         }
         else
         {
            req_dofs[i][send_proc] = hypre_CTAlloc(HYPRE_Int, num_req_dofs[i][send_proc], HYPRE_MEMORY_HOST);
            req_dof_dist[i][send_proc] = hypre_CTAlloc(HYPRE_Int, num_req_dofs[i][send_proc],
                                                       HYPRE_MEMORY_HOST);
         }
      }
   }
   // Fill the req dof info
   for (i = 0; i < hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)); i++)
   {
      if (add_flag_requests[i])
      {
         neighbor_global_index = hypre_ParCSRMatrixColMapOffd(A)[i];
         recv_proc = hypre_BoomerAMGDD_GetDofRecvProc(i, A);
         if (hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][recv_proc] != hypre_AMGDDCommPkgSendProcs(
                compGridCommPkg)[level][send_proc])
         {
            req_dofs[recv_proc][send_proc][ req_cnt[recv_proc] ] = neighbor_global_index;
            req_dof_dist[recv_proc][send_proc][ req_cnt[recv_proc] ] = add_flag_requests[i];
            req_cnt[recv_proc]++;
         }
      }
   }
   // Clean up memory
   hypre_TFree(req_cnt, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_FindNeighborProcessors( hypre_ParCSRMatrix      *A,
                                          hypre_AMGDDCommPkg      *compGridCommPkg,
                                          HYPRE_Int             ***distances_ptr,
                                          hypre_UnorderedIntMap ***send_dof_maps_ptr,
                                          HYPRE_Int               *send_proc_capacity_ptr,
                                          HYPRE_Int              **send_dof_capacities_ptr,
                                          HYPRE_Int               *recv_proc_capacity_ptr,
                                          HYPRE_Int             ***starting_dofs_ptr,
                                          HYPRE_Int              **num_starting_dofs_ptr,
                                          HYPRE_Int                level,
                                          HYPRE_Int                max_distance )
{
   HYPRE_Int **distances = *distances_ptr;
   hypre_UnorderedIntMap **send_dof_maps = *send_dof_maps_ptr;
   HYPRE_Int send_proc_capacity = *send_proc_capacity_ptr;
   HYPRE_Int *send_dof_capacities = *send_dof_capacities_ptr;
   HYPRE_Int recv_proc_capacity = *recv_proc_capacity_ptr;
   HYPRE_Int **starting_dofs = *starting_dofs_ptr;
   HYPRE_Int *num_starting_dofs = *num_starting_dofs_ptr;

   hypre_ParCSRCommPkg *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int csr_num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
   HYPRE_Int csr_num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);

   HYPRE_Int i, j, k;

   // Nodes to request from other processors. Note, requests are only issued to processors within distance 1, i.e. within the original communication stencil for A
   HYPRE_Int **num_req_dofs = hypre_CTAlloc(HYPRE_Int*, csr_num_recvs, HYPRE_MEMORY_HOST);
   HYPRE_Int ***req_dofs = hypre_CTAlloc(HYPRE_Int**, csr_num_recvs, HYPRE_MEMORY_HOST);
   HYPRE_Int ***req_dof_dist = hypre_CTAlloc(HYPRE_Int**, csr_num_recvs, HYPRE_MEMORY_HOST);
   for (i = 0; i < csr_num_recvs; i++)
   {
      num_req_dofs[i] = hypre_CTAlloc(HYPRE_Int, hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level],
                                      HYPRE_MEMORY_HOST);
      req_dofs[i] = hypre_CTAlloc(HYPRE_Int*, hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level],
                                  HYPRE_MEMORY_HOST);
      req_dof_dist[i] = hypre_CTAlloc(HYPRE_Int*, hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level],
                                      HYPRE_MEMORY_HOST);
   }

   HYPRE_Int *add_flag = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A), HYPRE_MEMORY_HOST);
   HYPRE_Int *add_flag_requests = hypre_CTAlloc(HYPRE_Int,
                                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST);

   // Recursively search through the operator stencil to find longer distance neighboring dofs
   // Loop over longdistance send procs
   for (i = 0; i < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; i++)
   {
      if (num_starting_dofs[i])
      {
         // Initialize the add_flag at the starting dofs
         for (j = 0; j < num_starting_dofs[i]; j++)
         {
            HYPRE_Int idx = starting_dofs[i][j];
            HYPRE_Int send_dof = hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][i][level][idx];
            add_flag[send_dof] = distances[i][idx];
         }
         // Recursively search for longer distance dofs
         for (j = 0; j < num_starting_dofs[i]; j++)
         {
            HYPRE_Int idx = starting_dofs[i][j];
            HYPRE_Int send_dof = hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][i][level][idx];
            hypre_BoomerAMGDD_RecursivelyFindNeighborNodes(send_dof, distances[i][idx] - 1, A, add_flag,
                                                           add_flag_requests);
         }
         num_starting_dofs[i] = 0;
         hypre_TFree(starting_dofs[i], HYPRE_MEMORY_HOST);
         starting_dofs[i] = NULL;
         // Update the send flag and request dofs
         hypre_BoomerAMGDD_AddToSendAndRequestDofs(A, add_flag, add_flag_requests, level, i, compGridCommPkg,
                                                   distances, send_dof_maps, send_dof_capacities, csr_num_recvs, num_req_dofs, req_dofs, req_dof_dist);
         // Reset add flags
         hypre_Memset(add_flag, 0, sizeof(HYPRE_Int)*hypre_ParCSRMatrixNumRows(A), HYPRE_MEMORY_HOST);
         hypre_Memset(add_flag_requests, 0,
                      sizeof(HYPRE_Int)*hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST);
      }
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
   hypre_TFree(add_flag_requests, HYPRE_MEMORY_HOST);

   //////////////////////////////////////////////////
   // Communicate newly connected longer-distance processors to send procs:
   // sending to current longdistance send_procs and receiving from current
   // longdistance recv_procs
   //////////////////////////////////////////////////

   // Get the sizes
   hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request,
                                               hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level] + hypre_AMGDDCommPkgNumRecvProcs(
                                                  compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   hypre_MPI_Status *statuses = hypre_CTAlloc(hypre_MPI_Status,
                                              hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level] + hypre_AMGDDCommPkgNumRecvProcs(
                                                 compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   HYPRE_Int request_cnt = 0;

   HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int,
                                         hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   for (i = 0; i < hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level]; i++)
   {
      hypre_MPI_Irecv(&(recv_sizes[i]), 1, HYPRE_MPI_INT,
                      hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 6, hypre_MPI_COMM_WORLD,
                      &(requests[request_cnt++]));
   }
   HYPRE_Int *send_sizes = hypre_CTAlloc(HYPRE_Int,
                                         hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   for (i = 0; i < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; i++)
   {
      for (j = 0; j < csr_num_recvs; j++)
      {
         if (num_req_dofs[j][i])
         {
            send_sizes[i]++;
         }
      }
      hypre_MPI_Isend(&(send_sizes[i]), 1, HYPRE_MPI_INT,
                      hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 6, hypre_MPI_COMM_WORLD,
                      &(requests[request_cnt++]));
   }

   // Wait
   hypre_MPI_Waitall(hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level] +
                     hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level], requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request,
                            hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level] + hypre_AMGDDCommPkgNumRecvProcs(
                               compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status,
                            hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level] + hypre_AMGDDCommPkgNumRecvProcs(
                               compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   request_cnt = 0;

   // Allocate and post the recvs
   HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*,
                                            hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   for (i = 0; i < hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level]; i++)
   {
      recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_sizes[i], HYPRE_MEMORY_HOST);
      hypre_MPI_Irecv(recv_buffers[i], recv_sizes[i], HYPRE_MPI_INT,
                      hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 7, hypre_MPI_COMM_WORLD,
                      &(requests[request_cnt++]));
   }
   // Setup and send the send buffers
   HYPRE_Int **send_buffers = hypre_CTAlloc(HYPRE_Int*,
                                            hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level], HYPRE_MEMORY_HOST);
   for (i = 0; i < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; i++)
   {
      send_buffers[i] = hypre_CTAlloc(HYPRE_Int, send_sizes[i], HYPRE_MEMORY_HOST);
      HYPRE_Int inner_cnt = 0;
      for (j = 0; j < csr_num_recvs; j++)
      {
         if (num_req_dofs[j][i])
         {
            send_buffers[i][inner_cnt++] = hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][j];
         }

      }
      hypre_MPI_Isend(send_buffers[i], send_sizes[i], HYPRE_MPI_INT,
                      hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 7, hypre_MPI_COMM_WORLD,
                      &(requests[request_cnt++]));
   }

   // Wait
   hypre_MPI_Waitall(hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level] +
                     hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level], requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);

   // Update recv_procs
   HYPRE_Int old_num_recv_procs = hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level];
   for (i = 0; i < old_num_recv_procs; i++)
   {
      for (j = 0; j < recv_sizes[i]; j++)
      {
         // For each incoming longer-distance recv proc, need to check whether it is already accounted for
         // !!! Optimization: can add a hypre set here for looking up previous recv procs if necessary (replace linear search)
         HYPRE_Int accounted_for = 0;
         for (k = 0; k < hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level]; k++)
         {
            if (hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][k] == recv_buffers[i][j])
            {
               accounted_for = 1;
               break;
            }
         }
         if (!accounted_for)
         {
            // Check whether we need to reallocate
            if (hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level] == recv_proc_capacity)
            {
               recv_proc_capacity *= 2;
               hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level] = hypre_TReAlloc(hypre_AMGDDCommPkgRecvProcs(
                                                                                       compGridCommPkg)[level], HYPRE_Int, recv_proc_capacity, HYPRE_MEMORY_HOST);
            }
            hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][ hypre_AMGDDCommPkgNumRecvProcs(
                                                                    compGridCommPkg)[level]++ ] = recv_buffers[i][j];
         }
      }
   }

   // Clean up memory
   for (i = 0; i < old_num_recv_procs; i++)
   {
      hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
   }
   for (i = 0; i < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; i++)
   {
      hypre_TFree(send_buffers[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_sizes, HYPRE_MEMORY_HOST);

   //////////////////////////////////////////////////
   // Communicate request dofs to processors that I recv from: sending to request_procs and receiving from distance 1 send procs
   //////////////////////////////////////////////////

   // Count up the send size: 1 + sum_{destination_procs}(2 + 2*num_requested_dofs)
   // send_buffer = [num destination procs, [request info for proc], [request info for proc], ... ]
   // [request info for proc] = [proc id, num requested dofs, [(dof index, distance), (dof index, distance), ...] ]

   // Exchange message sizes
   send_sizes = hypre_CTAlloc(HYPRE_Int, csr_num_recvs, HYPRE_MEMORY_HOST);
   recv_sizes = hypre_CTAlloc(HYPRE_Int, csr_num_sends, HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request, csr_num_sends + csr_num_recvs, HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status, csr_num_sends + csr_num_recvs, HYPRE_MEMORY_HOST);
   request_cnt = 0;
   for (i = 0; i < csr_num_sends; i++)
   {
      hypre_MPI_Irecv(&(recv_sizes[i]), 1, HYPRE_MPI_INT, hypre_ParCSRCommPkgSendProc(commPkg, i), 4,
                      hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   for (i = 0; i < csr_num_recvs; i++)
   {
      send_sizes[i]++;
      for (j = 0; j < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; j++)
      {
         if (num_req_dofs[i][j])
         {
            send_sizes[i] += 2 + 2 * num_req_dofs[i][j];
         }
      }
      hypre_MPI_Isend(&(send_sizes[i]), 1, HYPRE_MPI_INT, hypre_ParCSRCommPkgRecvProc(commPkg, i), 4,
                      hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }

   // Wait on the recv sizes, then free and re-allocate the requests and statuses
   hypre_MPI_Waitall(csr_num_sends + csr_num_recvs, requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request, csr_num_sends + csr_num_recvs, HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status, csr_num_sends + csr_num_recvs, HYPRE_MEMORY_HOST);
   request_cnt = 0;

   // Allocate recv buffers and post the recvs
   recv_buffers = hypre_CTAlloc(HYPRE_Int*, csr_num_sends, HYPRE_MEMORY_HOST);
   for (i = 0; i < csr_num_sends; i++)
   {
      recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_sizes[i], HYPRE_MEMORY_HOST);
      hypre_MPI_Irecv(recv_buffers[i], recv_sizes[i], HYPRE_MPI_INT, hypre_ParCSRCommPkgSendProc(commPkg,
                                                                                                 i), 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }

   // Setup the send buffer and post the sends
   send_buffers = hypre_CTAlloc(HYPRE_Int*, csr_num_recvs, HYPRE_MEMORY_HOST);
   for (i = 0; i < csr_num_recvs; i++)
   {
      send_buffers[i] = hypre_CTAlloc(HYPRE_Int, send_sizes[i], HYPRE_MEMORY_HOST);
      HYPRE_Int inner_cnt = 1;
      for (j = 0; j < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; j++)
      {
         if (num_req_dofs[i][j])
         {
            send_buffers[i][0]++;
            send_buffers[i][inner_cnt++] = hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][j];
            send_buffers[i][inner_cnt++] = num_req_dofs[i][j];
            for (k = 0; k < num_req_dofs[i][j]; k++)
            {
               send_buffers[i][inner_cnt++] = req_dofs[i][j][k];
               send_buffers[i][inner_cnt++] = req_dof_dist[i][j][k];
            }
         }
      }
      hypre_MPI_Isend(send_buffers[i], send_sizes[i], HYPRE_MPI_INT, hypre_ParCSRCommPkgRecvProc(commPkg,
                                                                                                 i), 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   // Free the req dof info
   for (i = 0; i < csr_num_recvs; i++)
   {
      for (j = 0; j < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; j++)
      {
         if (req_dofs[i][j])
         {
            hypre_TFree(req_dofs[i][j], HYPRE_MEMORY_HOST);
            hypre_TFree(req_dof_dist[i][j], HYPRE_MEMORY_HOST);
         }
      }
      hypre_TFree(num_req_dofs[i], HYPRE_MEMORY_HOST);
      hypre_TFree(req_dofs[i], HYPRE_MEMORY_HOST);
      hypre_TFree(req_dof_dist[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(num_req_dofs, HYPRE_MEMORY_HOST);
   hypre_TFree(req_dofs, HYPRE_MEMORY_HOST);
   hypre_TFree(req_dof_dist, HYPRE_MEMORY_HOST);

   // Wait
   hypre_MPI_Waitall(csr_num_sends + csr_num_recvs, requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);

   // Update send_proc_dofs and starting_dofs
   // Loop over send_proc's, i.e. the processors that we just received from
   for (i = 0; i < csr_num_sends; i++)
   {
      HYPRE_Int cnt = 0;
      HYPRE_Int num_destination_procs = recv_buffers[i][cnt++];
      HYPRE_Int destination_proc;
      for (destination_proc = 0; destination_proc < num_destination_procs; destination_proc++)
      {
         // Get destination proc id and the number of requested dofs
         HYPRE_Int proc_id = recv_buffers[i][cnt++];
         HYPRE_Int num_requested_dofs = recv_buffers[i][cnt++];

         // create new map for this destination proc if it doesn't already exist
         HYPRE_Int new_proc = 0;
         HYPRE_Int p_idx = -1;
         for (j = 0; j < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; j++)
         {
            if (hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][j] == proc_id)
            {
               p_idx = j;
               break;
            }
         }
         if (p_idx < 0)
         {
            new_proc = 1;
            p_idx = hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level];
            if (p_idx == send_proc_capacity)
            {
               send_proc_capacity *= 2;
               hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level] = hypre_TReAlloc(hypre_AMGDDCommPkgSendProcs(
                                                                                       compGridCommPkg)[level], HYPRE_Int, send_proc_capacity, HYPRE_MEMORY_HOST);
               hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level] = hypre_TReAlloc(hypre_AMGDDCommPkgSendFlag(
                                                                                      compGridCommPkg)[level], HYPRE_Int**, send_proc_capacity, HYPRE_MEMORY_HOST);
               hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level] = hypre_TReAlloc(
                                                                           hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level], HYPRE_Int*, send_proc_capacity,
                                                                           HYPRE_MEMORY_HOST);
               starting_dofs = hypre_TReAlloc(starting_dofs, HYPRE_Int*, send_proc_capacity, HYPRE_MEMORY_HOST);
               num_starting_dofs = hypre_TReAlloc(num_starting_dofs, HYPRE_Int, send_proc_capacity,
                                                  HYPRE_MEMORY_HOST);
               send_dof_capacities = hypre_TReAlloc(send_dof_capacities, HYPRE_Int, send_proc_capacity,
                                                    HYPRE_MEMORY_HOST);
               distances = hypre_TReAlloc(distances, HYPRE_Int*, send_proc_capacity, HYPRE_MEMORY_HOST);
               send_dof_maps = hypre_TReAlloc(send_dof_maps, hypre_UnorderedIntMap*, send_proc_capacity,
                                              HYPRE_MEMORY_HOST);
            }
            hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]++;
            hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][p_idx] = proc_id;
            hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][p_idx] = hypre_CTAlloc(HYPRE_Int*,
                                                                                      hypre_AMGDDCommPkgNumLevels(compGridCommPkg), HYPRE_MEMORY_HOST);
            hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][p_idx] = hypre_CTAlloc(HYPRE_Int,
                                                                                          hypre_AMGDDCommPkgNumLevels(compGridCommPkg), HYPRE_MEMORY_HOST);
            send_dof_capacities[p_idx] = num_requested_dofs;
            hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][p_idx][level] = hypre_CTAlloc(HYPRE_Int,
                                                                                             send_dof_capacities[p_idx], HYPRE_MEMORY_HOST);
            distances[p_idx] = hypre_CTAlloc(HYPRE_Int, send_dof_capacities[p_idx], HYPRE_MEMORY_HOST);
            send_dof_maps[p_idx] = hypre_CTAlloc(hypre_UnorderedIntMap, 1, HYPRE_MEMORY_HOST);
            hypre_UnorderedIntMapCreate(send_dof_maps[p_idx], 2 * max_distance * num_requested_dofs,
                                        16 * hypre_NumThreads()); // !!! Is this a "safe" upper bound on map size?
            starting_dofs[p_idx] = hypre_CTAlloc(HYPRE_Int, num_requested_dofs, HYPRE_MEMORY_HOST);
            num_starting_dofs[p_idx] = 0;
         }
         else
         {
            if (starting_dofs[p_idx])
            {
               starting_dofs[p_idx] = hypre_TReAlloc(starting_dofs[p_idx], HYPRE_Int,
                                                     num_starting_dofs[p_idx] + num_requested_dofs, HYPRE_MEMORY_HOST);
            }
            else
            {
               starting_dofs[p_idx] = hypre_CTAlloc(HYPRE_Int, num_requested_dofs, HYPRE_MEMORY_HOST);
            }
         }

         // Loop over the requested dofs for this destination proc
         HYPRE_Int j;
         for (j = 0; j < num_requested_dofs; j++)
         {
            // Get the local index for this dof on this processor
            HYPRE_Int req_dof_local_index = recv_buffers[i][cnt++] - hypre_ParCSRMatrixFirstRowIndex(A);
            HYPRE_Int req_dof_incoming_dist = recv_buffers[i][cnt++];

            // If this proc alreay has send dofs, look up to see whether this dof is already accounted for
            HYPRE_Int d_idx = -1;
            if (!new_proc)
            {
               d_idx = hypre_UnorderedIntMapGet(send_dof_maps[p_idx], req_dof_local_index);
            }
            // If dof is not found, then add local index and distance info, and add to starting dofs
            if (d_idx < 0)
            {
               d_idx = hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][p_idx][level];
               // Realloc if necessary
               if (d_idx == send_dof_capacities[p_idx])
               {
                  send_dof_capacities[p_idx] *= 2;
                  hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][p_idx][level] = hypre_TReAlloc(
                                                                                        hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][p_idx][level],
                                                                                        HYPRE_Int, send_dof_capacities[p_idx], HYPRE_MEMORY_HOST);
                  distances[p_idx] = hypre_TReAlloc(distances[p_idx], HYPRE_Int, send_dof_capacities[p_idx],
                                                    HYPRE_MEMORY_HOST);
               }
               hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][p_idx][level]++;
               hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][p_idx][level][d_idx] = req_dof_local_index;
               hypre_UnorderedIntMapPutIfAbsent(send_dof_maps[p_idx], req_dof_local_index, d_idx);
               distances[p_idx][d_idx] = req_dof_incoming_dist;
               starting_dofs[p_idx][ num_starting_dofs[p_idx]++ ] = d_idx;
            }
            // If dof is found, but at a closer distance (larger dist value), then update distances and add to starting dofs
            else if (distances[p_idx][d_idx] < req_dof_incoming_dist)
            {
               distances[p_idx][d_idx] = req_dof_incoming_dist;
               starting_dofs[p_idx][ num_starting_dofs[p_idx]++ ] = d_idx;
            }
         }
      }
   }

   // Clean up memory
   for (i = 0; i < csr_num_sends; i++)
   {
      hypre_TFree(recv_buffers[i], HYPRE_MEMORY_HOST);
   }
   for (i = 0; i < csr_num_recvs; i++)
   {
      hypre_TFree(send_buffers[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_sizes, HYPRE_MEMORY_HOST);

   // Return ptrs in case of reallocation
   (*distances_ptr) = distances;
   (*send_dof_maps_ptr) = send_dof_maps;
   (*send_proc_capacity_ptr) = send_proc_capacity;
   (*send_dof_capacities_ptr) = send_dof_capacities;
   (*recv_proc_capacity_ptr) = recv_proc_capacity;
   (*starting_dofs_ptr) = starting_dofs;
   (*num_starting_dofs_ptr) = num_starting_dofs;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A,
                                                  hypre_AMGDDCommPkg *compGridCommPkg,
                                                  HYPRE_Int           level,
                                                  HYPRE_Int          *padding,
                                                  HYPRE_Int           num_ghost_layers)
{
   hypre_ParCSRCommPkg     *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               start, finish;
   HYPRE_Int               i, j;
   HYPRE_Int               num_levels = hypre_AMGDDCommPkgNumLevels(compGridCommPkg);
   HYPRE_Int               max_distance = padding[level] + num_ghost_layers;

   // Get the default (distance 1) number of send and recv procs
   HYPRE_Int               csr_num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
   HYPRE_Int               csr_num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);

   // If csr_num_sends and csr_num_recvs are zero, then simply note that in compGridCommPkg and we are done
   if (csr_num_sends == 0 && csr_num_recvs == 0)
   {
      hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level] = 0;
      hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level] = 0;
   }
   else
   {
      // Initialize send info (send procs, send dofs, starting dofs, distances, map for lookups in send dofs)
      hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level] = csr_num_sends;
      HYPRE_Int send_proc_capacity = 2 * csr_num_sends;
      hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, send_proc_capacity,
                                                                          HYPRE_MEMORY_HOST);
      hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int*,
                                                                             send_proc_capacity, HYPRE_MEMORY_HOST);
      hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int**, send_proc_capacity,
                                                                         HYPRE_MEMORY_HOST);
      HYPRE_Int *num_starting_dofs = hypre_CTAlloc(HYPRE_Int, send_proc_capacity, HYPRE_MEMORY_HOST);
      HYPRE_Int *send_dof_capacities = hypre_CTAlloc(HYPRE_Int, send_proc_capacity, HYPRE_MEMORY_HOST);
      HYPRE_Int **starting_dofs = hypre_CTAlloc(HYPRE_Int*, send_proc_capacity, HYPRE_MEMORY_HOST);
      HYPRE_Int **distances = hypre_CTAlloc(HYPRE_Int*, send_proc_capacity, HYPRE_MEMORY_HOST);
      hypre_UnorderedIntMap **send_dof_maps = hypre_CTAlloc(hypre_UnorderedIntMap*, send_proc_capacity,
                                                            HYPRE_MEMORY_HOST);
      for (i = 0; i < csr_num_sends; i++)
      {
         hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i] = hypre_ParCSRCommPkgSendProc(commPkg, i);
         start = hypre_ParCSRCommPkgSendMapStart(commPkg, i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg, i + 1);
         HYPRE_Int num_send_dofs = finish - start;
         send_dof_capacities[i] = max_distance * num_send_dofs;
         num_starting_dofs[i] = num_send_dofs;
         hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][i] = hypre_CTAlloc(HYPRE_Int, num_levels,
                                                                                   HYPRE_MEMORY_HOST);
         hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][i][level] = num_send_dofs;
         hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][i] = hypre_CTAlloc(HYPRE_Int*, num_levels,
                                                                               HYPRE_MEMORY_HOST);
         hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][i][level] = hypre_CTAlloc(HYPRE_Int,
                                                                                      send_dof_capacities[i], HYPRE_MEMORY_HOST);
         starting_dofs[i] = hypre_CTAlloc(HYPRE_Int, num_send_dofs, HYPRE_MEMORY_HOST);
         distances[i] = hypre_CTAlloc(HYPRE_Int, send_dof_capacities[i], HYPRE_MEMORY_HOST);
         send_dof_maps[i] = hypre_CTAlloc(hypre_UnorderedIntMap, 1, HYPRE_MEMORY_HOST);
         hypre_UnorderedIntMapCreate(send_dof_maps[i],
                                     2 * max_distance * num_send_dofs, // !!! Is this a "safe" upper bound on the map size?
                                     16 * hypre_NumThreads());
         for (j = start; j < finish; j++)
         {
            HYPRE_Int send_dof = hypre_ParCSRCommPkgSendMapElmt(commPkg, j);
            hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][i][level][j - start] = send_dof;
            starting_dofs[i][j - start] = j - start;
            distances[i][j - start] = max_distance;
            hypre_UnorderedIntMapPutIfAbsent(send_dof_maps[i], send_dof, j - start);
         }
      }

      //Initialize the recv_procs
      hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level] = csr_num_recvs;
      HYPRE_Int recv_proc_capacity = 2 * csr_num_recvs;
      hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, recv_proc_capacity,
                                                                          HYPRE_MEMORY_HOST);
      for (i = 0; i < csr_num_recvs; i++)
      {
         hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i] = hypre_ParCSRCommPkgRecvProc(commPkg, i);
      }

      // Iteratively communicate with longer and longer distance neighbors to grow the communication stencils
      for (i = 0; i < max_distance - 1; i++)
      {
         hypre_BoomerAMGDD_FindNeighborProcessors(A,
                                                  compGridCommPkg,
                                                  &distances,
                                                  &send_dof_maps,
                                                  &send_proc_capacity,
                                                  &send_dof_capacities,
                                                  &recv_proc_capacity,
                                                  &starting_dofs,
                                                  &num_starting_dofs,
                                                  level, max_distance);
      }
      // Update sendflag to encode ghost layers with negative mapped indices and enforce global index ordering beyond original send dofs
      for (i = 0; i < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; i++)
      {
         HYPRE_Int num_orig_sends = 0;
         if (i < csr_num_sends)
         {
            num_orig_sends = hypre_ParCSRCommPkgSendMapStart(commPkg,
                                                             i + 1) - hypre_ParCSRCommPkgSendMapStart(commPkg, i);
         }
         hypre_qsort0(hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][i][level], num_orig_sends,
                      hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][i][level] - 1);

         for (j = num_orig_sends; j < hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[level][i][level]; j++)
         {
            if (distances[i][j] <= num_ghost_layers)
            {
               hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[level][i][level][j] = -(hypre_AMGDDCommPkgSendFlag(
                                                                                      compGridCommPkg)[level][i][level][j] + 1 );
            }
         }
      }
      // Clean up memory
      for (i = 0; i < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level]; i++)
      {
         if (starting_dofs[i])
         {
            hypre_TFree(starting_dofs[i], HYPRE_MEMORY_HOST);
         }
         if (distances[i])
         {
            hypre_TFree(distances[i], HYPRE_MEMORY_HOST);
         }
         if (send_dof_maps[i])
         {
            hypre_UnorderedIntMapDestroy(send_dof_maps[i]);
            hypre_TFree(send_dof_maps[i], HYPRE_MEMORY_HOST);
         }
      }
      hypre_TFree(num_starting_dofs, HYPRE_MEMORY_HOST);
      hypre_TFree(send_dof_capacities, HYPRE_MEMORY_HOST);
      hypre_TFree(starting_dofs, HYPRE_MEMORY_HOST);
      hypre_TFree(distances, HYPRE_MEMORY_HOST);
      hypre_TFree(send_dof_maps, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_UnpackRecvBuffer( hypre_ParAMGDDData *amgdd_data,
                                    HYPRE_Int          *recv_buffer,
                                    HYPRE_Int         **A_tmp_info,
                                    HYPRE_Int          *recv_map_send_buffer_size,
                                    HYPRE_Int          *nodes_added_on_level,
                                    HYPRE_Int           current_level,
                                    HYPRE_Int           buffer_number )
{
   // recv_buffer = [ num_psi_levels , [level] , [level] , ... ]
   // level = [ num send nodes, [global indices] , [coarse global indices] , [A row sizes] , [A col ind] ]

   hypre_ParAMGData        *amg_data        = hypre_ParAMGDDDataAMG(amgdd_data);
   hypre_AMGDDCompGrid    **compGrid        = hypre_ParAMGDDDataCompGrid(amgdd_data);
   hypre_AMGDDCommPkg      *compGridCommPkg = hypre_ParAMGDDDataCommPkg(amgdd_data);
   hypre_ParCSRCommPkg     *commPkg         = hypre_ParCSRMatrixCommPkg(hypre_ParAMGDataAArray(
                                                                           amg_data)[current_level]);

   HYPRE_Int                num_levels      = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int            ****recv_map        = hypre_AMGDDCommPkgRecvMap(compGridCommPkg);
   HYPRE_Int             ***num_recv_nodes  = hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg);
   HYPRE_Int            ****recv_red_marker = hypre_AMGDDCommPkgRecvRedMarker(compGridCommPkg);

   HYPRE_Int                level, i, j, cnt;
   HYPRE_Int                num_psi_levels;
   HYPRE_Int                level_start;
   HYPRE_Int                add_node_cnt;

   // initialize the counter
   cnt = 0;

   // get the number of levels received
   num_psi_levels = recv_buffer[cnt++];

   // Init the recv_map_send_buffer_size !!! I think this can just be set a priori instead of counting it up in this function... !!!
   *recv_map_send_buffer_size = num_levels - current_level - 1;

   ////////////////////////////////////////////////////////////////////
   // Treat current_level specially: no redundancy here, and recv positions need to agree with original ParCSRCommPkg (extra comp grid points at the end)
   ////////////////////////////////////////////////////////////////////

   // Get the compgrid matrix, specifically the nonowned parts that will be added to
   hypre_AMGDDCompGridMatrix *A = hypre_AMGDDCompGridA(compGrid[current_level]);
   hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   hypre_CSRMatrix *nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   // get the number of nodes on this level
   num_recv_nodes[current_level][buffer_number][current_level] = recv_buffer[cnt++];
   nodes_added_on_level[current_level] += num_recv_nodes[current_level][buffer_number][current_level];

   // if necessary, reallocate more space for nonowned dofs
   HYPRE_Int max_nonowned = hypre_CSRMatrixNumRows(nonowned_diag);
   HYPRE_Int start_extra_dofs = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[current_level]);
   if (num_recv_nodes[current_level][buffer_number][current_level] + start_extra_dofs > max_nonowned)
   {
      HYPRE_Int new_size = (HYPRE_Int)hypre_ceil(1.5 * max_nonowned);
      if (new_size < num_recv_nodes[current_level][buffer_number][current_level] + start_extra_dofs)
      {
         new_size = num_recv_nodes[current_level][buffer_number][current_level] + start_extra_dofs;
      }
      hypre_AMGDDCompGridResize(compGrid[current_level], new_size,
                                current_level != num_levels - 1); // !!! Is there a better way to manage memory? !!!
   }

   // Get the original number of recv dofs in the ParCSRCommPkg (if this proc was recv'd from in original)
   HYPRE_Int num_original_recv_dofs = 0;
   if (commPkg)
      if (buffer_number < hypre_ParCSRCommPkgNumRecvs(commPkg))
      {
         num_original_recv_dofs = hypre_ParCSRCommPkgRecvVecStart(commPkg,
                                                                  buffer_number + 1) - hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number);
      }

   // Skip over original commPkg recv dofs !!! Optimization: can avoid sending GIDs here
   HYPRE_Int remaining_dofs = num_recv_nodes[current_level][buffer_number][current_level] -
                              num_original_recv_dofs;
   cnt += num_original_recv_dofs;

   // Setup the recv map on current level
   recv_map[current_level][buffer_number][current_level] = hypre_CTAlloc(HYPRE_Int,
                                                                         num_recv_nodes[current_level][buffer_number][current_level], HYPRE_MEMORY_HOST);
   for (i = 0; i < num_original_recv_dofs; i++)
   {
      recv_map[current_level][buffer_number][current_level][i] = i + hypre_ParCSRCommPkgRecvVecStart(
                                                                    commPkg, buffer_number) + hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]);
   }

   // Unpack global indices and setup sort and invsort
   hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[current_level]) += remaining_dofs;
   HYPRE_Int *sort_map = hypre_AMGDDCompGridNonOwnedSort(compGrid[current_level]);
   HYPRE_Int *inv_sort_map = hypre_AMGDDCompGridNonOwnedInvSort(compGrid[current_level]);
   HYPRE_Int *new_inv_sort_map = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(nonowned_diag),
                                               hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
   HYPRE_Int sort_cnt = 0;
   HYPRE_Int compGrid_cnt = 0;
   HYPRE_Int incoming_cnt = 0;
   while (incoming_cnt < remaining_dofs && compGrid_cnt < start_extra_dofs)
   {
      // !!! Optimization: don't have to do these assignments every time... probably doesn't save much (i.e. only update incoming_global_index when necessary, etc.)
      HYPRE_Int incoming_global_index = recv_buffer[cnt];
      HYPRE_Int compGrid_global_index = hypre_AMGDDCompGridNonOwnedGlobalIndices(
                                           compGrid[current_level])[ inv_sort_map[compGrid_cnt] ];

      HYPRE_Int incoming_is_real = 1;
      if (incoming_global_index < 0)
      {
         incoming_global_index = -(incoming_global_index + 1);
         incoming_is_real = 0;
      }

      if (incoming_global_index < compGrid_global_index)
      {
         // Set global index and real marker for incoming extra dof
         hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[current_level])[ incoming_cnt + start_extra_dofs ]
            = incoming_global_index;
         hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[current_level])[ incoming_cnt + start_extra_dofs ] =
            incoming_is_real;

         if (incoming_is_real)
         {
            recv_map[current_level][buffer_number][current_level][incoming_cnt + num_original_recv_dofs] =
               incoming_cnt + start_extra_dofs + hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]);
         }
         else
         {
            recv_map[current_level][buffer_number][current_level][incoming_cnt + num_original_recv_dofs] = -
                                                                                                           (incoming_cnt + start_extra_dofs + hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]) + 1);
         }

         sort_map[ incoming_cnt + start_extra_dofs ] = sort_cnt;
         new_inv_sort_map[sort_cnt] = incoming_cnt + start_extra_dofs;
         sort_cnt++;
         incoming_cnt++;
         cnt++;
      }
      else
      {
         sort_map[ inv_sort_map[compGrid_cnt] ] = sort_cnt;
         new_inv_sort_map[sort_cnt] = inv_sort_map[compGrid_cnt];
         compGrid_cnt++;
         sort_cnt++;
      }
   }
   while (incoming_cnt < remaining_dofs)
   {
      HYPRE_Int incoming_global_index = recv_buffer[cnt];
      HYPRE_Int incoming_is_real = 1;
      if (incoming_global_index < 0)
      {
         incoming_global_index = -(incoming_global_index + 1);
         incoming_is_real = 0;
      }

      hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[current_level])[ incoming_cnt + start_extra_dofs ]
         = incoming_global_index;
      hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[current_level])[ incoming_cnt + start_extra_dofs ] =
         incoming_is_real;

      if (incoming_is_real)
      {
         recv_map[current_level][buffer_number][current_level][incoming_cnt + num_original_recv_dofs] =
            incoming_cnt + start_extra_dofs + hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]);
      }
      else
      {
         recv_map[current_level][buffer_number][current_level][incoming_cnt + num_original_recv_dofs] = -
                                                                                                        (incoming_cnt + start_extra_dofs + hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]) + 1);
      }

      sort_map[ incoming_cnt + start_extra_dofs ] = sort_cnt;
      new_inv_sort_map[sort_cnt] = incoming_cnt + start_extra_dofs;
      sort_cnt++;
      incoming_cnt++;
      cnt++;
   }
   while (compGrid_cnt < start_extra_dofs)
   {
      sort_map[ inv_sort_map[compGrid_cnt] ] = sort_cnt;
      new_inv_sort_map[sort_cnt] = inv_sort_map[compGrid_cnt];
      compGrid_cnt++;
      sort_cnt++;
   }

   hypre_TFree(inv_sort_map, hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
   hypre_AMGDDCompGridNonOwnedInvSort(compGrid[current_level]) = new_inv_sort_map;

   // Unpack coarse global indices (need these for original commPkg recvs as well).
   // NOTE: store global indices for now, will be adjusted to local indices during SetupLocalIndices
   if (current_level != num_levels - 1)
   {
      for (i = 0; i < num_original_recv_dofs; i++)
      {
         HYPRE_Int coarse_index = recv_buffer[cnt++];
         if (coarse_index != -1) { coarse_index = -(coarse_index + 2); } // Marking coarse indices that need setup by negative mapping
         hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[current_level])[i +
                                                                           hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number)] = coarse_index;
      }
      for (i = 0; i < remaining_dofs; i++)
      {
         HYPRE_Int coarse_index = recv_buffer[cnt++];
         if (coarse_index != -1) { coarse_index = -(coarse_index + 2); } // Marking coarse indices that need setup by negative mapping
         hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[current_level])[i + start_extra_dofs] =
            coarse_index;
      }
   }

   // Unpack the col indices of A
   HYPRE_Int row_sizes_start = cnt;
   cnt += num_recv_nodes[current_level][buffer_number][current_level];

   // Setup col indices for original commPkg dofs
   for (i = 0; i < num_original_recv_dofs; i++)
   {
      HYPRE_Int diag_rowptr = hypre_CSRMatrixI(nonowned_diag)[ hypre_ParCSRCommPkgRecvVecStart(commPkg,
                                                                                               buffer_number) + i ];
      HYPRE_Int offd_rowptr = hypre_CSRMatrixI(nonowned_offd)[ hypre_ParCSRCommPkgRecvVecStart(commPkg,
                                                                                               buffer_number) + i ];

      HYPRE_Int row_size = recv_buffer[ i + row_sizes_start ];
      for (j = 0; j < row_size; j++)
      {
         HYPRE_Int incoming_index = recv_buffer[cnt++];

         // Incoming is a global index (could be owned or nonowned)
         if (incoming_index < 0)
         {
            incoming_index = -(incoming_index + 1);
            // See whether global index is owned on this proc (if so, can directly setup appropriate local index)
            if (incoming_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[current_level]) &&
                incoming_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[current_level]))
            {
               // Add to offd
               if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_offd))
               {
                  hypre_CSRMatrixResize(nonowned_offd, hypre_CSRMatrixNumRows(nonowned_offd),
                                        hypre_CSRMatrixNumCols(nonowned_offd),
                                        (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_offd) + 1));
               }
               hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = incoming_index -
                                                                hypre_AMGDDCompGridFirstGlobalIndex(compGrid[current_level]);
            }
            else
            {
               // Add to diag (global index, not in buffer, so we store global index and get a local index during SetupLocalIndices)
               if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
               {
                  hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc_v2(
                                                                                                 hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int,
                                                                                                 hypre_CSRMatrixNumNonzeros(nonowned_diag), HYPRE_Int,
                                                                                                 (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1),
                                                                                                 hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
                  hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag),
                                        hypre_CSRMatrixNumCols(nonowned_diag),
                                        (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1));
               }
               hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(
                  compGrid[current_level])[ hypre_AMGDDCompGridNumMissingColIndices(compGrid[current_level])++ ] =
                     diag_rowptr;
               hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = -(incoming_index + 1);
            }
         }
         // Incoming is an index to dofs within the buffer (by construction, nonowned)
         else
         {
            // Add to diag (index is within buffer, so we can directly go to local index)
            if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
            {
               hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc_v2(
                                                                                              hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int,
                                                                                              hypre_CSRMatrixNumNonzeros(nonowned_diag), HYPRE_Int,
                                                                                              (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1),
                                                                                              hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
               hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag),
                                     hypre_CSRMatrixNumCols(nonowned_diag),
                                     (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1));
            }
            if (incoming_index < num_original_recv_dofs)
            {
               hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = incoming_index + hypre_ParCSRCommPkgRecvVecStart(
                                                                   commPkg, buffer_number);
            }
            else
            {
               hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = incoming_index - num_original_recv_dofs +
                                                                start_extra_dofs;
            }
         }
      }

      // Update row pointers
      hypre_CSRMatrixI(nonowned_diag)[ hypre_ParCSRCommPkgRecvVecStart(commPkg,
                                                                       buffer_number) + i + 1 ] = diag_rowptr;
      hypre_CSRMatrixI(nonowned_offd)[ hypre_ParCSRCommPkgRecvVecStart(commPkg,
                                                                       buffer_number) + i + 1 ] = offd_rowptr;
   }

   // Temporary storage for extra comp grid dofs on this level (will be setup after all recv's during SetupLocalIndices)
   // A_tmp_info[buffer_number] = [ size, [row], size, [row], ... ]
   HYPRE_Int A_tmp_info_size = 1 + remaining_dofs;

   for (i = num_original_recv_dofs; i < num_recv_nodes[current_level][buffer_number][current_level];
        i++)
   {
      HYPRE_Int row_size = recv_buffer[ i + row_sizes_start ];
      A_tmp_info_size += row_size;
   }
   A_tmp_info[buffer_number] = hypre_CTAlloc(HYPRE_Int, A_tmp_info_size,
                                             hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
   HYPRE_Int A_tmp_info_cnt = 0;
   A_tmp_info[buffer_number][A_tmp_info_cnt++] = remaining_dofs;
   for (i = num_original_recv_dofs; i < num_recv_nodes[current_level][buffer_number][current_level];
        i++)
   {
      HYPRE_Int row_size = recv_buffer[ i + row_sizes_start ];
      A_tmp_info[buffer_number][A_tmp_info_cnt++] = row_size;
      for (j = 0; j < row_size; j++)
      {
         A_tmp_info[buffer_number][A_tmp_info_cnt++] = recv_buffer[cnt++];
      }
   }

   ////////////////////////////////////////////////////////////////////
   // loop over coarser psi levels
   ////////////////////////////////////////////////////////////////////

   for (level = current_level + 1; level < current_level + num_psi_levels; level++)
   {
      // get the number of nodes on this level
      num_recv_nodes[current_level][buffer_number][level] = recv_buffer[cnt++];
      level_start = cnt;
      *recv_map_send_buffer_size += num_recv_nodes[current_level][buffer_number][level];

      A = hypre_AMGDDCompGridA(compGrid[level]);
      nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
      nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

      HYPRE_Int num_nonowned = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]);
      HYPRE_Int diag_rowptr = hypre_CSRMatrixI(nonowned_diag)[ num_nonowned ];
      HYPRE_Int offd_rowptr = hypre_CSRMatrixI(nonowned_offd)[ num_nonowned ];

      // Incoming nodes and existing (non-owned) nodes in the comp grid are both sorted by global index, so here we merge these lists together (getting rid of redundant nodes along the way)
      add_node_cnt = 0;

      // NOTE: Don't free incoming_dest because we set that as recv_map and use it outside this function
      HYPRE_Int *incoming_dest = hypre_CTAlloc(HYPRE_Int,
                                               num_recv_nodes[current_level][buffer_number][level], HYPRE_MEMORY_HOST);
      recv_red_marker[current_level][buffer_number][level] = hypre_CTAlloc(HYPRE_Int,
                                                                           num_recv_nodes[current_level][buffer_number][level], HYPRE_MEMORY_HOST);

      // if necessary, reallocate more space for compGrid
      if (num_recv_nodes[current_level][buffer_number][level] + num_nonowned > hypre_CSRMatrixNumRows(
             nonowned_diag))
      {
         HYPRE_Int new_size = (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumRows(nonowned_diag));
         if (new_size < num_recv_nodes[current_level][buffer_number][level] + num_nonowned)
         {
            new_size = num_recv_nodes[current_level][buffer_number][level] + num_nonowned;
         }
         hypre_AMGDDCompGridResize(compGrid[level], new_size,
                                   level != num_levels - 1); // !!! Is there a better way to manage memory? !!!
      }

      sort_map = hypre_AMGDDCompGridNonOwnedSort(compGrid[level]);
      inv_sort_map = hypre_AMGDDCompGridNonOwnedInvSort(compGrid[level]);
      new_inv_sort_map = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(nonowned_diag),
                                       hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      sort_cnt = 0;
      compGrid_cnt = 0;
      incoming_cnt = 0;
      HYPRE_Int dest = num_nonowned;

      while (incoming_cnt < num_recv_nodes[current_level][buffer_number][level] &&
             compGrid_cnt < num_nonowned)
      {
         HYPRE_Int incoming_global_index = recv_buffer[cnt];
         HYPRE_Int incoming_is_real = 1;
         if (incoming_global_index < 0)
         {
            incoming_global_index = -(incoming_global_index + 1);
            incoming_is_real = 0;
         }

         // If incoming is owned, go on to the next
         if (incoming_global_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]) &&
             incoming_global_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[level]))
         {
            recv_red_marker[current_level][buffer_number][level][incoming_cnt] = 1;
            if (incoming_is_real)
            {
               incoming_dest[incoming_cnt] = incoming_global_index - hypre_AMGDDCompGridFirstGlobalIndex(
                                                compGrid[level]);   // Save location info for use below
            }
            else
            {
               incoming_dest[incoming_cnt] = -(incoming_global_index - hypre_AMGDDCompGridFirstGlobalIndex(
                                                  compGrid[level]) + 1);   // Save location info for use below
            }
            incoming_cnt++;
            cnt++;
         }
         // Otherwise, merge
         else
         {
            HYPRE_Int compGrid_global_index = hypre_AMGDDCompGridNonOwnedGlobalIndices(
                                                 compGrid[level])[ inv_sort_map[compGrid_cnt] ];

            if (incoming_global_index < compGrid_global_index)
            {
               sort_map[dest] = sort_cnt;
               new_inv_sort_map[sort_cnt] = dest;
               if (incoming_is_real)
               {
                  incoming_dest[incoming_cnt] = dest + hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
               }
               else
               {
                  incoming_dest[incoming_cnt] = -(dest + hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) + 1);
               }
               sort_cnt++;
               incoming_cnt++;
               dest++;
               cnt++;
               add_node_cnt++;
            }
            else if (incoming_global_index > compGrid_global_index)
            {
               sort_map[ inv_sort_map[compGrid_cnt] ] = sort_cnt;
               new_inv_sort_map[sort_cnt] = inv_sort_map[compGrid_cnt];
               compGrid_cnt++;
               sort_cnt++;
            }
            else
            {
               if (incoming_is_real &&
                   !hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[ inv_sort_map[compGrid_cnt] ])
               {
                  hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[ inv_sort_map[compGrid_cnt] ] = 1;
                  incoming_dest[incoming_cnt] = inv_sort_map[compGrid_cnt] + hypre_AMGDDCompGridNumOwnedNodes(
                                                   compGrid[level]); // Incoming real dof received to existing ghost location
                  incoming_cnt++;
                  cnt++;
               }
               else
               {
                  recv_red_marker[current_level][buffer_number][level][incoming_cnt] = 1;
                  if (incoming_is_real)
                  {
                     incoming_dest[incoming_cnt] = inv_sort_map[compGrid_cnt] + hypre_AMGDDCompGridNumOwnedNodes(
                                                      compGrid[level]);   // Save location info for use below
                  }
                  else
                  {
                     incoming_dest[incoming_cnt] = -(inv_sort_map[compGrid_cnt] + hypre_AMGDDCompGridNumOwnedNodes(
                                                        compGrid[level]) + 1);   // Save location info for use below
                  }
                  incoming_cnt++;
                  cnt++;
               }
            }
         }
      }
      while (incoming_cnt < num_recv_nodes[current_level][buffer_number][level])
      {
         HYPRE_Int incoming_global_index = recv_buffer[cnt];
         HYPRE_Int incoming_is_real = 1;
         if (incoming_global_index < 0)
         {
            incoming_global_index = -(incoming_global_index + 1);
            incoming_is_real = 0;
         }

         // If incoming is owned, go on to the next
         if (incoming_global_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]) &&
             incoming_global_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[level]))
         {
            recv_red_marker[current_level][buffer_number][level][incoming_cnt] = 1;
            if (incoming_is_real)
            {
               incoming_dest[incoming_cnt] = incoming_global_index - hypre_AMGDDCompGridFirstGlobalIndex(
                                                compGrid[level]);   // Save location info for use below
            }
            else
            {
               incoming_dest[incoming_cnt] = -(incoming_global_index - hypre_AMGDDCompGridFirstGlobalIndex(
                                                  compGrid[level]) + 1);   // Save location info for use below
            }
            incoming_cnt++;
            cnt++;
         }
         else
         {
            sort_map[dest] = sort_cnt;
            new_inv_sort_map[sort_cnt] = dest;
            if (incoming_is_real)
            {
               incoming_dest[incoming_cnt] = dest + hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
            }
            else
            {
               incoming_dest[incoming_cnt] = -(dest + hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) + 1);
            }
            sort_cnt++;
            incoming_cnt++;
            dest++;
            cnt++;
            add_node_cnt++;
         }
      }
      while (compGrid_cnt < num_nonowned)
      {
         sort_map[ inv_sort_map[compGrid_cnt] ] = sort_cnt;
         new_inv_sort_map[sort_cnt] = inv_sort_map[compGrid_cnt];
         compGrid_cnt++;
         sort_cnt++;
      }

      nodes_added_on_level[level] += add_node_cnt;

      // Free the old inv sort map and set new
      hypre_TFree(inv_sort_map, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      hypre_AMGDDCompGridNonOwnedInvSort(compGrid[level]) = new_inv_sort_map;

      // Set recv_map[current_level] to incoming_dest
      recv_map[current_level][buffer_number][level] = incoming_dest;

      // Now copy in the new nodes to their appropriate positions
      cnt = level_start;
      for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++)
      {
         if (!recv_red_marker[current_level][buffer_number][level][i])
         {
            dest = incoming_dest[i];
            if (dest < 0) { dest = -(dest + 1); }
            dest -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
            HYPRE_Int global_index = recv_buffer[cnt];
            if (global_index < 0)
            {
               global_index = -(global_index + 1);
               hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[ dest ] = 0;
            }
            else { hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[ dest ] = 1; }
            hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level])[ dest ] = global_index;
         }
         cnt++;
      }
      if (level != num_levels - 1)
      {
         for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++)
         {
            if (!recv_red_marker[current_level][buffer_number][level][i])
            {
               dest = incoming_dest[i];
               if (dest < 0) { dest = -(dest + 1); }
               dest -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
               HYPRE_Int coarse_index = recv_buffer[cnt];
               if (coarse_index != -1) { coarse_index = -(coarse_index + 2); } // Marking coarse indices that need setup by negative mapping
               hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level])[ dest ] = coarse_index;
            }
            cnt++;
         }
      }

      // Setup col indices
      row_sizes_start = cnt;
      cnt += num_recv_nodes[current_level][buffer_number][level];
      for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++)
      {
         HYPRE_Int row_size = recv_buffer[ i + row_sizes_start ];

         // !!! Optimization: (probably small gain) right now, I disregard incoming info for real overwriting ghost (internal buf connectivity could be used to avoid a few binary searches later)
         dest = incoming_dest[i];
         if (dest < 0) { dest = -(dest + 1); }
         dest -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);

         if (dest >= num_nonowned)
         {
            for (j = 0; j < row_size; j++)
            {
               HYPRE_Int incoming_index = recv_buffer[cnt++];

               // Incoming is a global index (could be owned or nonowned)
               if (incoming_index < 0)
               {
                  incoming_index = -(incoming_index + 1);
                  // See whether global index is owned on this proc (if so, can directly setup appropriate local index)
                  if (incoming_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]) &&
                      incoming_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[level]))
                  {
                     // Add to offd
                     if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_offd))
                     {
                        hypre_CSRMatrixResize(nonowned_offd, hypre_CSRMatrixNumRows(nonowned_offd),
                                              hypre_CSRMatrixNumCols(nonowned_offd),
                                              (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_offd) + 1));
                     }
                     hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = incoming_index -
                                                                      hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]);
                  }
                  else
                  {
                     // Add to diag (global index, not in buffer, so we store global index and get a local index during SetupLocalIndices)
                     if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
                     {
                        hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[level]) = hypre_TReAlloc_v2(
                                                                                               hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[level]), HYPRE_Int,
                                                                                               hypre_CSRMatrixNumNonzeros(nonowned_diag), HYPRE_Int,
                                                                                               (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1),
                                                                                               hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
                        hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag),
                                              hypre_CSRMatrixNumCols(nonowned_diag),
                                              (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1));
                     }
                     hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(
                        compGrid[level])[ hypre_AMGDDCompGridNumMissingColIndices(compGrid[level])++ ] = diag_rowptr;
                     hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = -(incoming_index + 1);
                  }
               }
               // Incoming is an index to dofs within the buffer (could be owned or nonowned)
               else
               {
                  HYPRE_Int local_index = incoming_dest[ incoming_index ];
                  if (local_index < 0) { local_index = -(local_index + 1); }

                  // Check whether dof is owned or nonowned
                  if (local_index < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
                  {
                     // Add to offd
                     if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_offd))
                     {
                        hypre_CSRMatrixResize(nonowned_offd, hypre_CSRMatrixNumRows(nonowned_offd),
                                              hypre_CSRMatrixNumCols(nonowned_offd),
                                              (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_offd) + 1));
                     }
                     hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = local_index;
                  }
                  else
                  {
                     // Add to diag (index is within buffer, so we can directly go to local index)
                     if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
                     {
                        hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[level]) = hypre_TReAlloc_v2(
                                                                                               hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[level]), HYPRE_Int,
                                                                                               hypre_CSRMatrixNumNonzeros(nonowned_diag), HYPRE_Int,
                                                                                               (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1),
                                                                                               hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
                        hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag),
                                              hypre_CSRMatrixNumCols(nonowned_diag),
                                              (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1));
                     }
                     hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = local_index - hypre_AMGDDCompGridNumOwnedNodes(
                                                                         compGrid[level]);
                  }
               }
            }
            // Update row pointers
            hypre_CSRMatrixI(nonowned_diag)[ dest + 1 ] = diag_rowptr;
            hypre_CSRMatrixI(nonowned_offd)[ dest + 1 ] = offd_rowptr;
         }
         else
         {
            cnt += row_size;
         }
      }

      hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]) += add_node_cnt;
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_PackColInd( HYPRE_Int           *send_flag,
                              HYPRE_Int            num_send_nodes,
                              HYPRE_Int           *add_flag,
                              hypre_AMGDDCompGrid *compGrid,
                              HYPRE_Int           *send_buffer,
                              HYPRE_Int            starting_cnt)
{
   HYPRE_Int i, j, send_elmt, add_flag_index;
   HYPRE_Int cnt = starting_cnt;
   HYPRE_Int total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid) +
                               hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);
   for (i = 0; i < num_send_nodes; i++)
   {
      send_elmt = send_flag[i];
      if (send_elmt < 0) { send_elmt = -(send_elmt + 1); }

      // Owned point
      if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid))
      {
         hypre_CSRMatrix *diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid));
         hypre_CSRMatrix *offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid));
         // Get diag connections
         for (j = hypre_CSRMatrixI(diag)[send_elmt]; j < hypre_CSRMatrixI(diag)[send_elmt + 1]; j++)
         {
            add_flag_index = hypre_CSRMatrixJ(diag)[j];
            if (add_flag[add_flag_index] > 0)
            {
               send_buffer[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
            }
            else
            {
               send_buffer[cnt++] = -(add_flag_index + hypre_AMGDDCompGridFirstGlobalIndex(
                                         compGrid) + 1); // -(GID + 1)
            }
         }
         // Get offd connections
         for (j = hypre_CSRMatrixI(offd)[send_elmt]; j < hypre_CSRMatrixI(offd)[send_elmt + 1]; j++)
         {
            add_flag_index = hypre_CSRMatrixJ(offd)[j] + hypre_AMGDDCompGridNumOwnedNodes(compGrid);
            if (add_flag[add_flag_index] > 0)
            {
               send_buffer[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
            }
            else
            {
               send_buffer[cnt++] = -(hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)[ hypre_CSRMatrixJ(
                                                                                             offd)[j] ] + 1); // -(GID + 1)
            }
         }
      }
      // NonOwned point
      else if (send_elmt < total_num_nodes)
      {
         HYPRE_Int nonowned_index = send_elmt - hypre_AMGDDCompGridNumOwnedNodes(compGrid);
         hypre_CSRMatrix *diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid));
         hypre_CSRMatrix *offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid));
         // Get diag connections
         for (j = hypre_CSRMatrixI(diag)[nonowned_index]; j < hypre_CSRMatrixI(diag)[nonowned_index + 1];
              j++)
         {
            if (hypre_CSRMatrixJ(diag)[j] >= 0)
            {
               add_flag_index = hypre_CSRMatrixJ(diag)[j] + hypre_AMGDDCompGridNumOwnedNodes(compGrid);
               if (add_flag[add_flag_index] > 0)
               {
                  send_buffer[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  send_buffer[cnt++] = -(hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)[ hypre_CSRMatrixJ(
                                                                                                diag)[j] ] + 1); // -(GID + 1)
               }
            }
            else
            {
               send_buffer[cnt++] = hypre_CSRMatrixJ(diag)[j]; // -(GID + 1)
            }
         }
         // Get offd connections
         for (j = hypre_CSRMatrixI(offd)[nonowned_index]; j < hypre_CSRMatrixI(offd)[nonowned_index + 1];
              j++)
         {
            add_flag_index = hypre_CSRMatrixJ(offd)[j];
            if (add_flag[add_flag_index] > 0)
            {
               send_buffer[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
            }
            else
            {
               send_buffer[cnt++] = -(add_flag_index + hypre_AMGDDCompGridFirstGlobalIndex(
                                         compGrid) + 1); // -(GID + 1)
            }
         }
      }
      else { send_flag[i] = send_elmt - total_num_nodes; }
   }

   return cnt;
}

HYPRE_Int
hypre_BoomerAMGDD_MarkCoarse(HYPRE_Int  *list,
                             HYPRE_Int  *marker,
                             HYPRE_Int  *owned_coarse_indices,
                             HYPRE_Int  *nonowned_coarse_indices,
                             HYPRE_Int  *sort_map,
                             HYPRE_Int   num_owned,
                             HYPRE_Int   total_num_nodes,
                             HYPRE_Int   num_owned_coarse,
                             HYPRE_Int   list_size,
                             HYPRE_Int   dist,
                             HYPRE_Int   use_sort,
                             HYPRE_Int  *nodes_to_add)
{
   HYPRE_Int i, coarse_index;
   for (i = 0; i < list_size; i++)
   {
      HYPRE_Int idx = list[i];
      if (idx >= 0)
      {
         if (idx >= total_num_nodes)
         {
            idx -= total_num_nodes;
         }
         if (idx < num_owned)
         {
            coarse_index = owned_coarse_indices[idx];
            if (coarse_index >= 0)
            {
               marker[ coarse_index ] = dist;
               (*nodes_to_add) = 1;
            }
         }
         else
         {
            idx -= num_owned;
            coarse_index = nonowned_coarse_indices[idx];
            if (coarse_index >= 0)
            {
               if (use_sort)
               {
                  coarse_index = sort_map[ coarse_index ] + num_owned_coarse;
               }
               else
               {
                  coarse_index = coarse_index + num_owned_coarse;
               }
               marker[ coarse_index ] = dist;
               (*nodes_to_add) = 1;
            }
         }
      }
   }
   return hypre_error_flag;
}

HYPRE_Int*
hypre_BoomerAMGDD_AddFlagToSendFlag( hypre_AMGDDCompGrid *compGrid,
                                     HYPRE_Int           *add_flag,
                                     HYPRE_Int           *num_send_nodes,
                                     HYPRE_Int            num_ghost_layers )
{
   HYPRE_Int i, cnt, add_flag_index;
   HYPRE_Int total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid) +
                               hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);
   for (i = 0; i < total_num_nodes; i++)
   {
      if (add_flag[i] > 0)
      {
         (*num_send_nodes)++;
      }
   }

   HYPRE_Int *inv_sort_map = hypre_AMGDDCompGridNonOwnedInvSort(compGrid);
   HYPRE_Int *send_flag = hypre_CTAlloc( HYPRE_Int, (*num_send_nodes), HYPRE_MEMORY_HOST );
   cnt =  0;
   i = 0;
   // First the nonowned indices coming before the owned block
   if (hypre_AMGDDCompGridNumNonOwnedNodes(compGrid))
   {
      while (hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)[inv_sort_map[i]] <
             hypre_AMGDDCompGridFirstGlobalIndex(compGrid))
      {
         add_flag_index = i + hypre_AMGDDCompGridNumOwnedNodes(compGrid);
         if (add_flag[add_flag_index] > num_ghost_layers)
         {
            send_flag[cnt] = inv_sort_map[i] + hypre_AMGDDCompGridNumOwnedNodes(compGrid);
            cnt++;
         }
         else if (add_flag[add_flag_index] > 0)
         {
            send_flag[cnt] = -(inv_sort_map[i] + hypre_AMGDDCompGridNumOwnedNodes(compGrid) + 1);
            cnt++;
         }
         i++;
         if (i == hypre_AMGDDCompGridNumNonOwnedNodes(compGrid)) { break; }
      }
   }
   // Then the owned block
   for (add_flag_index = 0; add_flag_index < hypre_AMGDDCompGridNumOwnedNodes(compGrid);
        add_flag_index++)
   {
      if (add_flag[add_flag_index] > num_ghost_layers)
      {
         send_flag[cnt] = add_flag_index;
         cnt++;
      }
      else if (add_flag[add_flag_index] > 0)
      {
         send_flag[cnt] = -(add_flag_index + 1);
         cnt++;
      }
   }
   // Finally the nonowned indices coming after the owned block
   while (i < hypre_AMGDDCompGridNumNonOwnedNodes(compGrid))
   {
      add_flag_index = i + hypre_AMGDDCompGridNumOwnedNodes(compGrid);
      if (add_flag[add_flag_index] > num_ghost_layers)
      {
         send_flag[cnt] = inv_sort_map[i] + hypre_AMGDDCompGridNumOwnedNodes(compGrid);
         cnt++;
      }
      else if (add_flag[add_flag_index] > 0)
      {
         send_flag[cnt] = -(inv_sort_map[i] + hypre_AMGDDCompGridNumOwnedNodes(compGrid) + 1);
         cnt++;
      }
      i++;
   }

   return send_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_SubtractLists( hypre_AMGDDCompGrid *compGrid,
                                 HYPRE_Int           *current_list,
                                 HYPRE_Int           *current_list_length,
                                 HYPRE_Int           *prev_list,
                                 HYPRE_Int            prev_list_length )
{
   // send_flag's are in global index ordering on each level, so can merge
   HYPRE_Int prev_cnt = 0;
   HYPRE_Int current_cnt = 0;
   HYPRE_Int new_cnt = 0;
   while (current_cnt < (*current_list_length) && prev_cnt < prev_list_length)
   {
      // Get the global indices
      HYPRE_Int current_global_index = hypre_BoomerAMGDD_LocalToGlobalIndex(compGrid,
                                                                            current_list[current_cnt]);
      HYPRE_Int prev_global_index = hypre_BoomerAMGDD_LocalToGlobalIndex(compGrid, prev_list[prev_cnt]);

      // Do the merge
      if (current_global_index > prev_global_index)
      {
         prev_cnt++;
      }
      else if (current_global_index < prev_global_index)
      {
         current_list[new_cnt] = current_list[current_cnt];
         new_cnt++;
         current_cnt++;
      }
      else
      {
         // Special treatment for ghosts sent later as real
         if (prev_list[prev_cnt] < 0 && current_list[current_cnt] >= 0)
         {
            // This is the case of real dof sent to overwrite ghost.
            // Current list is a positive local index here. Map beyond the range of total dofs to mark.
            if (current_list[current_cnt] < hypre_AMGDDCompGridNumOwnedNodes(compGrid) +
                hypre_AMGDDCompGridNumNonOwnedNodes(compGrid))
            {
               current_list[new_cnt] = current_list[current_cnt] + hypre_AMGDDCompGridNumOwnedNodes(
                                          compGrid) + hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);
            }
            else
            {
               current_list[new_cnt] = current_list[current_cnt];
            }
            new_cnt++;
            current_cnt++;
            prev_cnt++;
         }
         else
         {
            prev_cnt++;
            current_cnt++;
         }
      }
   }
   while (current_cnt < (*current_list_length))
   {
      current_list[new_cnt] = current_list[current_cnt];
      new_cnt++;
      current_cnt++;
   }
   (*current_list_length) = new_cnt;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_RemoveRedundancy( hypre_ParAMGData      *amg_data,
                                    HYPRE_Int          ****send_flag,
                                    HYPRE_Int           ***num_send_nodes,
                                    hypre_AMGDDCompGrid  **compGrid,
                                    hypre_AMGDDCommPkg    *compGridCommPkg,
                                    HYPRE_Int              current_level,
                                    HYPRE_Int              proc,
                                    HYPRE_Int              level )
{
   HYPRE_Int current_send_proc = hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[current_level][proc];
   HYPRE_Int prev_proc, prev_level;
   for (prev_level = current_level + 1; prev_level <= level; prev_level++)
   {
      hypre_ParCSRCommPkg *original_commPkg = hypre_ParCSRMatrixCommPkg(hypre_ParAMGDataAArray(
                                                                           amg_data)[prev_level]);
      for (prev_proc = 0; prev_proc < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[prev_level];
           prev_proc++)
      {
         if (hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc)
         {
            HYPRE_Int prev_list_end = num_send_nodes[prev_level][prev_proc][level];
            if (prev_level == level)
            {
               HYPRE_Int original_proc;
               for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumSends(original_commPkg);
                    original_proc++)
               {
                  if (hypre_ParCSRCommPkgSendProc(original_commPkg, original_proc) == current_send_proc)
                  {
                     prev_list_end = hypre_ParCSRCommPkgSendMapStart(original_commPkg,
                                                                     original_proc + 1) - hypre_ParCSRCommPkgSendMapStart(original_commPkg, original_proc);
                     break;
                  }
               }
            }

            hypre_BoomerAMGDD_SubtractLists(compGrid[level],
                                            send_flag[current_level][proc][level],
                                            &(num_send_nodes[current_level][proc][level]),
                                            send_flag[prev_level][prev_proc][level],
                                            prev_list_end);

            if (num_send_nodes[prev_level][prev_proc][level] - prev_list_end > 0)
            {
               hypre_BoomerAMGDD_SubtractLists(compGrid[level],
                                               send_flag[current_level][proc][level],
                                               &(num_send_nodes[current_level][proc][level]),
                                               &(send_flag[prev_level][prev_proc][level][prev_list_end]),
                                               num_send_nodes[prev_level][prev_proc][level] - prev_list_end);
            }
         }
      }

      for (prev_proc = 0; prev_proc < hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[prev_level];
           prev_proc++)
      {
         if (hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc)
         {
            HYPRE_Int prev_list_end = hypre_AMGDDCommPkgNumRecvNodes(
                                         compGridCommPkg)[prev_level][prev_proc][level];
            if (prev_level == level)
            {
               HYPRE_Int original_proc;
               for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumRecvs(original_commPkg);
                    original_proc++)
               {
                  if (hypre_ParCSRCommPkgRecvProc(original_commPkg, original_proc) == current_send_proc)
                  {
                     prev_list_end = hypre_ParCSRCommPkgRecvVecStart(original_commPkg,
                                                                     original_proc + 1) - hypre_ParCSRCommPkgRecvVecStart(original_commPkg, original_proc);
                     break;
                  }
               }
            }

            hypre_BoomerAMGDD_SubtractLists(compGrid[level],
                                            send_flag[current_level][proc][level],
                                            &(num_send_nodes[current_level][proc][level]),
                                            hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[prev_level][prev_proc][level],
                                            prev_list_end);

            if (hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level] - prev_list_end >
                0)
            {
               hypre_BoomerAMGDD_SubtractLists(compGrid[level],
                                               send_flag[current_level][proc][level],
                                               &(num_send_nodes[current_level][proc][level]),
                                               &(hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[prev_level][prev_proc][level][prev_list_end]),
                                               hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level] - prev_list_end);
            }
         }
      }
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_RecursivelyBuildPsiComposite( HYPRE_Int            node,
                                                HYPRE_Int            m,
                                                hypre_AMGDDCompGrid *compGrid,
                                                HYPRE_Int           *add_flag,
                                                HYPRE_Int            use_sort)
{
   HYPRE_Int i, index, sort_index;
   HYPRE_Int error_code = 0;

   HYPRE_Int *sort_map = hypre_AMGDDCompGridNonOwnedSort(compGrid);

   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   HYPRE_Int owned;
   if (node < hypre_AMGDDCompGridNumOwnedNodes(compGrid))
   {
      owned = 1;
      diag = hypre_AMGDDCompGridMatrixOwnedDiag( hypre_AMGDDCompGridA(compGrid) );
      offd = hypre_AMGDDCompGridMatrixOwnedOffd( hypre_AMGDDCompGridA(compGrid) );
   }
   else
   {
      owned = 0;
      node = node - hypre_AMGDDCompGridNumOwnedNodes(compGrid);
      diag = hypre_AMGDDCompGridMatrixNonOwnedDiag( hypre_AMGDDCompGridA(compGrid) );
      offd = hypre_AMGDDCompGridMatrixNonOwnedOffd( hypre_AMGDDCompGridA(compGrid) );
   }

   // Look at neighbors in diag
   for (i = hypre_CSRMatrixI(diag)[node]; i < hypre_CSRMatrixI(diag)[node + 1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_CSRMatrixJ(diag)[i];
      if (index >= 0)
      {
         if (owned)
         {
            sort_index = index;
         }
         else
         {
            if (use_sort) { sort_index = sort_map[index] + hypre_AMGDDCompGridNumOwnedNodes(compGrid); }
            else { sort_index = index + hypre_AMGDDCompGridNumOwnedNodes(compGrid); }
            index += hypre_AMGDDCompGridNumOwnedNodes(compGrid);
         }

         // If we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m - 1 > 0) { error_code = hypre_BoomerAMGDD_RecursivelyBuildPsiComposite(index, m - 1, compGrid, add_flag, use_sort); }
         }
      }
      else
      {
         error_code = 1;
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "WARNING: Negative col index encountered during hypre_BoomerAMGDD_RecursivelyBuildPsiComposite().\n");
      }
   }

   // Look at neighbors in offd
   for (i = hypre_CSRMatrixI(offd)[node]; i < hypre_CSRMatrixI(offd)[node + 1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_CSRMatrixJ(offd)[i];
      if (index >= 0)
      {
         if (!owned)
         {
            sort_index = index;
         }
         else
         {
            if (use_sort) { sort_index = sort_map[index] + hypre_AMGDDCompGridNumOwnedNodes(compGrid); }
            else { sort_index = index + hypre_AMGDDCompGridNumOwnedNodes(compGrid); }
            index += hypre_AMGDDCompGridNumOwnedNodes(compGrid);
         }

         // If we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m - 1 > 0) { error_code = hypre_BoomerAMGDD_RecursivelyBuildPsiComposite(index, m - 1, compGrid, add_flag, use_sort); }
         }
      }
      else
      {
         error_code = 1;
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "WARNING: Negative col index encountered during hypre_BoomerAMGDD_RecursivelyBuildPsiComposite().\n");
      }
   }

   return error_code;
}

HYPRE_Int*
hypre_BoomerAMGDD_PackSendBuffer( hypre_ParAMGDDData *amgdd_data,
                                  HYPRE_Int           proc,
                                  HYPRE_Int           current_level,
                                  HYPRE_Int          *padding,
                                  HYPRE_Int          *send_flag_buffer_size )
{
   // send_buffer = [ num_psi_levels , [level] , [level] , ... ]
   // level = [ num send nodes, [global indices] , [coarse global indices] , [A row sizes] , [A col ind: either global indices or local col indices within buffer] ]
   hypre_ParAMGData      *amg_data         = hypre_ParAMGDDDataAMG(amgdd_data);
   hypre_AMGDDCompGrid  **compGrid         = hypre_ParAMGDDDataCompGrid(amgdd_data);
   hypre_AMGDDCommPkg    *compGridCommPkg  = hypre_ParAMGDDDataCommPkg(amgdd_data);
   HYPRE_Int              num_levels       = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int              num_ghost_layers = hypre_ParAMGDDDataNumGhostLayers(amgdd_data);

   HYPRE_Int          ****send_flag        = hypre_AMGDDCommPkgSendFlag(compGridCommPkg);
   HYPRE_Int           ***num_send_nodes   = hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg);
   HYPRE_Int            **send_buffer_size = hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg);
   HYPRE_Int            **add_flag;
   HYPRE_Int             *send_buffer;
   hypre_CSRMatrix       *diag;
   hypre_CSRMatrix       *offd;

   HYPRE_MemoryLocation   memory_location;
   HYPRE_Int              level, i, ii, cnt, row_length, send_elmt, add_flag_index;
   HYPRE_Int              nodes_to_add = 0;
   HYPRE_Int              num_psi_levels = 1;
   HYPRE_Int              total_num_nodes;
   HYPRE_Int              nonowned_index;
   HYPRE_Int              nonowned_coarse_index;

   // initialize send map buffer size
   (*send_flag_buffer_size) = num_levels - current_level - 1;

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Mark the nodes to send (including Psi_c grid plus ghost nodes)
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Count up the buffer size for the starting nodes
   add_flag = hypre_CTAlloc(HYPRE_Int *, num_levels, HYPRE_MEMORY_HOST);
   total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]) +
                     hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[current_level]);
   memory_location = hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]);
   add_flag[current_level] = hypre_CTAlloc(HYPRE_Int, total_num_nodes, memory_location);

   send_buffer_size[current_level][proc] += 2;
   if (current_level != num_levels - 1)
   {
      send_buffer_size[current_level][proc] += 3 * num_send_nodes[current_level][proc][current_level];
   }
   else
   {
      send_buffer_size[current_level][proc] += 2 * num_send_nodes[current_level][proc][current_level];
   }

   for (i = 0; i < num_send_nodes[current_level][proc][current_level]; i++)
   {
      send_elmt = send_flag[current_level][proc][current_level][i];
      if (send_elmt < 0)
      {
         send_elmt = -(send_elmt + 1);
      }
      add_flag[current_level][send_elmt] = i + 1;

      diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid[current_level]));
      offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid[current_level]));
      send_buffer_size[current_level][proc] += hypre_CSRMatrixI(diag)[send_elmt + 1] - hypre_CSRMatrixI(
                                                  diag)[send_elmt];
      send_buffer_size[current_level][proc] += hypre_CSRMatrixI(offd)[send_elmt + 1] - hypre_CSRMatrixI(
                                                  offd)[send_elmt];
   }

   // Add the nodes listed by the coarse grid counterparts if applicable
   // Note that the compGridCommPkg is set up to list all nodes within the padding plus ghost layers
   if (current_level != num_levels - 1)
   {
      total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level + 1]) +
                        hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[current_level + 1]);
      memory_location = hypre_AMGDDCompGridMemoryLocation(compGrid[current_level + 1]);
      add_flag[current_level + 1] = hypre_CTAlloc(HYPRE_Int, total_num_nodes, memory_location);

      total_num_nodes  = hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]) +
                         hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[current_level]);
      hypre_BoomerAMGDD_MarkCoarse(send_flag[current_level][proc][current_level],
                                   add_flag[current_level + 1],
                                   hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[current_level]),
                                   hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[current_level]),
                                   hypre_AMGDDCompGridNonOwnedSort(compGrid[current_level + 1]),
                                   hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]),
                                   total_num_nodes,
                                   hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level + 1]),
                                   num_send_nodes[current_level][proc][current_level],
                                   padding[current_level + 1] + num_ghost_layers + 1,
                                   1,
                                   &nodes_to_add);
   }

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Now build out the psi_c composite grid (along with required ghost nodes) on coarser levels
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   for (level = current_level + 1; level < num_levels; level++)
   {
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         num_psi_levels++;
         send_buffer_size[current_level][proc]++;
         nodes_to_add = 0;

         // if we need coarse info, allocate space for the add flag on the next level
         if (level != num_levels - 1)
         {
            total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level + 1]) +
                              hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level + 1]);
            memory_location = hypre_AMGDDCompGridMemoryLocation(compGrid[current_level + 1]);
            add_flag[level + 1] = hypre_CTAlloc(HYPRE_Int, total_num_nodes, memory_location);
         }

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) +
                           hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]);
         for (i = 0; i < total_num_nodes; i++)
         {
            if (i < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
            {
               add_flag_index = i;
            }
            else
            {
               ii = i - hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
               add_flag_index = hypre_AMGDDCompGridNonOwnedSort(compGrid[level])[ii] +
                                hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
            }

            if (add_flag[level][add_flag_index] == padding[level] + num_ghost_layers + 1)
            {
               hypre_BoomerAMGDD_RecursivelyBuildPsiComposite(i,
                                                              padding[level] + num_ghost_layers,
                                                              compGrid[level],
                                                              add_flag[level],
                                                              1);
            }
         }

         send_flag[current_level][proc][level] = hypre_BoomerAMGDD_AddFlagToSendFlag(compGrid[level],
                                                                                     add_flag[level],
                                                                                     &(num_send_nodes[current_level][proc][level]),
                                                                                     num_ghost_layers);

         // Compare with previous send/recvs to eliminate redundant info
         hypre_BoomerAMGDD_RemoveRedundancy(amg_data,
                                            send_flag,
                                            num_send_nodes,
                                            compGrid,
                                            compGridCommPkg,
                                            current_level,
                                            proc,
                                            level);

         // Mark the points to start from on the next level
         if (level != num_levels - 1)
         {
            total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) +
                              hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]);
            hypre_BoomerAMGDD_MarkCoarse(send_flag[current_level][proc][level],
                                         add_flag[level + 1],
                                         hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level]),
                                         hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level]),
                                         hypre_AMGDDCompGridNonOwnedSort(compGrid[level + 1]),
                                         hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]),
                                         total_num_nodes,
                                         hypre_AMGDDCompGridNumOwnedNodes(compGrid[level + 1]),
                                         num_send_nodes[current_level][proc][level],
                                         padding[level + 1] + num_ghost_layers + 1,
                                         1,
                                         &nodes_to_add);
         }

         // Count up the buffer sizes and adjust the add_flag
         total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) +
                           hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]);

         hypre_Memset(add_flag[level], 0, sizeof(HYPRE_Int)*total_num_nodes, memory_location);
         (*send_flag_buffer_size) += num_send_nodes[current_level][proc][level];
         if (level != num_levels - 1)
         {
            send_buffer_size[current_level][proc] += 3 * num_send_nodes[current_level][proc][level];
         }
         else
         {
            send_buffer_size[current_level][proc] += 2 * num_send_nodes[current_level][proc][level];
         }

         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0)
            {
               send_elmt = -(send_elmt + 1);
            }

            if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
            {
               add_flag[level][send_elmt] = i + 1;
               diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
               offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
               send_buffer_size[current_level][proc] += hypre_CSRMatrixI(diag)[send_elmt + 1] - hypre_CSRMatrixI(
                                                           diag)[send_elmt];
               send_buffer_size[current_level][proc] += hypre_CSRMatrixI(offd)[send_elmt + 1] - hypre_CSRMatrixI(
                                                           offd)[send_elmt];
            }
            else if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) +
                     hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]))
            {
               add_flag[level][send_elmt] = i + 1;
               send_elmt -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
               diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
               offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
               send_buffer_size[current_level][proc] += hypre_CSRMatrixI(diag)[send_elmt + 1] - hypre_CSRMatrixI(
                                                           diag)[send_elmt];
               send_buffer_size[current_level][proc] += hypre_CSRMatrixI(offd)[send_elmt + 1] - hypre_CSRMatrixI(
                                                           offd)[send_elmt];
            }
            else
            {
               send_elmt -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) +
                            hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]);
               add_flag[level][send_elmt] = i + 1;
            }
         }
      }
      else
      {
         break;
      }
   }

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Pack the buffer
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   send_buffer = hypre_CTAlloc(HYPRE_Int, send_buffer_size[current_level][proc], HYPRE_MEMORY_HOST);
   send_buffer[0] = num_psi_levels;
   cnt = 1;
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) +
                        hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]);

      // store the number of nodes on this level
      send_buffer[cnt++] = num_send_nodes[current_level][proc][level];

      // copy all global indices
      for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
      {
         send_elmt = send_flag[current_level][proc][level][i];
         if (send_elmt < 0)
         {
            send_elmt = -(send_elmt + 1);

            if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
            {
               send_buffer[cnt++] = -(send_elmt + hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]) + 1);
            }
            else
            {
               send_buffer[cnt++] = -(hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level])[ send_elmt -
                                                                                                 hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) ] + 1);
            }
         }
         else
         {
            if (send_elmt >= total_num_nodes)
            {
               send_elmt -= total_num_nodes;
            }

            if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
            {
               send_buffer[cnt++] = send_elmt + hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]);
            }
            else
            {
               send_buffer[cnt++] = hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level])[send_elmt -
                                                                                              hypre_AMGDDCompGridNumOwnedNodes(compGrid[level])];
            }
         }
      }

      // if not on last level, copy coarse gobal indices
      if (level != num_levels - 1)
      {
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0)
            {
               send_elmt = -(send_elmt + 1);
            }
            else if (send_elmt >= total_num_nodes)
            {
               send_elmt -= total_num_nodes;
            }

            if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
            {
               if (hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level])[ send_elmt ] >= 0)
               {
                  send_buffer[cnt++] = hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level])[send_elmt] +
                                       hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level + 1]);
               }
               else
               {
                  send_buffer[cnt++] = hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level])[send_elmt];
               }
            }
            else
            {
               nonowned_index = send_elmt - hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
               nonowned_coarse_index = hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level])[nonowned_index];

               if (nonowned_coarse_index >= 0)
               {
                  send_buffer[cnt++] = hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level
                                                                                         + 1])[ nonowned_coarse_index ];
               }
               else if (nonowned_coarse_index == -1)
               {
                  send_buffer[cnt++] = nonowned_coarse_index;
               }
               else
               {
                  send_buffer[cnt++] = -(nonowned_coarse_index + 2);
               }
            }
         }
      }

      // store the row length for matrix A
      for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
      {
         send_elmt = send_flag[current_level][proc][level][i];
         if (send_elmt < 0)
         {
            send_elmt = -(send_elmt + 1);
         }
         if (send_elmt < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
         {
            diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
            offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
            row_length = hypre_CSRMatrixI(diag)[send_elmt + 1] - hypre_CSRMatrixI(diag)[send_elmt]
                         + hypre_CSRMatrixI(offd)[send_elmt + 1] - hypre_CSRMatrixI(offd)[send_elmt];
         }
         else if (send_elmt < total_num_nodes)
         {
            nonowned_index = send_elmt - hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
            diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
            offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
            row_length = hypre_CSRMatrixI(diag)[nonowned_index + 1] - hypre_CSRMatrixI(diag)[nonowned_index]
                         + hypre_CSRMatrixI(offd)[nonowned_index + 1] - hypre_CSRMatrixI(offd)[nonowned_index];
         }
         else
         {
            row_length = 0;
            /* send_flag[current_level][proc][level][i] -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]) + hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]); */
         }
         send_buffer[cnt++] = row_length;
      }

      // copy indices for matrix A (local connectivity within buffer where available, global index otherwise)
      cnt = hypre_BoomerAMGDD_PackColInd(send_flag[current_level][proc][level],
                                         num_send_nodes[current_level][proc][level],
                                         add_flag[level],
                                         compGrid[level],
                                         send_buffer,
                                         cnt);
   }

   // Clean up memory
   for (level = 0; level < num_levels; level++)
   {
      if (add_flag[level])
      {
         hypre_TFree(add_flag[level], hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      }
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);

   // Return the send buffer
   return send_buffer;
}

HYPRE_Int
hypre_BoomerAMGDD_PackRecvMapSendBuffer( HYPRE_Int   *recv_map_send_buffer,
                                         HYPRE_Int  **recv_red_marker,
                                         HYPRE_Int   *num_recv_nodes,
                                         HYPRE_Int   *recv_buffer_size,
                                         HYPRE_Int    current_level,
                                         HYPRE_Int    num_levels )
{
   HYPRE_Int  level, i, cnt, num_nodes;

   cnt = 0;
   *recv_buffer_size = 0;
   for (level = current_level + 1; level < num_levels; level++)
   {
      // if there were nodes in psiComposite on this level
      if (recv_red_marker[level])
      {
         // store the number of nodes on this level
         num_nodes = num_recv_nodes[level];
         recv_map_send_buffer[cnt++] = num_nodes;

         for (i = 0; i < num_nodes; i++)
         {
            // store the map values for each node
            recv_map_send_buffer[cnt++] = recv_red_marker[level][i];
         }
      }
      // otherwise record that there were zero nodes on this level
      else { recv_map_send_buffer[cnt++] = 0; }
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_UnpackSendFlagBuffer( hypre_AMGDDCompGrid **compGrid,
                                        HYPRE_Int            *send_flag_buffer,
                                        HYPRE_Int           **send_flag,
                                        HYPRE_Int            *num_send_nodes,
                                        HYPRE_Int            *send_buffer_size,
                                        HYPRE_Int             current_level,
                                        HYPRE_Int             num_levels )
{
   HYPRE_UNUSED_VAR(compGrid);

   HYPRE_Int level, i, cnt, num_nodes;

   cnt = 0;
   *send_buffer_size = 0;
   for (level = current_level + 1; level < num_levels; level++)
   {
      num_nodes = send_flag_buffer[cnt++];
      num_send_nodes[level] = 0;

      for (i = 0; i < num_nodes; i++)
      {
         if (send_flag_buffer[cnt++] == 0)
         {
            send_flag[level][ num_send_nodes[level]++ ] = send_flag[level][i];
            (*send_buffer_size)++;
         }
      }

      send_flag[level] = hypre_TReAlloc(send_flag[level], HYPRE_Int, num_send_nodes[level],
                                        HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_CommunicateRemainingMatrixInfo( hypre_ParAMGDDData* amgdd_data )
{
   hypre_ParAMGData     *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);
   hypre_AMGDDCompGrid **compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data);
   hypre_AMGDDCommPkg   *compGridCommPkg = hypre_ParAMGDDDataCommPkg(amgdd_data);
   HYPRE_Int             num_levels = hypre_AMGDDCommPkgNumLevels(compGridCommPkg);
   HYPRE_Int             amgdd_start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);

   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int *P_row_cnt = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int *R_row_cnt = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int *A_row_cnt = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

   HYPRE_Int outer_level, proc, level, i, j;

   for (outer_level = num_levels - 1; outer_level >= amgdd_start_level; outer_level--)
   {

      // Initialize nonowned matrices for P (and R)
      if (outer_level != num_levels - 1)
      {
         hypre_CSRMatrix *P_diag_original = hypre_ParCSRMatrixDiag(hypre_ParAMGDataPArray(
                                                                      amg_data)[outer_level]);
         hypre_CSRMatrix *P_offd_original = hypre_ParCSRMatrixOffd(hypre_ParAMGDataPArray(
                                                                      amg_data)[outer_level]);
         HYPRE_Int ave_nnz_per_row = 1;
         if (hypre_ParAMGDataPMaxElmts(amg_data))
         {
            ave_nnz_per_row = hypre_ParAMGDataPMaxElmts(amg_data);
         }
         else if (hypre_CSRMatrixNumRows(P_diag_original))
         {
            ave_nnz_per_row = (HYPRE_Int) (hypre_CSRMatrixNumNonzeros(P_diag_original) / hypre_CSRMatrixNumRows(
                                              P_diag_original));
         }
         HYPRE_Int max_nonowned_diag_nnz = hypre_AMGDDCompGridNumNonOwnedNodes(
                                              compGrid[outer_level]) * ave_nnz_per_row;
         HYPRE_Int max_nonowned_offd_nnz = hypre_CSRMatrixNumNonzeros(P_offd_original);
         hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[outer_level])) =
            hypre_CSRMatrixCreate(hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[outer_level]),
                                  hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[outer_level + 1]), max_nonowned_diag_nnz);
         hypre_CSRMatrixInitialize(hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(
                                                                            compGrid[outer_level])));
         hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(compGrid[outer_level])) =
            hypre_CSRMatrixCreate(hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[outer_level]),
                                  hypre_AMGDDCompGridNumOwnedNodes(compGrid[outer_level + 1]), max_nonowned_offd_nnz);
         hypre_CSRMatrixInitialize(hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(
                                                                            compGrid[outer_level])));
      }
      if (hypre_ParAMGDataRestriction(amg_data) && outer_level != 0)
      {
         hypre_CSRMatrix *R_diag_original = hypre_ParCSRMatrixDiag(hypre_ParAMGDataPArray(
                                                                      amg_data)[outer_level - 1]);
         hypre_CSRMatrix *R_offd_original = hypre_ParCSRMatrixOffd(hypre_ParAMGDataPArray(
                                                                      amg_data)[outer_level - 1]);
         HYPRE_Int ave_nnz_per_row = 1;
         if (hypre_CSRMatrixNumRows(R_diag_original))
         {
            ave_nnz_per_row = (HYPRE_Int) (hypre_CSRMatrixNumNonzeros(R_diag_original) / hypre_CSRMatrixNumRows(
                                              R_diag_original));
         }
         HYPRE_Int max_nonowned_diag_nnz = hypre_AMGDDCompGridNumNonOwnedNodes(
                                              compGrid[outer_level]) * ave_nnz_per_row;
         HYPRE_Int max_nonowned_offd_nnz = hypre_CSRMatrixNumNonzeros(R_offd_original);
         hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[outer_level - 1])) =
            hypre_CSRMatrixCreate(hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[outer_level]),
                                  hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[outer_level - 1]), max_nonowned_diag_nnz);
         hypre_CSRMatrixInitialize(hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(
                                                                            compGrid[outer_level - 1])));
         hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(compGrid[outer_level - 1])) =
            hypre_CSRMatrixCreate(hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[outer_level]),
                                  hypre_AMGDDCompGridNumOwnedNodes(compGrid[outer_level - 1]), max_nonowned_offd_nnz);
         hypre_CSRMatrixInitialize(hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(
                                                                            compGrid[outer_level - 1])));
      }

      // Get send/recv info from the comp grid comm pkg
      HYPRE_Int num_send_procs = hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[outer_level];
      HYPRE_Int num_recv_procs = hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[outer_level];
      HYPRE_Int *send_procs = hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[outer_level];
      HYPRE_Int *recv_procs = hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[outer_level];

      if (num_send_procs || num_recv_procs)
      {
         ////////////////////////////////////
         // Get the buffer sizes
         ////////////////////////////////////

         HYPRE_Int *send_sizes = hypre_CTAlloc(HYPRE_Int, 2 * num_send_procs, HYPRE_MEMORY_HOST);
         for (proc = 0; proc < num_send_procs; proc++)
         {
            for (level = outer_level; level < num_levels; level++)
            {
               HYPRE_Int idx;
               HYPRE_Int A_row_size = 0;
               HYPRE_Int P_row_size = 0;
               HYPRE_Int R_row_size = 0;
               for (i = 0; i < hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  idx = hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                  if (idx < 0) { idx = -(idx + 1); }

                  // Owned diag and offd
                  if (idx < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
                  {
                     diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
                     offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
                     A_row_size = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                  + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                     if (level != num_levels - 1)
                     {
                        diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridP(compGrid[level]));
                        offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]));
                        P_row_size = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                     + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                     }
                     if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
                     {
                        diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridR(compGrid[level - 1]));
                        offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridR(compGrid[level - 1]));
                        R_row_size = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                     + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                     }
                  }
                  // Nonowned diag and offd
                  else
                  {
                     idx -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
                     // Count diag and offd
                     diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
                     offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
                     A_row_size = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                  + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                     if (level != num_levels - 1)
                     {
                        diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[level]));
                        offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]));
                        P_row_size = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                     + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                     }
                     if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
                     {
                        diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[level - 1]));
                        offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(compGrid[level - 1]));
                        R_row_size = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                     + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                     }
                  }

                  send_sizes[2 * proc] += A_row_size + P_row_size + R_row_size;
                  send_sizes[2 * proc + 1] += A_row_size + P_row_size + R_row_size;
               }
               if (level != num_levels - 1) { send_sizes[2 * proc] += hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; }
               if (hypre_ParAMGDataRestriction(amg_data) && level != 0) { send_sizes[2 * proc] += hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; }
            }
         }


         HYPRE_Int **int_recv_buffers = hypre_CTAlloc(HYPRE_Int*, num_recv_procs, HYPRE_MEMORY_HOST);
         HYPRE_Complex **complex_recv_buffers = hypre_CTAlloc(HYPRE_Complex*, num_recv_procs,
                                                              HYPRE_MEMORY_HOST);

         // Communicate buffer sizes
         hypre_MPI_Request *size_requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs,
                                                          HYPRE_MEMORY_HOST);
         HYPRE_Int request_cnt = 0;
         hypre_MPI_Status *size_statuses = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs,
                                                         HYPRE_MEMORY_HOST);
         HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int, 2 * num_recv_procs, HYPRE_MEMORY_HOST);

         for (proc = 0; proc < num_recv_procs; proc++)
         {
            hypre_MPI_Irecv(&(recv_sizes[2 * proc]), 2, HYPRE_MPI_INT, recv_procs[proc], 1,
                            hypre_MPI_COMM_WORLD,
                            &(size_requests[request_cnt++]));
         }
         for (proc = 0; proc < num_send_procs; proc++)
         {
            hypre_MPI_Isend(&(send_sizes[2 * proc]), 2, HYPRE_MPI_INT, send_procs[proc], 1,
                            hypre_MPI_COMM_WORLD,
                            &(size_requests[request_cnt++]));
         }

         ////////////////////////////////////
         // Pack buffers
         ////////////////////////////////////

         // int_send_buffer = [ [level] , [level] , ... , [level] ]
         // level = [ [A col ind], [P_rows], ( [R_rows] ) ]
         // P_row = [ row_size, [col_ind] ]
         // complex_send_buffer = [ [level] , [level] , ... , [level] ]
         // level = [ [A_data] , [P_data], ( [R_data] ) ]

         hypre_MPI_Request *buf_requests = hypre_CTAlloc(hypre_MPI_Request,
                                                         2 * (num_send_procs + num_recv_procs), HYPRE_MEMORY_HOST);
         request_cnt = 0;
         hypre_MPI_Status *buf_statuses = hypre_CTAlloc(hypre_MPI_Status,
                                                        2 * (num_send_procs + num_recv_procs), HYPRE_MEMORY_HOST);
         HYPRE_Int **int_send_buffers = hypre_CTAlloc(HYPRE_Int*, num_send_procs, HYPRE_MEMORY_HOST);
         HYPRE_Complex **complex_send_buffers = hypre_CTAlloc(HYPRE_Complex*, num_send_procs,
                                                              HYPRE_MEMORY_HOST);
         for (proc = 0; proc < num_send_procs; proc++)
         {
            int_send_buffers[proc] = hypre_CTAlloc(HYPRE_Int, send_sizes[2 * proc], HYPRE_MEMORY_HOST);
            complex_send_buffers[proc] = hypre_CTAlloc(HYPRE_Complex, send_sizes[2 * proc + 1],
                                                       HYPRE_MEMORY_HOST);

            HYPRE_Int int_cnt = 0;
            HYPRE_Int complex_cnt = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               // Pack A
               for (i = 0; i < hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  HYPRE_Int idx = hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                  if (idx < 0) { idx = -(idx + 1); }

                  // Owned diag and offd
                  if (idx < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
                  {
                     diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
                     offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
                     for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx + 1]; j++)
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j] + hypre_AMGDDCompGridFirstGlobalIndex(
                                                               compGrid[level]);
                        complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                     }
                     for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx + 1]; j++)
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_AMGDDCompGridNonOwnedGlobalIndices(
                                                               compGrid[level])[ hypre_CSRMatrixJ(offd)[j] ];
                        complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                     }
                  }
                  // Nonowned diag and offd
                  else
                  {
                     idx -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);

                     diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
                     offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
                     for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx + 1]; j++)
                     {
                        if (hypre_CSRMatrixJ(diag)[j] < 0) { int_send_buffers[proc][int_cnt++] = -(hypre_CSRMatrixJ(diag)[j] + 1); }
                        else { int_send_buffers[proc][int_cnt++] = hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(diag)[j] ]; }
                        complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                     }
                     for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx + 1]; j++)
                     {
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j] + hypre_AMGDDCompGridFirstGlobalIndex(
                                                               compGrid[level]);
                        complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                     }
                  }
               }
               // Pack P
               if (level != num_levels - 1)
               {
                  for (i = 0; i < hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) { idx = -(idx + 1); }

                     // Owned diag and offd
                     if (idx < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
                     {
                        diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridP(compGrid[level]));
                        offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]));
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                                            + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                        for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx + 1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j] + hypre_AMGDDCompGridFirstGlobalIndex(
                                                                  compGrid[level + 1]);
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                        }
                        for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx + 1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                        }
                     }
                     // Nonowned diag and offd
                     else
                     {
                        idx -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
                        diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[level]));
                        offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]));
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                                            + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                        for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx + 1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                        }
                        for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx + 1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j] + hypre_AMGDDCompGridFirstGlobalIndex(
                                                                  compGrid[level + 1]);
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                        }
                     }
                  }
               }
               // Pack R
               if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
               {
                  for (i = 0; i < hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) { idx = -(idx + 1); }

                     // Owned diag and offd
                     if (idx < hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
                     {
                        diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridR(compGrid[level - 1]));
                        offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridR(compGrid[level - 1]));
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                                            + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                        for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx + 1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j] + hypre_AMGDDCompGridFirstGlobalIndex(
                                                                  compGrid[level - 1]);
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                        }
                        for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx + 1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                        }
                     }
                     // Nonowned diag and offd
                     else
                     {
                        idx -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
                        diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[level - 1]));
                        offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(compGrid[level - 1]));
                        int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixI(diag)[idx + 1] - hypre_CSRMatrixI(diag)[idx]
                                                            + hypre_CSRMatrixI(offd)[idx + 1] - hypre_CSRMatrixI(offd)[idx];
                        for (j = hypre_CSRMatrixI(diag)[idx]; j < hypre_CSRMatrixI(diag)[idx + 1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(diag)[j];
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(diag)[j];
                        }
                        for (j = hypre_CSRMatrixI(offd)[idx]; j < hypre_CSRMatrixI(offd)[idx + 1]; j++)
                        {
                           int_send_buffers[proc][int_cnt++] = hypre_CSRMatrixJ(offd)[j] + hypre_AMGDDCompGridFirstGlobalIndex(
                                                                  compGrid[level - 1]);
                           complex_send_buffers[proc][complex_cnt++] = hypre_CSRMatrixData(offd)[j];
                        }
                     }
                  }
               }
            }
         }

         ////////////////////////////////////
         // Communicate
         ////////////////////////////////////

         for (proc = 0; proc < num_send_procs; proc++)
         {
            hypre_MPI_Isend(int_send_buffers[proc], send_sizes[2 * proc], HYPRE_MPI_INT, send_procs[proc], 2,
                            hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
            hypre_MPI_Isend(complex_send_buffers[proc], send_sizes[2 * proc + 1], HYPRE_MPI_COMPLEX,
                            send_procs[proc], 3, hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
         }

         // Wait on buffer sizes
         hypre_MPI_Waitall( num_send_procs + num_recv_procs, size_requests, size_statuses );

         // Allocate and post recvs
         for (proc = 0; proc < num_recv_procs; proc++)
         {
            int_recv_buffers[proc] = hypre_CTAlloc(HYPRE_Int, recv_sizes[2 * proc], HYPRE_MEMORY_HOST);
            complex_recv_buffers[proc] = hypre_CTAlloc(HYPRE_Complex, recv_sizes[2 * proc + 1],
                                                       HYPRE_MEMORY_HOST);
            hypre_MPI_Irecv(int_recv_buffers[proc], recv_sizes[2 * proc], HYPRE_MPI_INT, recv_procs[proc], 2,
                            hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
            hypre_MPI_Irecv(complex_recv_buffers[proc], recv_sizes[2 * proc + 1], HYPRE_MPI_COMPLEX,
                            recv_procs[proc], 3, hypre_MPI_COMM_WORLD, &(buf_requests[request_cnt++]));
         }

         // Wait on buffers
         hypre_MPI_Waitall( 2 * (num_send_procs + num_recv_procs), buf_requests, buf_statuses );

         for (proc = 0; proc < num_send_procs; proc++) { hypre_TFree(int_send_buffers[proc], HYPRE_MEMORY_HOST); }
         for (proc = 0; proc < num_send_procs; proc++) { hypre_TFree(complex_send_buffers[proc], HYPRE_MEMORY_HOST); }
         hypre_TFree(int_send_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(complex_send_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(size_requests, HYPRE_MEMORY_HOST);
         hypre_TFree(size_statuses, HYPRE_MEMORY_HOST);
         hypre_TFree(buf_requests, HYPRE_MEMORY_HOST);
         hypre_TFree(buf_statuses, HYPRE_MEMORY_HOST);

         // P_tmp_info[buffer_number] = [ size, [row], size, [row], ... ]
         HYPRE_Int **P_tmp_info_int = NULL;
         HYPRE_Complex **P_tmp_info_complex = NULL;
         HYPRE_Int P_tmp_info_size = 0;
         HYPRE_Int P_tmp_info_cnt = 0;
         if (outer_level != num_levels - 1)
         {
            for (proc = 0; proc < num_recv_procs; proc++)
            {
               P_tmp_info_size += hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][outer_level];
            }
            P_tmp_info_size -= hypre_CSRMatrixNumCols(hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(
                                                                                            compGrid[outer_level])));
            P_tmp_info_int = hypre_CTAlloc(HYPRE_Int*, P_tmp_info_size, HYPRE_MEMORY_HOST);
            P_tmp_info_complex = hypre_CTAlloc(HYPRE_Complex*, P_tmp_info_size, HYPRE_MEMORY_HOST);
         }
         // R_tmp_info[buffer_number] = [ size, [row], size, [row], ... ]
         HYPRE_Int **R_tmp_info_int = NULL;
         HYPRE_Complex **R_tmp_info_complex = NULL;
         HYPRE_Int R_tmp_info_size = 0;
         HYPRE_Int R_tmp_info_cnt = 0;
         if (hypre_ParAMGDataRestriction(amg_data) && outer_level != 0)
         {
            for (proc = 0; proc < num_recv_procs; proc++)
            {
               R_tmp_info_size += hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][outer_level];
            }
            R_tmp_info_size -= hypre_CSRMatrixNumCols(hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(
                                                                                            compGrid[outer_level])));
            R_tmp_info_int = hypre_CTAlloc(HYPRE_Int*, R_tmp_info_size, HYPRE_MEMORY_HOST);
            R_tmp_info_complex = hypre_CTAlloc(HYPRE_Complex*, R_tmp_info_size, HYPRE_MEMORY_HOST);
         }

         ////////////////////////////////////
         // Unpack recvs
         ////////////////////////////////////

         for (proc = 0; proc < num_recv_procs; proc++)
         {
            HYPRE_Int int_cnt = 0;
            HYPRE_Int complex_cnt = 0;

            for (level = outer_level; level < num_levels; level++)
            {
               for (i = 0; i < hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
               {
                  HYPRE_Int idx = hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];

                  if (idx < 0) { idx = -(idx + 1); }

                  // !!! Optimization: I send (and setup) A info twice for ghosts overwritten as real
                  // Unpack A data
                  diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
                  offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));
                  HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[idx];
                  HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[idx];

                  while (diag_rowptr < hypre_CSRMatrixI(diag)[idx + 1] ||
                         offd_rowptr < hypre_CSRMatrixI(offd)[idx + 1])
                  {
                     HYPRE_Int incoming_index = int_recv_buffers[proc][int_cnt++];

                     // See whether global index is owned
                     if (incoming_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]) &&
                         incoming_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[level]))
                     {
                        // Don't overwrite data if already accounted for (ordering can change and screw things up)
                        if (level == outer_level || idx == A_row_cnt[level])
                        {
                           hypre_CSRMatrixData(offd)[offd_rowptr++] = complex_recv_buffers[proc][complex_cnt++];
                        }
                        else
                        {
                           complex_cnt++;
                           offd_rowptr++;
                        }
                     }
                     else
                     {
                        // Don't overwrite data if already accounted for (ordering can change and screw things up)
                        if (level == outer_level || idx == A_row_cnt[level])
                        {
                           hypre_CSRMatrixData(diag)[diag_rowptr++] = complex_recv_buffers[proc][complex_cnt++];
                        }
                        else
                        {
                           complex_cnt++;
                           diag_rowptr++;
                        }
                     }
                  }
                  if (level != outer_level && idx == A_row_cnt[level]) { A_row_cnt[level]++; }
               }

               if (level == outer_level) { A_row_cnt[level] += hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; }

               // Unpack P data and col indices
               if (level != num_levels - 1)
               {
                  diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[level]));
                  offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]));

                  for (i = 0; i < hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) { idx = -(idx + 1); }

                     // Setup orig commPkg recv dofs
                     if (idx == P_row_cnt[level])
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];

                        HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[idx];
                        HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[idx];

                        for (j = 0; j < row_size; j++)
                        {
                           HYPRE_Int incoming_index = int_recv_buffers[proc][int_cnt++];

                           // See whether global index is owned
                           if (incoming_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level + 1]) &&
                               incoming_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[level + 1]))
                           {
                              if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(offd))
                              {
                                 hypre_CSRMatrixResize(offd, hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd),
                                                       (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(offd) + 1));
                              }
                              hypre_CSRMatrixJ(offd)[offd_rowptr] = incoming_index - hypre_AMGDDCompGridFirstGlobalIndex(
                                                                       compGrid[level + 1]);
                              hypre_CSRMatrixData(offd)[offd_rowptr] = complex_recv_buffers[proc][complex_cnt++];
                              offd_rowptr++;
                           }
                           else
                           {
                              if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(diag))
                              {
                                 hypre_CSRMatrixResize(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag),
                                                       (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(diag) + 1));
                              }
                              hypre_CSRMatrixJ(diag)[diag_rowptr] = incoming_index;
                              hypre_CSRMatrixData(diag)[diag_rowptr] = complex_recv_buffers[proc][complex_cnt++];
                              diag_rowptr++;
                           }
                        }
                        hypre_CSRMatrixI(diag)[idx + 1] = diag_rowptr;
                        hypre_CSRMatrixI(offd)[idx + 1] = offd_rowptr;

                        P_row_cnt[level]++;
                     }
                     // Store info for later setup on current outer level
                     else if (level == outer_level)
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                        P_tmp_info_int[P_tmp_info_cnt] = hypre_CTAlloc(HYPRE_Int, row_size + 1, HYPRE_MEMORY_HOST);
                        P_tmp_info_complex[P_tmp_info_cnt] = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
                        P_tmp_info_int[P_tmp_info_cnt][0] = row_size;
                        for (j = 0; j < row_size; j++)
                        {
                           P_tmp_info_int[P_tmp_info_cnt][j + 1] = int_recv_buffers[proc][int_cnt++];
                           P_tmp_info_complex[P_tmp_info_cnt][j] = complex_recv_buffers[proc][complex_cnt++];
                        }
                        P_tmp_info_cnt++;
                     }
                     // Otherwise, simply advance counters appropriately
                     else
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                        for (j = 0; j < row_size; j++)
                        {
                           int_cnt++;
                           complex_cnt++;
                        }
                     }
                  }
               }
               // Unpack R data and col indices
               if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
               {
                  diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[level - 1]));
                  offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(compGrid[level - 1]));

                  for (i = 0; i < hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
                  {
                     HYPRE_Int idx = hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
                     if (idx < 0) { idx = -(idx + 1); }

                     // Setup orig commPkg recv dofs
                     if (idx == R_row_cnt[level - 1])
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];

                        HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[idx];
                        HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[idx];

                        for (j = 0; j < row_size; j++)
                        {
                           HYPRE_Int incoming_index = int_recv_buffers[proc][int_cnt++];

                           // See whether global index is owned
                           if (incoming_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level - 1]) &&
                               incoming_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[level - 1]))
                           {
                              if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(offd))
                              {
                                 hypre_CSRMatrixResize(offd, hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd),
                                                       (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(offd) + 1));
                              }
                              hypre_CSRMatrixJ(offd)[offd_rowptr] = incoming_index - hypre_AMGDDCompGridFirstGlobalIndex(
                                                                       compGrid[level - 1]);
                              hypre_CSRMatrixData(offd)[offd_rowptr] = complex_recv_buffers[proc][complex_cnt++];
                              offd_rowptr++;
                           }
                           else
                           {
                              if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(diag))
                              {
                                 hypre_CSRMatrixResize(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag),
                                                       (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(diag) + 1));
                              }
                              hypre_CSRMatrixJ(diag)[diag_rowptr] = incoming_index;
                              hypre_CSRMatrixData(diag)[diag_rowptr] = complex_recv_buffers[proc][complex_cnt++];
                              diag_rowptr++;
                           }
                        }
                        hypre_CSRMatrixI(diag)[idx + 1] = diag_rowptr;
                        hypre_CSRMatrixI(offd)[idx + 1] = offd_rowptr;

                        R_row_cnt[level - 1]++;
                     }
                     // Store info for later setup on current outer level
                     else if (level == outer_level)
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                        R_tmp_info_int[R_tmp_info_cnt] = hypre_CTAlloc(HYPRE_Int, row_size + 1, HYPRE_MEMORY_HOST);
                        R_tmp_info_complex[R_tmp_info_cnt] = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
                        R_tmp_info_int[R_tmp_info_cnt][0] = row_size;
                        for (j = 0; j < row_size; j++)
                        {
                           R_tmp_info_int[R_tmp_info_cnt][j + 1] = int_recv_buffers[proc][int_cnt++];
                           R_tmp_info_complex[R_tmp_info_cnt][j] = complex_recv_buffers[proc][complex_cnt++];
                        }
                        R_tmp_info_cnt++;
                     }
                     // Otherwise, simply advance counters appropriately
                     else
                     {
                        HYPRE_Int row_size = int_recv_buffers[proc][int_cnt++];
                        for (j = 0; j < row_size; j++)
                        {
                           int_cnt++;
                           complex_cnt++;
                        }
                     }
                  }
               }
            }
         }

         // Setup temporary info for P on current level
         if (outer_level != num_levels - 1)
         {
            diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[outer_level]));
            offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(compGrid[outer_level]));

            HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[ P_row_cnt[outer_level] ];
            HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[ P_row_cnt[outer_level] ];

            for (i = 0; i < P_tmp_info_size; i++)
            {
               if (P_tmp_info_int[i])
               {
                  HYPRE_Int row_size = P_tmp_info_int[i][0];
                  for (j = 0; j < row_size; j++)
                  {
                     HYPRE_Int incoming_index = P_tmp_info_int[i][j + 1];

                     // See whether global index is owned
                     if (incoming_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[outer_level + 1]) &&
                         incoming_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[outer_level + 1]))
                     {
                        if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(offd))
                        {
                           hypre_CSRMatrixResize(offd, hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd),
                                                 (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(offd) + 1));
                        }
                        hypre_CSRMatrixJ(offd)[offd_rowptr] = incoming_index - hypre_AMGDDCompGridFirstGlobalIndex(
                                                                 compGrid[outer_level + 1]);
                        hypre_CSRMatrixData(offd)[offd_rowptr] = P_tmp_info_complex[i][j];
                        offd_rowptr++;
                     }
                     else
                     {
                        if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(diag))
                        {
                           hypre_CSRMatrixResize(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag),
                                                 (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(diag) + 1));
                        }
                        hypre_CSRMatrixJ(diag)[diag_rowptr] = incoming_index;
                        hypre_CSRMatrixData(diag)[diag_rowptr] = P_tmp_info_complex[i][j];
                        diag_rowptr++;
                     }

                  }
                  hypre_CSRMatrixI(diag)[P_row_cnt[outer_level] + 1] = diag_rowptr;
                  hypre_CSRMatrixI(offd)[P_row_cnt[outer_level] + 1] = offd_rowptr;
                  P_row_cnt[outer_level]++;

                  hypre_TFree(P_tmp_info_int[i], HYPRE_MEMORY_HOST);
                  hypre_TFree(P_tmp_info_complex[i], HYPRE_MEMORY_HOST);
               }
            }

            hypre_TFree(P_tmp_info_int, HYPRE_MEMORY_HOST);
            hypre_TFree(P_tmp_info_complex, HYPRE_MEMORY_HOST);
         }
         // Setup temporary info for R on current level
         if (hypre_ParAMGDataRestriction(amg_data) && outer_level != 0)
         {
            diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[outer_level - 1]));
            offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(compGrid[outer_level - 1]));

            HYPRE_Int diag_rowptr = hypre_CSRMatrixI(diag)[ R_row_cnt[outer_level - 1] ];
            HYPRE_Int offd_rowptr = hypre_CSRMatrixI(offd)[ R_row_cnt[outer_level - 1] ];

            for (i = 0; i < R_tmp_info_size; i++)
            {
               if (R_tmp_info_int[i])
               {
                  HYPRE_Int row_size = R_tmp_info_int[i][0];
                  for (j = 0; j < row_size; j++)
                  {
                     HYPRE_Int incoming_index = R_tmp_info_int[i][j + 1];

                     // See whether global index is owned
                     if (incoming_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[outer_level - 1]) &&
                         incoming_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[outer_level - 1]))
                     {
                        if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(offd))
                        {
                           hypre_CSRMatrixResize(offd, hypre_CSRMatrixNumRows(offd), hypre_CSRMatrixNumCols(offd),
                                                 (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(offd) + 1));
                        }
                        hypre_CSRMatrixJ(offd)[offd_rowptr] = incoming_index - hypre_AMGDDCompGridFirstGlobalIndex(
                                                                 compGrid[outer_level - 1]);
                        hypre_CSRMatrixData(offd)[offd_rowptr] = R_tmp_info_complex[i][j];
                        offd_rowptr++;
                     }
                     else
                     {
                        if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(diag))
                        {
                           hypre_CSRMatrixResize(diag, hypre_CSRMatrixNumRows(diag), hypre_CSRMatrixNumCols(diag),
                                                 (HYPRE_Int)hypre_ceil(1.5 * hypre_CSRMatrixNumNonzeros(diag) + 1));
                        }
                        hypre_CSRMatrixJ(diag)[diag_rowptr] = incoming_index;
                        hypre_CSRMatrixData(diag)[diag_rowptr] = R_tmp_info_complex[i][j];
                        diag_rowptr++;
                     }

                  }
                  hypre_CSRMatrixI(diag)[R_row_cnt[outer_level - 1] + 1] = diag_rowptr;
                  hypre_CSRMatrixI(offd)[R_row_cnt[outer_level - 1] + 1] = offd_rowptr;
                  R_row_cnt[outer_level - 1]++;

                  hypre_TFree(R_tmp_info_int[i], HYPRE_MEMORY_HOST);
                  hypre_TFree(R_tmp_info_complex[i], HYPRE_MEMORY_HOST);
               }
            }

            hypre_TFree(R_tmp_info_int, HYPRE_MEMORY_HOST);
            hypre_TFree(R_tmp_info_complex, HYPRE_MEMORY_HOST);
         }

         // Clean up memory
         for (proc = 0; proc < num_recv_procs; proc++) { hypre_TFree(int_recv_buffers[proc], HYPRE_MEMORY_HOST); }
         for (proc = 0; proc < num_recv_procs; proc++) { hypre_TFree(complex_recv_buffers[proc], HYPRE_MEMORY_HOST); }
         hypre_TFree(int_recv_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(complex_recv_buffers, HYPRE_MEMORY_HOST);
         hypre_TFree(send_sizes, HYPRE_MEMORY_HOST);
         hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
      }
   }

   // Clean up memory
   hypre_TFree(P_row_cnt, HYPRE_MEMORY_HOST);
   hypre_TFree(R_row_cnt, HYPRE_MEMORY_HOST);
   hypre_TFree(A_row_cnt, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_FixUpRecvMaps( hypre_AMGDDCompGrid **compGrid,
                                 hypre_AMGDDCommPkg   *compGridCommPkg,
                                 HYPRE_Int             start_level,
                                 HYPRE_Int             num_levels )
{
   HYPRE_Int  ****recv_red_marker;

   HYPRE_Int      proc;
   HYPRE_Int      inner_level;
   HYPRE_Int      num_nodes;
   HYPRE_Int      redundant;
   HYPRE_Int      map_val;
   HYPRE_Int      level, i;

   // Initial fix up of recv map:
   // Get rid of redundant recvs and index from beginning of nonowned (instead of owned)
   if (compGridCommPkg)
   {
      recv_red_marker = hypre_AMGDDCommPkgRecvRedMarker(compGridCommPkg);

      for (level = start_level; level < num_levels; level++)
      {
         for (proc = 0; proc < hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level]; proc++)
         {
            for (inner_level = level; inner_level < num_levels; inner_level++)
            {
               // if there were nodes in psiComposite on this level
               if (hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level])
               {
                  // store the number of nodes on this level
                  num_nodes = hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[level][proc][inner_level];
                  hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[level][proc][inner_level] = 0;

                  for (i = 0; i < num_nodes; i++)
                  {
                     if (inner_level == level)
                     {
                        redundant = 0;
                     }
                     else
                     {
                        redundant = recv_red_marker[level][proc][inner_level][i];
                     }

                     if (!redundant)
                     {
                        map_val = hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level][i];
                        if (map_val < 0)
                        {
                           map_val += hypre_AMGDDCompGridNumOwnedNodes(compGrid[inner_level]);
                        }
                        else
                        {
                           map_val -= hypre_AMGDDCompGridNumOwnedNodes(compGrid[inner_level]);
                        }
                        hypre_AMGDDCommPkgRecvMap(
                           compGridCommPkg)[level][proc][inner_level][ hypre_AMGDDCommPkgNumRecvNodes(
                                                                          compGridCommPkg)[level][proc][inner_level]++ ] = map_val;
                     }
                  }
                  hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level] =
                     hypre_TReAlloc(hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[level][proc][inner_level],
                                    HYPRE_Int,
                                    hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[level][proc][inner_level],
                                    HYPRE_MEMORY_HOST);
               }
            }
         }
      }
   }

   return hypre_error_flag;
}
