/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)
extern "C"
{
   __global__
   void ExpandPaddingNewKernel(HYPRE_Int *add_flag,
                        HYPRE_Int *owned_col_ind,
                        HYPRE_Int *nonowned_col_ind,
                        HYPRE_Int *owned_row_ptr,
                        HYPRE_Int *nonowned_row_ptr,
                        HYPRE_Int add_flag_size,
                        HYPRE_Int add_flag_start,
                        HYPRE_Int num_owned,
                        HYPRE_Int dist)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      HYPRE_Int j, jj;
      if (i < add_flag_size)
      {
         if (add_flag[i + add_flag_start] == dist)
         {
            HYPRE_Int start = owned_row_ptr[i];
            HYPRE_Int end = owned_row_ptr[i+1];
            for (jj = start; jj < end; jj++)
            {
               j = owned_col_ind[jj];
               add_flag[j] = max(add_flag[j], dist-1);
            }
            start = nonowned_row_ptr[i];
            end = nonowned_row_ptr[i+1];
            for (jj = start; jj < end; jj++)
            {
               j = nonowned_col_ind[jj];
               if (j >= 0) add_flag[j + num_owned] = max(add_flag[j + num_owned], dist-1);
            }
         }     
      } 
   }

   __global__
   void PackColIndNewKernel(HYPRE_Int *local_indices, 
                        HYPRE_Int *destination,
                        HYPRE_Int *offsets,
                        HYPRE_Int *owned_diag_col_ind,
                        HYPRE_Int *owned_offd_col_ind,
                        HYPRE_Int *nonowned_diag_col_ind,
                        HYPRE_Int *nonowned_offd_col_ind,
                        HYPRE_Int *owned_diag_row_ptr,
                        HYPRE_Int *owned_offd_row_ptr,
                        HYPRE_Int *nonowned_diag_row_ptr,
                        HYPRE_Int *nonowned_offd_row_ptr,
                        HYPRE_Int *nonowned_global_ind,
                        HYPRE_Int *add_flag,
                        HYPRE_Int local_indices_size,
                        HYPRE_Int num_owned,
                        HYPRE_Int first_global_index)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      HYPRE_Int j;
      if (i < local_indices_size)
      {
         HYPRE_Int cnt = offsets[i];
         HYPRE_Int send_elmt = local_indices[i];
         if (send_elmt < 0) send_elmt = -(send_elmt+1);

         if (send_elmt < num_owned)
         {
            HYPRE_Int start = owned_diag_row_ptr[send_elmt];
            HYPRE_Int end = owned_diag_row_ptr[send_elmt+1];
            for (j = start; j < end; j++)
            {
               HYPRE_Int add_flag_index = owned_diag_col_ind[j];
               if (add_flag[add_flag_index] > 0)
               {
                  destination[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  destination[cnt++] = -(add_flag_index + first_global_index + 1); // -(GID + 1)
               }
            }
            start = owned_offd_row_ptr[send_elmt];
            end = owned_offd_row_ptr[send_elmt+1];
            for (j = start; j < end; j++)
            {
               HYPRE_Int add_flag_index = owned_offd_col_ind[j] + num_owned;
               if (add_flag[add_flag_index] > 0)
               {
                  destination[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  destination[cnt++] = -(nonowned_global_ind[ owned_offd_col_ind[j] ] + 1); // -(GID + 1)
               }
            }
         }
         else
         {
            send_elmt -= num_owned;
            HYPRE_Int start = nonowned_diag_row_ptr[send_elmt];
            HYPRE_Int end = nonowned_diag_row_ptr[send_elmt+1];
            for (j = start; j < end; j++)
            {
               if (nonowned_diag_col_ind[j] >= 0)
               {
                  HYPRE_Int add_flag_index = nonowned_diag_col_ind[j] + num_owned;
                  if (add_flag[add_flag_index] > 0)
                  {
                     destination[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
                  }
                  else
                  {
                     destination[cnt++] = -(nonowned_global_ind[ nonowned_diag_col_ind[j] ] + 1); // -(GID + 1)
                  }
               }
               else
               {
                  destination[cnt++] = nonowned_diag_col_ind[j]; // -(GID + 1)
               }
            }
            start = nonowned_offd_row_ptr[send_elmt];
            end = nonowned_offd_row_ptr[send_elmt+1];
            for (j = start; j < end; j++)
            {
               HYPRE_Int add_flag_index = nonowned_offd_col_ind[j];
               if (add_flag[add_flag_index] > 0)
               {
                  destination[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  destination[cnt++] = -(add_flag_index + first_global_index + 1); // -(GID + 1)
               }
            }
         }
      } 
   }

   __global__
   void AddFlagToSendFlagKernel(HYPRE_Int *marker, HYPRE_Int *aux_marker, HYPRE_Int *list, HYPRE_Int marker_size, HYPRE_Int ghost_dist)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < marker_size)
      {
          if (marker[i] > ghost_dist) list[ aux_marker[i] ] = i;
          else if (marker[i] > 0) list[ aux_marker[i] ] = -(i + 1);
      }
   }

   struct greater_than_zero
   {
      __host__ __device__
      HYPRE_Int operator()(HYPRE_Int x)
      {
         return x > 0;
      }
   };

   struct less_than_zero
   {
      __host__ __device__
      HYPRE_Int operator()(HYPRE_Int x)
      {
         return x < 0;
      }
   };

   struct greater_than_equal_zero
   {
      __host__ __device__
      HYPRE_Int operator()(HYPRE_Int x)
      {
         return x >= 0;
      }
   };

   __global__
   void GetGlobalIndexKernel(HYPRE_Int *lid, HYPRE_Int *gid, HYPRE_Int *nonowned_gid, HYPRE_Int list_size, HYPRE_Int owned_size, HYPRE_Int first_owned_gid)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < list_size)
      {
         HYPRE_Int local_index = lid[i];
         if (local_index < 0) local_index = -(local_index+1);
         if (local_index < owned_size) gid[i] = local_index + first_owned_gid;
         else gid[i] = nonowned_gid[local_index - owned_size];
      }
   }

   void SortSendFlagGPU(hypre_AMGDDCompGrid *compGrid, HYPRE_Int *list, HYPRE_Int list_size)
   {
      const HYPRE_Int tpb=64;
      HYPRE_Int num_blocks = list_size/tpb+1;

      // Setup the keys as global indices
      HYPRE_Int *gid = hypre_CTAlloc(HYPRE_Int, list_size, HYPRE_MEMORY_DEVICE);
      GetGlobalIndexKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(list, gid, hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid), list_size, hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridFirstGlobalIndex(compGrid));
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */
      
      // Sort the list by global index
      thrust::sort_by_key(thrust::cuda::par.on(HYPRE_STREAM(1)), gid, gid + list_size, list);
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */
      hypre_TFree(gid, HYPRE_MEMORY_DEVICE);
   }

   HYPRE_Int* AddFlagToSendFlagGPU(hypre_AMGDDCompGrid *compGrid,
                        HYPRE_Int *add_flag,
                        HYPRE_Int *num_send_nodes,
                        HYPRE_Int num_ghost_layers)
   {
      const HYPRE_Int tpb=64;
      HYPRE_Int total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid) + hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);
      HYPRE_Int num_blocks = total_num_nodes/tpb+1;
      
      // Generate array that contains gather locations 
      HYPRE_Int *gather_locations = hypre_CTAlloc(HYPRE_Int, total_num_nodes, HYPRE_MEMORY_DEVICE);
      thrust::transform_exclusive_scan(thrust::cuda::par.on(HYPRE_STREAM(1)), 
                                          add_flag, 
                                          add_flag + total_num_nodes, 
                                          gather_locations, 
                                          greater_than_zero(),  
                                          0, 
                                          thrust::plus<HYPRE_Int>());
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));

      // Allocate send_flag
      if (add_flag[total_num_nodes-1] == 0) (*num_send_nodes) = gather_locations[total_num_nodes-1];
      else (*num_send_nodes) = gather_locations[total_num_nodes-1] + 1;
      HYPRE_Int *send_flag = hypre_CTAlloc(HYPRE_Int, (*num_send_nodes), HYPRE_MEMORY_DEVICE);

      // Collapse from marker to send_flag
      AddFlagToSendFlagKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(add_flag, gather_locations, send_flag, total_num_nodes, num_ghost_layers);
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));

      hypre_TFree(gather_locations, HYPRE_MEMORY_DEVICE);
      
      SortSendFlagGPU(compGrid, send_flag,(*num_send_nodes));
     
      return send_flag;
   }

   __global__
   void ListToMarkerKernel(HYPRE_Int *list, HYPRE_Int *marker, HYPRE_Int list_size)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < list_size)
      {
          HYPRE_Int marker_idx = list[i];
          if (marker_idx < 0) marker_idx = -(marker_idx + 1);
          marker[ marker_idx ] = i + 1;
      }
   }

   __global__
   void GetCoarseIndicesKernel(HYPRE_Int *list,
              HYPRE_Int *coarse_list,
              HYPRE_Int *owned_coarse_indices,
              HYPRE_Int *nonowned_coarse_indices,
              HYPRE_Int num_owned,
              HYPRE_Int num_owned_coarse,
              HYPRE_Int list_size)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < list_size)
      {
         HYPRE_Int idx = list[i];
         if (idx < num_owned)
         {
            coarse_list[i] = owned_coarse_indices[idx];
         }
         else
         {
            idx -= num_owned;
            HYPRE_Int coarse_index = nonowned_coarse_indices[idx];
            if (coarse_index >= 0)
               coarse_list[i] = coarse_index + num_owned_coarse;
            else
               coarse_list[i] = coarse_index;
         }
      }
   }

   struct not_in_range
   {
      HYPRE_Int lower;
      HYPRE_Int upper;
      __host__ __device__
      bool operator()(const int x)
      {
         return (x < lower || x >= upper);
      }
   };
   struct minus_equal
   {
      HYPRE_Int a;
      __host__ __device__
      HYPRE_Int operator()(const HYPRE_Int x)
      {
         return x - a;
      }
   };

   void MarkCoarseGPU(hypre_AMGDDCompGrid *compGrid,
                        HYPRE_Int *send_flag,
                        HYPRE_Int num_send_nodes,
                        HYPRE_Int *add_flag, 
                        HYPRE_Int num_owned_coarse,
                        HYPRE_Int dist)
   {
      // Owned points 
      not_in_range owned_range;
      owned_range.lower = 0;
      owned_range.upper = hypre_AMGDDCompGridNumOwnedNodes(compGrid);
      thrust::device_vector<HYPRE_Int> owned_real_indices_d(num_send_nodes); 
      thrust::device_vector<HYPRE_Int> owned_coarse_indices_d(num_send_nodes); 

      // !!! Debug
      HYPRE_Int myid;
      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
      if (myid == 1)
      {
         HYPRE_Int i;
         printf("send flag = ");
         for (i = 0; i < num_send_nodes; i++)
            printf("%d ", send_flag[i]);
         printf("\n");
         printf("sequential coarse indices = ");
         for (i = 0; i < num_send_nodes; i++)
            if (send_flag[i] >= 0)
               printf("%d ", hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)[ send_flag[i] ]);
         printf("\n");
      }

      auto owned_real_indices_end = thrust::remove_copy_if(thrust::cuda::par.on(HYPRE_STREAM(1)),
                                                         send_flag,
                                                         send_flag + num_send_nodes,
                                                         send_flag,
                                                         owned_real_indices_d.begin(),
                                                         owned_range); 	
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));
      HYPRE_Int num_owned_real = owned_real_indices_end - owned_real_indices_d.begin();

      // !!! Debug
      /* if (myid == 1) */
      /* { */
      /*    thrust::host_vector<HYPRE_Int> owned_real_indices_h = owned_real_indices_d; */
      /*    HYPRE_Int i; */
      /*    printf("owned real indices = "); */
      /*    for (auto it = owned_real_indices_h.begin(); it != owned_real_indices_h.end(); ++it) */
      /*       printf("%d, ", *it); */
      /*    printf("\n"); */
      /* } */

      thrust::gather(thrust::cuda::par.on(HYPRE_STREAM(1)),
                        owned_real_indices_d.begin(),
                        owned_real_indices_end,
                        hypre_AMGDDCompGridOwnedCoarseIndices(compGrid),
                        owned_coarse_indices_d.begin());
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));

      // !!! Debug
      if (myid == 1)
      {
         thrust::host_vector<HYPRE_Int> owned_coarse_indices_h = owned_coarse_indices_d;
         HYPRE_Int i;
         printf("    owned coarse indices = ");
         for (auto it = owned_coarse_indices_h.begin(); it != owned_coarse_indices_h.end(); ++it)
            printf("%d ", *it);
         printf("\n");
      }

      thrust::constant_iterator<HYPRE_Int> dist_it(dist);
      thrust::scatter_if(thrust::cuda::par.on(HYPRE_STREAM(1)),
                           dist_it,
                           dist_it + (owned_real_indices_end - owned_real_indices_d.begin()),
                           owned_coarse_indices_d.begin(),
                           owned_coarse_indices_d.begin(),
                           add_flag,
                           greater_than_equal_zero());
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));

      // !!! Debug
      if (myid == 1)
      {
         HYPRE_Int i;
         printf("add_flag = ");
         for (i = 0; i < num_owned_coarse; i++)
            printf("%d, ", add_flag[i]);
         printf("\n");
      }
      // Nonowned points 
      /* not_in_range nonowned_range; */
      /* nonowned_range.lower = 0; */
      /* nonowned_range.upper = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid); */
      /* minus_equal minus_num_owned; */
      /* minus_num_owned.a = hypre_AMGDDCompGridNumOwnedNodes(compGrid); */
      /* thrust::device_vector<HYPRE_Int> nonowned_scatter_indices_d(num_send_nodes); */ 
      /* thrust::gather_if(thrust::cuda::par.on(HYPRE_STREAM(2)), */
      /*                      thrust::make_transform_iterator(send_flag, minus_num_owned), */
      /*                      thrust::make_transform_iterator(send_flag + num_send_nodes, minus_num_owned), */
      /*                      thrust::make_transform_iterator(send_flag, minus_num_owned), */
	                        /* hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid), */
      /*                      nonowned_scatter_indices_d.begin(), */
      /*                      nonowned_range); */
      /* thrust::scatter(thrust::cuda::par.on(HYPRE_STREAM(2)), */
      /*                      dist_it, */
      /*                      dist_it + num_send_nodes, */
      /*                      nonowned_scatter_indices_d.begin(), */
      /*                      add_flag + num_owned_coarse); */
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(2))); */

   }

   __global__
   void MarkCoarseKernel(HYPRE_Int *list,
              HYPRE_Int *marker,
              HYPRE_Int *owned_coarse_indices,
              HYPRE_Int *nonowned_coarse_indices,
              HYPRE_Int num_owned,
              HYPRE_Int num_owned_coarse,
              HYPRE_Int list_size,
              HYPRE_Int dist,
              HYPRE_Int *nodes_to_add,
              HYPRE_Int myid)
   {

      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < list_size)
      {
         HYPRE_Int idx = list[i];
         if (idx >= 0)
         {
            if (idx < num_owned)
            {
               HYPRE_Int coarse_index = owned_coarse_indices[idx];
               if (coarse_index >= 0)
               {
                  marker[ coarse_index ] = dist;
                  (*nodes_to_add) = 1;
               }
            }
            else
            {
               idx -= num_owned;
               if (nonowned_coarse_indices[idx] >= 0)
               {
                  HYPRE_Int coarse_index = nonowned_coarse_indices[idx] + num_owned_coarse;
                  marker[ coarse_index ] = dist;
                  (*nodes_to_add) = 1;
               }
            }
         }
      }
   }

   __global__
   void LocalToGlobalIndexKernel(HYPRE_Int *local_indices, HYPRE_Int *global_indices, HYPRE_Int local_indices_size, HYPRE_Int num_owned, HYPRE_Int first_global, HYPRE_Int *nonowned_global_indices)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int local_index = local_indices[i];
         if (local_index < 0) local_index = -(local_index + 1);

         if (local_index < num_owned)
            global_indices[i] = local_index + first_global;
         else
            global_indices[i] = nonowned_global_indices[local_index - num_owned];
      }
   }

   __global__
   void PackCoarseGlobalIndicesKernel(HYPRE_Int *local_indices, HYPRE_Int *destination, HYPRE_Int local_indices_size, HYPRE_Int num_owned, HYPRE_Int first_global_coarse, HYPRE_Int *owned_coarse_indices, HYPRE_Int *nonowned_coarse_indices, HYPRE_Int *nonowned_global_indices_coarse)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int send_elmt = local_indices[i];
         if (send_elmt < 0) send_elmt = -(send_elmt + 1);

         if (send_elmt < num_owned)
         {
            if (owned_coarse_indices[ send_elmt ] >= 0)
               destination[i] = owned_coarse_indices[ send_elmt ] + first_global_coarse;
            else
               destination[i] = owned_coarse_indices[ send_elmt ];
         }
         else 
         {
            HYPRE_Int nonowned_index = send_elmt - num_owned;
            HYPRE_Int nonowned_coarse_index = nonowned_coarse_indices[ nonowned_index ];
            
            if (nonowned_coarse_index >= 0)
            {
               destination[i] = nonowned_global_indices_coarse[ nonowned_coarse_index ];
            }
            else if (nonowned_coarse_index == -1)
               destination[i] = nonowned_coarse_index;
            else
               destination[i] = -(nonowned_coarse_index+2);
         }
      }
   }

   __global__
   void LocalToGlobalIndexMarkGhostKernel(HYPRE_Int *local_indices, HYPRE_Int *global_indices, HYPRE_Int local_indices_size, HYPRE_Int num_owned, HYPRE_Int first_global, HYPRE_Int *nonowned_global_indices)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int local_index = local_indices[i];
         if (local_index < 0)
         {
            local_index = -(local_index + 1);
            if (local_index < num_owned)
               global_indices[i] = -(local_index + first_global + 1);
            else
               global_indices[i] = -(nonowned_global_indices[local_index - num_owned] + 1);
         }
         else
         {
            if (local_index < num_owned)
               global_indices[i] = local_index + first_global;
            else
               global_indices[i] = nonowned_global_indices[local_index - num_owned];
         }
      }
   }

   __global__
   void GetRowSizesKernel(HYPRE_Int *local_indices, HYPRE_Int *row_sizes, 
         HYPRE_Int *owned_diag_row_ptr,
         HYPRE_Int *owned_offd_row_ptr,
         HYPRE_Int *nonowned_diag_row_ptr,
         HYPRE_Int *nonowned_offd_row_ptr,
         HYPRE_Int local_indices_size, HYPRE_Int num_owned)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int send_elmt = local_indices[i];
         if (send_elmt < 0) send_elmt = -(send_elmt + 1);

         if (send_elmt < num_owned)
         {
            row_sizes[i] += owned_diag_row_ptr[send_elmt+1] - owned_diag_row_ptr[send_elmt];
            row_sizes[i] += owned_offd_row_ptr[send_elmt+1] - owned_offd_row_ptr[send_elmt];
         }
         else
         {
            send_elmt -= num_owned;
            row_sizes[i] += nonowned_diag_row_ptr[send_elmt+1] - nonowned_diag_row_ptr[send_elmt];
            row_sizes[i] += nonowned_offd_row_ptr[send_elmt+1] - nonowned_offd_row_ptr[send_elmt];
         }
      }
   }

   __global__
   void LocalToGlobalIndexRemoveGhostKernel(HYPRE_Int *local_indices, HYPRE_Int *aux_local_indices, HYPRE_Int *global_indices, HYPRE_Int local_indices_size, HYPRE_Int num_owned, HYPRE_Int first_global, HYPRE_Int *nonowned_global_indices)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int local_index = local_indices[i];
         if (local_index >= 0)
         {
            if (local_index < num_owned)
               global_indices[aux_local_indices[i]] = local_index + first_global;
            else
               global_indices[aux_local_indices[i]] = nonowned_global_indices[local_index - num_owned];
         }
      }
   }

   HYPRE_Int LocalToGlobalIndexRemoveGhostGPU(HYPRE_Int *local_indices, 
         HYPRE_Int **global_indices, 
         HYPRE_Int local_indices_size, 
         HYPRE_Int num_owned, 
         HYPRE_Int first_global, 
         HYPRE_Int *nonowned_global_indices)
   {
      const HYPRE_Int tpb=64;
      HYPRE_Int num_blocks = local_indices_size/tpb+1;
      
      HYPRE_Int *aux_local_indices = NULL;
      HYPRE_Int global_indices_size;
      // If not keeping ghost dofs, need to get locations to gather to and new size for global indices
      aux_local_indices = hypre_CTAlloc(HYPRE_Int, local_indices_size, HYPRE_MEMORY_DEVICE);
      greater_than_equal_zero gtez;
      thrust::transform(thrust::cuda::par.on(HYPRE_STREAM(1)), local_indices, local_indices + local_indices_size, aux_local_indices, gtez);
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */
      thrust::exclusive_scan(thrust::cuda::par.on(HYPRE_STREAM(1)), aux_local_indices, aux_local_indices + local_indices_size, aux_local_indices);
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */
      if (local_indices[local_indices_size-1] < 0) global_indices_size = aux_local_indices[local_indices_size-1];
      else global_indices_size = aux_local_indices[local_indices_size-1]+ 1;
      (*global_indices) = hypre_CTAlloc(HYPRE_Int, global_indices_size, HYPRE_MEMORY_DEVICE);
      LocalToGlobalIndexRemoveGhostKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(local_indices, aux_local_indices, (*global_indices), local_indices_size, num_owned, first_global, nonowned_global_indices);
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */

      return global_indices_size;
   }

   void RemoveRedundancyGPU(hypre_ParAMGData *amg_data,
                              hypre_AMGDDCompGrid **compGrid,
                              hypre_AMGDDCommPkg *compGridCommPkg,
                              HYPRE_Int ****send_flag,
                              HYPRE_Int ****recv_map,
                              HYPRE_Int ***num_send_nodes,
                              HYPRE_Int current_level,
                              HYPRE_Int proc,
                              HYPRE_Int level,
                              cudaStream_t stream)
   {

      const HYPRE_Int tpb=64;

      HYPRE_Int   myid;
      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
      // !!! Debug
      /* if (myid == 0) printf("In RemoveRedundancyGPU() %d, %d, %d\n", current_level, proc, level); */

      // Get the global indices for the send_flag
      HYPRE_Int *send_gid = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_DEVICE);
      HYPRE_Int *keys_result = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_DEVICE); 
      HYPRE_Int num_blocks = num_send_nodes[current_level][proc][level]/tpb+1;
      LocalToGlobalIndexKernel<<<num_blocks,tpb,0,stream>>>(send_flag[current_level][proc][level], send_gid, num_send_nodes[current_level][proc][level], hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]), hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]), hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]));
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */

      // Will need a copy of the send flag
      HYPRE_Int *current_send_flag = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_DEVICE);
      
      // Eliminate redundant send info by comparing with previous send_flags and recv_maps
      HYPRE_Int current_send_proc = hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[current_level][proc];
      HYPRE_Int prev_proc, prev_level;
      for (prev_level = current_level+1; prev_level <= level; prev_level++)
      {
         if (num_send_nodes[current_level][proc][level])
         {
            for (prev_proc = 0; prev_proc < hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[prev_level]; prev_proc++)
            {
               if (hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc && num_send_nodes[prev_level][prev_proc][level])
               {
                  HYPRE_Int original_send_size = 0;
                  if (prev_level == level) 
                  {
                     hypre_ParCSRCommPkg *original_commPkg = hypre_ParCSRMatrixCommPkg(hypre_ParAMGDataAArray(amg_data)[prev_level]);
                     HYPRE_Int original_proc;
                     for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumSends(original_commPkg); original_proc++)
                     {
                        if (hypre_ParCSRCommPkgSendProc(original_commPkg, original_proc) == current_send_proc) 
                        {
                           original_send_size = hypre_ParCSRCommPkgSendMapStart(original_commPkg, original_proc+1) - hypre_ParCSRCommPkgSendMapStart(original_commPkg, original_proc);
                           break;
                        }
                     }
                  }

                  // Get the global indices (to be used as keys). NOTE: need to get size of prev send (we exclude ghosts when comparing), combine this with the info in original_send_size above.
                  cudaMemcpyAsync(current_send_flag, send_flag[current_level][proc][level], sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                  HYPRE_Int *prev_send_flag = send_flag[prev_level][prev_proc][level];

                  HYPRE_Int *prev_send_gid;
                  HYPRE_Int prev_send_gid_size = LocalToGlobalIndexRemoveGhostGPU(prev_send_flag, &prev_send_gid, num_send_nodes[prev_level][prev_proc][level], hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]), hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]), hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]));
                  thrust::pair<HYPRE_Int*,HYPRE_Int*> end = thrust::set_difference_by_key(thrust::cuda::par.on(stream), send_gid, send_gid + num_send_nodes[current_level][proc][level], 
                        prev_send_gid + original_send_size, prev_send_gid + prev_send_gid_size, current_send_flag, prev_send_flag, keys_result, send_flag[current_level][proc][level]);
                  /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                  num_send_nodes[current_level][proc][level] = end.second - send_flag[current_level][proc][level]; // NOTE: no realloc of send_flag here... do that later I guess when final size determined?
                  cudaMemcpyAsync(send_gid, keys_result, sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);

                  if (original_send_size)
                  {
                     cudaMemcpyAsync(current_send_flag, send_flag[current_level][proc][level], sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                     end = thrust::set_difference_by_key(thrust::cuda::par.on(stream), send_gid, send_gid + num_send_nodes[current_level][proc][level], 
                        prev_send_gid, prev_send_gid + original_send_size, current_send_flag, prev_send_flag, keys_result, send_flag[current_level][proc][level]);
                     /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                     num_send_nodes[current_level][proc][level] = end.second - send_flag[current_level][proc][level]; // NOTE: no realloc of send_flag here... do that later I guess when final size determined?
                     cudaMemcpyAsync(send_gid, keys_result, sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                  }
                  hypre_TFree(prev_send_gid, HYPRE_MEMORY_DEVICE);
               }
            }
         }

         if (num_send_nodes[current_level][proc][level])
         {
            for (prev_proc = 0; prev_proc < hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[prev_level]; prev_proc++)
            {
               if (hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc && hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level])
               {
                  HYPRE_Int original_send_size = 0;
                  if (prev_level == level) 
                  {
                     hypre_ParCSRCommPkg *original_commPkg = hypre_ParCSRMatrixCommPkg(hypre_ParAMGDataAArray(amg_data)[prev_level]);
                     HYPRE_Int original_proc;
                     for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumRecvs(original_commPkg); original_proc++)
                     {
                        if (hypre_ParCSRCommPkgRecvProc(original_commPkg, original_proc) == current_send_proc) 
                        {
                           original_send_size = hypre_ParCSRCommPkgRecvVecStart(original_commPkg, original_proc+1) - hypre_ParCSRCommPkgRecvVecStart(original_commPkg, original_proc);
                           break;
                        }
                     }
                  }

                  // Get the global indices (to be used as keys). NOTE: need to get size of prev send (we exclude ghosts when comparing), combine this with the info in original_send_size above.
                  cudaMemcpyAsync(current_send_flag, send_flag[current_level][proc][level], sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                  HYPRE_Int *prev_recv_map = hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[prev_level][prev_proc][level];
                  /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */

                  HYPRE_Int *prev_recv_gid;
                  HYPRE_Int prev_recv_gid_size = LocalToGlobalIndexRemoveGhostGPU(prev_recv_map, &prev_recv_gid, hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level], hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]), hypre_AMGDDCompGridFirstGlobalIndex(compGrid[level]), hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]));
                  thrust::pair<HYPRE_Int*,HYPRE_Int*> end = thrust::set_difference_by_key(thrust::cuda::par.on(stream), send_gid, send_gid + num_send_nodes[current_level][proc][level], 
                        prev_recv_gid + original_send_size, prev_recv_gid + prev_recv_gid_size, current_send_flag, prev_recv_map, keys_result, send_flag[current_level][proc][level]);
                  /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                  num_send_nodes[current_level][proc][level] = end.second - send_flag[current_level][proc][level]; // NOTE: no realloc of send_flag here... do that later I guess when final size determined?
                  cudaMemcpyAsync(send_gid, keys_result, sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);

                  if (original_send_size)
                  {
                     cudaMemcpyAsync(current_send_flag, send_flag[current_level][proc][level], sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                     /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                     end = thrust::set_difference_by_key(thrust::cuda::par.on(stream), send_gid, send_gid + num_send_nodes[current_level][proc][level], 
                        prev_recv_gid, prev_recv_gid + original_send_size, current_send_flag, prev_recv_map, keys_result, send_flag[current_level][proc][level]);
                     /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                     num_send_nodes[current_level][proc][level] = end.second - send_flag[current_level][proc][level]; // NOTE: no realloc of send_flag here... do that later I guess when final size determined?
                     cudaMemcpyAsync(send_gid, keys_result, sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                  }
                  hypre_TFree(prev_recv_gid, HYPRE_MEMORY_DEVICE);
               }
            }
         }
      }
      hypre_TFree(current_send_flag, HYPRE_MEMORY_DEVICE);
      hypre_TFree(send_gid, HYPRE_MEMORY_DEVICE);
      hypre_TFree(keys_result, HYPRE_MEMORY_DEVICE);
      hypre_CheckErrorDevice(cudaStreamSynchronize(stream));
   }

}

#endif
