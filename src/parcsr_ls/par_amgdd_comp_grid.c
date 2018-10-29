/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Member functions for hypre_ParCompGrid and hypre_ParCompGridCommPkg classes.
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include <stdio.h>
#include <math.h>

HYPRE_Int
RecursivelyMarkRealDofs(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, HYPRE_Int padding);

hypre_ParCompGrid *
hypre_ParCompGridCreate ()
{
   hypre_ParCompGrid      *compGrid;

   compGrid = hypre_CTAlloc(hypre_ParCompGrid, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridNumNodes(compGrid) = 0;
   hypre_ParCompGridNumOwnedNodes(compGrid) = 0;
   hypre_ParCompGridNumRealNodes(compGrid) = 0;
   hypre_ParCompGridMemSize(compGrid) = 0;
   hypre_ParCompGridAMemSize(compGrid) = 0;
   hypre_ParCompGridPMemSize(compGrid) = 0;   
   hypre_ParCompGridU(compGrid) = NULL;
   hypre_ParCompGridF(compGrid) = NULL;
   hypre_ParCompGridGlobalIndices(compGrid) = NULL;
   hypre_ParCompGridCoarseGlobalIndices(compGrid) = NULL;
   hypre_ParCompGridCoarseLocalIndices(compGrid) = NULL;
   hypre_ParCompGridRealDofMarker(compGrid) = NULL;
   hypre_ParCompGridCoarseResidualMarker(compGrid) = NULL;

   hypre_ParCompGridARowPtr(compGrid) = NULL;
   hypre_ParCompGridAColInd(compGrid) = NULL;
   hypre_ParCompGridAGlobalColInd(compGrid) = NULL;
   hypre_ParCompGridAData(compGrid) = NULL;
   hypre_ParCompGridPRowPtr(compGrid) = NULL;
   hypre_ParCompGridPColInd(compGrid) = NULL;
   hypre_ParCompGridPData(compGrid) = NULL;

   return compGrid;
}

HYPRE_Int
hypre_ParCompGridDestroy ( hypre_ParCompGrid *compGrid )
{
   HYPRE_Int      i;
   
   if (hypre_ParCompGridU(compGrid))
   {
      hypre_TFree(hypre_ParCompGridU(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridF(compGrid))
   {
      hypre_TFree(hypre_ParCompGridF(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridGlobalIndices(compGrid))
   {
      hypre_TFree(hypre_ParCompGridGlobalIndices(compGrid), HYPRE_MEMORY_HOST);
   }
   
   if (hypre_ParCompGridCoarseGlobalIndices(compGrid))
   {
      hypre_TFree(hypre_ParCompGridCoarseGlobalIndices(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridCoarseLocalIndices(compGrid))
   {
      hypre_TFree(hypre_ParCompGridCoarseLocalIndices(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridRealDofMarker(compGrid))
   {
      hypre_TFree(hypre_ParCompGridRealDofMarker(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridCoarseResidualMarker(compGrid))
   {
      hypre_TFree(hypre_ParCompGridCoarseResidualMarker(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridARowPtr(compGrid))
   {
      hypre_TFree(hypre_ParCompGridARowPtr(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridAColInd(compGrid))
   {
      hypre_TFree(hypre_ParCompGridAColInd(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridAGlobalColInd(compGrid))
   {
      hypre_TFree(hypre_ParCompGridAGlobalColInd(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridAData(compGrid))
   {
      hypre_TFree(hypre_ParCompGridAData(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridPRowPtr(compGrid))
   {
      hypre_TFree(hypre_ParCompGridPRowPtr(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridPColInd(compGrid))
   {
      hypre_TFree(hypre_ParCompGridPColInd(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridPData(compGrid))
   {
      hypre_TFree(hypre_ParCompGridPData(compGrid), HYPRE_MEMORY_HOST);
   }

   hypre_TFree(compGrid, HYPRE_MEMORY_HOST);   
   

   return 0;
}

HYPRE_Int
hypre_ParCompGridInitialize ( hypre_ParCompGrid *compGrid, hypre_ParVector *residual, HYPRE_Int *CF_marker_array, HYPRE_Int coarseStart, hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int         i,j;

   hypre_Vector *residual_local = hypre_ParVectorLocalVector(residual);
   HYPRE_Int         num_nodes = hypre_VectorSize(residual_local);   

   hypre_ParCompGridNumNodes(compGrid) = num_nodes;
   hypre_ParCompGridNumOwnedNodes(compGrid) = num_nodes;
   hypre_ParCompGridNumRealNodes(compGrid) = num_nodes;
   hypre_ParCompGridMemSize(compGrid) = 2*num_nodes;

   HYPRE_Int A_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(A) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(A) );
   hypre_ParCompGridAMemSize(compGrid) = 2*A_nnz;
   HYPRE_Int P_nnz;
   if (P)
   {
      P_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(P) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(P) );
      hypre_ParCompGridPMemSize(compGrid) = 2*P_nnz;
   }

   // Allocate space for the info on the comp nodes
   HYPRE_Int        *global_indices_comp = hypre_CTAlloc(HYPRE_Int, 2*num_nodes, HYPRE_MEMORY_HOST);
   HYPRE_Int        *coarse_global_indices_comp = NULL; 
   HYPRE_Int        *coarse_local_indices_comp = NULL;
   if ( CF_marker_array )
   {
      coarse_global_indices_comp = hypre_CTAlloc(HYPRE_Int, 2*num_nodes, HYPRE_MEMORY_HOST); 
      coarse_local_indices_comp = hypre_CTAlloc(HYPRE_Int, 2*num_nodes, HYPRE_MEMORY_HOST);
   }
   HYPRE_Int        *A_rowptr = hypre_CTAlloc(HYPRE_Int, 2*(num_nodes+1), HYPRE_MEMORY_HOST);
   HYPRE_Int        *A_colind = hypre_CTAlloc(HYPRE_Int, 2*A_nnz, HYPRE_MEMORY_HOST);
   HYPRE_Int        *A_global_colind = hypre_CTAlloc(HYPRE_Int, 2*A_nnz, HYPRE_MEMORY_HOST);
   HYPRE_Complex    *A_data = hypre_CTAlloc(HYPRE_Complex, 2*A_nnz, HYPRE_MEMORY_HOST);
   HYPRE_Int        *P_rowptr = NULL;
   HYPRE_Int        *P_colind = NULL;
   HYPRE_Complex    *P_data = NULL;
   if (P)
   {
      P_rowptr = hypre_CTAlloc(HYPRE_Int, 2*(num_nodes+1), HYPRE_MEMORY_HOST);
      P_colind = hypre_CTAlloc(HYPRE_Int, 2*P_nnz, HYPRE_MEMORY_HOST);
      P_data = hypre_CTAlloc(HYPRE_Complex, 2*P_nnz, HYPRE_MEMORY_HOST);
   }

   // Set up temporary arrays for getting rows of matrix A
   HYPRE_Int         row_size;
   HYPRE_Int         *row_col_ind;
   HYPRE_Complex     *row_values;


   // Initialize composite grid data to the given information
   HYPRE_Int        coarseIndexCounter = 0;

   for (i = 0; i < num_nodes; i++)
   {
      global_indices_comp[i] = hypre_ParVectorFirstIndex(residual) + i;
      if ( CF_marker_array ) // if there is a CF_marker_array for this level (i.e. unless we are on the coarsest level)
      {
         if ( CF_marker_array[i] == 1 )
         {
            coarse_global_indices_comp[i] = coarseIndexCounter + coarseStart;
            coarse_local_indices_comp[i] = coarseIndexCounter;
            coarseIndexCounter++;
         }
         else 
         {
               coarse_global_indices_comp[i] = -1;
               coarse_local_indices_comp[i] = -1;
         }
      }
      else coarse_global_indices_comp = coarse_local_indices_comp = NULL;
      
      // Setup row of matrix A
      hypre_ParCSRMatrixGetRow( A, global_indices_comp[i], &row_size, &row_col_ind, &row_values );
      A_rowptr[i+1] = A_rowptr[i] + row_size;
      for (j = A_rowptr[i]; j < A_rowptr[i+1]; j++)
      {
         A_data[j] = row_values[j - A_rowptr[i]];

         HYPRE_Int global_index = row_col_ind[j - A_rowptr[i]];
         A_global_colind[j] = global_index;

         if ( global_index >= hypre_ParVectorFirstIndex(residual) && global_index <= hypre_ParVectorLastIndex(residual) )
            A_colind[j] = global_index - hypre_ParVectorFirstIndex(residual);
         else A_colind[j] = -1;
      }
      hypre_ParCSRMatrixRestoreRow( A, i, &row_size, &row_col_ind, &row_values );

      if (P)
      {
         // Setup row of matrix A
         hypre_ParCSRMatrixGetRow( P, global_indices_comp[i], &row_size, &row_col_ind, &row_values );
         P_rowptr[i+1] = P_rowptr[i] + row_size;
         for (j = P_rowptr[i]; j < P_rowptr[i+1]; j++)
         {
            P_data[j] = row_values[j - P_rowptr[i]];
            P_colind[j] = row_col_ind[j - P_rowptr[i]];
         }
         hypre_ParCSRMatrixRestoreRow( P, i, &row_size, &row_col_ind, &row_values );
      }
   }

   // Set attributes for compGrid
   hypre_ParCompGridU(compGrid) = hypre_CTAlloc(HYPRE_Complex, 2*num_nodes, HYPRE_MEMORY_HOST);
   hypre_ParCompGridF(compGrid) = hypre_CTAlloc(HYPRE_Complex, 2*num_nodes, HYPRE_MEMORY_HOST);
   hypre_ParCompGridGlobalIndices(compGrid) = global_indices_comp;
   hypre_ParCompGridCoarseGlobalIndices(compGrid) = coarse_global_indices_comp;
   hypre_ParCompGridCoarseLocalIndices(compGrid) = coarse_local_indices_comp;
   hypre_ParCompGridARowPtr(compGrid) = A_rowptr;
   hypre_ParCompGridAColInd(compGrid) = A_colind;
   hypre_ParCompGridAGlobalColInd(compGrid) = A_global_colind;
   hypre_ParCompGridAData(compGrid) = A_data;
   hypre_ParCompGridPRowPtr(compGrid) = P_rowptr;
   hypre_ParCompGridPColInd(compGrid) = P_colind;
   hypre_ParCompGridPData(compGrid) = P_data;

   // cleanup memory
   hypre_TFree( row_col_ind, HYPRE_MEMORY_HOST );
   hypre_TFree( row_values, HYPRE_MEMORY_HOST );

   return 0;
}

HYPRE_Int
hypre_ParCompGridFinalize( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int transition_level, HYPRE_Int debug )
{
   HYPRE_Int delete_global_indices = 1;

   if (debug)
   {
      delete_global_indices = 0;
   }

   HYPRE_Int i,j,k,cnt,level;

   // Clean up memory for things we don't need anymore
   for (level = 0; level < transition_level; level++)
   {
      if (hypre_ParCompGridGlobalIndices(compGrid[level]) && delete_global_indices)
      {
         hypre_TFree(hypre_ParCompGridGlobalIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridGlobalIndices(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridAGlobalColInd(compGrid[level]) && delete_global_indices)
      {
         hypre_TFree(hypre_ParCompGridAGlobalColInd(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridAGlobalColInd(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridCoarseGlobalIndices(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridCoarseGlobalIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridCoarseGlobalIndices(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridCoarseLocalIndices(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridCoarseLocalIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridCoarseLocalIndices(compGrid[level]) = NULL;
      }
   }

   // Setup the coarse residual marker for use in FAC cycles
   for (level = 1; level < transition_level; level++)
   {
      hypre_ParCompGridCoarseResidualMarker(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST); // mark the coarse dofs as we restrict (or don't) to make sure they are all updated appropriately: 0 = nothing has happened yet, 1 = has incomplete residual info, 2 = restricted to from fine grid

      // Look at fine grid A matrix
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level-1]); i++)
      {
         // Loop over entries in A
         for (j = hypre_ParCompGridARowPtr(compGrid[level-1])[i]; j < hypre_ParCompGridARowPtr(compGrid[level-1])[i+1]; j++)
         {
            // If -1 index encountered, mark the coarse grid connections to this node (don't want to restrict to these)
            if ( hypre_ParCompGridAColInd(compGrid[level-1])[j] < 0 )
            {
               for (k = hypre_ParCompGridPRowPtr(compGrid[level-1])[i]; k < hypre_ParCompGridPRowPtr(compGrid[level-1])[i+1]; k++)
               {
                  if (hypre_ParCompGridPColInd(compGrid[level-1])[k] >= 0)
                     hypre_ParCompGridCoarseResidualMarker(compGrid[level])[ hypre_ParCompGridPColInd(compGrid[level-1])[k] ] = 1; // Mark coarse dofs that we don't want to restrict to from fine grid
               }
               break;
            }
         }
      }
      
      // Mark where we have complete residual information
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level-1]); i++)
      {
         // Loop over entries in P
         for (j = hypre_ParCompGridPRowPtr(compGrid[level-1])[i]; j < hypre_ParCompGridPRowPtr(compGrid[level-1])[i+1]; j++)
         {
            if (hypre_ParCompGridPColInd(compGrid[level-1])[j] >= 0)
            {
               // Add contribution to restricted residual where appropriate
               if (hypre_ParCompGridCoarseResidualMarker(compGrid[level])[ hypre_ParCompGridPColInd(compGrid[level-1])[j] ] != 1) 
               {
                  hypre_ParCompGridCoarseResidualMarker(compGrid[level])[ hypre_ParCompGridPColInd(compGrid[level-1])[j] ] = 2; // Mark coarse dofs that successfully recieve their value from restriction from the fine grid
               }
            }
         }
      }
   }

   // Setup the coarse residual marker for use in FAC cycles
   if (transition_level != num_levels)
   {
      hypre_ParCompGridCoarseResidualMarker(compGrid[transition_level]) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[transition_level]), HYPRE_MEMORY_HOST); // mark the coarse dofs as we restrict (or don't) to make sure they are all updated appropriately: 0 = nothing has happened yet, 1 = has incomplete residual info, 2 = restricted to from fine grid

      // Look at fine grid A matrix
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[transition_level-1]); i++)
      {
         // Loop over entries in A
         for (j = hypre_ParCompGridARowPtr(compGrid[transition_level-1])[i]; j < hypre_ParCompGridARowPtr(compGrid[transition_level-1])[i+1]; j++)
         {
            // If -1 index encountered, mark the coarse grid connections to this node (don't want to restrict to these)
            if ( hypre_ParCompGridAColInd(compGrid[transition_level-1])[j] < 0 )
            {
               for (k = hypre_ParCompGridPRowPtr(compGrid[transition_level-1])[i]; k < hypre_ParCompGridPRowPtr(compGrid[transition_level-1])[i+1]; k++)
               {
                  hypre_ParCompGridCoarseResidualMarker(compGrid[transition_level])[ hypre_ParCompGridPColInd(compGrid[transition_level-1])[k] ] = 1; // Mark coarse dofs that we don't want to restrict to from fine grid
               }
               break;
            }
         }
      }
      
      // Mark where we have complete residual information
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[transition_level-1]); i++)
      {
         // Loop over entries in P
         for (j = hypre_ParCompGridPRowPtr(compGrid[transition_level-1])[i]; j < hypre_ParCompGridPRowPtr(compGrid[transition_level-1])[i+1]; j++)
         {
            // Add contribution to restricted residual where appropriate
            if (hypre_ParCompGridPColInd(compGrid[transition_level-1])[j] < 0) printf("P has -1 index found in comp grid finalize\n");
            if (hypre_ParCompGridCoarseResidualMarker(compGrid[transition_level])[ hypre_ParCompGridPColInd(compGrid[transition_level-1])[j] ] != 1) 
            {
               hypre_ParCompGridCoarseResidualMarker(compGrid[transition_level])[ hypre_ParCompGridPColInd(compGrid[transition_level-1])[j] ] = 2; // Mark coarse dofs that successfully recieve their value from restriction from the fine grid
            }
         }
      }
   }

   return 0;
}

HYPRE_Int
hypre_ParCompGridSetupRealDofMarker( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int padding )
{
   HYPRE_Int *add_flag_fine = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[0]), HYPRE_MEMORY_HOST);
   HYPRE_Int *add_flag_coarse;

   HYPRE_Int i,level;
   for (level = 0; level < num_levels-1; level++)
   {
      // Allocate the real dof marker
      hypre_ParCompGridRealDofMarker(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_HOST);

      // Mark the real dofs
      if (level == 0)
      {
         for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid[level]); i++) add_flag_fine[i] = padding + 1;
      }
      else
      {
         add_flag_fine = add_flag_coarse;
      }

      // Expand by the padding on this level and add coarse grid counterparts
      if (level != num_levels-2) add_flag_coarse = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level+1]), HYPRE_MEMORY_HOST);
      else add_flag_coarse = NULL;
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
      {
         if (add_flag_fine[i] == padding + 1)
         {
            // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
            hypre_ParCompGridRealDofMarker(compGrid[level])[i] = 1;
            RecursivelyMarkRealDofs(i, padding, compGrid[level], add_flag_fine, add_flag_coarse, padding);
         }
      }


      hypre_TFree(add_flag_fine, HYPRE_MEMORY_HOST);
   }

   return 0;
}

HYPRE_Int
hypre_ParCompGridSetSizeMatricesOnly ( hypre_ParCompGrid *compGrid, HYPRE_Int num_nodes, HYPRE_Int A_nnz, HYPRE_Int P_nnz )
{
   hypre_ParCompGridNumNodes(compGrid) = num_nodes;
   hypre_ParCompGridMemSize(compGrid) = num_nodes;
   hypre_ParCompGridAMemSize(compGrid) = A_nnz;
   hypre_ParCompGridPMemSize(compGrid) = P_nnz;

   hypre_ParCompGridU(compGrid) = hypre_CTAlloc(HYPRE_Complex, num_nodes, HYPRE_MEMORY_HOST);
   hypre_ParCompGridF(compGrid) = hypre_CTAlloc(HYPRE_Complex, num_nodes, HYPRE_MEMORY_HOST);
   
   hypre_ParCompGridARowPtr(compGrid) = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_HOST);
   hypre_ParCompGridAColInd(compGrid) = hypre_CTAlloc(HYPRE_Int, A_nnz, HYPRE_MEMORY_HOST);
   hypre_ParCompGridAData(compGrid) = hypre_CTAlloc(HYPRE_Complex, A_nnz, HYPRE_MEMORY_HOST);

   if (P_nnz)
   {
      hypre_ParCompGridPRowPtr(compGrid) = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_HOST);
      hypre_ParCompGridPColInd(compGrid) = hypre_CTAlloc(HYPRE_Int, P_nnz, HYPRE_MEMORY_HOST);
      hypre_ParCompGridPData(compGrid) = hypre_CTAlloc(HYPRE_Complex, P_nnz, HYPRE_MEMORY_HOST);      
   }

   return 0;
}

HYPRE_Int
hypre_ParCompGridResize ( hypre_ParCompGrid *compGrid, HYPRE_Int new_size, HYPRE_Int need_coarse_info, HYPRE_Int type )
{
   // This function reallocates exactly enough memory to hold a comp grid of size new_size
   // num_nodes and mem_size are set to new_size. Use this when exact size of new comp grid is known.
   HYPRE_Int      i;

   // Reallocate num nodes
   if (type == 0)
   {
      // Re allocate to given size
      hypre_ParCompGridU(compGrid) = hypre_TReAlloc(hypre_ParCompGridU(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridF(compGrid) = hypre_TReAlloc(hypre_ParCompGridF(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridGlobalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridARowPtr(compGrid) = hypre_TReAlloc(hypre_ParCompGridARowPtr(compGrid), HYPRE_Int, new_size+1, HYPRE_MEMORY_HOST);
      if (need_coarse_info)
      {
         hypre_ParCompGridCoarseGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridCoarseGlobalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
         hypre_ParCompGridCoarseLocalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridCoarseLocalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
         hypre_ParCompGridPRowPtr(compGrid) = hypre_TReAlloc(hypre_ParCompGridPRowPtr(compGrid), HYPRE_Int, new_size+1, HYPRE_MEMORY_HOST);
      }
      hypre_ParCompGridMemSize(compGrid) = new_size;
      hypre_ParCompGridNumNodes(compGrid) = new_size;    
   }
   // Reallocate A matrix
   else if (type == 1)
   {
      hypre_ParCompGridAColInd(compGrid) = hypre_TReAlloc(hypre_ParCompGridAColInd(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridAGlobalColInd(compGrid) = hypre_TReAlloc(hypre_ParCompGridAGlobalColInd(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridAData(compGrid) = hypre_TReAlloc(hypre_ParCompGridAData(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridAMemSize(compGrid) = new_size;
   }
   // Reallocate P matrix
   else if (type == 2)
   {
      hypre_ParCompGridPColInd(compGrid) = hypre_TReAlloc(hypre_ParCompGridPColInd(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridPData(compGrid) = hypre_TReAlloc(hypre_ParCompGridPData(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridPMemSize(compGrid) = new_size;
   }

   return 0;
}

HYPRE_Int 
hypre_ParCompGridSetupLocalIndices( hypre_ParCompGrid **compGrid, HYPRE_Int *num_added_nodes, HYPRE_Int transition_level, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index )
{
   // when nodes are added to a composite grid, global info is copied over, but local indices must be generated appropriately for all added nodes
   // this must be done on each level as info is added to correctly construct subsequent Psi_c grids
   // also done after each ghost layer is added
   HYPRE_Int      level,i,j,k,l;
   HYPRE_Int      row_size, global_index, coarse_global_index, local_index, insert_row_size;



   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);


   for (level = 0; level < transition_level; level++)
   {
      // If we have added nodes on this level
      if (num_added_nodes[level])
      {
         // loop over indices of non-owned nodes on this level
         for (i = hypre_ParCompGridNumOwnedNodes(compGrid[level]); i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            // fix up the local indices for the matrix A row info
            for (j = hypre_ParCompGridARowPtr(compGrid[level])[i]; j < hypre_ParCompGridARowPtr(compGrid[level])[i+1]; j++)
            {
               // initialize local index to -1 (it will be set to another value only if we can find the global index somewhere in this comp grid)
               global_index = hypre_ParCompGridAGlobalColInd(compGrid[level])[j];
               local_index = -global_index-1;
               
               // if global index of j'th row entry is owned by this proc, then local index is calculable
               if ( global_index >= proc_first_index[level] && global_index <= proc_last_index[level] )
               {
                  // set local index for entry in this row of the matrix
                  local_index = global_index - proc_first_index[level];
               }
               // otherwise find local index via binary search
               else
               {
                  local_index = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level], global_index, 0);
                  if (local_index == -1) local_index = -global_index-1;
               }
               hypre_ParCompGridAColInd(compGrid[level])[j] = local_index;



               // if (myid == 1 && level == 2) printf("local_index = %d, global_index = %d\n", local_index, global_index);



               // if we need to insert an entry into the matrix (!!! Note that I'm assuming a symmetric matrix here !!!)
               if ( local_index < hypre_ParCompGridNumOwnedNodes(compGrid[level]) && local_index >= 0 )
               {
                  // search over the row to find the appropriate global index and insert local index
                  for (k = hypre_ParCompGridARowPtr(compGrid[level])[local_index]; k < hypre_ParCompGridARowPtr(compGrid[level])[local_index+1]; k++)
                  {
                     if ( hypre_ParCompGridAGlobalColInd(compGrid[level])[k] == hypre_ParCompGridGlobalIndices(compGrid[level])[i] )
                     {
                        hypre_ParCompGridAColInd(compGrid[level])[k] = i;
                        break;
                     }
                  }
               }
            }
         }
      }

      
      // if we are not on the coarsest level
      if (level != transition_level-1)
      {
         if (num_added_nodes[level] || num_added_nodes[level+1])
         {
            // loop over indices of non-owned nodes on this level
            for (i = hypre_ParCompGridNumOwnedNodes(compGrid[level]); i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
            {
               // fix up the coarse local indices
               coarse_global_index = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[i];

               // if this node is repeated on the next coarsest grid, figure out its local index
               if ( coarse_global_index != -1)
               {
                  hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level+1], coarse_global_index, 0);
               }
               // otherwise set it to -1
               else hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] = -1;
            }
         }
      }
   }

   return 0;
}

HYPRE_Int hypre_ParCompGridSetupLocalIndicesP( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels, HYPRE_Int transition_level )
{
   HYPRE_Int                  i,j,level,global_index,first,last;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   // if (myid == 0) printf("transition_level = %d\n", transition_level);

   for (level = 0; level < transition_level-1; level++)
   {
      // Get first and last owned global indices on the next level (domain of P)
      first = hypre_ParCompGridGlobalIndices(compGrid[level+1])[0];
      if (hypre_ParCompGridNumOwnedNodes(compGrid[level+1])) last = hypre_ParCompGridGlobalIndices(compGrid[level+1])[hypre_ParCompGridNumOwnedNodes(compGrid[level+1]) - 1];
      else last = -1;
      // Setup all local indices for all nodes (note that PColInd currently stores global indices)
      for (i = 0; i < hypre_ParCompGridPRowPtr(compGrid[level])[ hypre_ParCompGridNumNodes(compGrid[level]) ]; i++)
      {
         global_index = hypre_ParCompGridPColInd(compGrid[level])[i];
         // If global index is owned, simply calculate
         if (global_index >= first && global_index <= last) hypre_ParCompGridPColInd(compGrid[level])[i] = global_index - first;
         // Otherwise, binary search
         else hypre_ParCompGridPColInd(compGrid[level])[i] = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level+1], global_index, 0);
         if (hypre_ParCompGridPColInd(compGrid[level])[i] < 0) hypre_ParCompGridPColInd(compGrid[level])[i] = -global_index - 1;
      }
   }

   return 0;
}

HYPRE_Int hypre_ParCompGridLocalIndexBinarySearch( hypre_ParCompGrid *compGrid, HYPRE_Int global_index, HYPRE_Int allow_failed_search )
{
   HYPRE_Int      left = hypre_ParCompGridNumOwnedNodes(compGrid);
   HYPRE_Int      right = hypre_ParCompGridNumNodes(compGrid) - 1;
   HYPRE_Int      index;

   while (left <= right)
   {
      index = (left + right) / 2;
      if (hypre_ParCompGridGlobalIndices(compGrid)[index] < global_index) left = index + 1;
      else if (hypre_ParCompGridGlobalIndices(compGrid)[index] > global_index) right = index - 1;
      else return index;
   }

   // If binary search fails to find an exact match, return the index of the first element greater than global_index or -1
   if (allow_failed_search) return left;
   else return -1;
}

HYPRE_Int
hypre_ParCompGridDebugPrint ( hypre_ParCompGrid *compGrid, const char* filename )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Get composite grid information
   HYPRE_Int       num_nodes = hypre_ParCompGridNumNodes(compGrid);
   HYPRE_Int       num_owned_nodes = hypre_ParCompGridNumOwnedNodes(compGrid);
   HYPRE_Int       mem_size = hypre_ParCompGridMemSize(compGrid);
   HYPRE_Int       A_mem_size = hypre_ParCompGridAMemSize(compGrid);
   HYPRE_Int       P_mem_size = hypre_ParCompGridPMemSize(compGrid);

   HYPRE_Complex     *u = hypre_ParCompGridU(compGrid);
   HYPRE_Complex     *f = hypre_ParCompGridF(compGrid);

   HYPRE_Int        *global_indices = hypre_ParCompGridGlobalIndices(compGrid);
   HYPRE_Int        *coarse_global_indices = hypre_ParCompGridCoarseGlobalIndices(compGrid);
   HYPRE_Int        *coarse_local_indices = hypre_ParCompGridCoarseLocalIndices(compGrid);
   HYPRE_Int        *real_dof_marker = hypre_ParCompGridRealDofMarker(compGrid);

   HYPRE_Int *A_rowptr = hypre_ParCompGridARowPtr(compGrid);
   HYPRE_Int *A_colind = hypre_ParCompGridAColInd(compGrid);
   HYPRE_Int *A_global_colind = hypre_ParCompGridAGlobalColInd(compGrid);
   HYPRE_Complex *A_data = hypre_ParCompGridAData(compGrid);
   HYPRE_Int *P_rowptr = hypre_ParCompGridPRowPtr(compGrid);
   HYPRE_Int *P_colind = hypre_ParCompGridPColInd(compGrid);
   HYPRE_Complex *P_data = hypre_ParCompGridPData(compGrid);

   HYPRE_Int         i,j;

   // Measure number of ghost nodes
   HYPRE_Int num_real = 0;
   if (real_dof_marker) for (i = 0; i < num_nodes; i++) if (real_dof_marker[i]) num_real++;

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   hypre_fprintf(file, "Num nodes: %d\nMem size: %d\nA Mem size: %d\nP Mem size: %d\nNum owned nodes: %d\nNum ghost dofs: %d\nNum real dofs: %d\n", 
      num_nodes, mem_size, A_mem_size, P_mem_size, num_owned_nodes, num_nodes - num_real, num_real);
   hypre_fprintf(file, "u:\n");
   for (i = 0; i < num_nodes; i++)
   {
      hypre_fprintf(file, "%.10f ", u[i]);
   }
   hypre_fprintf(file, "\n");
   hypre_fprintf(file, "f:\n");
   for (i = 0; i < num_nodes; i++)
   {
      hypre_fprintf(file, "%.10f ", f[i]);
   }
   hypre_fprintf(file, "\n");
   if (global_indices)
   {
      hypre_fprintf(file, "global_indices:\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", global_indices[i]);
      }
   }
   if (real_dof_marker)
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "real_dof_marker:\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", real_dof_marker[i]);
      }
   }
   if (coarse_global_indices)
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "coarse_global_indices:\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", coarse_global_indices[i]);
      }
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "coarse_local_indices:\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", coarse_local_indices[i]);
      }
      hypre_fprintf(file, "\n");
   }
   hypre_fprintf(file, "\n");

   if (A_rowptr)
   {
      hypre_fprintf(file, "\nA row pointer:\n");
      for (i = 0; i < num_nodes+1; i++) hypre_fprintf(file, "%d ", A_rowptr[i]);
      hypre_fprintf(file,"\n\n");
      hypre_fprintf(file, "A colind:\n");
      for (i = 0; i < A_rowptr[num_nodes]; i++) hypre_fprintf(file, "%d ", A_colind[i]);
      hypre_fprintf(file,"\n\n");
      hypre_fprintf(file, "A global colind:\n");
      for (i = 0; i < A_rowptr[num_nodes]; i++) hypre_fprintf(file, "%d ", A_global_colind[i]);
      hypre_fprintf(file,"\n\n");
      hypre_fprintf(file, "A data:\n");
      for (i = 0; i < A_rowptr[num_nodes]; i++) hypre_fprintf(file, "%f ", A_data[i]);
      if (P_rowptr)
      {
         hypre_fprintf(file,"\n\n");
         hypre_fprintf(file, "P row pointer:\n");
         for (i = 0; i < num_nodes+1; i++) hypre_fprintf(file, "%d ", P_rowptr[i]);
         hypre_fprintf(file,"\n\n");
         hypre_fprintf(file, "P colind:\n");
         for (i = 0; i < P_rowptr[num_nodes]; i++) hypre_fprintf(file, "%d ", P_colind[i]);
         hypre_fprintf(file,"\n\n");
         hypre_fprintf(file, "P data:\n");
         for (i = 0; i < P_rowptr[num_nodes]; i++) hypre_fprintf(file, "%f ", P_data[i]);
      }
   }

   fclose(file);

   return 0;

}

HYPRE_Int
hypre_ParCompGridPrintSolnRHS ( hypre_ParCompGrid *compGrid, const char* filename )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Get composite grid information
   HYPRE_Int       num_nodes = hypre_ParCompGridNumNodes(compGrid);
   HYPRE_Int       num_owned_nodes = hypre_ParCompGridNumOwnedNodes(compGrid);

   HYPRE_Complex     *u = hypre_ParCompGridU(compGrid);
   HYPRE_Complex     *f = hypre_ParCompGridF(compGrid);

   HYPRE_Int        *global_indices = hypre_ParCompGridGlobalIndices(compGrid);
   HYPRE_Int        *coarse_global_indices = hypre_ParCompGridCoarseGlobalIndices(compGrid);
   HYPRE_Int        *coarse_local_indices = hypre_ParCompGridCoarseLocalIndices(compGrid);
   HYPRE_Int        *ghost_marker = hypre_ParCompGridRealDofMarker(compGrid);

   HYPRE_Int         i;

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");\
   hypre_fprintf(file, "%d\n%d\n", num_nodes, num_owned_nodes);

   for (i = 0; i < num_nodes; i++)
   {
      hypre_fprintf(file, "%e ", u[i]);
   }
   hypre_fprintf(file, "\n");
   for (i = 0; i < num_nodes; i++)
   {
      hypre_fprintf(file, "%e ", f[i]);
   }
   hypre_fprintf(file, "\n");
   for (i = 0; i < num_nodes; i++)
   {
      hypre_fprintf(file, "%d ", global_indices[i]);
   }
   if (coarse_global_indices)
   {
      hypre_fprintf(file, "\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", ghost_marker[i]);
      }
      hypre_fprintf(file, "\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", coarse_global_indices[i]);
      }
      hypre_fprintf(file, "\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", coarse_local_indices[i]);
      }
      hypre_fprintf(file, "\n");
   }
   hypre_fprintf(file, "\n");

   fclose(file);

   return 0;

}

HYPRE_Int 
hypre_ParCompGridDumpSorted( hypre_ParCompGrid *compGrid, const char* filename)
{
   // Check whether we have anything to dump
   if (!hypre_ParCompGridGlobalIndices(compGrid))
   {
      printf("Trying to dump comp grid, but no global indices\n");
      return 0;
   }

   // Get composite grid information
   HYPRE_Int        *global_indices = hypre_ParCompGridGlobalIndices(compGrid);
   HYPRE_Int        *ghost_marker = hypre_ParCompGridRealDofMarker(compGrid);

   // Get the position where the owned nodes should go in order to output arrays sorted by global index
   HYPRE_Int insert_owned_position;
   if (hypre_ParCompGridNumOwnedNodes(compGrid))
   {
      HYPRE_Int first_owned = hypre_ParCompGridGlobalIndices(compGrid)[0];
      HYPRE_Int last_owned = hypre_ParCompGridGlobalIndices(compGrid)[ hypre_ParCompGridNumOwnedNodes(compGrid) - 1 ];
      HYPRE_Int first_nonowned = hypre_ParCompGridGlobalIndices(compGrid)[ hypre_ParCompGridNumOwnedNodes(compGrid) ];
      HYPRE_Int last_nonowned = hypre_ParCompGridGlobalIndices(compGrid)[ hypre_ParCompGridNumNodes(compGrid) - 1 ];

      // Find where to insert owned nodes in the list of all comp grid nodes (such that they are ordered according to global index)
      if (last_owned < first_nonowned) insert_owned_position = hypre_ParCompGridNumOwnedNodes(compGrid);
      else if (first_owned > last_nonowned) insert_owned_position = hypre_ParCompGridNumNodes(compGrid);
      else
      {
         // Binary search to find where to insert
         insert_owned_position = hypre_ParCompGridLocalIndexBinarySearch(compGrid, first_owned, 1);
      }
   }
   else insert_owned_position = 0;

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i;

   // Global indices
   for (i = hypre_ParCompGridNumOwnedNodes(compGrid); i < insert_owned_position; i++)
   {
      hypre_fprintf(file, "%d ", global_indices[i]);
   }
   for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
   {
      hypre_fprintf(file, "%d ", global_indices[i]);
   }
   for (i = insert_owned_position; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      hypre_fprintf(file, "%d ", global_indices[i]);
   }

   if (ghost_marker)
   {
      // Ghost marker
      hypre_fprintf(file, "\n");
      for (i = hypre_ParCompGridNumOwnedNodes(compGrid); i < insert_owned_position; i++)
      {
         hypre_fprintf(file, "%d ", ghost_marker[i]);
      }
      for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d ", ghost_marker[i]);
      }
      for (i = insert_owned_position; i < hypre_ParCompGridNumNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d ", ghost_marker[i]);
      }
      hypre_fprintf(file, "\n");
   }
   hypre_fprintf(file, "\n");

   fclose(file);

   return 0;
}

HYPRE_Int 
hypre_ParCompGridGlobalIndicesDump( hypre_ParCompGrid *compGrid, const char* filename)
{
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i;

   // Global indices
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      hypre_fprintf(file, "%d\n", hypre_ParCompGridGlobalIndices(compGrid)[i]);
   }

   fclose(file);

   return 0;
}

HYPRE_Int 
hypre_ParCompGridRealDofMarkerDump( hypre_ParCompGrid *compGrid, const char* filename)
{
   if (hypre_ParCompGridRealDofMarker(compGrid))
   {
      FILE             *file;
      file = fopen(filename,"w");
      HYPRE_Int i;

      // Global indices
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d\n", hypre_ParCompGridRealDofMarker(compGrid)[i]);
      }

      fclose(file);
   }

   return 0;
}

HYPRE_Int 
hypre_ParCompGridCoarseGlobalIndicesDump( hypre_ParCompGrid *compGrid, const char* filename)
{
      FILE             *file;
      file = fopen(filename,"w");
      HYPRE_Int i;

   if (hypre_ParCompGridCoarseGlobalIndices(compGrid))
   {
      // Global indices
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d\n", hypre_ParCompGridCoarseGlobalIndices(compGrid)[i]);
      }

      fclose(file);
   }

   return 0;
}

HYPRE_Int 
hypre_ParCompGridCoarseResidualMarkerDump( hypre_ParCompGrid *compGrid, const char* filename)
{
      FILE             *file;
      file = fopen(filename,"w");
      HYPRE_Int i;

   if (hypre_ParCompGridCoarseResidualMarker(compGrid))
   {
      // Global indices
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d\n", hypre_ParCompGridCoarseResidualMarker(compGrid)[i]);
      }

      fclose(file);
   }

   return 0;
}

HYPRE_Int 
hypre_ParCompGridUDump( hypre_ParCompGrid *compGrid, const char* filename)
{
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i;

   // Global indices
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      hypre_fprintf(file, "%.14e\n", hypre_ParCompGridU(compGrid)[i]);
   }

   fclose(file);

   return 0;
}

HYPRE_Int 
hypre_ParCompGridFDump( hypre_ParCompGrid *compGrid, const char* filename)
{
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i;

   // Global indices
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      hypre_fprintf(file, "%.14e\n", hypre_ParCompGridF(compGrid)[i]);
   }

   fclose(file);

   return 0;
}

HYPRE_Int
hypre_ParCompGridMatlabAMatrixDump( hypre_ParCompGrid *compGrid, const char* filename)
{
   // Get composite grid information
   HYPRE_Int       num_nodes = hypre_ParCompGridNumNodes(compGrid);

   HYPRE_Int                     *global_indices = hypre_ParCompGridGlobalIndices(compGrid);

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i,j,row_size;

   if (hypre_ParCompGridARowPtr(compGrid))
   {
      for (i = 0; i < num_nodes; i++)
      {
         for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
         {
            hypre_fprintf(file, "%d ", i);
            hypre_fprintf(file, "%d ", hypre_ParCompGridAColInd(compGrid)[j]);
            hypre_fprintf(file, "%e\n", hypre_ParCompGridAData(compGrid)[j]);
         }
      }
   }

   fclose(file);

   return 0;
}

HYPRE_Int
hypre_ParCompGridMatlabPMatrixDump( hypre_ParCompGrid *compGrid, const char* filename)
{
   // Get composite grid information
   HYPRE_Int       num_nodes = hypre_ParCompGridNumNodes(compGrid);

   HYPRE_Int                     *global_indices = hypre_ParCompGridGlobalIndices(compGrid);

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i,j,row_size;

   if (hypre_ParCompGridPRowPtr(compGrid))
   {
      for (i = 0; i < num_nodes; i++)
      {
         for (j = hypre_ParCompGridPRowPtr(compGrid)[i]; j < hypre_ParCompGridPRowPtr(compGrid)[i+1]; j++)
         {
            hypre_fprintf(file, "%d ", i);
            hypre_fprintf(file, "%d ", hypre_ParCompGridPColInd(compGrid)[j]);
            hypre_fprintf(file, "%e\n", hypre_ParCompGridPData(compGrid)[j]);
         }
      }
   }

   fclose(file);

   return 0;
}

hypre_ParCompGridCommPkg*
hypre_ParCompGridCommPkgCreate()
{
   hypre_ParCompGridCommPkg   *compGridCommPkg;

   compGridCommPkg = hypre_CTAlloc(hypre_ParCompGridCommPkg, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridCommPkgNumLevels(compGridCommPkg) = 0;
   hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg) = -1;
   hypre_ParCompGridCommPkgUseAllgatherv(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgTransitionResRecvSizes(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgTransitionResRecvDisps(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgNumSends(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgSendProcs(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgResDispls(compGridCommPkg) = NULL;

   return compGridCommPkg;
}

hypre_ParCompGridCommPkg*
hypre_ParCompGridCommPkgCreateAndAllocate(HYPRE_Int num_levels)
{
   hypre_ParCompGridCommPkg   *compGridCommPkg;

   compGridCommPkg = hypre_CTAlloc(hypre_ParCompGridCommPkg, 1, HYPRE_MEMORY_HOST);


   hypre_ParCompGridCommPkgNumLevels(compGridCommPkg) = num_levels;
   hypre_ParCompGridCommPkgUseAllgatherv(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumSends(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgResDispls(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);

   return compGridCommPkg;
}

HYPRE_Int
hypre_ParCompGridCommPkgDestroy( hypre_ParCompGridCommPkg *compGridCommPkg )
{
   HYPRE_Int   num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int         i, j, k;

   if ( hypre_ParCompGridCommPkgUseAllgatherv(compGridCommPkg) )
   {
      hypre_TFree(hypre_ParCompGridCommPkgUseAllgatherv(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgTransitionResRecvSizes(compGridCommPkg) )
   {
      hypre_TFree(hypre_ParCompGridCommPkgTransitionResRecvSizes(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgTransitionResRecvDisps(compGridCommPkg) )
   {
      hypre_TFree(hypre_ParCompGridCommPkgTransitionResRecvDisps(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgSendProcs(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgSendProcs(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         for (j = 0; j < hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[i]; j++)
         {
            for (k = 0; k < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); k++)
            {
               if ( hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[i][j][k] ) hypre_TFree( hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[i][j][k], HYPRE_MEMORY_HOST );
            }
            hypre_TFree( hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_ParCompGridCommPkgSendFlag(compGridCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         for (j = 0; j < hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[i]; j++)
         {
            for (k = 0; k < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); k++)
            {
               if ( hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[i][j][k] ) hypre_TFree( hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[i][j][k], HYPRE_MEMORY_HOST );
            }
            hypre_TFree( hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_ParCompGridCommPkgRecvMap(compGridCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         for (j = 0; j < hypre_ParCompGridCommPkgNumSends(compGridCommPkg)[i]; j++)
         {
            hypre_TFree( hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         for (j = 0; j < hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg)[i]; j++)
         {
            hypre_TFree( hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_ParCompGridCommPkgNumSends(compGridCommPkg) )
   {
      hypre_TFree( hypre_ParCompGridCommPkgNumSends(compGridCommPkg), HYPRE_MEMORY_HOST );
   }
   
   if  ( hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg) )
   {
      hypre_TFree( hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree( hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree( hypre_ParCompGridCommPkgResRecvcounts(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgResDispls(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree( hypre_ParCompGridCommPkgResDispls(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree( hypre_ParCompGridCommPkgResDispls(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   hypre_TFree(compGridCommPkg, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int
RecursivelyMarkRealDofs(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int *add_flag_coarse, HYPRE_Int padding)
{
   HYPRE_Int         i,index,coarse_grid_index;
   HYPRE_Int error_code = 0;

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
            hypre_ParCompGridRealDofMarker(compGrid)[index] = 1;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) error_code = RecursivelyMarkRealDofs(index, m-1, compGrid, add_flag, add_flag_coarse, padding);
         }
         // If m = 1, we won't do another recursive call, so make sure to flag the coarse grid here if applicable
         if (m == 1 && add_flag_coarse)
         {
            coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid)[index];
            if ( coarse_grid_index != -1 ) 
            {
               // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
               add_flag_coarse[ coarse_grid_index ] = padding+1;
            }
         }
      }
      else
      {
         error_code = 1; 
      }
   }

   // Flag this node on the next coarsest level 
   if (add_flag_coarse)
   {
      coarse_grid_index = hypre_ParCompGridCoarseLocalIndices(compGrid)[node];
      if ( coarse_grid_index != -1 ) 
      {
         // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
         add_flag_coarse[ coarse_grid_index ] = padding+1;
      }
   }

   return error_code;
}