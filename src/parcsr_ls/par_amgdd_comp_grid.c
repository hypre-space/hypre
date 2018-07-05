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
 * Member functions for hypre_ParCompGrid and hypre_ParCompMatrixRow classes.
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include <stdio.h>
#include <math.h>

hypre_ParCompGrid *
hypre_ParCompGridCreate ()
{
   hypre_ParCompGrid      *compGrid;

   compGrid = hypre_CTAlloc(hypre_ParCompGrid, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridNumNodes(compGrid) = 0;
   hypre_ParCompGridNumOwnedNodes(compGrid) = 0;
   hypre_ParCompGridMemSize(compGrid) = 0;
   hypre_ParCompGridU(compGrid) = NULL;
   hypre_ParCompGridF(compGrid) = NULL;
   hypre_ParCompGridGlobalIndices(compGrid) = NULL;
   hypre_ParCompGridCoarseGlobalIndices(compGrid) = NULL;
   hypre_ParCompGridCoarseLocalIndices(compGrid) = NULL;
   hypre_ParCompGridGhostMarker(compGrid) = NULL;
   hypre_ParCompGridARows(compGrid) = NULL;
   hypre_ParCompGridPRows(compGrid) = NULL;

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

   if (hypre_ParCompGridGhostMarker(compGrid))
   {
      hypre_TFree(hypre_ParCompGridGhostMarker(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridARows(compGrid))
   {
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
      {
         hypre_ParCompMatrixRowDestroy(hypre_ParCompGridARows(compGrid)[i]);
      }
      hypre_TFree(hypre_ParCompGridARows(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridPRows(compGrid))
   {
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
      {
         hypre_ParCompMatrixRowDestroy(hypre_ParCompGridPRows(compGrid)[i]);
      }
      hypre_TFree(hypre_ParCompGridPRows(compGrid), HYPRE_MEMORY_HOST);
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
   // Access the residual data
   hypre_Vector      *residual_local = hypre_ParVectorLocalVector(residual);
   HYPRE_Complex     *residual_data = hypre_VectorData(residual_local);

   HYPRE_Int         num_nodes = hypre_VectorSize(residual_local);   
   hypre_ParCompGridNumNodes(compGrid) = num_nodes;
   hypre_ParCompGridNumOwnedNodes(compGrid) = num_nodes;
   hypre_ParCompGridMemSize(compGrid) = 2*num_nodes;


   // Allocate space for the info on the comp nodes
   HYPRE_Complex    *u_comp = hypre_CTAlloc(HYPRE_Complex, 2*num_nodes, HYPRE_MEMORY_HOST);
   HYPRE_Complex    *residual_comp = hypre_CTAlloc(HYPRE_Complex, 2*num_nodes, HYPRE_MEMORY_HOST);
   HYPRE_Int        *global_indices_comp = hypre_CTAlloc(HYPRE_Int, 2*num_nodes, HYPRE_MEMORY_HOST);
   HYPRE_Int        *coarse_global_indices_comp = NULL; 
   HYPRE_Int        *coarse_local_indices_comp = NULL;
   hypre_ParCompMatrixRow        **P_rows = NULL;
   if ( CF_marker_array )
   {
      coarse_global_indices_comp = hypre_CTAlloc(HYPRE_Int, 2*num_nodes, HYPRE_MEMORY_HOST); 
      coarse_local_indices_comp = hypre_CTAlloc(HYPRE_Int, 2*num_nodes, HYPRE_MEMORY_HOST);
      P_rows = hypre_CTAlloc(hypre_ParCompMatrixRow*, 2*num_nodes, HYPRE_MEMORY_HOST);   
   }
   hypre_ParCompMatrixRow        **A_rows = hypre_CTAlloc(hypre_ParCompMatrixRow*, 2*num_nodes, HYPRE_MEMORY_HOST);
   
   // Set up temporary arrays for getting rows of matrix A
   HYPRE_Int         *row_size = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_HOST);
   HYPRE_Int         **row_col_ind = hypre_CTAlloc(HYPRE_Int*, 1, HYPRE_MEMORY_HOST);
   HYPRE_Int         **local_row_col_ind = hypre_CTAlloc(HYPRE_Int*, 1, HYPRE_MEMORY_HOST);
   HYPRE_Complex     **row_values = hypre_CTAlloc(HYPRE_Complex*, 1, HYPRE_MEMORY_HOST);


   // Initialize composite grid data to the given information
   HYPRE_Int        coarseIndexCounter = 0;

   for (i = 0; i < num_nodes; i++)
   {
      residual_comp[i] = residual_data[i];
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
      A_rows[i] = hypre_ParCompMatrixRowCreate();
      // use GetRow routine for ParCSRMatrix to get col_indices and data from appropriate row of A
      hypre_ParCSRMatrixGetRow( A, global_indices_comp[i], row_size, row_col_ind, row_values );
      // Allocate space for the row information
      hypre_ParCompMatrixRowSize(A_rows[i]) = *row_size;
      hypre_ParCompMatrixRowData(A_rows[i]) = hypre_CTAlloc(HYPRE_Complex, *row_size, HYPRE_MEMORY_HOST);
      hypre_ParCompMatrixRowGlobalIndices(A_rows[i]) = hypre_CTAlloc(HYPRE_Int, *row_size, HYPRE_MEMORY_HOST);
      hypre_ParCompMatrixRowLocalIndices(A_rows[i]) = hypre_CTAlloc(HYPRE_Int, *row_size, HYPRE_MEMORY_HOST);
      // Set the row data and col indices
      for (j = 0; j < *row_size; j++)
      {
         hypre_ParCompMatrixRowData(A_rows[i])[j] = (*row_values)[j];
         hypre_ParCompMatrixRowGlobalIndices(A_rows[i])[j] = (*row_col_ind)[j];
      }
      // Restore matrix row
      hypre_ParCSRMatrixRestoreRow( A, i, row_size, row_col_ind, row_values );

      // Setup row of matrix P
      if (P)
      {
         P_rows[i] = hypre_ParCompMatrixRowCreate();
         // use GetRow routine for ParCSRMatrix to get col_indices and data from appropriate row of P
         hypre_ParCSRMatrixGetRow( P, global_indices_comp[i], row_size, row_col_ind, row_values );
         // Allocate space for the row information
         hypre_ParCompMatrixRowSize(P_rows[i]) = *row_size;
         hypre_ParCompMatrixRowData(P_rows[i]) = hypre_CTAlloc(HYPRE_Complex, *row_size, HYPRE_MEMORY_HOST);
         hypre_ParCompMatrixRowGlobalIndices(P_rows[i]) = hypre_CTAlloc(HYPRE_Int, *row_size, HYPRE_MEMORY_HOST);
         hypre_ParCompMatrixRowLocalIndices(P_rows[i]) = hypre_CTAlloc(HYPRE_Int, *row_size, HYPRE_MEMORY_HOST);
         // Set the row data and col indices
         for (j = 0; j < *row_size; j++)
         {
            hypre_ParCompMatrixRowData(P_rows[i])[j] = (*row_values)[j];
            hypre_ParCompMatrixRowGlobalIndices(P_rows[i])[j] = (*row_col_ind)[j];
         }
         // Restore matrix row
         hypre_ParCSRMatrixRestoreRow( P, i, row_size, row_col_ind, row_values );
      }
   }

   // Now that all initial rows have been added to local matrix, set the local indices
   for (i = 0; i < num_nodes; i++)
   {
      *row_size = hypre_ParCompMatrixRowSize(A_rows[i]);
      *row_col_ind = hypre_ParCompMatrixRowGlobalIndices(A_rows[i]);
      *local_row_col_ind = hypre_ParCompMatrixRowLocalIndices(A_rows[i]);

      for (j = 0; j < *row_size; j++)
      {
         // if global col index is on this comp grid, set appropriate local index
         if ( (*row_col_ind)[j] >= hypre_ParVectorFirstIndex(residual) && (*row_col_ind)[j] <= hypre_ParVectorLastIndex(residual) )
            (*local_row_col_ind)[j] = (*row_col_ind)[j] - hypre_ParVectorFirstIndex(residual);
         // else, set local index to -1
         else (*local_row_col_ind)[j] = -1;
      }

      if (P)
      {
         *row_size = hypre_ParCompMatrixRowSize(P_rows[i]);
         *row_col_ind = hypre_ParCompMatrixRowGlobalIndices(P_rows[i]);
         *local_row_col_ind = hypre_ParCompMatrixRowLocalIndices(P_rows[i]);
         for (j = 0; j < *row_size; j++)
         {
            // if global col index is on this comp grid on next level down, set appropriate local index
            if ( (*row_col_ind)[j] >= coarseStart && (*row_col_ind)[j] < coarseStart + hypre_ParCSRMatrixNumCols(P) )
               (*local_row_col_ind)[j] = (*row_col_ind)[j] - coarseStart;
            // else, set local index to -1
            else 
            {
               (*local_row_col_ind)[j] = -1;
            }
         }
      }
   }

   // Set attributes for compGrid
   hypre_ParCompGridU(compGrid) = u_comp;
   hypre_ParCompGridF(compGrid) = residual_comp;
   hypre_ParCompGridGlobalIndices(compGrid) = global_indices_comp;
   hypre_ParCompGridCoarseGlobalIndices(compGrid) = coarse_global_indices_comp;
   hypre_ParCompGridCoarseLocalIndices(compGrid) = coarse_local_indices_comp;
   if (CF_marker_array) hypre_ParCompGridGhostMarker(compGrid) = hypre_CTAlloc( HYPRE_Int, 2*num_nodes, HYPRE_MEMORY_HOST );
   else hypre_ParCompGridGhostMarker(compGrid) = NULL;
   hypre_ParCompGridARows(compGrid) = A_rows;
   hypre_ParCompGridPRows(compGrid) = P_rows;

   // cleanup memory
   hypre_TFree( row_size, HYPRE_MEMORY_HOST );
   hypre_TFree( row_col_ind, HYPRE_MEMORY_HOST );
   hypre_TFree( local_row_col_ind, HYPRE_MEMORY_HOST );
   hypre_TFree( row_values, HYPRE_MEMORY_HOST );

   return 0;
}

HYPRE_Int
hypre_ParCompGridSetSize ( hypre_ParCompGrid *compGrid, HYPRE_Int size, HYPRE_Int need_coarse_info )
{
   hypre_ParCompGridNumNodes(compGrid) = size;
   hypre_ParCompGridNumOwnedNodes(compGrid) = 0;
   hypre_ParCompGridMemSize(compGrid) = size;

   hypre_ParCompGridU(compGrid) = hypre_CTAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
   hypre_ParCompGridF(compGrid) = hypre_CTAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
   hypre_ParCompGridGlobalIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
   if (need_coarse_info)
   {
      hypre_ParCompGridGhostMarker(compGrid) = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCoarseGlobalIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCoarseLocalIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridPRows(compGrid) = hypre_CTAlloc(hypre_ParCompMatrixRow*, size, HYPRE_MEMORY_HOST);
   }
   hypre_ParCompGridARows(compGrid) = hypre_CTAlloc(hypre_ParCompMatrixRow*, size, HYPRE_MEMORY_HOST);
   
   return 0;
}

HYPRE_Int
hypre_ParCompGridDynamicResize ( hypre_ParCompGrid *compGrid, HYPRE_Int need_coarse_info )
{
   // This function doubles allocated memory if num_nodes is close to mem_size (final size of comp grid is unknown)
   // num_nodes is not reset, since we don't know how many actual new nodes will be added
   HYPRE_Int      i;
   HYPRE_Int      num_nodes = hypre_ParCompGridNumNodes(compGrid);
   HYPRE_Int      mem_size = hypre_ParCompGridMemSize(compGrid);

   // If getting close to full capacity, re allocate space for the info on the comp nodes
   if (num_nodes > mem_size - 2)
   {
      hypre_ParCompGridU(compGrid) = hypre_TReAlloc(hypre_ParCompGridU(compGrid), HYPRE_Complex, 2*mem_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridF(compGrid) = hypre_TReAlloc(hypre_ParCompGridF(compGrid), HYPRE_Complex, 2*mem_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridGlobalIndices(compGrid), HYPRE_Int, 2*mem_size, HYPRE_MEMORY_HOST);
      if (need_coarse_info)
      {
         hypre_ParCompGridGhostMarker(compGrid) = hypre_TReAlloc(hypre_ParCompGridGhostMarker(compGrid), HYPRE_Int, 2*mem_size, HYPRE_MEMORY_HOST);
         hypre_ParCompGridCoarseGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridCoarseGlobalIndices(compGrid), HYPRE_Int, 2*mem_size, HYPRE_MEMORY_HOST);
         hypre_ParCompGridCoarseLocalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridCoarseLocalIndices(compGrid), HYPRE_Int, 2*mem_size, HYPRE_MEMORY_HOST);
         hypre_ParCompGridPRows(compGrid) = hypre_TReAlloc(hypre_ParCompGridPRows(compGrid), hypre_ParCompMatrixRow*, 2*mem_size, HYPRE_MEMORY_HOST);
      }
      hypre_ParCompGridARows(compGrid) = hypre_TReAlloc(hypre_ParCompGridARows(compGrid), hypre_ParCompMatrixRow*, 2*mem_size, HYPRE_MEMORY_HOST);
      // make sure new pointers realloc'd are set to null
      for (i = mem_size; i < 2*mem_size; i++)
      {
         hypre_ParCompGridARows(compGrid)[i] = NULL;
         if (need_coarse_info) hypre_ParCompGridPRows(compGrid)[i] = NULL;
      } 
      hypre_ParCompGridMemSize(compGrid) = 2*mem_size;    
   }

   return 0;
}

HYPRE_Int
hypre_ParCompGridResize ( hypre_ParCompGrid *compGrid, HYPRE_Int new_size, HYPRE_Int need_coarse_info )
{
   // This function reallocates exactly enough memory to hold a comp grid of size new_size
   // num_nodes and mem_size are set to new_size. Use this when exact size of new comp grid is known.
   HYPRE_Int      i;
   HYPRE_Int      mem_size = hypre_ParCompGridMemSize(compGrid);

   // Re allocate to given size
   hypre_ParCompGridU(compGrid) = hypre_TReAlloc(hypre_ParCompGridU(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
   hypre_ParCompGridF(compGrid) = hypre_TReAlloc(hypre_ParCompGridF(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
   hypre_ParCompGridGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridGlobalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
   if (need_coarse_info)
   {
      hypre_ParCompGridGhostMarker(compGrid) = hypre_TReAlloc(hypre_ParCompGridGhostMarker(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCoarseGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridCoarseGlobalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCoarseLocalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridCoarseLocalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridPRows(compGrid) = hypre_TReAlloc(hypre_ParCompGridPRows(compGrid), hypre_ParCompMatrixRow*, new_size, HYPRE_MEMORY_HOST);
   }
   hypre_ParCompGridARows(compGrid) = hypre_TReAlloc(hypre_ParCompGridARows(compGrid), hypre_ParCompMatrixRow*, new_size, HYPRE_MEMORY_HOST);
   // make sure new pointers realloc'd are set to null
   for (i = mem_size; i < new_size; i++)
   {
      hypre_ParCompGridARows(compGrid)[i] = NULL;
      if (need_coarse_info) hypre_ParCompGridPRows(compGrid)[i] = NULL;
   } 
   hypre_ParCompGridMemSize(compGrid) = new_size;
   hypre_ParCompGridNumNodes(compGrid) = new_size;    

   return 0;
}

HYPRE_Int
hypre_ParCompGridCopyNode ( hypre_ParCompGrid *compGrid, hypre_ParCompGrid *compGridCopy, HYPRE_Int index, HYPRE_Int copyIndex)
{
   // this copies information on the node in 'compGrid' with index 'index' to 'compGridCopy' at location 'copyIndex'
   HYPRE_Int      row_size, j;

   // copy F, GlobalIndices, and Coarse Indices
   hypre_ParCompGridU(compGridCopy)[copyIndex] = hypre_ParCompGridU(compGrid)[index];
   hypre_ParCompGridF(compGridCopy)[copyIndex] = hypre_ParCompGridF(compGrid)[index];
   hypre_ParCompGridGlobalIndices(compGridCopy)[copyIndex] = hypre_ParCompGridGlobalIndices(compGrid)[index];
   if (hypre_ParCompGridCoarseGlobalIndices(compGrid))
   {
      hypre_ParCompGridGhostMarker(compGridCopy)[copyIndex] = hypre_ParCompGridGhostMarker(compGrid)[index];
      hypre_ParCompGridCoarseGlobalIndices(compGridCopy)[copyIndex] = hypre_ParCompGridCoarseGlobalIndices(compGrid)[index];
      hypre_ParCompGridCoarseLocalIndices(compGridCopy)[copyIndex] = hypre_ParCompGridCoarseLocalIndices(compGrid)[index];
   }

   // copy matrix A info
   if (hypre_ParCompGridARows(compGridCopy)[copyIndex]) hypre_ParCompMatrixRowDestroy(hypre_ParCompGridARows(compGridCopy)[copyIndex]);
   hypre_ParCompGridARows(compGridCopy)[copyIndex] = hypre_ParCompMatrixRowCreate();

   row_size = hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGrid)[index]);
   hypre_ParCompMatrixRowSize(hypre_ParCompGridARows(compGridCopy)[copyIndex]) = row_size;
   hypre_ParCompMatrixRowData(hypre_ParCompGridARows(compGridCopy)[copyIndex]) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
   hypre_ParCompMatrixRowGlobalIndices(hypre_ParCompGridARows(compGridCopy)[copyIndex]) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
   hypre_ParCompMatrixRowLocalIndices(hypre_ParCompGridARows(compGridCopy)[copyIndex]) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

   for (j = 0; j < row_size; j++)
   {
      hypre_ParCompMatrixRowData(hypre_ParCompGridARows(compGridCopy)[copyIndex])[j] = hypre_ParCompMatrixRowData(hypre_ParCompGridARows(compGrid)[index])[j];
      hypre_ParCompMatrixRowGlobalIndices(hypre_ParCompGridARows(compGridCopy)[copyIndex])[j] = hypre_ParCompMatrixRowGlobalIndices(hypre_ParCompGridARows(compGrid)[index])[j];
      hypre_ParCompMatrixRowLocalIndices(hypre_ParCompGridARows(compGridCopy)[copyIndex])[j] = hypre_ParCompMatrixRowLocalIndices(hypre_ParCompGridARows(compGrid)[index])[j];
   }

   // copy matrix P info (if not on coarsest grid)
   if (hypre_ParCompGridPRows(compGrid))
   {
      if (hypre_ParCompGridPRows(compGridCopy)[copyIndex]) hypre_ParCompMatrixRowDestroy(hypre_ParCompGridPRows(compGridCopy)[copyIndex]);
      hypre_ParCompGridPRows(compGridCopy)[copyIndex] = hypre_ParCompMatrixRowCreate();

      row_size = hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGrid)[index]);
      hypre_ParCompMatrixRowSize(hypre_ParCompGridPRows(compGridCopy)[copyIndex]) = row_size;
      hypre_ParCompMatrixRowData(hypre_ParCompGridPRows(compGridCopy)[copyIndex]) = hypre_CTAlloc(HYPRE_Complex, row_size, HYPRE_MEMORY_HOST);
      hypre_ParCompMatrixRowGlobalIndices(hypre_ParCompGridPRows(compGridCopy)[copyIndex]) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);
      hypre_ParCompMatrixRowLocalIndices(hypre_ParCompGridPRows(compGridCopy)[copyIndex]) = hypre_CTAlloc(HYPRE_Int, row_size, HYPRE_MEMORY_HOST);

      for (j = 0; j < row_size; j++)
      {
         hypre_ParCompMatrixRowData(hypre_ParCompGridPRows(compGridCopy)[copyIndex])[j] = hypre_ParCompMatrixRowData(hypre_ParCompGridPRows(compGrid)[index])[j];
         hypre_ParCompMatrixRowGlobalIndices(hypre_ParCompGridPRows(compGridCopy)[copyIndex])[j] = hypre_ParCompMatrixRowGlobalIndices(hypre_ParCompGridPRows(compGrid)[index])[j];
         hypre_ParCompMatrixRowLocalIndices(hypre_ParCompGridPRows(compGridCopy)[copyIndex])[j] = hypre_ParCompMatrixRowLocalIndices(hypre_ParCompGridPRows(compGrid)[index])[j];
      }
   }

   return 0;
}

HYPRE_Int 
hypre_ParCompGridSetupLocalIndices( hypre_ParCompGrid **compGrid, HYPRE_Int *num_added_nodes, HYPRE_Int num_levels, HYPRE_Int *proc_first_index, HYPRE_Int *proc_last_index )
{
   // when nodes are added to a composite grid, global info is copied over, but local indices must be generated appropriately for all added nodes
   // this must be done on each level as info is added to correctly construct subsequent Psi_c grids
   // also done after each ghost layer is added
   HYPRE_Int      level,i,j,k,l;
   hypre_ParCompMatrixRow     *row, *insert_row;
   HYPRE_Int      row_size, global_index, coarse_global_index, local_index, insert_row_size;

   // HYPRE_Int   myid, num_procs;
   // hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   // hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // for (level = 0; level < num_levels; level++)
   // {
   //    // loop over indices of nodes added to the comp grid on this level
   //    for (i = hypre_ParCompGridNumNodes(compGrid[level]) - num_added_nodes[level]; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
   //    {
   //       // fix up the local indices for the matrix A row info
   //       row = hypre_ParCompGridARows(compGrid[level])[i];
   //       row_size = hypre_ParCompMatrixRowSize(row);
   //       for (j = 0; j < row_size; j++)
   //       {
   //          // initialize local index to -1 (it will be set to another value only if we can find the global index somewhere in this comp grid)
   //          hypre_ParCompMatrixRowLocalIndices(row)[j] = -1;

   //          global_index = hypre_ParCompMatrixRowGlobalIndices(row)[j];

   //          // if global index of j'th row entry is owned by this proc, then local index is calculable
   //          if ( global_index >= proc_first_index[level] && global_index <= proc_last_index[level] )
   //          {
   //             // set local index for entry in this row of the matrix
   //             local_index = global_index - proc_first_index[level];
   //             hypre_ParCompMatrixRowLocalIndices(row)[j] = local_index;

   //             // if we need to insert an entry into the matrix (!!! Note that I'm assuming a symmetric matrix here !!!)
   //             if ( local_index < hypre_ParCompGridNumNodes(compGrid[level]) - num_added_nodes[level] )
   //             {
   //                // get the row to insert into and its size
   //                insert_row = hypre_ParCompGridARows(compGrid[level])[local_index];
   //                insert_row_size = hypre_ParCompMatrixRowSize(insert_row);
   //                // search over the row to find the appropriate global index and insert local index
   //                for (k = 0; k < insert_row_size; k++)
   //                {
   //                   if ( hypre_ParCompMatrixRowGlobalIndices(insert_row)[k] == hypre_ParCompGridGlobalIndices(compGrid[level])[i] )
   //                   {
   //                      hypre_ParCompMatrixRowLocalIndices(insert_row)[k] = i;
   //                      break;
   //                   }
   //                }
   //             }
   //          }
   //          // otherwise search over added nodes
   //          else
   //          {
   //             for (k = hypre_ParCompGridNumNodes(compGrid[level]) - 1; k >= hypre_ParCompGridNumOwnedNodes(compGrid[level]); k--) // !!! Linear search !!! Note: doing the search backward (hopefully shorter)
   //             {
   //                if ( global_index == hypre_ParCompGridGlobalIndices(compGrid[level])[k] )
   //                {
   //                   local_index = k;
   //                   hypre_ParCompMatrixRowLocalIndices(row)[j] = local_index;

   //                   // if we need to insert an entry into the matrix
   //                   if ( local_index < hypre_ParCompGridNumNodes(compGrid[level]) - num_added_nodes[level] )
   //                   {
   //                      // get the row to insert into and its size
   //                      insert_row = hypre_ParCompGridARows(compGrid[level])[local_index];
   //                      insert_row_size = hypre_ParCompMatrixRowSize(insert_row);
   //                      // search over the row to find the appropriate global index and insert local index
   //                      for (l = 0; l < insert_row_size; l++)
   //                      {
   //                         if ( hypre_ParCompMatrixRowGlobalIndices(insert_row)[l] == hypre_ParCompGridGlobalIndices(compGrid[level])[i] )
   //                         {
   //                            hypre_ParCompMatrixRowLocalIndices(insert_row)[l] = i;
   //                            break;
   //                         }
   //                      }
   //                   }
   //                   break;
   //                }
   //             }
   //          }
   //       }

   //       // if we are not on the coarsest level
   //       if (level != num_levels-1)
   //       {
   //          // fix up the coarse local indices
   //          coarse_global_index = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[i];

   //          // if this node is repeated on the next coarsest grid, figure out its local index
   //          if ( coarse_global_index != -1)
   //          {
   //             // if global coarse index is owned by this proc on next level down, then local coarse index is easy
   //             if ( coarse_global_index >= proc_first_index[level+1] && coarse_global_index <= proc_last_index[level+1] )
   //             {
   //                hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] = coarse_global_index - proc_first_index[level+1];
   //             }
   //             // otherwise, have to search over added nodes
   //             else
   //             {
   //                for (j = hypre_ParCompGridNumNodes(compGrid[level+1]) - 1; j >= hypre_ParCompGridNumOwnedNodes(compGrid[level+1]); j--) // Note: doing the search backward (hopefully shorter)
   //                {
   //                   if ( coarse_global_index == hypre_ParCompGridGlobalIndices(compGrid[level+1])[j] )
   //                   {
   //                      hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] = j;
   //                      break;
   //                   }
   //                }
   //             }

   //             if ( hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] < 0 ) printf("Warning: coarse point beneath fine point isn't in comp grid\n");
   //          }
   //          // otherwise set it to -1
   //          else hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] = -1;
            
   //          // fix up the local indices for the matrix P row info
   //          row = hypre_ParCompGridPRows(compGrid[level])[i];
   //          row_size = hypre_ParCompMatrixRowSize(row);
   //          for (j = 0; j < row_size; j++)
   //          {
   //             // initialize local index to -1 (it will be set to another value only if we can find the global index somewhere in this comp grid one level down)
   //             hypre_ParCompMatrixRowLocalIndices(row)[j] = -1;

   //             global_index = hypre_ParCompMatrixRowGlobalIndices(row)[j];
   //             // if global index of j'th row entry is owned by this proc one level down, then local index is easy
   //             if ( global_index >= proc_first_index[level+1] && global_index <= proc_last_index[level+1] )
   //             {
   //                // set local index for entry in this row of the matrix
   //                local_index = global_index - proc_first_index[level+1];
   //                hypre_ParCompMatrixRowLocalIndices(row)[j] = local_index;
   //             }
   //             // otherwise, have to search over nodes
   //             else
   //             {
   //                for (k = hypre_ParCompGridNumNodes(compGrid[level+1]) - 1; k >= hypre_ParCompGridNumOwnedNodes(compGrid[level+1]); k--) // !!! Linear search !!! Note: doing the search backward (hopefully shorter)
   //                {
   //                   if ( global_index == hypre_ParCompGridGlobalIndices(compGrid[level+1])[k] )
   //                   {
   //                      local_index = k;
   //                      hypre_ParCompMatrixRowLocalIndices(row)[j] = local_index;
   //                      break;
   //                   }
   //                }
   //             }
   //          }
   //       }
   //       // Insert into P one level up (this level is in the domain of P one level up) !!! Linear search. Whoa... this is pretty bad: we are basically looping over all of P for each node added. Yikes... !!!
   //       if (level != 0)
   //       {
   //          // Search over old rows of P to find where we need to update local indices to account for this new added node
   //          for (j = 0; j < hypre_ParCompGridNumNodes(compGrid[level-1]) - num_added_nodes[level-1]; j++)
   //          {
   //             row = hypre_ParCompGridPRows(compGrid[level-1])[j];
   //             row_size = hypre_ParCompMatrixRowSize(row);
   //             for (k = 0; k < row_size; k++)
   //             {
   //                if (hypre_ParCompGridGlobalIndices(compGrid[level])[i] == hypre_ParCompMatrixRowGlobalIndices(row)[k] )
   //                {
   //                   hypre_ParCompMatrixRowLocalIndices(row)[k] = i;
   //                   break;
   //                }
   //             }
   //          } 
   //       }
   //    }
   // }



   for (level = 0; level < num_levels; level++)
   {
      // If we have added nodes on this level
      if (num_added_nodes[level])
      {
         // loop over indices of non-owned nodes on this level
         for (i = hypre_ParCompGridNumOwnedNodes(compGrid[level]); i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            // fix up the local indices for the matrix A row info
            row = hypre_ParCompGridARows(compGrid[level])[i];
            row_size = hypre_ParCompMatrixRowSize(row);
            for (j = 0; j < row_size; j++)
            {
               // initialize local index to -1 (it will be set to another value only if we can find the global index somewhere in this comp grid)
               local_index = -1;

               global_index = hypre_ParCompMatrixRowGlobalIndices(row)[j];

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
               }
               hypre_ParCompMatrixRowLocalIndices(row)[j] = local_index;

               // if we need to insert an entry into the matrix (!!! Note that I'm assuming a symmetric matrix here !!!)
               if ( local_index < hypre_ParCompGridNumOwnedNodes(compGrid[level]) && local_index >= 0 )
               {
                  // get the row to insert into and its size
                  insert_row = hypre_ParCompGridARows(compGrid[level])[local_index];
                  insert_row_size = hypre_ParCompMatrixRowSize(insert_row);
                  // search over the row to find the appropriate global index and insert local index
                  for (k = 0; k < insert_row_size; k++)
                  {
                     if ( hypre_ParCompMatrixRowGlobalIndices(insert_row)[k] == hypre_ParCompGridGlobalIndices(compGrid[level])[i] )
                     {
                        hypre_ParCompMatrixRowLocalIndices(insert_row)[k] = i;
                        break;
                     }
                  }
               }
            }

            // if we are not on the coarsest level
            if (level != num_levels-1)
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

HYPRE_Int hypre_ParCompGridSetupLocalIndicesP( hypre_ParCompGrid **compGrid, HYPRE_Int num_levels )
{
   HYPRE_Int                  i,j,level,global_index,first,last;
   hypre_ParCompMatrixRow     *row;

   for (level = 0; level < num_levels-1; level++)
   {
      // Get first and last owned global indices on the next level (domain of P)
      first = hypre_ParCompGridGlobalIndices(compGrid[level+1])[0];
      if (hypre_ParCompGridNumOwnedNodes(compGrid[level+1])) last = hypre_ParCompGridGlobalIndices(compGrid[level+1])[hypre_ParCompGridNumOwnedNodes(compGrid[level+1]) - 1];
      else last = 0;
      // Look for missing local indices in the owned nodes
      for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid[level]); i++)
      {
         row = hypre_ParCompGridPRows(compGrid[level])[i];
         for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
         {
            if (hypre_ParCompMatrixRowLocalIndices(row)[j] < 0) 
            {
               // Binary search to find local index
               hypre_ParCompMatrixRowLocalIndices(row)[j] = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level+1], hypre_ParCompMatrixRowGlobalIndices(row)[j], 0);
            }
         }
      }
      // Setup all local indices for non-owned nodes
      for (i = hypre_ParCompGridNumOwnedNodes(compGrid[level]); i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
      {
         row = hypre_ParCompGridPRows(compGrid[level])[i];
         for (j = 0; j < hypre_ParCompMatrixRowSize(row); j++)
         {
            global_index = hypre_ParCompMatrixRowGlobalIndices(row)[j];
            // If global index is owned, simply calculate
            if (global_index >= first && global_index <= last) hypre_ParCompMatrixRowLocalIndices(row)[j] = global_index - first;
            // Otherwise, binary search
            else hypre_ParCompMatrixRowLocalIndices(row)[j] = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level+1], global_index, 0);
         }
      }
   }
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

   HYPRE_Complex     *u = hypre_ParCompGridU(compGrid);
   HYPRE_Complex     *f = hypre_ParCompGridF(compGrid);

   HYPRE_Int        *global_indices = hypre_ParCompGridGlobalIndices(compGrid);
   HYPRE_Int        *coarse_global_indices = hypre_ParCompGridCoarseGlobalIndices(compGrid);
   HYPRE_Int        *coarse_local_indices = hypre_ParCompGridCoarseLocalIndices(compGrid);
   HYPRE_Int        *ghost_marker = hypre_ParCompGridGhostMarker(compGrid);

   hypre_ParCompMatrixRow  **A_rows = hypre_ParCompGridARows(compGrid);
   hypre_ParCompMatrixRow  **P_rows = hypre_ParCompGridPRows(compGrid);

   HYPRE_Int         i,j;

   // Print info on how to read generated files


   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   hypre_fprintf(file, "Num nodes: %d\nMem size: %d\nNum owned nodes: %d\n", num_nodes, mem_size, num_owned_nodes);
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
   hypre_fprintf(file, "global_indices:\n");
   for (i = 0; i < num_nodes; i++)
   {
      hypre_fprintf(file, "%d ", global_indices[i]);
   }
   if (coarse_global_indices)
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "ghost_marker:\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", ghost_marker[i]);
      }
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

   hypre_fprintf(file, "Rows of comp matrix A: size, data, global indices, local indices\n");
   HYPRE_Int         matrix_row_size;
   HYPRE_Complex     *matrix_row_data;
   HYPRE_Int         *matrix_row_global_indices;
   HYPRE_Int         *matrix_row_local_indices;
   for (i = 0; i < num_nodes; i++)
   {
      matrix_row_size = hypre_ParCompMatrixRowSize(A_rows[i]);
      matrix_row_data = hypre_ParCompMatrixRowData(A_rows[i]);
      matrix_row_global_indices = hypre_ParCompMatrixRowGlobalIndices(A_rows[i]);
      matrix_row_local_indices = hypre_ParCompMatrixRowLocalIndices(A_rows[i]);
      hypre_fprintf(file, "row %d:\n",i);
      hypre_fprintf(file, "%d\n", matrix_row_size);
      for (j = 0; j < matrix_row_size; j++)
      {
         hypre_fprintf(file, "%.2f ", matrix_row_data[j]);
      }
      hypre_fprintf(file, "\n");
      for (j = 0; j < matrix_row_size; j++)
      {
         hypre_fprintf(file, "%d ", matrix_row_global_indices[j]);
      }
      hypre_fprintf(file, "\n");
      for (j = 0; j < matrix_row_size; j++)
      {
         hypre_fprintf(file, "%d ", matrix_row_local_indices[j]);
      }
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "\n");
   }

   if (P_rows)
   {
      hypre_fprintf(file, "Rows of comp matrix P: size, data, global indices, local indices\n");
      for (i = 0; i < num_nodes; i++)
      {
         matrix_row_size = hypre_ParCompMatrixRowSize(P_rows[i]);
         matrix_row_data = hypre_ParCompMatrixRowData(P_rows[i]);
         matrix_row_global_indices = hypre_ParCompMatrixRowGlobalIndices(P_rows[i]);
         matrix_row_local_indices = hypre_ParCompMatrixRowLocalIndices(P_rows[i]);
         hypre_fprintf(file, "row %d:\n",i);
         hypre_fprintf(file, "%d\n", matrix_row_size);
         for (j = 0; j < matrix_row_size; j++)
         {
            hypre_fprintf(file, "%.2f ", matrix_row_data[j]);
         }
         hypre_fprintf(file, "\n");
         for (j = 0; j < matrix_row_size; j++)
         {
            hypre_fprintf(file, "%d ", matrix_row_global_indices[j]);
         }
         hypre_fprintf(file, "\n");
         for (j = 0; j < matrix_row_size; j++)
         {
            hypre_fprintf(file, "%d ", matrix_row_local_indices[j]);
         }
         hypre_fprintf(file, "\n");
         hypre_fprintf(file, "\n");
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
   HYPRE_Int        *ghost_marker = hypre_ParCompGridGhostMarker(compGrid);

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
hypre_ParCompGridMatlabPlot( hypre_ParCompGrid *compGrid, const char* filename)
{

   // Get composite grid information
   HYPRE_Int       num_nodes = hypre_ParCompGridNumNodes(compGrid);

   HYPRE_Int        *global_indices = hypre_ParCompGridGlobalIndices(compGrid);
   HYPRE_Int        *coarse_global_indices = hypre_ParCompGridCoarseGlobalIndices(compGrid);
   HYPRE_Int        *ghost_marker = hypre_ParCompGridGhostMarker(compGrid);

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i;

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
   }
   hypre_fprintf(file, "\n");

   fclose(file);

   return 0;
}

HYPRE_Int
hypre_ParCompGridMatlabAMatrixDump( hypre_ParCompGrid *compGrid, const char* filename)
{
   // Get composite grid information
   HYPRE_Int       num_nodes = hypre_ParCompGridNumNodes(compGrid);

   HYPRE_Int                     *global_indices = hypre_ParCompGridGlobalIndices(compGrid);
   hypre_ParCompMatrixRow        **A_rows = hypre_ParCompGridARows(compGrid);

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i,j,row_size;

   for (i = 0; i < num_nodes; i++)
   {
      row_size = hypre_ParCompMatrixRowSize(A_rows[i]);
      for (j = 0; j < row_size; j++)
      {
         hypre_fprintf(file, "%d ", global_indices[i]);
         hypre_fprintf(file, "%d ", hypre_ParCompMatrixRowGlobalIndices(A_rows[i])[j]);
         hypre_fprintf(file, "%f\n", hypre_ParCompMatrixRowData(A_rows[i])[j]);
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
   hypre_ParCompMatrixRow        **P_rows = hypre_ParCompGridPRows(compGrid);

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i,j,row_size;

   for (i = 0; i < num_nodes; i++)
   {
      row_size = hypre_ParCompMatrixRowSize(P_rows[i]);
      for (j = 0; j < row_size; j++)
      {
         hypre_fprintf(file, "%d ", global_indices[i]);
         hypre_fprintf(file, "%d ", hypre_ParCompMatrixRowGlobalIndices(P_rows[i])[j]);
         hypre_fprintf(file, "%f\n", hypre_ParCompMatrixRowData(P_rows[i])[j]);
      }
   }

   fclose(file);

   return 0;
}

hypre_ParCompMatrixRow *
hypre_ParCompMatrixRowCreate ()
{
   hypre_ParCompMatrixRow      *row;

   row = hypre_CTAlloc(hypre_ParCompMatrixRow, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompMatrixRowSize(row) = 0;

   hypre_ParCompMatrixRowData(row) = NULL;
   hypre_ParCompMatrixRowGlobalIndices(row) = NULL;
   hypre_ParCompMatrixRowLocalIndices(row) = NULL;

   return row;
}

HYPRE_Int
hypre_ParCompMatrixRowDestroy ( hypre_ParCompMatrixRow *row )
{
   if (hypre_ParCompMatrixRowData(row))
   {
      hypre_TFree(hypre_ParCompMatrixRowData(row), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompMatrixRowGlobalIndices(row))
   {
      hypre_TFree(hypre_ParCompMatrixRowGlobalIndices(row), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompMatrixRowLocalIndices(row))
   {
      hypre_TFree(hypre_ParCompMatrixRowLocalIndices(row), HYPRE_MEMORY_HOST);
   }

   hypre_TFree(row, HYPRE_MEMORY_HOST);

   return 0;
}

hypre_ParCompGridCommPkg*
hypre_ParCompGridCommPkgCreate()
{
   hypre_ParCompGridCommPkg   *compGridCommPkg;

   compGridCommPkg = hypre_CTAlloc(hypre_ParCompGridCommPkg, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridCommPkgNumLevels(compGridCommPkg) = 0;
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
   hypre_ParCompGridCommPkgSendFlag(compGridCommPkg) = NULL;
   hypre_ParCompGridCommPkgRecvMap(compGridCommPkg) = NULL;

   return compGridCommPkg;
}

HYPRE_Int
hypre_ParCompGridCommPkgDestroy( hypre_ParCompGridCommPkg *compGridCommPkg )
{
   HYPRE_Int   num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   HYPRE_Int         i, j, k;

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

   if ( hypre_ParCompGridCommPkgNumSends(compGridCommPkg) )
   {
      hypre_TFree( hypre_ParCompGridCommPkgNumSends(compGridCommPkg), HYPRE_MEMORY_HOST );
   }
   
   if  (hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg) )
   {
      hypre_TFree( hypre_ParCompGridCommPkgNumRecvs(compGridCommPkg), HYPRE_MEMORY_HOST );
   }


   hypre_TFree(compGridCommPkg, HYPRE_MEMORY_HOST);

   return 0;
}
