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
RecursivelyMarkGhostDofs(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid);

hypre_ParCompGridMatrix* hypre_ParCompGridMatrixCreate()
{
   hypre_ParCompGridMatrix *matrix = hypre_CTAlloc(hypre_ParCompGridMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridMatrixOwnedDiag(matrix) = NULL;
   hypre_ParCompGridMatrixOwnedOffd(matrix) = NULL;
   hypre_ParCompGridMatrixNonOwnedDiag(matrix) = NULL;
   hypre_ParCompGridMatrixNonOwnedOffd(matrix) = NULL;

   hypre_ParCompGridMatrixOwnsOwnedMatrices(matrix) = 0;

   return matrix;
}

HYPRE_Int hypre_ParCompGridMatrixDestroy(hypre_ParCompGridMatrix *matrix)
{
   if (hypre_ParCompGridMatrixOwnsOwnedMatrices(matrix))
   {
      if (hypre_ParCompGridMatrixOwnedDiag(matrix)) hypre_CSRMatrixDestroy(hypre_ParCompGridMatrixOwnedDiag(matrix));
      if (hypre_ParCompGridMatrixOwnedOffd(matrix)) hypre_CSRMatrixDestroy(hypre_ParCompGridMatrixOwnedOffd(matrix));
   }
   if (hypre_ParCompGridMatrixNonOwnedDiag(matrix)) hypre_CSRMatrixDestroy(hypre_ParCompGridMatrixNonOwnedDiag(matrix));
   if (hypre_ParCompGridMatrixNonOwnedOffd(matrix)) hypre_CSRMatrixDestroy(hypre_ParCompGridMatrixNonOwnedOffd(matrix));

   hypre_TFree(matrix, HYPRE_MEMORY_HOST);

   return 0;
}

hypre_ParCompGridVector *hypre_ParCompGridVectorCreate()
{
   hypre_ParCompGridVector *vector;

   hypre_ParCompGridVectorOwned(vector) = NULL;
   hypre_ParCompGridVectorNonOwned(vector) = NULL;

   hypre_ParCompGridVectorOwnsOwnedVector(vector) = 0;

   return vector;
}

HYPRE_Int hypre_ParCompGridVectorDestroy(hypre_ParCompGridVector *vector)
{
   if (hypre_ParCompGridVectorOwnsOwnedVector(vector))
   {
      if (hypre_ParCompGridVectorOwned(vector)) hypre_SeqVectorDestroy(hypre_ParCompGridVectorOwned(vector));
   }
   if (hypre_ParCompGridVectorNonOwned(vector)) hypre_SeqVectorDestroy(hypre_ParCompGridVectorNonOwned(vector));

   hypre_TFree(vector, HYPRE_MEMORY_HOST);

   return 0;
}

hypre_ParCompGrid *
hypre_ParCompGridCreate ()
{
   hypre_ParCompGrid      *compGrid;

   compGrid = hypre_CTAlloc(hypre_ParCompGrid, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridNumNodes(compGrid) = 0;
   hypre_ParCompGridNumOwnedBlocks(compGrid) = 0;
   hypre_ParCompGridOwnedBlockStarts(compGrid) = NULL;
   hypre_ParCompGridNumRealNodes(compGrid) = 0;
   hypre_ParCompGridNumCPoints(compGrid) = 0;
   hypre_ParCompGridMemSize(compGrid) = 0;
   hypre_ParCompGridAMemSize(compGrid) = 0;
   hypre_ParCompGridPMemSize(compGrid) = 0;
   hypre_ParCompGridRMemSize(compGrid) = 0;
   hypre_ParCompGridU(compGrid) = NULL;
   hypre_ParCompGridF(compGrid) = NULL;
   hypre_ParCompGridT(compGrid) = NULL;
   hypre_ParCompGridS(compGrid) = NULL;
   hypre_ParCompGridQ(compGrid) = NULL;
   hypre_ParCompGridTemp(compGrid) = NULL;
   hypre_ParCompGridTemp2(compGrid) = NULL;
   hypre_ParCompGridTemp3(compGrid) = NULL;
   hypre_ParCompGridL1Norms(compGrid) = NULL;
   hypre_ParCompGridCFMarkerArray(compGrid) = NULL;
   hypre_ParCompGridCMask(compGrid) = NULL;
   hypre_ParCompGridFMask(compGrid) = NULL;
   hypre_ParCompGridChebyCoeffs(compGrid) = NULL;
   hypre_ParCompGridGlobalIndices(compGrid) = NULL;
   hypre_ParCompGridCoarseGlobalIndices(compGrid) = NULL;
   hypre_ParCompGridCoarseLocalIndices(compGrid) = NULL;
   hypre_ParCompGridRealDofMarker(compGrid) = NULL;
   hypre_ParCompGridEdgeIndices(compGrid) = NULL;
   hypre_ParCompGridSortMap(compGrid) = NULL;
   hypre_ParCompGridInvSortMap(compGrid) = NULL;
   hypre_ParCompGridRelaxOrdering(compGrid) = NULL;

   hypre_ParCompGridARowPtr(compGrid) = NULL;
   hypre_ParCompGridAColInd(compGrid) = NULL;
   hypre_ParCompGridAGlobalColInd(compGrid) = NULL;
   hypre_ParCompGridAData(compGrid) = NULL;
   hypre_ParCompGridPRowPtr(compGrid) = NULL;
   hypre_ParCompGridPColInd(compGrid) = NULL;
   hypre_ParCompGridPData(compGrid) = NULL;
   hypre_ParCompGridRRowPtr(compGrid) = NULL;
   hypre_ParCompGridRColInd(compGrid) = NULL;
   hypre_ParCompGridRData(compGrid) = NULL;

   hypre_ParCompGridA(compGrid) = NULL;
   hypre_ParCompGridAReal(compGrid) = NULL;
   hypre_ParCompGridP(compGrid) = NULL;
   hypre_ParCompGridR(compGrid) = NULL;

   return compGrid;
}

HYPRE_Int
hypre_ParCompGridDestroy ( hypre_ParCompGrid *compGrid )
{
   
   if (hypre_ParCompGridOwnedBlockStarts(compGrid))
   {
      hypre_TFree(hypre_ParCompGridOwnedBlockStarts(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridL1Norms(compGrid))
   {
      hypre_TFree(hypre_ParCompGridL1Norms(compGrid), HYPRE_MEMORY_SHARED);
   }

   if (hypre_ParCompGridCFMarkerArray(compGrid))
   {
      hypre_TFree(hypre_ParCompGridCFMarkerArray(compGrid), HYPRE_MEMORY_SHARED);
   }

   if (hypre_ParCompGridCMask(compGrid))
   {
      hypre_TFree(hypre_ParCompGridCMask(compGrid), HYPRE_MEMORY_SHARED);
   }

   if (hypre_ParCompGridFMask(compGrid))
   {
      hypre_TFree(hypre_ParCompGridFMask(compGrid), HYPRE_MEMORY_SHARED);
   }

   if (hypre_ParCompGridChebyCoeffs(compGrid))
   {
      hypre_TFree(hypre_ParCompGridChebyCoeffs(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridU(compGrid))
   {
      hypre_SeqVectorDestroy(hypre_ParCompGridU(compGrid));
   }

   if (hypre_ParCompGridF(compGrid))
   {
      hypre_SeqVectorDestroy(hypre_ParCompGridF(compGrid));
   }

   if (hypre_ParCompGridT(compGrid))
   {
      hypre_SeqVectorDestroy(hypre_ParCompGridT(compGrid));
   }

   if (hypre_ParCompGridS(compGrid))
   {
      hypre_SeqVectorDestroy(hypre_ParCompGridS(compGrid));
   }

   if (hypre_ParCompGridQ(compGrid))
   {
      hypre_SeqVectorDestroy(hypre_ParCompGridQ(compGrid));
   }

   if (hypre_ParCompGridTemp(compGrid))
   {
      hypre_SeqVectorDestroy(hypre_ParCompGridTemp(compGrid));
   }

   if (hypre_ParCompGridTemp2(compGrid))
   {
      hypre_SeqVectorDestroy(hypre_ParCompGridTemp2(compGrid));
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

   if (hypre_ParCompGridEdgeIndices(compGrid))
   {
      hypre_TFree(hypre_ParCompGridEdgeIndices(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridSortMap(compGrid))
   {
      hypre_TFree(hypre_ParCompGridSortMap(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridInvSortMap(compGrid))
   {
      hypre_TFree(hypre_ParCompGridInvSortMap(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridRelaxOrdering(compGrid))
   {
      hypre_TFree(hypre_ParCompGridRelaxOrdering(compGrid), HYPRE_MEMORY_HOST);
   }

   if (hypre_ParCompGridA(compGrid))
   {
      hypre_CSRMatrixDestroy(hypre_ParCompGridA(compGrid));
   }

   if (hypre_ParCompGridP(compGrid))
   {
      hypre_CSRMatrixDestroy(hypre_ParCompGridP(compGrid));
   }

   if (hypre_ParCompGridR(compGrid))
   {
      hypre_CSRMatrixDestroy(hypre_ParCompGridR(compGrid));
   }

   hypre_TFree(compGrid, HYPRE_MEMORY_HOST);   
   

   return 0;
}

HYPRE_Int
hypre_ParCompGridInitializeNew( hypre_ParAMGData *amg_data, HYPRE_Int padding, HYPRE_Int level, HYPRE_Int symmetric )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int         i,j;

   // Get info from the amg data structure
   hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];
   HYPRE_Int *CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data)[level];
   hypre_CSRMatrix *A_diag_original = hypre_ParCSRMatrixDiag( hypre_ParAMGDataAArray(amg_data)[level] );
   hypre_CSRMatrix *A_offd_original = hypre_ParCSRMatrixOffd( hypre_ParAMGDataAArray(amg_data)[level] );
   hypre_ParCompGridFirstGlobalIndex(compGrid) = hypre_ParVectorFirstIndex(hypre_ParAMGDataFArray(amg_data)[level]);
   hypre_ParCompGridLastGlobalIndex(compGrid) = hypre_ParVectorLastIndex(hypre_ParAMGDataFArray(amg_data)[level]);
   hypre_ParCompGridNumOwnedNodes(compGrid) = hypre_VectorSize(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[level]));
   hypre_ParCompGridNumNonOwnedNodes(compGrid) = 0;

   // !!! Check on how good a guess this is for eventual size of the nononwed dofs and nnz
   HYPRE_Int max_nonowned = 2 * (padding + hypre_ParAMGDataAMGDDNumGhostLayers(amg_data)) * hypre_CSRMatrixNumCols(A_offd_original);
   HYPRE_Int ave_nnz_per_row = 0;
   if (hypre_CSRMatrixNumRows(A_diag_original)) ave_nnz_per_row = (HYPRE_Int) (hypre_CSRMatrixNumNonzeros(A_diag_original) / hypre_CSRMatrixNumRows(A_diag_original));
   HYPRE_Int max_nonowned_diag_nnz = max_nonowned * ave_nnz_per_row;
   HYPRE_Int max_nonowned_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd_original);

   // Setup CompGridMatrix A
   hypre_ParCompGridMatrix *A = hypre_ParCompGridMatrixCreate();
   hypre_ParCompGridMatrixOwnedDiag(A) = A_diag_original;
   hypre_ParCompGridMatrixOwnedOffd(A) = A_offd_original;
   hypre_ParCompGridMatrixOwnsOwnedMatrices(A) = 0;
   hypre_ParCompGridMatrixNonOwnedDiag(A) = hypre_CSRMatrixCreate(max_nonowned, max_nonowned, max_nonowned_diag_nnz);
   hypre_CSRMatrixInitialize(hypre_ParCompGridMatrixNonOwnedDiag(A));
   hypre_ParCompGridMatrixNonOwnedOffd(A) = hypre_CSRMatrixCreate(max_nonowned, hypre_ParCompGridNumOwnedNodes(compGrid), max_nonowned_offd_nnz);
   hypre_CSRMatrixInitialize(hypre_ParCompGridMatrixNonOwnedOffd(A));
   hypre_ParCompGridANew(compGrid) = A;

   // !!! Symmetric: in the symmetric case we can go ahead and just setup nonowned_offd 

   // Setup CompGridMatrix P and R if appropriate (!!! Don't actually need to do this here, I guess)
   if (level != hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      hypre_ParCompGridMatrix *P = hypre_ParCompGridMatrixCreate();
      hypre_ParCompGridMatrixOwnedDiag(P) = hypre_ParCSRMatrixDiag( hypre_ParAMGDataPArray(amg_data)[level] );
      hypre_ParCompGridMatrixOwnedOffd(P) = hypre_ParCSRMatrixOffd( hypre_ParAMGDataPArray(amg_data)[level] );
      hypre_ParCompGridMatrixOwnsOwnedMatrices(P) = 0;
      hypre_ParCompGridPNew(compGrid) = P;
   }
   if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
   {
      // NOTE: want to associate rows of R with comp grid points, so need to take R from one level finer
      hypre_ParCompGridMatrix *R = hypre_ParCompGridMatrixCreate();
      hypre_ParCompGridMatrixOwnedDiag(R) = hypre_ParCSRMatrixDiag( hypre_ParAMGDataRArray(amg_data)[level-1] );
      hypre_ParCompGridMatrixOwnedOffd(R) = hypre_ParCSRMatrixOffd( hypre_ParAMGDataRArray(amg_data)[level-1] );
      hypre_ParCompGridMatrixOwnsOwnedMatrices(R) = 0;
      hypre_ParCompGridRNew(compGrid) = R;
   }

   // Allocate some extra arrays used during AMG-DD setup
   hypre_ParCompGridNonOwnedGlobalIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, HYPRE_MEMORY_HOST);
   hypre_ParCompGridNonOwnedRealMarker(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, HYPRE_MEMORY_HOST);
   hypre_ParCompGridNonOwnedSort(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, HYPRE_MEMORY_HOST);
   hypre_ParCompGridNonOwnedInvSort(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, HYPRE_MEMORY_HOST);
   hypre_ParCompGridNonOwnedDiagMissingColIndics(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, HYPRE_MEMORY_HOST);

   // TODO: initialize nonownedglobalindices???


   if (level != hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      hypre_ParCompGridNonOwnedCoarseIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, HYPRE_MEMORY_HOST);
      hypre_ParCompGridOwnedCoarseIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid), HYPRE_MEMORY_HOST);

      // Setup the owned coarse indices
      if ( CF_marker_array )
      {
         HYPRE_Int coarseIndexCounter = 0;
         for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
         {
            if ( CF_marker_array[i] == 1 )
            {
               hypre_ParCompGridOwnedCoarseIndices(compGrid)[i] = coarseIndexCounter++;
            }
            else 
            {
               hypre_ParCompGridOwnedCoarseIndices(compGrid)[i] = -1;
            }
         }
      }
      else 
      {
         for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
         {
            hypre_ParCompGridOwnedCoarseIndices(compGrid)[i] = -1;
         }
      }
   }

   return 0;
}

HYPRE_Int
hypre_ParCompGridInitialize ( hypre_ParAMGData *amg_data, HYPRE_Int padding, HYPRE_Int level, HYPRE_Int symmetric )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int         i,j;

   // Get info from the amg data structure
   hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];
   hypre_ParVector *residual = hypre_ParAMGDataFArray(amg_data)[level];
   HYPRE_Int *CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data)[level];
   hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
   hypre_ParCSRMatrix *P = NULL;
   hypre_ParCSRMatrix *R = NULL;
   HYPRE_Int coarseStart = 0;
   if (level != hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      P = hypre_ParAMGDataPArray(amg_data)[level];
      coarseStart = hypre_ParVectorFirstIndex(hypre_ParAMGDataFArray(amg_data)[level+1]);
   }
   if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
   {
      R = hypre_ParAMGDataRArray(amg_data)[level-1]; // NOTE: want to associate rows of R with comp grid points, so need to take R from one level finer
   }

   hypre_Vector *residual_local = hypre_ParVectorLocalVector(residual);
   HYPRE_Int         num_nodes = hypre_VectorSize(residual_local);

   HYPRE_Int         mem_size = num_nodes + 2 * (padding + hypre_ParAMGDataAMGDDNumGhostLayers(amg_data)) * hypre_CSRMatrixNumCols( hypre_ParCSRMatrixOffd(A) );
   HYPRE_Real        over_allocation_factor = (HYPRE_Real) mem_size;
   if (num_nodes > 0) over_allocation_factor = ((HYPRE_Real) mem_size) / ((HYPRE_Real) num_nodes);

   hypre_ParCompGridNumNodes(compGrid) = num_nodes;
   hypre_ParCompGridNumOwnedBlocks(compGrid) = 1;
   hypre_ParCompGridOwnedBlockStarts(compGrid) = hypre_CTAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
   hypre_ParCompGridOwnedBlockStarts(compGrid)[0] = 0;
   hypre_ParCompGridOwnedBlockStarts(compGrid)[1] = num_nodes;
   hypre_ParCompGridNumRealNodes(compGrid) = num_nodes;
   hypre_ParCompGridMemSize(compGrid) = mem_size;

   HYPRE_Int A_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(A) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(A) );
   hypre_ParCompGridAMemSize(compGrid) = ceil(over_allocation_factor*A_nnz);

   // Allocate space for the info on the comp nodes
   HYPRE_Int        *global_indices_comp = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
   HYPRE_Int        *real_dof_marker = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
   HYPRE_Int        *edge_indices = NULL;
   if (!symmetric) edge_indices = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
   HYPRE_Int        *sort_map = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
   HYPRE_Int        *inv_sort_map = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
   HYPRE_Int        *coarse_global_indices_comp = NULL; 
   HYPRE_Int        *coarse_local_indices_comp = NULL;
   
   // Initialize edge indices using offd from A
   HYPRE_Int num_edge_indices = 0;
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   if (!symmetric)
   {
      for (i = 0; i < num_nodes; i++)
         if (hypre_CSRMatrixI(offd)[i+1] > hypre_CSRMatrixI(offd)[i]) 
            edge_indices[num_edge_indices++] = i;
   }

   if (level != hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      coarse_global_indices_comp = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST); 
      coarse_local_indices_comp = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
   }
   HYPRE_Int        *A_rowptr = hypre_CTAlloc(HYPRE_Int, mem_size+1, HYPRE_MEMORY_HOST);
   HYPRE_Int        *A_colind = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridAMemSize(compGrid), HYPRE_MEMORY_HOST);
   HYPRE_Int        *A_global_colind = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridAMemSize(compGrid), HYPRE_MEMORY_HOST);
   HYPRE_Complex    *A_data = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridAMemSize(compGrid), HYPRE_MEMORY_HOST);
   HYPRE_Int        *P_rowptr = NULL;
   HYPRE_Int        *P_colind = NULL;
   HYPRE_Complex    *P_data = NULL;
   HYPRE_Int        *R_rowptr = NULL;
   HYPRE_Int        *R_colind = NULL;
   HYPRE_Complex    *R_data = NULL;
   if (P)
   {
      HYPRE_Int P_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(P) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(P) );
      hypre_ParCompGridPMemSize(compGrid) = ceil(over_allocation_factor*P_nnz);
      P_rowptr = hypre_CTAlloc(HYPRE_Int, mem_size+1, HYPRE_MEMORY_HOST);
      P_colind = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridPMemSize(compGrid), HYPRE_MEMORY_HOST);
      P_data = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridPMemSize(compGrid), HYPRE_MEMORY_HOST);
   }
   if (R)
   {
      HYPRE_Int R_nnz = hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(R) ) + hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(R) );
      hypre_ParCompGridRMemSize(compGrid) = ceil(over_allocation_factor*R_nnz);
      R_rowptr = hypre_CTAlloc(HYPRE_Int, mem_size+1, HYPRE_MEMORY_HOST);
      R_colind = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridRMemSize(compGrid), HYPRE_MEMORY_HOST);
      R_data = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridRMemSize(compGrid), HYPRE_MEMORY_HOST);
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
      real_dof_marker[i] = 1;
      sort_map[i] = i;
      inv_sort_map[i] = i;
      if (level != hypre_ParAMGDataNumLevels(amg_data) - 1) 
      {
         if ( CF_marker_array )
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
         // Setup row of matrix P
         hypre_ParCSRMatrixGetRow( P, global_indices_comp[i], &row_size, &row_col_ind, &row_values );
         P_rowptr[i+1] = P_rowptr[i] + row_size;
         for (j = P_rowptr[i]; j < P_rowptr[i+1]; j++)
         {
            P_data[j] = row_values[j - P_rowptr[i]];
            P_colind[j] = row_col_ind[j - P_rowptr[i]];
         }
         hypre_ParCSRMatrixRestoreRow( P, i, &row_size, &row_col_ind, &row_values );
      }

      if (R)
      {         
         // Setup row of matrix R
         hypre_ParCSRMatrixGetRow( R, global_indices_comp[i], &row_size, &row_col_ind, &row_values );
         R_rowptr[i+1] = R_rowptr[i] + row_size;
         for (j = R_rowptr[i]; j < R_rowptr[i+1]; j++)
         {
            R_data[j] = row_values[j - R_rowptr[i]];
            R_colind[j] = row_col_ind[j - R_rowptr[i]];
         }
         hypre_ParCSRMatrixRestoreRow( R, i, &row_size, &row_col_ind, &row_values );
      }
   }

   // Set attributes for compGrid
   hypre_ParCompGridGlobalIndices(compGrid) = global_indices_comp;
   hypre_ParCompGridRealDofMarker(compGrid) = real_dof_marker;
   hypre_ParCompGridNumEdgeIndices(compGrid) = num_edge_indices;
   hypre_ParCompGridEdgeIndices(compGrid) = edge_indices;
   hypre_ParCompGridSortMap(compGrid) = sort_map;
   hypre_ParCompGridInvSortMap(compGrid) = inv_sort_map;
   hypre_ParCompGridCoarseGlobalIndices(compGrid) = coarse_global_indices_comp;
   hypre_ParCompGridCoarseLocalIndices(compGrid) = coarse_local_indices_comp;
   hypre_ParCompGridARowPtr(compGrid) = A_rowptr;
   hypre_ParCompGridAColInd(compGrid) = A_colind;
   hypre_ParCompGridAGlobalColInd(compGrid) = A_global_colind;
   hypre_ParCompGridAData(compGrid) = A_data;
   hypre_ParCompGridPRowPtr(compGrid) = P_rowptr;
   hypre_ParCompGridPColInd(compGrid) = P_colind;
   hypre_ParCompGridPData(compGrid) = P_data;
   hypre_ParCompGridRRowPtr(compGrid) = R_rowptr;
   hypre_ParCompGridRColInd(compGrid) = R_colind;
   hypre_ParCompGridRData(compGrid) = R_data;

   return 0;
}

HYPRE_Int 
hypre_ParCompGridSetupRelax( hypre_ParAMGData *amg_data )
{
   HYPRE_Int level, i, j;

   if (hypre_ParAMGDataFACRelaxType(amg_data) == 0) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_Jacobi;
   else if (hypre_ParAMGDataFACRelaxType(amg_data) == 1) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_GaussSeidel;
   else if (hypre_ParAMGDataFACRelaxType(amg_data) == 2) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_Cheby;
   else if (hypre_ParAMGDataFACRelaxType(amg_data) == 3) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_CFL1Jacobi; 
   else if (hypre_ParAMGDataFACRelaxType(amg_data) == 4) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_OrderedGaussSeidel; 

   for (level = hypre_ParAMGDataAMGDDStartLevel(amg_data); level < hypre_ParAMGDataNumLevels(amg_data); level++)
   {
      hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];

      if (hypre_ParAMGDataFACRelaxType(amg_data) == 2)
      {
         // Setup chebyshev coefficients
         hypre_CSRMatrix *A = hypre_ParCompGridA(compGrid);
         HYPRE_Real    *coefs = hypre_ParCompGridChebyCoeffs(compGrid);
         HYPRE_Int     scale = hypre_ParAMGDataChebyScale(amg_data);
         HYPRE_Int     order = hypre_ParAMGDataChebyOrder(amg_data);

         // Select submatrix of real to real connections
         HYPRE_Int nnz = 0;
         for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++)
         {
            for (j = hypre_CSRMatrixI(A)[i]; j < hypre_CSRMatrixI(A)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A)[j] < hypre_ParCompGridNumRealNodes(compGrid)) nnz++;
            }
         }
         HYPRE_Int *A_real_i = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumRealNodes(compGrid)+1, HYPRE_MEMORY_SHARED);
         HYPRE_Int *A_real_j = hypre_CTAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_SHARED);
         HYPRE_Complex *A_real_data = hypre_CTAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_SHARED);
         nnz = 0;
         for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++)
         {
            for (j = hypre_CSRMatrixI(A)[i]; j < hypre_CSRMatrixI(A)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A)[j] < hypre_ParCompGridNumRealNodes(compGrid))
               {
                  A_real_j[nnz] = hypre_CSRMatrixJ(A)[j];
                  A_real_data[nnz] = hypre_CSRMatrixData(A)[j];
                  nnz++;
               }
            }
            A_real_i[i+1] = nnz;
         }

         HYPRE_BigInt *row_starts = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
         row_starts[0] = 0;
         row_starts[1] = hypre_ParCompGridNumRealNodes(compGrid);
         hypre_ParCSRMatrix *A_real = hypre_ParCSRMatrixCreate( MPI_COMM_SELF,
                             (HYPRE_BigInt) hypre_ParCompGridNumRealNodes(compGrid),
                             (HYPRE_BigInt) hypre_ParCompGridNumRealNodes(compGrid),
                             row_starts,
                             NULL,
                             0,
                             nnz,
                             0 );
         hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_real)) = A_real_i;
         hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A_real)) = A_real_j;
         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_real)) = A_real_data;
         hypre_CSRMatrixInitialize(hypre_ParCSRMatrixOffd(A_real));
         hypre_ParCSRMatrixColMapOffd(A_real) = hypre_CTAlloc(HYPRE_BigInt, 0, HYPRE_MEMORY_HOST);

         HYPRE_Real max_eig, min_eig = 0;

         if (hypre_ParAMGDataChebyEigEst(amg_data)) hypre_ParCSRMaxEigEstimateCG(A_real, scale, hypre_ParAMGDataChebyEigEst(amg_data), &max_eig, &min_eig);
         else hypre_ParCSRMaxEigEstimate(A_real, scale, &max_eig);

         HYPRE_Real *dummy_ptr;
         hypre_ParCSRRelax_Cheby_Setup(hypre_ParAMGDataAArray(amg_data)[level], 
                               max_eig,      
                               min_eig,     
                               hypre_ParAMGDataChebyFraction(amg_data),   
                               order,
                               0,
                               hypre_ParAMGDataChebyVariant(amg_data),           
                               &coefs,
                               &dummy_ptr);

         hypre_ParCompGridChebyCoeffs(compGrid) = coefs;

         hypre_ParCSRMatrixDestroy(A_real);

         // Calculate diagonal scaling values 
         hypre_ParCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_SHARED);
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
         {
            for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
            {
               if (hypre_ParCompGridAColInd(compGrid)[j] == i)
               {
                  hypre_ParCompGridL1Norms(compGrid)[i] = 1.0/sqrt(hypre_ParCompGridAData(compGrid)[j]);
                  break;
               }
            }
         }

         // Setup temporary/auxiliary vectors
         hypre_ParCompGridTemp(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumRealNodes(compGrid));
         hypre_SeqVectorInitialize(hypre_ParCompGridTemp(compGrid));

         hypre_ParCompGridTemp2(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
         hypre_SeqVectorInitialize(hypre_ParCompGridTemp2(compGrid));

         hypre_ParCompGridTemp3(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumRealNodes(compGrid));
         hypre_SeqVectorInitialize(hypre_ParCompGridTemp3(compGrid));
      }
      if (hypre_ParAMGDataFACRelaxType(amg_data) == 3)
      {
         // Calculate l1_norms
         hypre_ParCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_SHARED);
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
         {
            HYPRE_Int cf_diag = hypre_ParCompGridCFMarkerArray(compGrid)[i];
            for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
            {
               if (hypre_ParCompGridCFMarkerArray(compGrid)[ hypre_ParCompGridAColInd(compGrid)[j] ] == cf_diag) 
               {
                  hypre_ParCompGridL1Norms(compGrid)[i] += fabs(hypre_ParCompGridAData(compGrid)[j]);
               }
            }
         }
         // Setup temporary/auxiliary vectors
         hypre_ParCompGridTemp(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
         hypre_SeqVectorInitialize(hypre_ParCompGridTemp(compGrid));

         #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
         // Setup c and f point masks
         int num_c_points = 0;
         int num_f_points = 0;
         for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++) if (hypre_ParCompGridCFMarkerArray(compGrid)[i]) num_c_points++;
         num_f_points = hypre_ParCompGridNumRealNodes(compGrid) - num_c_points;
         hypre_ParCompGridNumCPoints(compGrid) = num_c_points;
         hypre_ParCompGridCMask(compGrid) = hypre_CTAlloc(int, num_c_points, HYPRE_MEMORY_SHARED);
         hypre_ParCompGridFMask(compGrid) = hypre_CTAlloc(int, num_f_points, HYPRE_MEMORY_SHARED);
         int c_cnt = 0, f_cnt = 0;
         for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++)
         {
            if (hypre_ParCompGridCFMarkerArray(compGrid)[i]) hypre_ParCompGridCMask(compGrid)[c_cnt++] = i;
            else hypre_ParCompGridFMask(compGrid)[f_cnt++] = i;
         }
         #endif
      }
   }


   return 0;
}

HYPRE_Int
hypre_ParCompGridFinalize( hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int start_level, HYPRE_Int num_levels, HYPRE_Int use_rd, HYPRE_Int debug )
{
   HYPRE_Int level, i, j;

   // Post process to remove -1 entries from matrices and reorder !!! Is there a more efficient way here? 
   for (level = start_level; level < num_levels; level++)
   {
      HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
      HYPRE_Int num_real_nodes = 0;
      for (i = 0; i < num_nodes; i++)
      {
         if (hypre_ParCompGridRealDofMarker(compGrid[level])[i]) num_real_nodes++;
      }
      hypre_ParCompGridNumRealNodes(compGrid[level]) = num_real_nodes;
      HYPRE_Int *new_indices = hypre_CTAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
      HYPRE_Int real_cnt = 0;
      HYPRE_Int ghost_cnt = 0;
      for (i = 0; i < num_nodes; i++)
      {
         if (hypre_ParCompGridRealDofMarker(compGrid[level])[i])
         {
            new_indices[i] = real_cnt++;
         }
         else new_indices[i] = num_real_nodes + ghost_cnt++;
      }

      // Transform indices in send_flag and recv_map
      if (compGridCommPkg)
      {
         HYPRE_Int outer_level;
         for (outer_level = start_level; outer_level < num_levels; outer_level++)
         {
            HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level];
            HYPRE_Int proc;
            for (proc = 0; proc < num_send_procs; proc++)
            {
               HYPRE_Int num_send_nodes = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level];
               for (i = 0; i < num_send_nodes; i++)
               {
                  if (hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i] >= 0)
                  {
                     hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i] = new_indices[hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i]];
                  }
               }
            }
            HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level];
            for (proc = 0; proc < num_recv_procs; proc++)
            {
               HYPRE_Int num_recv_nodes = hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level];
               for (i = 0; i < num_recv_nodes; i++)
               {
                  if (hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i] >= 0)
                  {
                     hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i] = new_indices[hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i]];
                  }
               }
            }
         }
      }

      // If global indices are still needed, transform these also
      if (debug)
      {
         HYPRE_Int *new_global_indices = hypre_CTAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_nodes; i++)
         {
            new_global_indices[ new_indices[i] ] = hypre_ParCompGridGlobalIndices(compGrid[level])[ i ];
         }
         hypre_TFree(hypre_ParCompGridGlobalIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridGlobalIndices(compGrid[level]) = new_global_indices;
      }

      // Setup cf marker array in correct ordering
      hypre_ParCompGridCFMarkerArray(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumNodes(compGrid[level]), HYPRE_MEMORY_SHARED);
      if (hypre_ParCompGridCoarseLocalIndices(compGrid[level]))
      {
         for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            if (hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] >= 0) hypre_ParCompGridCFMarkerArray(compGrid[level])[ new_indices[i] ] = 1;
         }
         hypre_TFree(hypre_ParCompGridCoarseLocalIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridCoarseLocalIndices(compGrid[level]) = NULL;
      }

      HYPRE_Int A_nnz = hypre_ParCompGridARowPtr(compGrid[level])[num_nodes];
      HYPRE_Int *new_A_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_SHARED);
      HYPRE_Int *new_A_colInd = hypre_CTAlloc(HYPRE_Int, A_nnz, HYPRE_MEMORY_SHARED);
      HYPRE_Complex *new_A_data = hypre_CTAlloc(HYPRE_Complex, A_nnz, HYPRE_MEMORY_SHARED);

      HYPRE_Int P_nnz;
      HYPRE_Int *new_P_rowPtr;
      HYPRE_Int *new_P_colInd;
      HYPRE_Complex *new_P_data;

      HYPRE_Int R_nnz;
      HYPRE_Int *new_R_rowPtr;
      HYPRE_Int *new_R_colInd;
      HYPRE_Complex *new_R_data;

      if (level != num_levels-1)
      {
         P_nnz = hypre_ParCompGridPRowPtr(compGrid[level])[num_nodes];
         new_P_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_SHARED);
         new_P_colInd = hypre_CTAlloc(HYPRE_Int, P_nnz, HYPRE_MEMORY_SHARED);
         new_P_data = hypre_CTAlloc(HYPRE_Complex, P_nnz, HYPRE_MEMORY_SHARED);
      }
      if (hypre_ParCompGridRRowPtr(compGrid[level]))
      {
         R_nnz = hypre_ParCompGridRRowPtr(compGrid[level])[num_nodes];
         new_R_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_SHARED);
         new_R_colInd = hypre_CTAlloc(HYPRE_Int, R_nnz, HYPRE_MEMORY_SHARED);
         new_R_data = hypre_CTAlloc(HYPRE_Complex, R_nnz, HYPRE_MEMORY_SHARED);
      }

      HYPRE_Int A_cnt = 0;
      HYPRE_Int P_cnt = 0;
      HYPRE_Int R_cnt = 0;
      HYPRE_Int node_cnt = 0;
      // Real nodes
      for (i = 0; i < num_nodes; i++)
      {
         if (hypre_ParCompGridRealDofMarker(compGrid[level])[i])
         {
            new_A_rowPtr[node_cnt] = A_cnt;
            for (j = hypre_ParCompGridARowPtr(compGrid[level])[i]; j < hypre_ParCompGridARowPtr(compGrid[level])[i+1]; j++)
            {
               if (hypre_ParCompGridAColInd(compGrid[level])[j] >= 0)
               {
                  new_A_colInd[A_cnt] = new_indices[ hypre_ParCompGridAColInd(compGrid[level])[j] ];
                  new_A_data[A_cnt] = hypre_ParCompGridAData(compGrid[level])[j];
                  A_cnt++;
               }
            }

            if (hypre_ParCompGridPRowPtr(compGrid[level]))
            {
               new_P_rowPtr[node_cnt] = P_cnt;
               for (j = hypre_ParCompGridPRowPtr(compGrid[level])[i]; j < hypre_ParCompGridPRowPtr(compGrid[level])[i+1]; j++)
               {
                  if (hypre_ParCompGridPColInd(compGrid[level])[j] >= 0)
                  {
                     new_P_colInd[P_cnt] = hypre_ParCompGridPColInd(compGrid[level])[j];
                     new_P_data[P_cnt] = hypre_ParCompGridPData(compGrid[level])[j];
                     P_cnt++;
                  }
               }
            }
            if (hypre_ParCompGridRRowPtr(compGrid[level]))
            {
               new_R_rowPtr[node_cnt] = R_cnt;
               for (j = hypre_ParCompGridRRowPtr(compGrid[level])[i]; j < hypre_ParCompGridRRowPtr(compGrid[level])[i+1]; j++)
               {
                  if (hypre_ParCompGridRColInd(compGrid[level])[j] >= 0)
                  {
                     new_R_colInd[R_cnt] = hypre_ParCompGridRColInd(compGrid[level])[j];
                     new_R_data[R_cnt] = hypre_ParCompGridRData(compGrid[level])[j];
                     R_cnt++;
                  }
               }
            }
            node_cnt++;
         }
      }
      // Ghost nodes
      for (i = 0; i < num_nodes; i++)
      {
         if (!hypre_ParCompGridRealDofMarker(compGrid[level])[i])
         {
            new_A_rowPtr[node_cnt] = A_cnt;
            for (j = hypre_ParCompGridARowPtr(compGrid[level])[i]; j < hypre_ParCompGridARowPtr(compGrid[level])[i+1]; j++)
            {
               if (hypre_ParCompGridAColInd(compGrid[level])[j] >= 0)
               {
                  new_A_colInd[A_cnt] = new_indices[hypre_ParCompGridAColInd(compGrid[level])[j]];
                  new_A_data[A_cnt] = hypre_ParCompGridAData(compGrid[level])[j];
                  A_cnt++;
               }
            }

            if (hypre_ParCompGridPRowPtr(compGrid[level]))
            {
               new_P_rowPtr[node_cnt] = P_cnt;
               for (j = hypre_ParCompGridPRowPtr(compGrid[level])[i]; j < hypre_ParCompGridPRowPtr(compGrid[level])[i+1]; j++)
               {
                  if (hypre_ParCompGridPColInd(compGrid[level])[j] >= 0)
                  {
                     new_P_colInd[P_cnt] = hypre_ParCompGridPColInd(compGrid[level])[j];
                     new_P_data[P_cnt] = hypre_ParCompGridPData(compGrid[level])[j];
                     P_cnt++;
                  }
               }
            }
            if (hypre_ParCompGridRRowPtr(compGrid[level]))
            {
               new_R_rowPtr[node_cnt] = R_cnt;
               for (j = hypre_ParCompGridRRowPtr(compGrid[level])[i]; j < hypre_ParCompGridRRowPtr(compGrid[level])[i+1]; j++)
               {
                  if (hypre_ParCompGridRColInd(compGrid[level])[j] >= 0)
                  {
                     new_R_colInd[R_cnt] = hypre_ParCompGridRColInd(compGrid[level])[j];
                     new_R_data[R_cnt] = hypre_ParCompGridRData(compGrid[level])[j];
                     R_cnt++;
                  }
               }
            }
            node_cnt++;
         }
      }
      new_A_rowPtr[num_nodes] = A_cnt;
      if (hypre_ParCompGridPRowPtr(compGrid[level])) new_P_rowPtr[num_nodes] = P_cnt;
      if (hypre_ParCompGridRRowPtr(compGrid[level])) new_R_rowPtr[num_nodes] = R_cnt;

      // Fix up P col indices on finer level
      if (level != start_level)
      {
         if (hypre_ParCompGridPRowPtr(compGrid[level-1]))
         {
            for (i = 0; i < hypre_ParCompGridPRowPtr(compGrid[level-1])[ hypre_ParCompGridNumNodes(compGrid[level-1]) ]; i++)
            {
               hypre_ParCompGridPColInd(compGrid[level-1])[i] = new_indices[ hypre_ParCompGridPColInd(compGrid[level-1])[i] ];
            }
         }
      }
      // Fix up R col indices on coarser level
      if (level != num_levels-1)
      {
         if (hypre_ParCompGridRRowPtr(compGrid[level+1]))
         {
            for (i = 0; i < hypre_ParCompGridRRowPtr(compGrid[level+1])[ hypre_ParCompGridNumNodes(compGrid[level+1]) ]; i++)
            {
               if (hypre_ParCompGridRColInd(compGrid[level+1])[i] >= 0)
                  hypre_ParCompGridRColInd(compGrid[level+1])[i] = new_indices[ hypre_ParCompGridRColInd(compGrid[level+1])[i] ];
            }
         }
      }

      // Clean up memory, deallocate old arrays and reset pointers to new arrays
      hypre_TFree(hypre_ParCompGridARowPtr(compGrid[level]), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParCompGridAColInd(compGrid[level]), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParCompGridAData(compGrid[level]), HYPRE_MEMORY_HOST);
      hypre_ParCompGridARowPtr(compGrid[level]) = new_A_rowPtr;
      hypre_ParCompGridAColInd(compGrid[level]) = new_A_colInd;
      hypre_ParCompGridAData(compGrid[level]) = new_A_data;

      if (hypre_ParCompGridPRowPtr(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridPRowPtr(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_ParCompGridPColInd(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_ParCompGridPData(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridPRowPtr(compGrid[level]) = new_P_rowPtr;
         hypre_ParCompGridPColInd(compGrid[level]) = new_P_colInd;
         hypre_ParCompGridPData(compGrid[level]) = new_P_data;
      }
      if (hypre_ParCompGridRRowPtr(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridRRowPtr(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_ParCompGridRColInd(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_ParCompGridRData(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridRRowPtr(compGrid[level]) = new_R_rowPtr;
         hypre_ParCompGridRColInd(compGrid[level]) = new_R_colInd;
         hypre_ParCompGridRData(compGrid[level]) = new_R_data;            
      }

      hypre_TFree(new_indices, HYPRE_MEMORY_HOST);
   }

   // Setup vectors and matrices
   HYPRE_Int total_num_nodes = 0;
   for (level = start_level; level < num_levels; level++)
   {
      HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
      HYPRE_Int num_real_nodes = hypre_ParCompGridNumRealNodes(compGrid[level]);
      total_num_nodes += num_nodes;
      HYPRE_Int A_nnz = hypre_ParCompGridARowPtr(compGrid[level])[num_nodes];
      HYPRE_Int A_real_nnz = hypre_ParCompGridARowPtr(compGrid[level])[num_real_nodes];
      
      hypre_ParCompGridA(compGrid[level]) = hypre_CSRMatrixCreate(num_nodes, num_nodes, A_nnz);
      hypre_CSRMatrixI(hypre_ParCompGridA(compGrid[level])) = hypre_ParCompGridARowPtr(compGrid[level]);
      hypre_CSRMatrixJ(hypre_ParCompGridA(compGrid[level])) = hypre_ParCompGridAColInd(compGrid[level]);
      hypre_CSRMatrixData(hypre_ParCompGridA(compGrid[level])) = hypre_ParCompGridAData(compGrid[level]);

      hypre_ParCompGridAReal(compGrid[level]) = hypre_CSRMatrixCreate(num_real_nodes, num_nodes, A_real_nnz);
      hypre_CSRMatrixI(hypre_ParCompGridAReal(compGrid[level])) = hypre_ParCompGridARowPtr(compGrid[level]);
      hypre_CSRMatrixJ(hypre_ParCompGridAReal(compGrid[level])) = hypre_ParCompGridAColInd(compGrid[level]);
      hypre_CSRMatrixData(hypre_ParCompGridAReal(compGrid[level])) = hypre_ParCompGridAData(compGrid[level]);

      if (level != num_levels-1)
      {
         HYPRE_Int P_nnz = hypre_ParCompGridPRowPtr(compGrid[level])[num_nodes];
         HYPRE_Int num_nodes_c = hypre_ParCompGridNumNodes(compGrid[level+1]);
         hypre_ParCompGridP(compGrid[level]) = hypre_CSRMatrixCreate(num_nodes, num_nodes_c, P_nnz);
         hypre_CSRMatrixI(hypre_ParCompGridP(compGrid[level])) = hypre_ParCompGridPRowPtr(compGrid[level]);
         hypre_CSRMatrixJ(hypre_ParCompGridP(compGrid[level])) = hypre_ParCompGridPColInd(compGrid[level]);
         hypre_CSRMatrixData(hypre_ParCompGridP(compGrid[level])) = hypre_ParCompGridPData(compGrid[level]);

         if (hypre_ParCompGridRRowPtr(compGrid[level+1]))
         {
            // NOTE: shifting R back one level up to agree with AMGData organization (that is, R[level] restricts from level to level+1)
            HYPRE_Int R_nnz = hypre_ParCompGridRRowPtr(compGrid[level+1])[num_nodes_c];
            hypre_ParCompGridR(compGrid[level]) = hypre_CSRMatrixCreate(num_nodes_c, num_nodes, R_nnz);
            hypre_CSRMatrixI(hypre_ParCompGridR(compGrid[level])) = hypre_ParCompGridRRowPtr(compGrid[level+1]);
            hypre_CSRMatrixJ(hypre_ParCompGridR(compGrid[level])) = hypre_ParCompGridRColInd(compGrid[level+1]);
            hypre_CSRMatrixData(hypre_ParCompGridR(compGrid[level])) = hypre_ParCompGridRData(compGrid[level+1]);
            // Move the comp grid raw data also
            hypre_ParCompGridRMemSize(compGrid[level]) = hypre_ParCompGridRMemSize(compGrid[level+1]);
            hypre_ParCompGridRRowPtr(compGrid[level]) = hypre_ParCompGridRRowPtr(compGrid[level+1]);
            hypre_ParCompGridRColInd(compGrid[level]) = hypre_ParCompGridRColInd(compGrid[level+1]);
            hypre_ParCompGridRData(compGrid[level]) = hypre_ParCompGridRData(compGrid[level+1]);
         }
         else hypre_CSRMatrixTranspose(hypre_ParCompGridP(compGrid[level]), &hypre_ParCompGridR(compGrid[level]), 1);
      }
      if (level == num_levels-1 && hypre_ParCompGridRRowPtr(compGrid[level]))
      {
         hypre_ParCompGridRMemSize(compGrid[level]) = 0;
         hypre_ParCompGridRRowPtr(compGrid[level]) = NULL;
         hypre_ParCompGridRColInd(compGrid[level]) = NULL;
         hypre_ParCompGridRData(compGrid[level]) = NULL;
      }

      hypre_ParCompGridU(compGrid[level]) = hypre_SeqVectorCreate(num_nodes);
      hypre_SeqVectorInitialize(hypre_ParCompGridU(compGrid[level]));

      if (level < num_levels)
      {
         hypre_ParCompGridS(compGrid[level]) = hypre_SeqVectorCreate(num_nodes);
         hypre_SeqVectorInitialize(hypre_ParCompGridS(compGrid[level]));

         hypre_ParCompGridT(compGrid[level]) = hypre_SeqVectorCreate(num_nodes);
         hypre_SeqVectorInitialize(hypre_ParCompGridT(compGrid[level]));
      }
   }

   // Allocate space for the rhs vectors, compGridF, as one big block of memory for better access when packing/unpack communication buffers
   HYPRE_Complex *f_data = hypre_CTAlloc(HYPRE_Complex, total_num_nodes, HYPRE_MEMORY_SHARED);

   total_num_nodes = 0;

   for (level = start_level; level < num_levels; level++)
   {
      HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);

      hypre_ParCompGridF(compGrid[level]) = hypre_SeqVectorCreate(num_nodes);
      if (level != 0) hypre_SeqVectorSetDataOwner(hypre_ParCompGridF(compGrid[level]), 0);
      hypre_VectorData(hypre_ParCompGridF(compGrid[level])) = &(f_data[total_num_nodes]);

      total_num_nodes += num_nodes;
   }

   if (use_rd)
   {
      // Allocate space for the update vectors, compGridQ, as one big block of memory for better access when packing/unpack communication buffers
      HYPRE_Complex *q_data = hypre_CTAlloc(HYPRE_Complex, total_num_nodes, HYPRE_MEMORY_SHARED);

      total_num_nodes = 0;

      for (level = start_level; level < num_levels; level++)
      {
         HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
         HYPRE_Int num_real_nodes = hypre_ParCompGridNumRealNodes(compGrid[level]);

         hypre_ParCompGridQ(compGrid[level]) = hypre_SeqVectorCreate(num_real_nodes);
         if (level != 0) hypre_SeqVectorSetDataOwner(hypre_ParCompGridQ(compGrid[level]), 0);
         hypre_VectorData(hypre_ParCompGridQ(compGrid[level])) = &(q_data[total_num_nodes]);

         total_num_nodes += num_nodes;
      }
   }

   // Clean up memory for things we don't need anymore
   for (level = start_level; level < num_levels; level++)
   {
      if (hypre_ParCompGridRealDofMarker(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridRealDofMarker(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridRealDofMarker(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridGlobalIndices(compGrid[level]) && !debug)
      {
         hypre_TFree(hypre_ParCompGridGlobalIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridGlobalIndices(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridAGlobalColInd(compGrid[level]) && !debug)
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

   return 0;
}

HYPRE_Int
hypre_ParCompGridSetSize ( hypre_ParCompGrid *compGrid, HYPRE_Int num_nodes, HYPRE_Int mem_size, HYPRE_Int A_nnz, HYPRE_Int P_nnz, HYPRE_Int full_comp_info )
{
   hypre_ParCompGridNumNodes(compGrid) = num_nodes;
   hypre_ParCompGridMemSize(compGrid) = mem_size;
   HYPRE_Real over_allocation_factor = mem_size;
   if (num_nodes > 0) over_allocation_factor = mem_size/num_nodes;
   hypre_ParCompGridAMemSize(compGrid) = ceil(A_nnz*over_allocation_factor);
   hypre_ParCompGridPMemSize(compGrid) = ceil(P_nnz*over_allocation_factor);
   
   hypre_ParCompGridARowPtr(compGrid) = hypre_CTAlloc(HYPRE_Int, mem_size+1, HYPRE_MEMORY_HOST);
   hypre_ParCompGridAColInd(compGrid) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridAMemSize(compGrid), HYPRE_MEMORY_HOST);
   if (full_comp_info) hypre_ParCompGridAGlobalColInd(compGrid) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridAMemSize(compGrid), HYPRE_MEMORY_HOST);
   hypre_ParCompGridAData(compGrid) = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridAMemSize(compGrid), HYPRE_MEMORY_HOST);

   if (full_comp_info)
   {
      hypre_ParCompGridGlobalIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
   }
   if (full_comp_info > 1)
   {
      hypre_ParCompGridCoarseGlobalIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCoarseLocalIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_HOST);
   }

   hypre_ParCompGridPRowPtr(compGrid) = hypre_CTAlloc(HYPRE_Int, mem_size+1, HYPRE_MEMORY_HOST);
   if (P_nnz)
   {
      hypre_ParCompGridPColInd(compGrid) = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridPMemSize(compGrid), HYPRE_MEMORY_HOST);
      hypre_ParCompGridPData(compGrid) = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridPMemSize(compGrid), HYPRE_MEMORY_HOST);      
   }

   return 0;
}

HYPRE_Int
hypre_ParCompGridResize ( hypre_ParCompGrid *compGrid, HYPRE_Int new_size, HYPRE_Int need_coarse_info, HYPRE_Int type, HYPRE_Int symmetric )
{
   // This function reallocates memory to hold a comp grid of size new_size
   // num_nodes and mem_size are set to new_size. Use this when exact size of new comp grid is known.

   // Reallocate num nodes
   if (type == 0)
   {
      // Re allocate to given size
      hypre_ParCompGridGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridGlobalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridRealDofMarker(compGrid) = hypre_TReAlloc(hypre_ParCompGridRealDofMarker(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      if (!symmetric) hypre_ParCompGridEdgeIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridEdgeIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridSortMap(compGrid) = hypre_TReAlloc(hypre_ParCompGridSortMap(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridInvSortMap(compGrid) = hypre_TReAlloc(hypre_ParCompGridInvSortMap(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridARowPtr(compGrid) = hypre_TReAlloc(hypre_ParCompGridARowPtr(compGrid), HYPRE_Int, new_size+1, HYPRE_MEMORY_HOST);
      if (need_coarse_info)
      {
         hypre_ParCompGridCoarseGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridCoarseGlobalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
         hypre_ParCompGridCoarseLocalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridCoarseLocalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
         hypre_ParCompGridPRowPtr(compGrid) = hypre_TReAlloc(hypre_ParCompGridPRowPtr(compGrid), HYPRE_Int, new_size+1, HYPRE_MEMORY_HOST);
      }
      if (hypre_ParCompGridRRowPtr(compGrid)) hypre_ParCompGridRRowPtr(compGrid) = hypre_TReAlloc(hypre_ParCompGridRRowPtr(compGrid), HYPRE_Int, new_size+1, HYPRE_MEMORY_HOST);
      hypre_ParCompGridMemSize(compGrid) = new_size;  
   }
   // Reallocate A matrix
   else if (type == 1)
   {
      hypre_ParCompGridAColInd(compGrid) = hypre_TReAlloc(hypre_ParCompGridAColInd(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridAGlobalColInd(compGrid) = hypre_TReAlloc(hypre_ParCompGridAGlobalColInd(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridAData(compGrid) = hypre_TReAlloc(hypre_ParCompGridAData(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridAMemSize(compGrid) = new_size;
   }   // Reallocate P matrix
   else if (type == 2)
   {
      hypre_ParCompGridPColInd(compGrid) = hypre_TReAlloc(hypre_ParCompGridPColInd(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridPData(compGrid) = hypre_TReAlloc(hypre_ParCompGridPData(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridPMemSize(compGrid) = new_size;
   }   // Reallocate P matrix
   else if (type == 3)
   {
      hypre_ParCompGridRColInd(compGrid) = hypre_TReAlloc(hypre_ParCompGridRColInd(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridRData(compGrid) = hypre_TReAlloc(hypre_ParCompGridRData(compGrid), HYPRE_Complex, new_size, HYPRE_MEMORY_HOST);
      hypre_ParCompGridRMemSize(compGrid) = new_size;
   }

   return 0;
}

HYPRE_Int 
hypre_ParCompGridSetupLocalIndices( hypre_ParCompGrid **compGrid, HYPRE_Int *nodes_added_on_level, HYPRE_Int start_level, HYPRE_Int num_levels, HYPRE_Int symmetric )
{
   // when nodes are added to a composite grid, global info is copied over, but local indices must be generated appropriately for all added nodes
   // this must be done on each level as info is added to correctly construct subsequent Psi_c grids
   // also done after each ghost layer is added
   HYPRE_Int      level,i,j,k;
   HYPRE_Int      global_index, local_index;

   HYPRE_Int bin_search_cnt = 0;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   for (level = start_level; level < num_levels; level++)
   {
      // If we have added nodes on this level
      if (nodes_added_on_level[level])
      {

         HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
         HYPRE_Int old_num_nodes = num_nodes - nodes_added_on_level[level];

         HYPRE_Int new_num_edge_indices = 0;
         if (!symmetric)
         {
            // Loop over previous edge dofs to fill in missing col indices
            for (i = 0; i < hypre_ParCompGridNumEdgeIndices(compGrid[level]); i++)
            {
               HYPRE_Int is_edge = 0;
               HYPRE_Int edge_index = hypre_ParCompGridEdgeIndices(compGrid[level])[i];

               // Loop over col indices of A at the edge indices
               for (j = hypre_ParCompGridARowPtr(compGrid[level])[edge_index]; j < hypre_ParCompGridARowPtr(compGrid[level])[edge_index+1]; j++)
               {
                  local_index = hypre_ParCompGridAColInd(compGrid[level])[j];
                  if (local_index < 0)
                  {
                     global_index = hypre_ParCompGridAGlobalColInd(compGrid[level])[j];
                     bin_search_cnt++;
                     local_index = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level], global_index, 0, num_nodes, hypre_ParCompGridInvSortMap(compGrid[level]));
                     if (local_index == -1) local_index = -global_index-1;
                     if (local_index < 0) is_edge = 1;
                     hypre_ParCompGridAColInd(compGrid[level])[j] = local_index;
                  }
               }
               if (is_edge) hypre_ParCompGridEdgeIndices(compGrid[level])[new_num_edge_indices++] = edge_index;
            }
         }

         // Loop over new nodes and setup
         // NOTE: don't forget to mark edges here as well
         for (i = old_num_nodes; i < num_nodes; i++)
         {
            HYPRE_Int is_edge = 0;
            for (j = hypre_ParCompGridARowPtr(compGrid[level])[i]; j < hypre_ParCompGridARowPtr(compGrid[level])[i+1]; j++)
            {
               global_index = hypre_ParCompGridAGlobalColInd(compGrid[level])[j];
               // local_index = -1;
               local_index = hypre_ParCompGridAColInd(compGrid[level])[j];
               if (local_index < 0)
               {
                  // If global index is owned, simply calculate
                  HYPRE_Int num_owned_blocks = hypre_ParCompGridNumOwnedBlocks(compGrid[level]);
                  for (k = 0; k < num_owned_blocks; k++)
                  {
                     if (hypre_ParCompGridOwnedBlockStarts(compGrid[level])[k+1] - hypre_ParCompGridOwnedBlockStarts(compGrid[level])[k] > 0)
                     {
                        HYPRE_Int low_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ hypre_ParCompGridOwnedBlockStarts(compGrid[level])[k] ];
                        HYPRE_Int high_global_index = hypre_ParCompGridGlobalIndices(compGrid[level])[ hypre_ParCompGridOwnedBlockStarts(compGrid[level])[k+1] - 1 ];
                        if ( global_index >= low_global_index && global_index <= high_global_index )
                        {
                           local_index = global_index - low_global_index + hypre_ParCompGridOwnedBlockStarts(compGrid[level])[k];
                        }
                     }
                  }
                  if (local_index == -1)
                  {
                     bin_search_cnt++;
                     local_index = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level], global_index, 0, num_nodes, hypre_ParCompGridInvSortMap(compGrid[level]));
                  }
                  if (local_index == -1) local_index = -global_index-1;
                  if (local_index < 0) is_edge = 1;
                  hypre_ParCompGridAColInd(compGrid[level])[j] = local_index;

                  // If symmetric, insert missing symmetric connection if necessary
                  if (symmetric && local_index >= 0)
                  {
                     for (k = hypre_ParCompGridARowPtr(compGrid[level])[local_index]; k < hypre_ParCompGridARowPtr(compGrid[level])[local_index+1]; k++)
                     {
                        if (hypre_ParCompGridAGlobalColInd(compGrid[level])[k] == hypre_ParCompGridGlobalIndices(compGrid[level])[i])
                        {
                           hypre_ParCompGridAColInd(compGrid[level])[k] = i;
                        }
                     }
                  }
               }
            }
            if (is_edge && !symmetric) hypre_ParCompGridEdgeIndices(compGrid[level])[new_num_edge_indices++] = i;
         }

         if (!symmetric) hypre_ParCompGridNumEdgeIndices(compGrid[level]) = new_num_edge_indices;
      }
      
      // if we are not on the coarsest level
      if (level != num_levels-1)
      {
         // loop over indices of non-owned nodes on this level 
         // !!! No guarantee that previous ghost dofs converted to real dofs have coarse local indices setup...
         // !!! Thus we go over all non-owned dofs here instead of just the added ones, but we only setup coarse local index where necessary.
         // !!! NOTE: can't use nodes_added_on_level here either because real overwritten by ghost doesn't count as added node (so you can miss setting these up)
         HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[level]);
         HYPRE_Int old_num_nodes = num_nodes - nodes_added_on_level[level];
         HYPRE_Int num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[hypre_ParCompGridNumOwnedBlocks(compGrid[level])];
         for (i = num_owned_nodes; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
         {
            // fix up the coarse local indices
            global_index = hypre_ParCompGridCoarseGlobalIndices(compGrid[level])[i];
            HYPRE_Int is_real = hypre_ParCompGridRealDofMarker(compGrid[level])[i];

            // setup coarse local index if necessary
            if (global_index >= 0 && is_real)
            {
               if (i < old_num_nodes) local_index = hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i];
               else local_index = -1;

               if (local_index < 0)
               {
                  // bin_search_cnt++;
                  hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level+1], global_index, 0, hypre_ParCompGridNumNodes(compGrid[level+1]), hypre_ParCompGridInvSortMap(compGrid[level+1]));
               }
               else if ( hypre_ParCompGridGlobalIndices(compGrid[level+1])[local_index] != global_index )
               {
                  // bin_search_cnt++;
                  hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level+1], global_index, 0, hypre_ParCompGridNumNodes(compGrid[level+1]), hypre_ParCompGridInvSortMap(compGrid[level+1]));
               }
            }
            else hypre_ParCompGridCoarseLocalIndices(compGrid[level])[i] = -1;
         }
      }
   }

   return bin_search_cnt;
}

HYPRE_Int hypre_ParCompGridSetupLocalIndicesP( hypre_ParCompGrid **compGrid, HYPRE_Int start_level, HYPRE_Int num_levels )
{
   HYPRE_Int                  i,j,level,global_index;

   for (level = start_level; level < num_levels; level++)
   {
      if (hypre_ParCompGridPRowPtr(compGrid[level]))
      {
         HYPRE_Int num_owned_blocks = hypre_ParCompGridNumOwnedBlocks(compGrid[level+1]);
         
         // Setup all local indices for all nodes (note that PColInd currently stores global indices)
         for (i = 0; i < hypre_ParCompGridPRowPtr(compGrid[level])[ hypre_ParCompGridNumNodes(compGrid[level]) ]; i++)
         {
            global_index = hypre_ParCompGridPColInd(compGrid[level])[i];
            hypre_ParCompGridPColInd(compGrid[level])[i] = -1;
            // If global index is owned, simply calculate
            for (j = 0; j < num_owned_blocks; j++)
            {
               if (hypre_ParCompGridOwnedBlockStarts(compGrid[level+1])[j+1] - hypre_ParCompGridOwnedBlockStarts(compGrid[level+1])[j] > 0)
               {
                  HYPRE_Int low_global_index = hypre_ParCompGridGlobalIndices(compGrid[level+1])[ hypre_ParCompGridOwnedBlockStarts(compGrid[level+1])[j] ];
                  HYPRE_Int high_global_index = hypre_ParCompGridGlobalIndices(compGrid[level+1])[ hypre_ParCompGridOwnedBlockStarts(compGrid[level+1])[j+1] - 1 ];
                  if ( global_index >= low_global_index && global_index <= high_global_index )
                  {
                     hypre_ParCompGridPColInd(compGrid[level])[i] = global_index - low_global_index + hypre_ParCompGridOwnedBlockStarts(compGrid[level+1])[j];
                  }
               }
            }
            // Otherwise, binary search
            if (hypre_ParCompGridPColInd(compGrid[level])[i] < 0) hypre_ParCompGridPColInd(compGrid[level])[i] = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level+1], global_index, 0, hypre_ParCompGridNumNodes(compGrid[level+1]), hypre_ParCompGridInvSortMap(compGrid[level+1]));
         }
      }

      if (hypre_ParCompGridRRowPtr(compGrid[level]))
      {
         HYPRE_Int num_owned_blocks = hypre_ParCompGridNumOwnedBlocks(compGrid[level-1]);

         // Setup all local indices for all nodes (note that RColInd currently stores global indices)
         for (i = 0; i < hypre_ParCompGridRRowPtr(compGrid[level])[ hypre_ParCompGridNumNodes(compGrid[level]) ]; i++)
         {
            global_index = hypre_ParCompGridRColInd(compGrid[level])[i];
            hypre_ParCompGridRColInd(compGrid[level])[i] = -1;
            // If global index is owned, simply calculate
            for (j = 0; j < num_owned_blocks; j++)
            {
               if (hypre_ParCompGridOwnedBlockStarts(compGrid[level-1])[j+1] - hypre_ParCompGridOwnedBlockStarts(compGrid[level-1])[j] > 0)
               {
                  HYPRE_Int low_global_index = hypre_ParCompGridGlobalIndices(compGrid[level-1])[ hypre_ParCompGridOwnedBlockStarts(compGrid[level-1])[j] ];
                  HYPRE_Int high_global_index = hypre_ParCompGridGlobalIndices(compGrid[level-1])[ hypre_ParCompGridOwnedBlockStarts(compGrid[level-1])[j+1] - 1 ];
                  if ( global_index >= low_global_index && global_index <= high_global_index )
                  {
                     hypre_ParCompGridRColInd(compGrid[level])[i] = global_index - low_global_index + hypre_ParCompGridOwnedBlockStarts(compGrid[level-1])[j];
                  }
               }
            }
            // Otherwise, binary search
            if (hypre_ParCompGridRColInd(compGrid[level])[i] < 0) hypre_ParCompGridRColInd(compGrid[level])[i] = hypre_ParCompGridLocalIndexBinarySearch(compGrid[level-1], global_index, 0, hypre_ParCompGridNumNodes(compGrid[level-1]), hypre_ParCompGridInvSortMap(compGrid[level-1]));
         }
      }
   }

   return 0;
}

HYPRE_Int hypre_ParCompGridLocalIndexBinarySearch( hypre_ParCompGrid *compGrid, HYPRE_Int global_index, HYPRE_Int start, HYPRE_Int end, HYPRE_Int *inv_map )
{
   HYPRE_Int      left = start;
   HYPRE_Int      right = end-1;
   HYPRE_Int      index, sorted_index;

   while (left <= right)
   {
      sorted_index = (left + right) / 2;
      if (inv_map) index = inv_map[sorted_index];
      else index = sorted_index;
      if (hypre_ParCompGridGlobalIndices(compGrid)[index] < global_index) left = sorted_index + 1;
      else if (hypre_ParCompGridGlobalIndices(compGrid)[index] > global_index) right = sorted_index - 1;
      else return index;
   }

   return -1;
}

HYPRE_Int
hypre_ParCompGridDebugPrint ( hypre_ParCompGrid *compGrid, const char* filename, HYPRE_Int coarse_num_nodes )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Get composite grid information
   HYPRE_Int       num_nodes = hypre_ParCompGridNumNodes(compGrid);
   HYPRE_Int       num_real = hypre_ParCompGridNumRealNodes(compGrid);
   HYPRE_Int       num_owned_blocks = hypre_ParCompGridNumOwnedBlocks(compGrid);
   HYPRE_Int       num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid)[hypre_ParCompGridNumOwnedBlocks(compGrid)];
   HYPRE_Int       mem_size = hypre_ParCompGridMemSize(compGrid);
   HYPRE_Int       A_mem_size = hypre_ParCompGridAMemSize(compGrid);
   HYPRE_Int       P_mem_size = hypre_ParCompGridPMemSize(compGrid);

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
   HYPRE_Int *R_rowptr = hypre_ParCompGridRRowPtr(compGrid);
   HYPRE_Int *R_colind = hypre_ParCompGridRColInd(compGrid);
   HYPRE_Complex *R_data = hypre_ParCompGridRData(compGrid);

   HYPRE_Int         i;

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   hypre_fprintf(file, "Num nodes: %d\nMem size: %d\nA Mem size: %d\nP Mem size: %d\nNum owned nodes: %d\nNum ghost dofs: %d\nNum real dofs: %d\n", 
      num_nodes, mem_size, A_mem_size, P_mem_size, num_owned_nodes, num_nodes - num_real, num_real);
   hypre_fprintf(file, "Num owned blocks = %d\n", num_owned_blocks);
   hypre_fprintf(file, "owned_block_starts = ");
   for (i = 0; i < num_owned_blocks+1; i++) hypre_fprintf(file, "%d ", hypre_ParCompGridOwnedBlockStarts(compGrid)[i]);
   hypre_fprintf(file,"\n");
   // hypre_fprintf(file, "u:\n");
   // for (i = 0; i < num_nodes; i++)
   // {
   //    hypre_fprintf(file, "%.10f ", u[i]);
   // }
   // hypre_fprintf(file, "\n");
   // hypre_fprintf(file, "f:\n");
   // for (i = 0; i < num_nodes; i++)
   // {
   //    hypre_fprintf(file, "%.10f ", f[i]);
   // }
   // hypre_fprintf(file, "\n");
   if (global_indices)
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "global_indices:\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", global_indices[i]);
      }
      hypre_fprintf(file, "\n");
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
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "coarse_local_indices:\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", coarse_local_indices[i]);
      }
      hypre_fprintf(file, "\n");
   }
   if (real_dof_marker)
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "real_dof_marker:\n");
      for (i = 0; i < num_nodes; i++)
      {
         hypre_fprintf(file, "%d ", real_dof_marker[i]);
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
      if (A_global_colind)
      {
         hypre_fprintf(file, "A global colind:\n");
         for (i = 0; i < A_rowptr[num_nodes]; i++) hypre_fprintf(file, "%d ", A_global_colind[i]);
         hypre_fprintf(file,"\n\n");
      }
      hypre_fprintf(file, "A data:\n");
      for (i = 0; i < A_rowptr[num_nodes]; i++) hypre_fprintf(file, "%f ", A_data[i]);
   }
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
   // if (R_rowptr && !hypre_ParCompGridR(compGrid)) // NOTE: depending on when this is called, R might have num_nodes rows or coarse_num_nodes rows...
   // {
   //    hypre_fprintf(file,"\n\n");
   //    hypre_fprintf(file, "R row pointer:\n");
   //    for (i = 0; i < num_nodes+1; i++) hypre_fprintf(file, "%d ", R_rowptr[i]);
   //    hypre_fprintf(file,"\n\n");
   //    hypre_fprintf(file, "R colind:\n");
   //    for (i = 0; i < R_rowptr[num_nodes]; i++) hypre_fprintf(file, "%d ", R_colind[i]);
   //    hypre_fprintf(file,"\n\n");
   //    hypre_fprintf(file, "R data:\n");
   //    for (i = 0; i < R_rowptr[num_nodes]; i++) hypre_fprintf(file, "%f ", R_data[i]);
   // }
   // if (hypre_ParCompGridR(compGrid)) // NOTE: depending on when this is called, R might have num_nodes rows or coarse_num_nodes rows...
   // {
   //    hypre_fprintf(file,"\n\n");
   //    hypre_fprintf(file, "R row pointer:\n");
   //    for (i = 0; i < coarse_num_nodes+1; i++) hypre_fprintf(file, "%d ", R_rowptr[i]);
   //    hypre_fprintf(file,"\n\n");
   //    hypre_fprintf(file, "R colind:\n");
   //    for (i = 0; i < R_rowptr[coarse_num_nodes]; i++) hypre_fprintf(file, "%d ", R_colind[i]);
   //    hypre_fprintf(file,"\n\n");
   //    hypre_fprintf(file, "R data:\n");
   //    for (i = 0; i < R_rowptr[coarse_num_nodes]; i++) hypre_fprintf(file, "%f ", R_data[i]);
   // }

   // hypre_fprintf(file, "\n");

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
hypre_ParCompGridMatlabAMatrixDump( hypre_ParCompGrid *compGrid, const char* filename)
{
   // Get composite grid information
   HYPRE_Int       num_nodes = hypre_ParCompGridNumNodes(compGrid);

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i,j;

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

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   HYPRE_Int i,j;

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
hypre_ParCompGridCommPkgCreate(HYPRE_Int num_levels)
{
   hypre_ParCompGridCommPkg   *compGridCommPkg;

   compGridCommPkg = hypre_CTAlloc(hypre_ParCompGridCommPkg, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridCommPkgNumLevels(compGridCommPkg) = num_levels;

   hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
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

   return compGridCommPkg;
}

HYPRE_Int
hypre_ParCompGridCommPkgDestroy( hypre_ParCompGridCommPkg *compGridCommPkg )
{
   HYPRE_Int         i, j, k;

   if ( hypre_ParCompGridCommPkgSendProcs(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgSendProcs(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         if (hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[i])
            hypre_TFree(hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[i], HYPRE_MEMORY_SHARED);
      }
      hypre_TFree(hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         if (hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[i])
            hypre_TFree(hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[i], HYPRE_MEMORY_SHARED);
      }
      hypre_TFree(hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         if (hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[i])
            hypre_TFree(hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[i], HYPRE_MEMORY_SHARED);
      }
      hypre_TFree(hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         if (hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[i])
            hypre_TFree(hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[i], HYPRE_MEMORY_SHARED);
      }
      hypre_TFree(hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg) )
   {
      for (i = 0; i < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); i++)
      {
         hypre_TFree(hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg), HYPRE_MEMORY_HOST);
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
         for (j = 0; j < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[i]; j++)
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
         for (j = 0; j < hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[i]; j++)
         {
            for (k = 0; k < hypre_ParCompGridCommPkgNumLevels(compGridCommPkg); k++)
            {
               if ( hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[i][j][k] ) hypre_TFree( hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[i][j][k], HYPRE_MEMORY_SHARED );
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
         for (j = 0; j < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[i]; j++)
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
         for (j = 0; j < hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[i]; j++)
         {
            hypre_TFree( hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg) )
   {
      hypre_TFree( hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg) )
   {
      hypre_TFree( hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg), HYPRE_MEMORY_HOST );
   }
   
   hypre_TFree(compGridCommPkg, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int
hypre_ParCompGridCommPkgFinalize(hypre_ParAMGData* amg_data, hypre_ParCompGridCommPkg *compGridCommPkg, hypre_ParCompGrid **compGrid)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   HYPRE_Int outer_level, proc, level, i;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int amgdd_start_level = hypre_ParAMGDataAMGDDStartLevel(amg_data);

   HYPRE_Int total_num_nodes = 0;
   HYPRE_Int *offsets = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      offsets[level] = total_num_nodes;
      total_num_nodes += hypre_ParCompGridNumNodes(compGrid[level]);
   }

   if (!hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)) hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   if (!hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)) hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   if (!hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)) hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   if (!hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)) hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);

   for (outer_level = amgdd_start_level; outer_level < num_levels; outer_level++)
   {
      // Finalize send info
      HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level];

      if (hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level]) hypre_TFree(hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level], HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level] = hypre_CTAlloc(HYPRE_Int, num_send_procs+1, HYPRE_MEMORY_SHARED);

      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][0] = 0;
      
      for (proc = 0; proc < num_send_procs; proc++)
      {
         hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][proc+1] = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][proc];

         for (level = outer_level; level < num_levels; level++)
         {
            for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
            {
               if (hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i] >= 0)
               {
                  hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][proc+1]++;
               }
            }
         }
      }

      if (hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[outer_level]) hypre_TFree(hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[outer_level], HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[outer_level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][num_send_procs], HYPRE_MEMORY_SHARED);

      HYPRE_Int num_send_nodes = 0;
      HYPRE_Int new_num_send_procs = 0;
      for (proc = 0; proc < num_send_procs; proc++)
      {
         for (level = outer_level; level < num_levels; level++)
         {
            for (i = 0; i < hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level]; i++)
            {
               if (hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i] >= 0)
               {
                  hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[outer_level][num_send_nodes++] = offsets[level] + hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i];
               }
            }
         }

         if (hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][proc+1] > hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][proc])
         {
            new_num_send_procs++;
         }
      }

      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level] = new_num_send_procs;
      HYPRE_Int new_cnt = 0;
      for (proc = 0; proc < num_send_procs; proc++)
      {
         if (hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][proc+1] > hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][proc])
         {
            hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[outer_level][new_cnt] = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[outer_level][proc];
            hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][new_cnt+1] = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[outer_level][proc+1];
            new_cnt++;
         }
      }

      // Finalize recv info
      HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level];

      if (hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level]) hypre_TFree(hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level], HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level] = hypre_CTAlloc(HYPRE_Int, num_recv_procs+1, HYPRE_MEMORY_SHARED);

      hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][0] = 0;
      
      for (proc = 0; proc < num_recv_procs; proc++)
      {
         hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1] = hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc];

         for (level = outer_level; level < num_levels; level++)
         {
            for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
            {
               if (hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i] >= 0)
               {
                  hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1]++;
               }
            }
         }
      }
      
      if (hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[outer_level]) hypre_TFree(hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[outer_level], HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[outer_level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][num_recv_procs], HYPRE_MEMORY_SHARED);

      HYPRE_Int num_recv_nodes = 0;
      HYPRE_Int new_num_recv_procs = 0;
      for (proc = 0; proc < num_recv_procs; proc++)
      {
         for (level = outer_level; level < num_levels; level++)
         {
            for (i = 0; i < hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level]; i++)
            {
               if (hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i] >= 0)
               {
                  hypre_ParCompGridCommPkgRecvMapElmts(compGridCommPkg)[outer_level][num_recv_nodes++] = offsets[level] + hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i];
               }
            }
         }

         if (hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1] > hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc])
         {
            new_num_recv_procs++;
         }
      }

      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level] = new_num_recv_procs;
      new_cnt = 0;
      for (proc = 0; proc < num_recv_procs; proc++)
      {
         if (hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1] > hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc])
         {
            hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[outer_level][new_cnt] = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[outer_level][proc];
            hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][new_cnt+1] = hypre_ParCompGridCommPkgRecvMapStarts(compGridCommPkg)[outer_level][proc+1];
            new_cnt++;
         }
      }
   }

   // TODO: I think I can free the recvmap and sendflag here

   return 0;
}