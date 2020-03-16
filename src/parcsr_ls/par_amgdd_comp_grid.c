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

HYPRE_Int LocalIndexBinarySearch( hypre_ParCompGrid *compGrid, HYPRE_Int global_index )
{
   HYPRE_Int      left = 0;
   HYPRE_Int      right = hypre_ParCompGridNumNonOwnedNodes(compGrid)-1;
   HYPRE_Int      index, sorted_index;
   HYPRE_Int      *inv_map = hypre_ParCompGridNonOwnedInvSort(compGrid);

   while (left <= right)
   {
      sorted_index = (left + right) / 2;
      index = inv_map[sorted_index];
      if (hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[index] < global_index) left = sorted_index + 1;
      else if (hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[index] > global_index) right = sorted_index - 1;
      else return index;
   }

   return -1;
}

hypre_ParCompGridMatrix* hypre_ParCompGridMatrixCreate()
{
   hypre_ParCompGridMatrix *matrix = hypre_CTAlloc(hypre_ParCompGridMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridMatrixOwnedDiag(matrix) = NULL;
   hypre_ParCompGridMatrixOwnedOffd(matrix) = NULL;
   hypre_ParCompGridMatrixNonOwnedDiag(matrix) = NULL;
   hypre_ParCompGridMatrixNonOwnedOffd(matrix) = NULL;

   hypre_ParCompGridMatrixOwnsOwnedMatrices(matrix) = 0;
   hypre_ParCompGridMatrixOwnsOffdColIndices(matrix) = 0;

   return matrix;
}

HYPRE_Int hypre_ParCompGridMatrixDestroy(hypre_ParCompGridMatrix *matrix)
{
   if (hypre_ParCompGridMatrixOwnsOwnedMatrices(matrix))
   {
      if (hypre_ParCompGridMatrixOwnedDiag(matrix)) hypre_CSRMatrixDestroy(hypre_ParCompGridMatrixOwnedDiag(matrix));
      if (hypre_ParCompGridMatrixOwnedOffd(matrix)) hypre_CSRMatrixDestroy(hypre_ParCompGridMatrixOwnedOffd(matrix));
   }
   else if (hypre_ParCompGridMatrixOwnsOffdColIndices(matrix))
   {
      if (hypre_CSRMatrixJ(hypre_ParCompGridMatrixOwnedOffd(matrix))) hypre_TFree(hypre_CSRMatrixJ(hypre_ParCompGridMatrixOwnedOffd(matrix)), HYPRE_MEMORY_SHARED);
   }
   if (hypre_ParCompGridMatrixNonOwnedDiag(matrix)) hypre_CSRMatrixDestroy(hypre_ParCompGridMatrixNonOwnedDiag(matrix));
   if (hypre_ParCompGridMatrixNonOwnedOffd(matrix)) hypre_CSRMatrixDestroy(hypre_ParCompGridMatrixNonOwnedOffd(matrix));

   hypre_TFree(matrix, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int hypre_ParCompGridMatvec( HYPRE_Complex alpha, hypre_ParCompGridMatrix *A, hypre_ParCompGridVector *x, HYPRE_Complex beta, hypre_ParCompGridVector *y)
{
   hypre_CSRMatrix *owned_diag = hypre_ParCompGridMatrixOwnedDiag(A);
   hypre_CSRMatrix *owned_offd = hypre_ParCompGridMatrixOwnedOffd(A);
   hypre_CSRMatrix *nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(A);
   hypre_CSRMatrix *nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(A);

   hypre_Vector *x_owned = hypre_ParCompGridVectorOwned(x);
   hypre_Vector *x_nonowned = hypre_ParCompGridVectorNonOwned(x);

   hypre_Vector *y_owned = hypre_ParCompGridVectorOwned(y);
   hypre_Vector *y_nonowned = hypre_ParCompGridVectorNonOwned(y);

   hypre_CSRMatrixMatvec(alpha, owned_diag, x_owned, beta, y_owned);
   hypre_CSRMatrixMatvec(alpha, owned_offd, x_nonowned, 0.0, y_owned);
   hypre_CSRMatrixMatvec(alpha, nonowned_diag, x_nonowned, beta, y_nonowned);
   hypre_CSRMatrixMatvec(alpha, nonowned_offd, x_owned, 0.0, y_nonowned);

   return 0;
}

hypre_ParCompGridVector *hypre_ParCompGridVectorCreate()
{
   hypre_ParCompGridVector *vector = hypre_CTAlloc(hypre_ParCompGridVector, 1, HYPRE_MEMORY_HOST);

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

HYPRE_Int hypre_ParCompGridVectorAxpy(HYPRE_Complex alpha, hypre_ParCompGridVector *x, hypre_ParCompGridVector *y )
{
   if (hypre_ParCompGridVectorOwned(x))
      hypre_SeqVectorAxpy(alpha, hypre_ParCompGridVectorOwned(x), hypre_ParCompGridVectorOwned(y));
   if (hypre_ParCompGridVectorNonOwned(x))
      hypre_SeqVectorAxpy(alpha, hypre_ParCompGridVectorNonOwned(x), hypre_ParCompGridVectorNonOwned(y));

   return 0;
}

HYPRE_Int hypre_ParCompGridVectorSetConstantValues(hypre_ParCompGridVector *vector, HYPRE_Complex value )
{
   if (hypre_ParCompGridVectorOwned(vector))
      hypre_SeqVectorSetConstantValues(hypre_ParCompGridVectorOwned(vector), value);
   if (hypre_ParCompGridVectorNonOwned(vector))
      hypre_SeqVectorSetConstantValues(hypre_ParCompGridVectorNonOwned(vector), value);

   return 0;
}

hypre_ParCompGrid *
hypre_ParCompGridCreate ()
{
   hypre_ParCompGrid      *compGrid;

   compGrid = hypre_CTAlloc(hypre_ParCompGrid, 1, HYPRE_MEMORY_HOST);

   hypre_ParCompGridFirstGlobalIndex(compGrid) = 0;
   hypre_ParCompGridLastGlobalIndex(compGrid) = 0;
   hypre_ParCompGridNumOwnedNodes(compGrid) = 0;
   hypre_ParCompGridNumNonOwnedNodes(compGrid) = 0;
   hypre_ParCompGridNumNonOwnedRealNodes(compGrid) = 0;
   hypre_ParCompGridNumOwnedCPoints(compGrid) = 0;
   hypre_ParCompGridNumNonOwnedCPoints(compGrid) = 0;
   hypre_ParCompGridNumMissingColIndices(compGrid) = 0;

   hypre_ParCompGridNonOwnedGlobalIndices(compGrid) = NULL;
   hypre_ParCompGridNonOwnedCoarseIndices(compGrid) = NULL;
   hypre_ParCompGridNonOwnedRealMarker(compGrid) = NULL;
   hypre_ParCompGridNonOwnedSort(compGrid) = NULL;
   hypre_ParCompGridNonOwnedInvSort(compGrid) = NULL;
   hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid) = NULL;

   hypre_ParCompGridOwnedCoarseIndices(compGrid) = NULL;

   hypre_ParCompGridANew(compGrid) = NULL;
   hypre_ParCompGridPNew(compGrid) = NULL;
   hypre_ParCompGridRNew(compGrid) = NULL;

   hypre_ParCompGridUNew(compGrid) = NULL;
   hypre_ParCompGridFNew(compGrid) = NULL;
   hypre_ParCompGridTNew(compGrid) = NULL;
   hypre_ParCompGridSNew(compGrid) = NULL;
   hypre_ParCompGridQNew(compGrid) = NULL;
   hypre_ParCompGridTempNew(compGrid) = NULL;
   hypre_ParCompGridTemp2New(compGrid) = NULL;
   hypre_ParCompGridTemp3New(compGrid) = NULL;



   return compGrid;
}

HYPRE_Int
hypre_ParCompGridDestroy ( hypre_ParCompGrid *compGrid )
{
   

   // !!! TODO NEW



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
   hypre_ParCompGridNumNonOwnedNodes(compGrid) = hypre_CSRMatrixNumCols(A_offd_original);
   hypre_ParCompGridNumMissingColIndices(compGrid) = 0;

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
   hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned_diag_nnz, HYPRE_MEMORY_HOST);

   // !!! Symmetric: in the symmetric case we can go ahead and just setup nonowned_offd 

   // Setup CompGridMatrix P and R if appropriate (!!! Don't actually need to do this here, I guess)
   if (level != hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      hypre_ParCompGridMatrix *P = hypre_ParCompGridMatrixCreate();
      hypre_ParCompGridMatrixOwnedDiag(P) = hypre_ParCSRMatrixDiag( hypre_ParAMGDataPArray(amg_data)[level] );
      // Use original rowptr and data from P, but need to use new col indices (init to global index, then setup local indices later)
      hypre_CSRMatrix *P_offd_original = hypre_ParCSRMatrixOffd( hypre_ParAMGDataPArray(amg_data)[level] );
      hypre_ParCompGridMatrixOwnedOffd(P) = hypre_CSRMatrixCreate(hypre_CSRMatrixNumRows(P_offd_original), hypre_CSRMatrixNumCols(P_offd_original), hypre_CSRMatrixNumNonzeros(P_offd_original));
      hypre_CSRMatrixI(hypre_ParCompGridMatrixOwnedOffd(P)) = hypre_CSRMatrixI(P_offd_original);
      hypre_CSRMatrixData(hypre_ParCompGridMatrixOwnedOffd(P)) = hypre_CSRMatrixData(P_offd_original);
      hypre_CSRMatrixOwnsData(hypre_ParCompGridMatrixOwnedOffd(P)) = 0;
      hypre_CSRMatrixJ(hypre_ParCompGridMatrixOwnedOffd(P)) = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumNonzeros(P_offd_original), HYPRE_MEMORY_SHARED);
      
      // Initialize P owned offd col ind to their global indices
      for (i = 0; i < hypre_CSRMatrixNumNonzeros(hypre_ParCompGridMatrixOwnedOffd(P)); i++)
         hypre_CSRMatrixJ(hypre_ParCompGridMatrixOwnedOffd(P))[i] = hypre_ParCSRMatrixColMapOffd( hypre_ParAMGDataPArray(amg_data)[level] )[ hypre_CSRMatrixJ(P_offd_original)[i] ];

      hypre_ParCompGridMatrixOwnsOwnedMatrices(P) = 0;
      hypre_ParCompGridMatrixOwnsOffdColIndices(P) = 1;
      hypre_ParCompGridPNew(compGrid) = P;
   }
   if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
   {
      // !!! TODO
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

   // Initialize nonowned global indices, real marker, and the sort and invsort arrays
   HYPRE_Int prev_gid = 0; // !!! Debug
   for (i = 0; i < hypre_CSRMatrixNumCols(A_offd_original); i++)
   {
      hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[i] = hypre_ParCSRMatrixColMapOffd( hypre_ParAMGDataAArray(amg_data)[level] )[i];
      
      // !!! Debug: expect gid ordering
      if (hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[i] < prev_gid) printf("HEY! ColMapOffd isn't in GID ordering\n");
      prev_gid = hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[i];

      // !!! Question: necessary to initialize sort and invsort? With my current PackSendBufferNew, I need values here
      hypre_ParCompGridNonOwnedSort(compGrid)[i] = i;
      hypre_ParCompGridNonOwnedInvSort(compGrid)[i] = i;

      // !!! Assume at least pad = 1 (i.e. the first layer of dofs are real)
      hypre_ParCompGridNonOwnedRealMarker(compGrid)[i] = 1;

   }

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
hypre_ParCompGridSetupRelax( hypre_ParAMGData *amg_data )
{
   HYPRE_Int level, i, j;

   if (hypre_ParAMGDataFACRelaxType(amg_data) == 0) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_Jacobi;
   else if (hypre_ParAMGDataFACRelaxType(amg_data) == 1) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_GaussSeidel;
   else if (hypre_ParAMGDataFACRelaxType(amg_data) == 2) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_Cheby;
   else if (hypre_ParAMGDataFACRelaxType(amg_data) == 3) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_CFL1Jacobi; 
   else if (hypre_ParAMGDataFACRelaxType(amg_data) == 4) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_OrderedGaussSeidel; 

   // for (level = hypre_ParAMGDataAMGDDStartLevel(amg_data); level < hypre_ParAMGDataNumLevels(amg_data); level++)
   // {
   //    hypre_ParCompGrid *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];

   //    if (hypre_ParAMGDataFACRelaxType(amg_data) == 2)
   //    {
   //       // Setup chebyshev coefficients
   //       hypre_CSRMatrix *A = hypre_ParCompGridA(compGrid);
   //       HYPRE_Real    *coefs = hypre_ParCompGridChebyCoeffs(compGrid);
   //       HYPRE_Int     scale = hypre_ParAMGDataChebyScale(amg_data);
   //       HYPRE_Int     order = hypre_ParAMGDataChebyOrder(amg_data);

   //       // Select submatrix of real to real connections
   //       HYPRE_Int nnz = 0;
   //       for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++)
   //       {
   //          for (j = hypre_CSRMatrixI(A)[i]; j < hypre_CSRMatrixI(A)[i+1]; j++)
   //          {
   //             if (hypre_CSRMatrixJ(A)[j] < hypre_ParCompGridNumRealNodes(compGrid)) nnz++;
   //          }
   //       }
   //       HYPRE_Int *A_real_i = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumRealNodes(compGrid)+1, HYPRE_MEMORY_SHARED);
   //       HYPRE_Int *A_real_j = hypre_CTAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_SHARED);
   //       HYPRE_Complex *A_real_data = hypre_CTAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_SHARED);
   //       nnz = 0;
   //       for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++)
   //       {
   //          for (j = hypre_CSRMatrixI(A)[i]; j < hypre_CSRMatrixI(A)[i+1]; j++)
   //          {
   //             if (hypre_CSRMatrixJ(A)[j] < hypre_ParCompGridNumRealNodes(compGrid))
   //             {
   //                A_real_j[nnz] = hypre_CSRMatrixJ(A)[j];
   //                A_real_data[nnz] = hypre_CSRMatrixData(A)[j];
   //                nnz++;
   //             }
   //          }
   //          A_real_i[i+1] = nnz;
   //       }

   //       HYPRE_BigInt *row_starts = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   //       row_starts[0] = 0;
   //       row_starts[1] = hypre_ParCompGridNumRealNodes(compGrid);
   //       hypre_ParCSRMatrix *A_real = hypre_ParCSRMatrixCreate( MPI_COMM_SELF,
   //                           (HYPRE_BigInt) hypre_ParCompGridNumRealNodes(compGrid),
   //                           (HYPRE_BigInt) hypre_ParCompGridNumRealNodes(compGrid),
   //                           row_starts,
   //                           NULL,
   //                           0,
   //                           nnz,
   //                           0 );
   //       hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_real)) = A_real_i;
   //       hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A_real)) = A_real_j;
   //       hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_real)) = A_real_data;
   //       hypre_CSRMatrixInitialize(hypre_ParCSRMatrixOffd(A_real));
   //       hypre_ParCSRMatrixColMapOffd(A_real) = hypre_CTAlloc(HYPRE_BigInt, 0, HYPRE_MEMORY_HOST);

   //       HYPRE_Real max_eig, min_eig = 0;

   //       if (hypre_ParAMGDataChebyEigEst(amg_data)) hypre_ParCSRMaxEigEstimateCG(A_real, scale, hypre_ParAMGDataChebyEigEst(amg_data), &max_eig, &min_eig);
   //       else hypre_ParCSRMaxEigEstimate(A_real, scale, &max_eig);

   //       HYPRE_Real *dummy_ptr;
   //       hypre_ParCSRRelax_Cheby_Setup(hypre_ParAMGDataAArray(amg_data)[level], 
   //                             max_eig,      
   //                             min_eig,     
   //                             hypre_ParAMGDataChebyFraction(amg_data),   
   //                             order,
   //                             0,
   //                             hypre_ParAMGDataChebyVariant(amg_data),           
   //                             &coefs,
   //                             &dummy_ptr);

   //       hypre_ParCompGridChebyCoeffs(compGrid) = coefs;

   //       hypre_ParCSRMatrixDestroy(A_real);

   //       // Calculate diagonal scaling values 
   //       hypre_ParCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_SHARED);
   //       for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   //       {
   //          for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
   //          {
   //             if (hypre_ParCompGridAColInd(compGrid)[j] == i)
   //             {
   //                hypre_ParCompGridL1Norms(compGrid)[i] = 1.0/sqrt(hypre_ParCompGridAData(compGrid)[j]);
   //                break;
   //             }
   //          }
   //       }

   //       // Setup temporary/auxiliary vectors
   //       hypre_ParCompGridTemp(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumRealNodes(compGrid));
   //       hypre_SeqVectorInitialize(hypre_ParCompGridTemp(compGrid));

   //       hypre_ParCompGridTemp2(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
   //       hypre_SeqVectorInitialize(hypre_ParCompGridTemp2(compGrid));

   //       hypre_ParCompGridTemp3(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumRealNodes(compGrid));
   //       hypre_SeqVectorInitialize(hypre_ParCompGridTemp3(compGrid));
   //    }
   //    if (hypre_ParAMGDataFACRelaxType(amg_data) == 3)
   //    {
   //       // Calculate l1_norms
   //       hypre_ParCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, hypre_ParCompGridNumNodes(compGrid), HYPRE_MEMORY_SHARED);
   //       for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   //       {
   //          HYPRE_Int cf_diag = hypre_ParCompGridCFMarkerArray(compGrid)[i];
   //          for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
   //          {
   //             if (hypre_ParCompGridCFMarkerArray(compGrid)[ hypre_ParCompGridAColInd(compGrid)[j] ] == cf_diag) 
   //             {
   //                hypre_ParCompGridL1Norms(compGrid)[i] += fabs(hypre_ParCompGridAData(compGrid)[j]);
   //             }
   //          }
   //       }
   //       // Setup temporary/auxiliary vectors
   //       hypre_ParCompGridTemp(compGrid) = hypre_SeqVectorCreate(hypre_ParCompGridNumNodes(compGrid));
   //       hypre_SeqVectorInitialize(hypre_ParCompGridTemp(compGrid));

   //       #if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   //       // Setup c and f point masks
   //       int num_c_points = 0;
   //       int num_f_points = 0;
   //       for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++) if (hypre_ParCompGridCFMarkerArray(compGrid)[i]) num_c_points++;
   //       num_f_points = hypre_ParCompGridNumRealNodes(compGrid) - num_c_points;
   //       hypre_ParCompGridNumCPoints(compGrid) = num_c_points;
   //       hypre_ParCompGridCMask(compGrid) = hypre_CTAlloc(int, num_c_points, HYPRE_MEMORY_SHARED);
   //       hypre_ParCompGridFMask(compGrid) = hypre_CTAlloc(int, num_f_points, HYPRE_MEMORY_SHARED);
   //       int c_cnt = 0, f_cnt = 0;
   //       for (i = 0; i < hypre_ParCompGridNumRealNodes(compGrid); i++)
   //       {
   //          if (hypre_ParCompGridCFMarkerArray(compGrid)[i]) hypre_ParCompGridCMask(compGrid)[c_cnt++] = i;
   //          else hypre_ParCompGridFMask(compGrid)[f_cnt++] = i;
   //       }
   //       #endif
   //    }
   // }


   return 0;
}

HYPRE_Int
hypre_ParCompGridFinalizeNew( hypre_ParAMGData *amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int start_level, HYPRE_Int num_levels, HYPRE_Int use_rd, HYPRE_Int debug )
{
   HYPRE_Int level, i, j;

   // Post process to remove -1 entries from matrices and reorder so that extra nodes are [real, ghost]
   for (level = start_level; level < num_levels; level++)
   {
      HYPRE_Int num_nonowned = hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
      HYPRE_Int num_owned = hypre_ParCompGridNumOwnedNodes(compGrid[level]);
      HYPRE_Int num_nonowned_real_nodes = 0;
      for (i = 0; i < num_nonowned; i++)
      {
         if (hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[i]) num_nonowned_real_nodes++;
      }
      hypre_ParCompGridNumNonOwnedRealNodes(compGrid[level]) = num_nonowned_real_nodes;
      HYPRE_Int *new_indices = hypre_CTAlloc(HYPRE_Int, num_nonowned, HYPRE_MEMORY_HOST);
      HYPRE_Int real_cnt = 0;
      HYPRE_Int ghost_cnt = 0;
      for (i = 0; i < num_nonowned; i++)
      {
         if (hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_indices[i] = real_cnt++;
         }
         else new_indices[i] = num_nonowned_real_nodes + ghost_cnt++;
      }

      // Transform indices in send_flag and recv_map
      if (compGridCommPkg)
      {
         HYPRE_Int outer_level;
         for (outer_level = start_level; outer_level < num_levels; outer_level++)
         {
            HYPRE_Int num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level];
            HYPRE_Int new_num_send_procs = 0;
            HYPRE_Int proc;
            for (proc = 0; proc < num_send_procs; proc++)
            {
               HYPRE_Int num_send_nodes = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level];
               HYPRE_Int new_num_send_nodes = 0;
               for (i = 0; i < num_send_nodes; i++)
               {
                  if (hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i] >= num_owned)
                  {
                     hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][new_num_send_nodes++] = new_indices[ hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level][i] - num_owned ] + num_owned;
                  }
               }
               hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[outer_level][proc][level] = new_num_send_nodes;
               if (new_num_send_nodes)
               {
                  hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][new_num_send_procs++][level] = hypre_TReAlloc(hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[outer_level][proc][level], HYPRE_Int, new_num_send_nodes, HYPRE_MEMORY_SHARED);
               }
            }
            hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[outer_level] = new_num_send_procs;

            HYPRE_Int num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level];
            HYPRE_Int new_num_recv_procs = 0;
            for (proc = 0; proc < num_recv_procs; proc++)
            {
               HYPRE_Int num_recv_nodes = hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level];
               HYPRE_Int new_num_recv_nodes = 0;
               for (i = 0; i < num_recv_nodes; i++)
               {
                  if (hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i] >= 0)
                  {
                     hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][new_num_recv_nodes++] = new_indices[hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level][i]];
                  }
               }
               hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[outer_level][proc][level] = new_num_recv_nodes;
               if (new_num_recv_nodes)
               {
                  hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][new_num_recv_procs++][level] = hypre_TReAlloc(hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[outer_level][proc][level], HYPRE_Int, new_num_recv_nodes, HYPRE_MEMORY_SHARED);
               }
            }
            hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[outer_level] = new_num_recv_procs;
         }
      }

      // If global indices are still needed, transform these also
      if (debug)
      {
         HYPRE_Int *new_global_indices = hypre_CTAlloc(HYPRE_Int, num_nonowned, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_nonowned; i++)
         {
            new_global_indices[ new_indices[i] ] = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ i ];
         }
         hypre_TFree(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]) = new_global_indices;
      }

      // Reorder nonowned matrices
      hypre_CSRMatrix *A_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid[level]));
      hypre_CSRMatrix *A_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid[level]));

      HYPRE_Int A_diag_nnz = hypre_CSRMatrixI(A_diag)[num_nonowned];
      HYPRE_Int *new_A_diag_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_SHARED);
      HYPRE_Int *new_A_diag_colInd = hypre_CTAlloc(HYPRE_Int, A_diag_nnz, HYPRE_MEMORY_SHARED);
      HYPRE_Complex *new_A_diag_data = hypre_CTAlloc(HYPRE_Complex, A_diag_nnz, HYPRE_MEMORY_SHARED);

      HYPRE_Int A_offd_nnz = hypre_CSRMatrixI(A_offd)[num_nonowned];
      HYPRE_Int *new_A_offd_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_SHARED);
      HYPRE_Int *new_A_offd_colInd = hypre_CTAlloc(HYPRE_Int, A_offd_nnz, HYPRE_MEMORY_SHARED);
      HYPRE_Complex *new_A_offd_data = hypre_CTAlloc(HYPRE_Complex, A_offd_nnz, HYPRE_MEMORY_SHARED);

      hypre_CSRMatrix *P_diag;
      hypre_CSRMatrix *P_offd;

      HYPRE_Int P_diag_nnz;
      HYPRE_Int *new_P_diag_rowPtr;
      HYPRE_Int *new_P_diag_colInd;
      HYPRE_Complex *new_P_diag_data;

      HYPRE_Int P_offd_nnz;
      HYPRE_Int *new_P_offd_rowPtr;
      HYPRE_Int *new_P_offd_colInd;
      HYPRE_Complex *new_P_offd_data;

      // TODO R
      // HYPRE_Int R_nnz;
      // HYPRE_Int *new_R_rowPtr;
      // HYPRE_Int *new_R_colInd;
      // HYPRE_Complex *new_R_data;

      if (level != num_levels-1)
      {
         P_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));
         P_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));

         P_diag_nnz = hypre_CSRMatrixI(P_diag)[num_nonowned];
         new_P_diag_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_SHARED);
         new_P_diag_colInd = hypre_CTAlloc(HYPRE_Int, P_diag_nnz, HYPRE_MEMORY_SHARED);
         new_P_diag_data = hypre_CTAlloc(HYPRE_Complex, P_diag_nnz, HYPRE_MEMORY_SHARED);

         P_offd_nnz = hypre_CSRMatrixI(P_offd)[num_nonowned];
         new_P_offd_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, HYPRE_MEMORY_SHARED);
         new_P_offd_colInd = hypre_CTAlloc(HYPRE_Int, P_offd_nnz, HYPRE_MEMORY_SHARED);
         new_P_offd_data = hypre_CTAlloc(HYPRE_Complex, P_offd_nnz, HYPRE_MEMORY_SHARED);
      }
      // TODO R
      // if (hypre_ParCompGridRRowPtr(compGrid[level]))
      // {
      //    R_nnz = hypre_ParCompGridRRowPtr(compGrid[level])[num_nodes];
      //    new_R_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nodes+1, HYPRE_MEMORY_SHARED);
      //    new_R_colInd = hypre_CTAlloc(HYPRE_Int, R_nnz, HYPRE_MEMORY_SHARED);
      //    new_R_data = hypre_CTAlloc(HYPRE_Complex, R_nnz, HYPRE_MEMORY_SHARED);
      // }

      HYPRE_Int A_diag_cnt = 0;
      HYPRE_Int A_offd_cnt = 0;
      HYPRE_Int P_diag_cnt = 0;
      HYPRE_Int P_offd_cnt = 0;
      HYPRE_Int R_diag_cnt = 0;
      HYPRE_Int R_offd_cnt = 0;
      HYPRE_Int node_cnt = 0;
      // Real nodes
      for (i = 0; i < num_nonowned; i++)
      {
         if (hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_A_diag_rowPtr[node_cnt] = A_diag_cnt;
            for (j = hypre_CSRMatrixI(A_diag)[i]; j < hypre_CSRMatrixI(A_diag)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A_diag)[j] >= 0)
               {
                  new_A_diag_colInd[A_diag_cnt] = new_indices[ hypre_CSRMatrixJ(A_diag)[j] ];
                  new_A_diag_data[A_diag_cnt] = hypre_CSRMatrixData(A_diag)[j];
                  A_diag_cnt++;
               }
            }
            new_A_offd_rowPtr[node_cnt] = A_offd_cnt;
            for (j = hypre_CSRMatrixI(A_offd)[i]; j < hypre_CSRMatrixI(A_offd)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A_offd)[j] >= 0)
               {
                  new_A_offd_colInd[A_offd_cnt] = hypre_CSRMatrixJ(A_offd)[j];
                  new_A_offd_data[A_offd_cnt] = hypre_CSRMatrixData(A_offd)[j];
                  A_offd_cnt++;
               }
            }

            if (level != num_levels-1)
            {
               new_P_diag_rowPtr[node_cnt] = P_diag_cnt;
               for (j = hypre_CSRMatrixI(P_diag)[i]; j < hypre_CSRMatrixI(P_diag)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(P_diag)[j] >= 0)
                  {
                     new_P_diag_colInd[P_diag_cnt] = hypre_CSRMatrixJ(P_diag)[j];
                     new_P_diag_data[P_diag_cnt] = hypre_CSRMatrixData(P_diag)[j];
                     P_diag_cnt++;
                  }
               }
               new_P_offd_rowPtr[node_cnt] = P_offd_cnt;
               for (j = hypre_CSRMatrixI(P_offd)[i]; j < hypre_CSRMatrixI(P_offd)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(P_offd)[j] >= 0)
                  {
                     new_P_offd_colInd[P_offd_cnt] = hypre_CSRMatrixJ(P_offd)[j];
                     new_P_offd_data[P_offd_cnt] = hypre_CSRMatrixData(P_offd)[j];
                     P_offd_cnt++;
                  }
               }
            }
            // TODO R
            // if (hypre_ParCompGridRRowPtr(compGrid[level]))
            // {
            //    new_R_rowPtr[node_cnt] = R_cnt;
            //    for (j = hypre_ParCompGridRRowPtr(compGrid[level])[i]; j < hypre_ParCompGridRRowPtr(compGrid[level])[i+1]; j++)
            //    {
            //       if (hypre_ParCompGridRColInd(compGrid[level])[j] >= 0)
            //       {
            //          new_R_colInd[R_cnt] = hypre_ParCompGridRColInd(compGrid[level])[j];
            //          new_R_data[R_cnt] = hypre_ParCompGridRData(compGrid[level])[j];
            //          R_cnt++;
            //       }
            //    }
            // }
            node_cnt++;
         }
      }
      // Ghost nodes
      for (i = 0; i < num_nonowned; i++)
      {
         if (!hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_A_diag_rowPtr[node_cnt] = A_diag_cnt;
            for (j = hypre_CSRMatrixI(A_diag)[i]; j < hypre_CSRMatrixI(A_diag)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A_diag)[j] >= 0)
               {
                  new_A_diag_colInd[A_diag_cnt] = new_indices[ hypre_CSRMatrixJ(A_diag)[j] ];
                  new_A_diag_data[A_diag_cnt] = hypre_CSRMatrixData(A_diag)[j];
                  A_diag_cnt++;
               }
            }
            new_A_offd_rowPtr[node_cnt] = A_offd_cnt;
            for (j = hypre_CSRMatrixI(A_offd)[i]; j < hypre_CSRMatrixI(A_offd)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A_offd)[j] >= 0)
               {
                  new_A_offd_colInd[A_offd_cnt] = hypre_CSRMatrixJ(A_offd)[j];
                  new_A_offd_data[A_offd_cnt] = hypre_CSRMatrixData(A_offd)[j];
                  A_offd_cnt++;
               }
            }

            if (level != num_levels-1)
            {
               new_P_diag_rowPtr[node_cnt] = P_diag_cnt;
               for (j = hypre_CSRMatrixI(P_diag)[i]; j < hypre_CSRMatrixI(P_diag)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(P_diag)[j] >= 0)
                  {
                     new_P_diag_colInd[P_diag_cnt] = hypre_CSRMatrixJ(P_diag)[j];
                     new_P_diag_data[P_diag_cnt] = hypre_CSRMatrixData(P_diag)[j];
                     P_diag_cnt++;
                  }
               }
               new_P_offd_rowPtr[node_cnt] = P_offd_cnt;
               for (j = hypre_CSRMatrixI(P_offd)[i]; j < hypre_CSRMatrixI(P_offd)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(P_offd)[j] >= 0)
                  {
                     new_P_offd_colInd[P_offd_cnt] = hypre_CSRMatrixJ(P_offd)[j];
                     new_P_offd_data[P_offd_cnt] = hypre_CSRMatrixData(P_offd)[j];
                     P_offd_cnt++;
                  }
               }
            }
            // TODO R
            // if (hypre_ParCompGridRRowPtr(compGrid[level]))
            // {
            //    new_R_rowPtr[node_cnt] = R_cnt;
            //    for (j = hypre_ParCompGridRRowPtr(compGrid[level])[i]; j < hypre_ParCompGridRRowPtr(compGrid[level])[i+1]; j++)
            //    {
            //       if (hypre_ParCompGridRColInd(compGrid[level])[j] >= 0)
            //       {
            //          new_R_colInd[R_cnt] = hypre_ParCompGridRColInd(compGrid[level])[j];
            //          new_R_data[R_cnt] = hypre_ParCompGridRData(compGrid[level])[j];
            //          R_cnt++;
            //       }
            //    }
            // }
            node_cnt++;
         }
      }
      new_A_diag_rowPtr[num_nonowned] = A_diag_cnt;
      new_A_offd_rowPtr[num_nonowned] = A_offd_cnt;
      if (level != num_levels-1)
      {
         new_P_diag_rowPtr[num_nonowned] = P_diag_cnt;
         new_P_offd_rowPtr[num_nonowned] = P_offd_cnt;
      }
      // TODO R
      // if (hypre_ParCompGridRRowPtr(compGrid[level])) new_R_rowPtr[num_nodes] = R_cnt;

      // Fix up P col indices on finer level
      if (level != start_level)
      {
         P_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level-1]));
         P_offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridPNew(compGrid[level-1]));

         for (i = 0; i < hypre_CSRMatrixI(P_diag)[ hypre_ParCompGridNumNonOwnedNodes(compGrid[level-1]) ]; i++)
         {
            hypre_CSRMatrixJ(P_diag)[i] = new_indices[ hypre_CSRMatrixJ(P_diag)[i] ];
         }
         // Also fix up owned offd col indices 
         for (i = 0; i < hypre_CSRMatrixI(P_offd)[ hypre_ParCompGridNumOwnedNodes(compGrid[level-1]) ]; i++)
         {
            hypre_CSRMatrixJ(P_offd)[i] = new_indices[ hypre_CSRMatrixJ(P_offd)[i] ];
         }
      }
      // TODO Fix up R col indices on coarser level (and owned offd)
      // if (level != num_levels-1)
      // {
      //    if (hypre_ParCompGridRRowPtr(compGrid[level+1]))
      //    {
      //       for (i = 0; i < hypre_ParCompGridRRowPtr(compGrid[level+1])[ hypre_ParCompGridNumNodes(compGrid[level+1]) ]; i++)
      //       {
      //          if (hypre_ParCompGridRColInd(compGrid[level+1])[i] >= 0)
      //             hypre_ParCompGridRColInd(compGrid[level+1])[i] = new_indices[ hypre_ParCompGridRColInd(compGrid[level+1])[i] ];
      //       }
      //    }
      // }

      // Clean up memory, deallocate old arrays and reset pointers to new arrays
      hypre_TFree(hypre_CSRMatrixI(A_diag), HYPRE_MEMORY_SHARED);
      hypre_TFree(hypre_CSRMatrixJ(A_diag), HYPRE_MEMORY_SHARED);
      hypre_TFree(hypre_CSRMatrixData(A_diag), HYPRE_MEMORY_SHARED);
      hypre_CSRMatrixI(A_diag) = new_A_diag_rowPtr;
      hypre_CSRMatrixJ(A_diag) = new_A_diag_colInd;
      hypre_CSRMatrixData(A_diag) = new_A_diag_data;
      hypre_CSRMatrixNumRows(A_diag) = num_nonowned;
      hypre_CSRMatrixNumCols(A_diag) = num_nonowned;
      hypre_CSRMatrixNumNonzeros(A_diag) = hypre_CSRMatrixI(A_diag)[num_nonowned];

      hypre_TFree(hypre_CSRMatrixI(A_offd), HYPRE_MEMORY_SHARED);
      hypre_TFree(hypre_CSRMatrixJ(A_offd), HYPRE_MEMORY_SHARED);
      hypre_TFree(hypre_CSRMatrixData(A_offd), HYPRE_MEMORY_SHARED);
      hypre_CSRMatrixI(A_offd) = new_A_offd_rowPtr;
      hypre_CSRMatrixJ(A_offd) = new_A_offd_colInd;
      hypre_CSRMatrixData(A_offd) = new_A_offd_data;
      hypre_CSRMatrixNumRows(A_offd) = num_nonowned;
      hypre_CSRMatrixNumCols(A_offd) = hypre_ParCompGridNumOwnedNodes(compGrid[level]);
      hypre_CSRMatrixNumNonzeros(A_offd) = hypre_CSRMatrixI(A_offd)[num_nonowned];

      if (level != num_levels-1)
      {
         P_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));
         P_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));

         hypre_TFree(hypre_CSRMatrixI(P_diag), HYPRE_MEMORY_SHARED);
         hypre_TFree(hypre_CSRMatrixJ(P_diag), HYPRE_MEMORY_SHARED);
         hypre_TFree(hypre_CSRMatrixData(P_diag), HYPRE_MEMORY_SHARED);
         hypre_CSRMatrixI(P_diag) = new_P_diag_rowPtr;
         hypre_CSRMatrixJ(P_diag) = new_P_diag_colInd;
         hypre_CSRMatrixData(P_diag) = new_P_diag_data;
         hypre_CSRMatrixNumRows(P_diag) = num_nonowned;
         hypre_CSRMatrixNumCols(P_diag) = hypre_ParCompGridNumNonOwnedNodes(compGrid[level+1]);
         hypre_CSRMatrixNumNonzeros(P_diag) = hypre_CSRMatrixI(P_diag)[num_nonowned];

         hypre_TFree(hypre_CSRMatrixI(P_offd), HYPRE_MEMORY_SHARED);
         hypre_TFree(hypre_CSRMatrixJ(P_offd), HYPRE_MEMORY_SHARED);
         hypre_TFree(hypre_CSRMatrixData(P_offd), HYPRE_MEMORY_SHARED);
         hypre_CSRMatrixI(P_offd) = new_P_offd_rowPtr;
         hypre_CSRMatrixJ(P_offd) = new_P_offd_colInd;
         hypre_CSRMatrixData(P_offd) = new_P_offd_data;
         hypre_CSRMatrixNumRows(P_offd) = num_nonowned;
         hypre_CSRMatrixNumCols(P_offd) = hypre_ParCompGridNumOwnedNodes(compGrid[level+1]);
         hypre_CSRMatrixNumNonzeros(P_offd) = hypre_CSRMatrixI(P_offd)[num_nonowned];

         hypre_CSRMatrixNumCols(hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridPNew(compGrid[level]))) = hypre_ParCompGridNumNonOwnedNodes(compGrid[level+1]);
      }
      // TODO R
      // if (hypre_ParCompGridRRowPtr(compGrid[level]))
      // {
      //    hypre_TFree(hypre_ParCompGridRRowPtr(compGrid[level]), HYPRE_MEMORY_HOST);
      //    hypre_TFree(hypre_ParCompGridRColInd(compGrid[level]), HYPRE_MEMORY_HOST);
      //    hypre_TFree(hypre_ParCompGridRData(compGrid[level]), HYPRE_MEMORY_HOST);
      //    hypre_ParCompGridRRowPtr(compGrid[level]) = new_R_rowPtr;
      //    hypre_ParCompGridRColInd(compGrid[level]) = new_R_colInd;
      //    hypre_ParCompGridRData(compGrid[level]) = new_R_data;            
      // }

      hypre_TFree(new_indices, HYPRE_MEMORY_HOST);

      // Setup comp grid vectors
      hypre_ParCompGridUNew(compGrid[level]) = hypre_ParCompGridVectorCreate();
      hypre_ParCompGridVectorOwned(hypre_ParCompGridUNew(compGrid[level])) = hypre_ParVectorLocalVector( hypre_ParAMGDataUArray(amg_data)[level] );
      hypre_ParCompGridVectorOwnsOwnedVector(hypre_ParCompGridUNew(compGrid[level])) = 0;
      hypre_ParCompGridVectorNonOwned(hypre_ParCompGridUNew(compGrid[level])) = hypre_SeqVectorCreate(num_nonowned);
      hypre_SeqVectorInitialize(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridUNew(compGrid[level])));

      hypre_ParCompGridFNew(compGrid[level]) = hypre_ParCompGridVectorCreate();
      hypre_ParCompGridVectorOwned(hypre_ParCompGridFNew(compGrid[level])) = hypre_ParVectorLocalVector( hypre_ParAMGDataFArray(amg_data)[level] );
      hypre_ParCompGridVectorOwnsOwnedVector(hypre_ParCompGridFNew(compGrid[level])) = 0;
      hypre_ParCompGridVectorNonOwned(hypre_ParCompGridFNew(compGrid[level])) = hypre_SeqVectorCreate(num_nonowned);
      hypre_SeqVectorInitialize(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridFNew(compGrid[level])));

      if (use_rd)
      {
         hypre_ParCompGridQNew(compGrid[level]) = hypre_ParCompGridVectorCreate();
         hypre_ParCompGridVectorOwned(hypre_ParCompGridQNew(compGrid[level])) = hypre_SeqVectorCreate(num_owned);
         hypre_SeqVectorInitialize(hypre_ParCompGridVectorOwned(hypre_ParCompGridQNew(compGrid[level])));
         hypre_ParCompGridVectorOwnsOwnedVector(hypre_ParCompGridQNew(compGrid[level])) = 1;
         hypre_ParCompGridVectorNonOwned(hypre_ParCompGridQNew(compGrid[level])) = hypre_SeqVectorCreate(num_nonowned);
         hypre_SeqVectorInitialize(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridQNew(compGrid[level])));
      }

      if (level < num_levels)
      {
         hypre_ParCompGridSNew(compGrid[level]) = hypre_ParCompGridVectorCreate();
         hypre_ParCompGridVectorOwned(hypre_ParCompGridSNew(compGrid[level])) = hypre_SeqVectorCreate(num_owned);
         hypre_SeqVectorInitialize(hypre_ParCompGridVectorOwned(hypre_ParCompGridSNew(compGrid[level])));
         hypre_ParCompGridVectorOwnsOwnedVector(hypre_ParCompGridSNew(compGrid[level])) = 1;
         hypre_ParCompGridVectorNonOwned(hypre_ParCompGridSNew(compGrid[level])) = hypre_SeqVectorCreate(num_nonowned);
         hypre_SeqVectorInitialize(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridSNew(compGrid[level])));

         hypre_ParCompGridTNew(compGrid[level]) = hypre_ParCompGridVectorCreate();
         hypre_ParCompGridVectorOwned(hypre_ParCompGridTNew(compGrid[level])) = hypre_SeqVectorCreate(num_owned);
         hypre_SeqVectorInitialize(hypre_ParCompGridVectorOwned(hypre_ParCompGridTNew(compGrid[level])));
         hypre_ParCompGridVectorOwnsOwnedVector(hypre_ParCompGridTNew(compGrid[level])) = 1;
         hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTNew(compGrid[level])) = hypre_SeqVectorCreate(num_nonowned);
         hypre_SeqVectorInitialize(hypre_ParCompGridVectorNonOwned(hypre_ParCompGridTNew(compGrid[level])));
      }

      // Free up arrays we no longer need
      if (hypre_ParCompGridNonOwnedRealMarker(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridNonOwnedRealMarker(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridNonOwnedRealMarker(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]) && !debug)
      {
         hypre_TFree(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridOwnedCoarseIndices(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridOwnedCoarseIndices(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridOwnedCoarseIndices(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridNonOwnedSort(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridNonOwnedSort(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridNonOwnedSort(compGrid[level]) = NULL;
      }
      if (hypre_ParCompGridNonOwnedInvSort(compGrid[level]))
      {
         hypre_TFree(hypre_ParCompGridNonOwnedInvSort(compGrid[level]), HYPRE_MEMORY_HOST);
         hypre_ParCompGridNonOwnedInvSort(compGrid[level]) = NULL;
      }
   }
   
   // Setup R = P^T if R not specified
   if (!hypre_ParAMGDataRestriction(amg_data))
   {
      for (level = start_level; level < num_levels-1; level++)
      {
         // !!! TODO: if BoomerAMG explicitly stores R = P^T, use those matrices in
         hypre_ParCompGridRNew(compGrid[level]) = hypre_ParCompGridMatrixCreate();
         hypre_ParCompGridMatrixOwnsOwnedMatrices(hypre_ParCompGridRNew(compGrid[level])) = 1;
         hypre_CSRMatrixTranspose(hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridPNew(compGrid[level])), 
                                  &hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridRNew(compGrid[level])), 1);
         hypre_CSRMatrixTranspose(hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid[level])), 
                                  &hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridRNew(compGrid[level])), 1);
         hypre_CSRMatrixTranspose(hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridPNew(compGrid[level])), 
                                  &hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridRNew(compGrid[level])), 1);
         hypre_CSRMatrixTranspose(hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level])), 
                                  &hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridRNew(compGrid[level])), 1);
      }
   }

   // Finish up comm pkg buffer sizes
   for (start_level = 0; start_level < num_levels; start_level++)
   {
      HYPRE_Int proc;
      for (proc = 0; proc < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[start_level]; proc++)
      {
         hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)[start_level][proc] = 0;
         for (level = start_level; level < num_levels; level++)
         {
            hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg)[start_level][proc] += hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[start_level][proc][level];
         }
      }
      for (proc = 0; proc < hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[start_level]; proc++)
      {
         hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg)[start_level][proc] = 0;
         for (level = start_level; level < num_levels; level++)
         {
            hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg)[start_level][proc] += hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[start_level][proc][level];
         }
      }
   }

   return 0;
}

HYPRE_Int
hypre_ParCompGridResizeNew ( hypre_ParCompGrid *compGrid, HYPRE_Int new_size, HYPRE_Int need_coarse_info )
{
   // This function reallocates memory to hold nonowned info for the comp grid

   hypre_ParCompGridNonOwnedGlobalIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridNonOwnedGlobalIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
   hypre_ParCompGridNonOwnedRealMarker(compGrid) = hypre_TReAlloc(hypre_ParCompGridNonOwnedRealMarker(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
   hypre_ParCompGridNonOwnedSort(compGrid) = hypre_TReAlloc(hypre_ParCompGridNonOwnedSort(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);
   hypre_ParCompGridNonOwnedInvSort(compGrid) = hypre_TReAlloc(hypre_ParCompGridNonOwnedInvSort(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);

   hypre_CSRMatrix *nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid));
   hypre_CSRMatrix *nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid));
   hypre_CSRMatrixResize(nonowned_diag, new_size, new_size, hypre_CSRMatrixNumNonzeros(nonowned_diag));
   hypre_CSRMatrixResize(nonowned_offd, new_size, hypre_CSRMatrixNumCols(nonowned_offd), hypre_CSRMatrixNumNonzeros(nonowned_offd));

   if (need_coarse_info)
   {
      hypre_ParCompGridNonOwnedCoarseIndices(compGrid) = hypre_TReAlloc(hypre_ParCompGridNonOwnedCoarseIndices(compGrid), HYPRE_Int, new_size, HYPRE_MEMORY_HOST);

      // !!! Double check... can just take care of this at the end, yeah?
      // nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid));
      // nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid));
      // hypre_CSRMatrixResize(nonowned_diag, new_size, hypre_CSRMatrixNumCols(nonowned_diag), hypre_CSRMatrixNumNonzeros(nonowned_diag));
      // hypre_CSRMatrixResize(nonowned_offd, new_size, hypre_CSRMatrixNumCols(nonowned_offd), hypre_CSRMatrixNumNonzeros(nonowned_offd));

   }

   // !!! Double check... can just take care of this at the end, yeah?
   // if (hypre_ParCompGridRNew(compGrid))
   // {
   //    nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridRNew(compGrid));
   //    nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridRNew(compGrid));
   //    hypre_CSRMatrixResize(nonowned_diag, new_size, hypre_CSRMatrixNumCols(nonowned_diag), hypre_CSRMatrixNumNonzeros(nonowned_diag));
   //    hypre_CSRMatrixResize(nonowned_offd, new_size, hypre_CSRMatrixNumCols(nonowned_offd), hypre_CSRMatrixNumNonzeros(nonowned_offd));
   // }

   return 0;
}

HYPRE_Int 
hypre_ParCompGridSetupLocalIndicesNew( hypre_ParCompGrid **compGrid, HYPRE_Int *nodes_added_on_level, HYPRE_Int ****recv_map,
   HYPRE_Int num_recv_procs, HYPRE_Int **A_tmp_info, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int symmetric )
{
   // when nodes are added to a composite grid, global info is copied over, but local indices must be generated appropriately for all added nodes
   // this must be done on each level as info is added to correctly construct subsequent Psi_c grids
   // also done after each ghost layer is added
   HYPRE_Int      level,proc,i,j,k;
   HYPRE_Int      global_index, local_index, coarse_index;

   HYPRE_Int bin_search_cnt = 0;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   // !!! Debug
   for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid[1]); i++)
   {
      if (hypre_ParCompGridNonOwnedCoarseIndices(compGrid[1])[i] >= hypre_ParCompGridNumNonOwnedNodes(compGrid[2]))
         printf("Rank %d, level %d, start of setup local current level %d, nonowned coarse index out of bounds: i = %d, coarse index = %d\n",
            myid, level, current_level, i, hypre_ParCompGridNonOwnedCoarseIndices(compGrid[1])[i]);
   }

   hypre_ParCompGridMatrix *A = hypre_ParCompGridANew(compGrid[current_level]);
   hypre_CSRMatrix *owned_offd = hypre_ParCompGridMatrixOwnedOffd(A);
   hypre_CSRMatrix *nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(A);
   hypre_CSRMatrix *nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(A);

   // On current_level, need to deal with A_tmp_info
   HYPRE_Int row = hypre_CSRMatrixNumCols(owned_offd)+1;
   HYPRE_Int diag_rowptr = hypre_CSRMatrixI(nonowned_diag)[ hypre_CSRMatrixNumCols(owned_offd) ];
   HYPRE_Int offd_rowptr = hypre_CSRMatrixI(nonowned_offd)[ hypre_CSRMatrixNumCols(owned_offd) ];
   for (proc = 0; proc < num_recv_procs; proc++)
   {
      HYPRE_Int cnt = 0;
      HYPRE_Int num_original_recv_dofs = A_tmp_info[proc][cnt++];
      HYPRE_Int remaining_dofs = A_tmp_info[proc][cnt++];

      for (i = 0; i < remaining_dofs; i++)
      {
         HYPRE_Int row_size = A_tmp_info[proc][cnt++];
         for (j = 0; j < row_size; j++)
         {
            HYPRE_Int incoming_index = A_tmp_info[proc][cnt++];

            // Incoming is a global index (could be owned or nonowned)
            if (incoming_index < 0)
            {
               incoming_index = -(incoming_index+1);
               // See whether global index is owned on this proc (if so, can directly setup appropriate local index)
               if (incoming_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[current_level]) && incoming_index <= hypre_ParCompGridLastGlobalIndex(compGrid[current_level]))
               {
                  // Add to offd
                  if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_offd))
                     hypre_CSRMatrixResize(nonowned_offd, hypre_CSRMatrixNumRows(nonowned_offd), hypre_CSRMatrixNumCols(nonowned_offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_offd)));
                  hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = incoming_index - hypre_ParCompGridFirstGlobalIndex(compGrid[current_level]);
               }
               else
               {
                  // Add to diag (global index, not in buffer, so need to do local binary search)
                  if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
                  {
                     hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag), hypre_CSRMatrixNumCols(nonowned_diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag)));
                     hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag)), HYPRE_MEMORY_HOST);
                  }
                  // If we dof not found in comp grid, then mark this as a missing connection
                  hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level])[ hypre_ParCompGridNumMissingColIndices(compGrid[current_level])++ ] = diag_rowptr;
                  hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = -(incoming_index+1);
               }
            }
            // Incoming is an index to dofs within the buffer (by construction, nonowned)
            else
            {
               // Add to diag (index is within buffer, so we can directly go to local index)
               if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
               {
                  hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag), hypre_CSRMatrixNumCols(nonowned_diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag)));
                  hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag)), HYPRE_MEMORY_HOST);
               }
               local_index = recv_map[current_level][proc][current_level][ incoming_index ];
               if (local_index < 0) local_index = -(local_index + 1);
               hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = local_index;
            }
         }

         // Update row pointers 
         hypre_CSRMatrixI(nonowned_offd)[ row ] = offd_rowptr;
         hypre_CSRMatrixI(nonowned_diag)[ row ] = diag_rowptr;
         row++;
      }
      hypre_TFree(A_tmp_info[proc], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(A_tmp_info, HYPRE_MEMORY_HOST);

   // Loop over levels from current to coarsest
   for (level = current_level; level < num_levels; level++)
   {
      A = hypre_ParCompGridANew(compGrid[level]);
      nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(A);

      // If we have added nodes on this level
      if (nodes_added_on_level[level])
      {
         // Look for missing col ind connections
         HYPRE_Int num_missing_col_ind = hypre_ParCompGridNumMissingColIndices(compGrid[level]);
         hypre_ParCompGridNumMissingColIndices(compGrid[level]) = 0;
         for (i = 0; i < num_missing_col_ind; i++)
         {
            j = hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level])[i];
            global_index = hypre_CSRMatrixJ(nonowned_diag)[ j ];

            // !!! Debug
            if (global_index >= 0)
            {
               printf("global_index >= 0 where I expected negative... should have been missing col ind\n");
               printf("Rank %d, level %d, global_index = %d, j = %d, num_missing_col_ind = %d\n", myid, level, global_index, j, num_missing_col_ind);
            }


            global_index = -(global_index+1);

            local_index = LocalIndexBinarySearch(compGrid[level], global_index);
            bin_search_cnt++;
            // If we dof not found in comp grid, then mark this as a missing connection
            if (local_index == -1)
            {
               local_index = -(global_index+1);
               hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level])[ hypre_ParCompGridNumMissingColIndices(compGrid[level])++ ] = j;
            }
            hypre_CSRMatrixJ(nonowned_diag)[ j ] = local_index;
         }
      }
     
      // if we are not on the coarsest level
      if (level != num_levels-1)
      {
         // loop over indices of non-owned nodes on this level 
         // !!! No guarantee that previous ghost dofs converted to real dofs have coarse local indices setup...
         // !!! Thus we go over all non-owned dofs here instead of just the added ones, but we only setup coarse local index where necessary.
         // !!! NOTE: can't use nodes_added_on_level here either because real overwritten by ghost doesn't count as added node (so you can miss setting these up)
         for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid[level]); i++)
         {
            // fix up the coarse local indices
            coarse_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level])[i];
            HYPRE_Int is_real = hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[i];

            // setup coarse local index if necessary
            if (coarse_index < -1 && is_real)
            {
               coarse_index = -(coarse_index+2); // Map back to regular global index
               local_index = LocalIndexBinarySearch(compGrid[level+1], coarse_index);
               bin_search_cnt++;

               // !!! Debug
               if (local_index < 0) printf("Rank %d: Error! Level %d: Could not find coasre index %d during setup local indices\n", myid, level, coarse_index);

               hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level])[i] = local_index;
               // !!! Debug
               if (local_index >= hypre_ParCompGridNumNonOwnedNodes(compGrid[level+1]))
                  printf("Rank %d, level %d, current level %d, nonowned coarse index out of bounds: local_index = %d\n",myid, level, current_level, local_index);
            }
         }
      }
   }

   // !!! Debug
   for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid[1]); i++)
   {
      if (hypre_ParCompGridNonOwnedCoarseIndices(compGrid[1])[i] >= hypre_ParCompGridNumNonOwnedNodes(compGrid[2]))
         printf("Rank %d, level %d, end of setup local current level %d, nonowned coarse index out of bounds: i = %d, coarse index = %d\n",
            myid, level, current_level, i, hypre_ParCompGridNonOwnedCoarseIndices(compGrid[1])[i]);
   }

   return bin_search_cnt;
}

HYPRE_Int hypre_ParCompGridSetupLocalIndicesPNew( hypre_ParCompGrid **compGrid, HYPRE_Int start_level, HYPRE_Int num_levels )
{
   HYPRE_Int                  i,level;

   for (level = start_level; level < num_levels-1; level++)
   {
      // Setup owned offd col indices
      hypre_CSRMatrix *owned_offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridPNew(compGrid[level]));

      for (i = 0; i < hypre_CSRMatrixI(owned_offd)[hypre_ParCompGridNumOwnedNodes(compGrid[level])]; i++)
      {
         HYPRE_Int local_index = LocalIndexBinarySearch(compGrid[level+1], hypre_CSRMatrixJ(owned_offd)[i]);
         if (local_index == -1) hypre_CSRMatrixJ(owned_offd)[i] = -(hypre_CSRMatrixJ(owned_offd)[i] + 1);
         else hypre_CSRMatrixJ(owned_offd)[i] = local_index;
      }

      // Setup nonowned diag col indices
      hypre_CSRMatrix *nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid[level]));

      for (i = 0; i < hypre_CSRMatrixI(nonowned_diag)[hypre_ParCompGridNumNonOwnedNodes(compGrid[level])]; i++)
      {
         HYPRE_Int local_index = LocalIndexBinarySearch(compGrid[level+1], hypre_CSRMatrixJ(nonowned_diag)[i]);
         if (local_index == -1) hypre_CSRMatrixJ(nonowned_diag)[i] = -(hypre_CSRMatrixJ(nonowned_diag)[i] + 1);
         else hypre_CSRMatrixJ(nonowned_diag)[i] = local_index;
      }
   }

   // !!! TODO R
   // if ()
   // {
   // }

   return 0;
}

HYPRE_Int
hypre_ParCompGridDebugPrintNew ( hypre_ParCompGrid *compGrid, const char* filename )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int         i;

   // Print info to given filename   
   FILE             *file;
   file = fopen(filename,"w");
   hypre_fprintf(file, "Num owned nodes: %d [%d - %d]\nNum nonowned nodes: %d\n", 
      hypre_ParCompGridNumOwnedNodes(compGrid), hypre_ParCompGridFirstGlobalIndex(compGrid),
      hypre_ParCompGridLastGlobalIndex(compGrid), hypre_ParCompGridNumNonOwnedNodes(compGrid));

   if (hypre_ParCompGridNonOwnedGlobalIndices(compGrid))
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "hypre_ParCompGridNonOwnedGlobalIndices(compGrid):\n");
      for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d ", hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[i]);
      }
      hypre_fprintf(file, "\n");
   }

   if (hypre_ParCompGridNonOwnedRealMarker(compGrid))
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "hypre_ParCompGridNonOwnedRealMarker(compGrid):\n");
      for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d ", hypre_ParCompGridNonOwnedRealMarker(compGrid)[i]);
      }
      hypre_fprintf(file, "\n");
   }

   if (hypre_ParCompGridNonOwnedSort(compGrid))
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "hypre_ParCompGridNonOwnedSort(compGrid):\n");
      for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d ", hypre_ParCompGridNonOwnedSort(compGrid)[i]);
      }
      hypre_fprintf(file, "\n");
   }

   if (hypre_ParCompGridNonOwnedInvSort(compGrid))
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "hypre_ParCompGridNonOwnedInvSort(compGrid):\n");
      for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d ", hypre_ParCompGridNonOwnedInvSort(compGrid)[i]);
      }
      hypre_fprintf(file, "\n");
   }

   if (hypre_ParCompGridOwnedCoarseIndices(compGrid))
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "hypre_ParCompGridOwnedCoarseIndices(compGrid):\n");
      for (i = 0; i < hypre_ParCompGridNumOwnedNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d ", hypre_ParCompGridOwnedCoarseIndices(compGrid)[i]);
      }
      hypre_fprintf(file, "\n");
   }

   if (hypre_ParCompGridNonOwnedCoarseIndices(compGrid))
   {
      hypre_fprintf(file, "\n");
      hypre_fprintf(file, "hypre_ParCompGridNonOwnedCoarseIndices(compGrid):\n");
      for (i = 0; i < hypre_ParCompGridNumNonOwnedNodes(compGrid); i++)
      {
         hypre_fprintf(file, "%d ", hypre_ParCompGridNonOwnedCoarseIndices(compGrid)[i]);
      }
      hypre_fprintf(file, "\n");
   }


   fclose(file);

   char matrix_filename[256];
   sprintf(matrix_filename, "%s_A_owned_diag", filename);
   hypre_CSRMatrixPrint(  hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridANew(compGrid)), matrix_filename);

   sprintf(matrix_filename, "%s_A_owned_offd", filename);
   hypre_CSRMatrixPrint(  hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridANew(compGrid)), matrix_filename);

   sprintf(matrix_filename, "%s_A_nonowned_diag", filename);
   hypre_CSRMatrixPrintCustom(  hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridANew(compGrid)), matrix_filename, hypre_ParCompGridNumNonOwnedNodes(compGrid));

   sprintf(matrix_filename, "%s_A_nonowned_offd", filename);
   hypre_CSRMatrixPrintCustom(  hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridANew(compGrid)), matrix_filename, hypre_ParCompGridNumNonOwnedNodes(compGrid));

   if (hypre_ParCompGridPNew(compGrid))
   {
      if (hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid)))
      {
         sprintf(matrix_filename, "%s_P_owned_diag", filename);
         hypre_CSRMatrixPrint(  hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridPNew(compGrid)), matrix_filename);

         sprintf(matrix_filename, "%s_P_owned_offd", filename);
         hypre_CSRMatrixPrint(  hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridPNew(compGrid)), matrix_filename);

         sprintf(matrix_filename, "%s_P_nonowned_diag", filename);
         hypre_CSRMatrixPrintCustom(  hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridPNew(compGrid)), matrix_filename, hypre_ParCompGridNumNonOwnedNodes(compGrid));

         sprintf(matrix_filename, "%s_P_nonowned_offd", filename);
         hypre_CSRMatrixPrintCustom(  hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridPNew(compGrid)), matrix_filename, hypre_ParCompGridNumNonOwnedNodes(compGrid));
      }
   }

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
