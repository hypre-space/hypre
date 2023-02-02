/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 test program
 *****************************************************************************/

#include <stdio.h>
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "mli_matrix.h"
#include "mli_matrix_misc.h"
#include "IJ_mv.h"
#include "seq_mv.h"
#include "parcsr_mv.h"
#include "parcsr_ls.h"
#include <mpi.h>

/******************************************************************************
 test program
 ----------------------------------------------------------------------------*/

extern void testOverLapped( HYPRE_ParCSRMatrix );
extern void testRAP( HYPRE_ParCSRMatrix );
extern void GenTridiagMatrix( HYPRE_ParCSRMatrix *Amat );
extern void GenLaplacian9pt( HYPRE_ParCSRMatrix *Amat );

/******************************************************************************
 main program 
 ----------------------------------------------------------------------------*/

main(int argc, char **argv)
{
   int                problem=2, test=2;
   HYPRE_ParCSRMatrix Amat;

   MPI_Init(&argc, &argv);
   switch (problem)
   {
      case 1 : GenTridiagMatrix( &Amat ); break;
      case 2 : GenLaplacian9pt( &Amat ); break;
   }
   switch (test)
   {
      case 1 : testOverLapped(Amat); break;
      case 2 : testRAP(Amat); break;
   }
   HYPRE_ParCSRMatrixDestroy( Amat );
   MPI_Finalize();
}

/******************************************************************************
 test the overlapped matrix
 ----------------------------------------------------------------------------*/

void testOverLapped( HYPRE_ParCSRMatrix HYPREA )
{
   int          extNRows, *extRowLengs, *extCols;
   double       *extVals;
   char         paramString[100];
   MLI_Function *funcPtr;
   MLI_Matrix   *mli_mat;

   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   mli_mat = new MLI_Matrix( HYPREA, paramString, funcPtr );
   MLI_Matrix_GetOverlappedMatrix(mli_mat, &extNRows, &extRowLengs,
                                  &extCols, &extVals);
      
   delete [] extRowLengs;
   delete [] extCols;
   delete [] extVals;
   delete mli_mat;
}

/******************************************************************************
 test matrix matrix product
 ----------------------------------------------------------------------------*/

void testRAP( HYPRE_ParCSRMatrix HYPREA )
{
   int          i, mypid;
   char         paramString[100];
   double       time1, time2, timeOld, timeNew;
   MLI_Function *funcPtr;
   MLI_Matrix   *mli_mat, *mli_Cmat;
   hypre_ParCSRMatrix *hypreRAP;

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   sprintf(paramString, "HYPRE_ParCSR" );
   mli_mat = new MLI_Matrix( HYPREA, paramString, funcPtr );

   timeOld = timeNew = 0.0;
/*
   for ( i = 0; i < 1; i++ )
   {
      if ( mypid == 0 ) printf("MatMatMult %d\n", i);
      MPI_Barrier(MPI_COMM_WORLD);
      time1 = MLI_Utils_WTime();
      hypreRAP = (hypre_ParCSRMatrix *)
                 hypre_ParMatmul( (hypre_ParCSRMatrix *) HYPREA,
                                  (hypre_ParCSRMatrix *) HYPREA);
      MPI_Barrier(MPI_COMM_WORLD);
      time2 = MLI_Utils_WTime();
      timeOld += time2 - time1;

      MPI_Barrier(MPI_COMM_WORLD);
      time1 = MLI_Utils_WTime();
      MLI_Matrix_MatMatMult(mli_mat, mli_mat, &mli_Cmat); 
      MPI_Barrier(MPI_COMM_WORLD);
      time2 = MLI_Utils_WTime();
      timeNew += time2 - time1;
      hypre_ParCSRMatrixDestroy( (hypre_ParCSRMatrix *) hypreRAP);
      delete mli_Cmat;
   }
   if ( mypid == 0 ) printf("Old MatMatMult time = %e\n", timeOld/10);
   if ( mypid == 0 ) printf("New MatMatMult time = %e\n", timeNew/10);
*/

   if ( mypid == 0 ) printf("Old MatMatMult\n");
   MPI_Barrier(MPI_COMM_WORLD);
   time1 = MLI_Utils_WTime();
   hypreRAP = (hypre_ParCSRMatrix *)
              hypre_ParMatmul( (hypre_ParCSRMatrix *) HYPREA,
                               (hypre_ParCSRMatrix *) HYPREA);
   MPI_Barrier(MPI_COMM_WORLD);
   time2 = MLI_Utils_WTime();
   if ( mypid == 0 ) printf("Old MatMatMult time = %e\n", time2-time1);

   sprintf(paramString, "HYPRE_ParCSR" );
   funcPtr = new MLI_Function();
   mli_Cmat = new MLI_Matrix( hypreRAP, paramString, funcPtr );
   mli_Cmat->print("OldAA");

   if ( mypid == 0 ) printf("New MatMatMult\n");
   MPI_Barrier(MPI_COMM_WORLD);
   time1 = MLI_Utils_WTime();
   MLI_Matrix_MatMatMult(mli_mat, mli_mat, &mli_Cmat); 
   MPI_Barrier(MPI_COMM_WORLD);
   time2 = MLI_Utils_WTime();
   if ( mypid == 0 ) printf("New MatMatMult time = %e\n", time2-time1);
   mli_Cmat->print("newAA");

   delete mli_mat;
}

/******************************************************************************
 set up a matrix from 9-point 2D Laplacian
 ----------------------------------------------------------------------------*/

void GenTridiagMatrix( HYPRE_ParCSRMatrix *Amat )
{
   int    mypid, nprocs, localNRows=10, length=7, *rowSizes, *colInd;
   int    ii, irow, irow2, rowIndex, ierr, globalNRows;
   int    firstRow, firstCol, lastRow, lastCol;
   double *colVal;
   HYPRE_IJMatrix IJA;

   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   firstRow    = mypid * localNRows;
   lastRow     = (mypid + 1) * localNRows - 1;
   firstCol    = firstRow;
   lastCol     = lastRow;
   globalNRows = localNRows * nprocs;
   ierr = HYPRE_IJMatrixCreate( MPI_COMM_WORLD, firstRow, lastRow,
                                firstCol, lastCol, &IJA );
   ierr += HYPRE_IJMatrixSetObjectType( IJA, HYPRE_PARCSR );
   rowSizes = new int[localNRows];
   for ( ii = 0; ii < localNRows; ii++ ) rowSizes[ii] = length;
   ierr = HYPRE_IJMatrixSetRowSizes ( IJA, (const int *) rowSizes );
   ierr = HYPRE_IJMatrixInitialize( IJA );
   delete [] rowSizes;
   colInd = new int[length];
   colVal = new double[length];
   for ( irow = 0; irow < localNRows; irow++ ) 
   {
      rowIndex = firstRow + irow;
      irow2    = 0;
      for ( ii = length-1; ii >= 0; ii-- )
      {
         colInd[irow2] = rowIndex - length/2 + ii;
         colVal[irow2] = -1;
         if ( colInd[irow2] == rowIndex ) colVal[irow2] = 10.0;
         if ( colInd[irow2] >= 0 && colInd[irow2] < globalNRows ) irow2++;
      }
      ierr += HYPRE_IJMatrixSetValues( IJA, 1, &irow2, &rowIndex,
                     (const int *) colInd, (const double *) colVal );
   }
   ierr += HYPRE_IJMatrixAssemble( IJA );
   delete [] colInd;
   delete [] colVal;
   ierr += HYPRE_IJMatrixGetObject( IJA, (void **) Amat);
}

/******************************************************************************
 set up a matrix from 9-point 2D Laplacian
 ----------------------------------------------------------------------------*/

void GenLaplacian9pt( HYPRE_ParCSRMatrix *Amat )
{
   int                nx, ny, px, py, mypx, mypy, mypid, nprocs;
   double             *values;
   HYPRE_ParCSRMatrix  A;

   /*-----------------------------------------------------------
    * get machine information
    *-----------------------------------------------------------*/

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid );
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs );

   /*-----------------------------------------------------------
    * set up grid and processor parameters
    *-----------------------------------------------------------*/

   nx = ny = 100;
   if ( nprocs > 1 ) px = 2;
   else              px = 1;
   py = nprocs / px;

   if ( (px*py) != nprocs)
   {
      printf("Error: Invalid processor topology \n");
      exit(1);
   }
   if (mypid == 0)
   {
      printf("  Laplacian 9pt:\n");
      printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      printf("    (px, py) = (%d, %d)\n", px, py);
   }
   mypx = mypid % px;
   mypy = ( mypid - mypx) / px;

   /*-----------------------------------------------------------
    * create matrix parameters 
    *-----------------------------------------------------------*/

   values = new double[2];
   values[1] = -1.0;
   values[0] = 0.0;
   if ( nx > 1 ) values[0] += 2.0;
   if ( ny > 1 ) values[0] += 2.0;
   if ( nx > 1 && ny > 1 ) values[0] += 4.0;
   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(MPI_COMM_WORLD,
                                  nx, ny, px, py, mypx, mypy, values);
   delete [] values;

   (*Amat) = A;
}

