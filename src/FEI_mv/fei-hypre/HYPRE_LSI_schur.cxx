/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "HYPRE_FEI_includes.h"

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#include "HYPRE_LSI_schur.h"

//******************************************************************************
// These are external functions needed internally here
//------------------------------------------------------------------------------

extern "C" {
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix**);
   int HYPRE_LSI_Search(int*, int, int);
   void qsort0(int *, int, int);
   void qsort1(int *, double *, int, int);
}

//******************************************************************************
// Constructor
//------------------------------------------------------------------------------

HYPRE_LSI_Schur::HYPRE_LSI_Schur()
{
   A11mat_        = NULL;
   A12mat_        = NULL;
   A22mat_        = NULL;
   Smat_          = NULL;
   Svec_          = NULL;
   APartition_    = NULL;
   P22LocalInds_  = NULL;
   P22GlobalInds_ = NULL;
   P22Offsets_    = NULL;
   P22Size_       = -1;
   P22GSize_      = -1;
   assembled_     = 0;
   outputLevel_   = 1;
   lookup_        = NULL;
   mpiComm_       = -1;
}

//******************************************************************************
// destructor
//------------------------------------------------------------------------------

HYPRE_LSI_Schur::~HYPRE_LSI_Schur()
{
   if ( A11mat_        != NULL ) HYPRE_IJMatrixDestroy(A11mat_);
   if ( A12mat_        != NULL ) HYPRE_IJMatrixDestroy(A12mat_);
   if ( A22mat_        != NULL ) HYPRE_IJMatrixDestroy(A22mat_);
   if ( APartition_    != NULL ) free( APartition_ );
   if ( P22LocalInds_  != NULL ) delete [] P22LocalInds_;
   if ( P22GlobalInds_ != NULL ) delete [] P22GlobalInds_;
   if ( P22Offsets_    != NULL ) delete [] P22Offsets_;
}

//******************************************************************************
// set lookup object
//------------------------------------------------------------------------------

int HYPRE_LSI_Schur::setLookup(Lookup *object)
{
   lookup_ = object;
   return 0;
}

//******************************************************************************
// Given a matrix A, compute the sizes and indices of the 2 x 2 blocks
// (P22Size_,P22GSize_,P22LocalInds_,P22GlobalInds_,P22Offsets_)
//------------------------------------------------------------------------------

int HYPRE_LSI_Schur::computeBlockInfo()
{
   int      mypid, nprocs, startRow, endRow, localNrows, irow, lastNodeNum;
   int      j, rowSize, *colInd, *dispArray, index, globalNrows, count;
   int      nodeNum;
   double   *colVal;

   //------------------------------------------------------------------
   // check that the system matrix has been set, clean up previous
   // allocations, and extract matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   startRow     = APartition_[mypid];
   endRow       = APartition_[mypid+1] - 1;
   localNrows   = endRow - startRow + 1;
   globalNrows  = APartition_[nprocs];
    
   //------------------------------------------------------------------
   // find the local size of the (2,2) block
   //------------------------------------------------------------------

   P22Size_    = count = 0;
   lastNodeNum = -1;
   for ( irow = startRow; irow <= endRow; irow++ )
   {
      nodeNum = lookup_->getAssociatedNodeNumber(irow);
      if ( nodeNum != lastNodeNum ) 
      {
         if (count == 1) break; 
         lastNodeNum = nodeNum; 
         count = 1;
      }
      else count++;
   }
   index = irow - 1;
   for ( irow = index; irow <= endRow; irow++ ) P22Size_++;

   if ( outputLevel_ > 0 )
      printf("%4d HYPRE_LSI_Schur : P22_size = %d\n", mypid, P22Size_);

   //------------------------------------------------------------------
   // allocate array for storing indices of (2,2) block variables 
   //------------------------------------------------------------------

   if ( P22Size_ > 0 ) P22LocalInds_ = new int[P22Size_];
   else                P22LocalInds_ = NULL; 

   //------------------------------------------------------------------
   // compose a local list of rows for the (2,2) block
   //------------------------------------------------------------------

   P22Size_    = count = 0;
   lastNodeNum = -1;
   for ( irow = startRow; irow <= endRow; irow++ )
   {
      nodeNum = lookup_->getAssociatedNodeNumber(irow);
      if ( nodeNum != lastNodeNum ) 
      {
         if (count == 1) break; 
         lastNodeNum = nodeNum; 
         count = 1;
      }
      else count++;
   }
   index = irow - 1;
   for ( irow = index; irow <= endRow; irow++ ) 
      P22LocalInds_[P22Size_++] = irow;

   //------------------------------------------------------------------
   // compose a global list of rows for the (2,2) block
   //------------------------------------------------------------------

   MPI_Allreduce(&P22Size_, &P22GSize_, 1, MPI_INT, MPI_SUM, mpiComm_);

   if ( outputLevel_ > 0 )
   {
      if ( P22GSize_ == 0 && mypid == 0 )
         printf("HYPRE_LSI_Schur WARNING : P22Size = 0 on all processors.\n");
   }
   if ( P22GSize_ == 0 )
   {
      if ( APartition_ != NULL ) free( APartition_ );
      APartition_ = NULL;
      return 1;
   }

   if ( P22GSize_ > 0 ) P22GlobalInds_ = new int[P22GSize_];
   else                 P22GlobalInds_ = NULL;
   dispArray   = new int[nprocs];
   P22Offsets_ = new int[nprocs];
   MPI_Allgather(&P22Size_, 1, MPI_INT, P22Offsets_, 1, MPI_INT, mpiComm_);
   dispArray[0] = 0;
   for ( j = 1; j < nprocs; j++ ) 
      dispArray[j] = dispArray[j-1] + P22Offsets_[j-1];
   MPI_Allgatherv(P22LocalInds_, P22Size_, MPI_INT, P22GlobalInds_,
                  P22Offsets_, dispArray, MPI_INT, mpiComm_);
   delete [] P22Offsets_;
   P22Offsets_ = dispArray;

   if ( outputLevel_ > 1 )
   {
      for ( j = 0; j < P22Size_; j++ )
         printf("%4d HYPRE_LSI_Schur : P22Inds %8d = %d\n", mypid,
                j, P22LocalInds_[j]);
   }
   return 0;
} 

//******************************************************************************
// Given a matrix A, build the 2 x 2 blocks
// (This function is to be called after computeBlockInfo
//------------------------------------------------------------------------------

int HYPRE_LSI_Schur::buildBlocks(HYPRE_IJMatrix Amat)
{
   int    irow, j, k, rowSize, *inds, ierr, mypid, nprocs, index, searchInd;
   int    ANRows, ANCols, AGNRows, AGNCols, AStartRow, AStartCol;
   int    A11NRows, A11NCols, A11GNRows, A11GNCols, A11StartRow, A11StartCol;
   int    A12NRows, A12NCols, A12GNRows, A12GNCols, A12StartRow, A12StartCol;
   int    A22NRows, A22NCols, A22GNRows, A22GNCols, A22StartRow, A22StartCol;
   int    *A11RowLengs, A11MaxRowLeng, A11RowCnt, A11NewSize, *A11_inds;
   int    *A12RowLengs, A12MaxRowLeng, A12RowCnt, A12NewSize, *A12_inds;
   int    *A22RowLengs, A22MaxRowLeng, A22RowCnt, A22NewSize, *A22_inds;
   double *vals, *A11_vals, *A12_vals, *A22_vals;
   char   fname[200];
   FILE   *fp;
   HYPRE_ParCSRMatrix Amat_csr, A11mat_csr, A22mat_csr, A12mat_csr;

   //------------------------------------------------------------------
   // extract information about the system matrix
   //------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   AStartRow = APartition_[mypid];
   AStartCol = AStartRow;
   ANRows    = APartition_[mypid+1] - AStartRow;
   ANCols    = ANRows;
   AGNRows   = APartition_[nprocs];
   AGNCols   = AGNRows;

   //------------------------------------------------------------------
   // calculate the dimensions of the 2 x 2 blocks
   //------------------------------------------------------------------

   A11NRows    = ANRows - P22Size_;
   A11NCols    = A11NRows;
   A11GNRows   = AGNRows - P22GSize_;
   A11GNCols   = A11GNRows;
   A11StartRow = AStartRow - P22Offsets_[mypid];
   A11StartCol = A11StartRow;

   A12NRows    = ANRows - P22Size_;
   A12NCols    = P22Size_;
   A12GNRows   = AGNRows - P22GSize_;
   A12GNCols   = P22GSize_;
   A12StartRow = AStartRow - P22Offsets_[mypid];
   A12StartCol = P22Offsets_[mypid];

   A22NRows    = P22Size_;
   A22NCols    = P22Size_;
   A22GNRows   = P22GSize_;
   A22GNCols   = P22GSize_;
   A22StartRow = P22Offsets_[mypid];
   A22StartCol = P22Offsets_[mypid];

   if ( outputLevel_ >= 1 )
   {
      printf("%4d HYPRE_LSI_Schur(1,1) : StartRow  = %d\n",mypid,A11StartRow);
      printf("%4d HYPRE_LSI_Schur(1,1) : GlobalDim = %d %d\n",mypid,A11GNRows, 
                                                   A11GNCols);
      printf("%4d HYPRE_LSI_Schur(1,1) : LocalDim  = %d %d\n",mypid,A11NRows, 
                                                   A11NCols);
      printf("%4d HYPRE_LSI_Schur(1,2) : StartRow  = %d\n",mypid,A12StartRow);
      printf("%4d HYPRE_LSI_Schur(1,2) : GlobalDim = %d %d\n",mypid,A12GNRows, 
                                                   A12GNCols);
      printf("%4d HYPRE_LSI_Schur(1,2) : LocalDim  = %d %d\n",mypid,A12NRows, 
                                                   A12NCols);
      printf("%4d HYPRE_LSI_Schur(2,2) : StartRow  = %d\n",mypid,A22StartRow);
      printf("%4d HYPRE_LSI_Schur(2,2) : GlobalDim = %d %d\n",mypid,A22GNRows, 
                                                   A22GNCols);
      printf("%4d HYPRE_LSI_Schur(2,2) : LocalDim  = %d %d\n",mypid,A22NRows, 
                                                   A22NCols);
   }

   //------------------------------------------------------------------
   // figure the row sizes of the block matrices
   //------------------------------------------------------------------

   A11RowLengs = new int[A11NRows];
   A12RowLengs = new int[A12NRows];
   A22RowLengs = new int[A22NRows];
   A11MaxRowLeng = 0;
   A12MaxRowLeng = 0;
   A22MaxRowLeng = 0;
   A11RowCnt = 0;
   A12RowCnt = 0;
   A22RowCnt = 0;
   HYPRE_IJMatrixGetObject(Amat, (void**) &Amat_csr);

   for ( irow = AStartRow; irow < AStartRow+ANRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Amat_csr, irow, &rowSize, &inds, &vals);
      searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
      if ( searchInd < 0 )   // A(1,1) or A(1,2) block
      {
         A11NewSize = A12NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = hypre_BinarySearch(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) A12NewSize++;
            else                A11NewSize++;
         }
         if ( A12NewSize == 0 ) A12NewSize = 1;
         A11RowLengs[A11RowCnt++] = A11NewSize;
         A12RowLengs[A12RowCnt++] = A12NewSize;
         A11MaxRowLeng = (A11NewSize > A11MaxRowLeng) ? 
                          A11NewSize : A11MaxRowLeng;
         A12MaxRowLeng = (A12NewSize > A12MaxRowLeng) ? 
                          A12NewSize : A12MaxRowLeng;
         if ( A11NewSize != 1 )
            printf("%4d HYPRE_LSI_Schur WARNING - A11 row length > 1 : %d\n",
                   irow);
      }
      else // A(2,2) block
      {
         A22NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = hypre_BinarySearch(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) A22NewSize++;
         }
         A22RowLengs[A22RowCnt++] = A22NewSize;
         A22MaxRowLeng = (A22NewSize > A22MaxRowLeng) ? 
                          A22NewSize : A22MaxRowLeng;
      }
      HYPRE_ParCSRMatrixRestoreRow(Amat_csr, irow, &rowSize, &inds, &vals);
   }

   //------------------------------------------------------------------
   // create matrix contexts for the blocks
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpiComm_, A11StartRow, A11StartRow+A11NRows-1,
                                A11StartCol, A11StartCol+A11NCols-1, &A11mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A11mat_, HYPRE_PARCSR);
   ierr  = HYPRE_IJMatrixSetRowSizes(A11mat_, A11RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A11mat_);
   assert(!ierr);
   delete [] A11RowLengs;
   ierr  = HYPRE_IJMatrixCreate(mpiComm_, A12StartRow, A12StartRow+A12NRows-1,
                                A12StartCol, A12StartCol+A12NCols-1, &A12mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A12mat_, HYPRE_PARCSR);
   ierr  = HYPRE_IJMatrixSetRowSizes(A12mat_, A12RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A12mat_);
   assert(!ierr);
   delete [] A12RowLengs;
   if ( A22MaxRowLeng > 0 )
   {
      ierr = HYPRE_IJMatrixCreate(mpiComm_,A22StartRow,A22StartRow+A22NRows-1,
                                A22StartCol, A22StartCol+A22NCols-1, &A22mat_);
      ierr += HYPRE_IJMatrixSetObjectType(A22mat_, HYPRE_PARCSR);
      ierr  = HYPRE_IJMatrixSetRowSizes(A22mat_, A22RowLengs);
      ierr += HYPRE_IJMatrixInitialize(A22mat_);
      assert(!ierr);
   }
   else A22mat_ = NULL;
   delete [] A22RowLengs;

   //------------------------------------------------------------------
   // load the matrices extracted from A
   //------------------------------------------------------------------

   A11_inds = new int[A11MaxRowLeng+1];
   A11_vals = new double[A11MaxRowLeng+1];
   A12_inds = new int[A12MaxRowLeng+1];
   A12_vals = new double[A12MaxRowLeng+1];
   A22_inds = new int[A22MaxRowLeng+1];
   A22_vals = new double[A22MaxRowLeng+1];

   A11RowCnt = A11StartRow;
   A12RowCnt = A12StartRow;
   A22RowCnt = A22StartRow;

   for ( irow = AStartRow; irow < AStartRow+ANRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Amat_csr, irow, &rowSize, &inds, &vals);
      searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
      if ( searchInd < 0 )   // A(1,1) or A(1,2) block
      {
         A11NewSize = A12NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = HYPRE_LSI_Search(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) // A(1,2) block 
            {
               A12_inds[A12NewSize] = searchInd;
               A12_vals[A12NewSize++] = vals[j];
            }
            else
            {
               searchInd = - searchInd - 1;
               if ( index == irow && vals[j] != 0.0 )
               {
                  A11_inds[A11NewSize] = index - searchInd;
                  A11_vals[A11NewSize++] = 1.0 / vals[j];
               }
            }
         }
         if ( A12NewSize == 0 )
         {
            A12_inds[0] = P22Offsets_[mypid];
            A12_vals[0] = 0.0;
            A12NewSize  = 1;
         }
         for ( k = 0; k < A11NewSize; k++ )
         {
            if ( A11_inds[k] < 0 || A11_inds[k] >= A11GNCols )
            {
               printf("%4d : A11 row %8d has invalid column %8d\n",mypid,
                      A11_inds[k]);
               exit(1);
            }
         }
         HYPRE_IJMatrixSetValues(A11mat_, 1, &A11NewSize, 
	                    (const int *) &A11RowCnt, (const int *) A11_inds,
                            (const double *) A11_vals);
         HYPRE_IJMatrixSetValues(A12mat_, 1, &A12NewSize, 
	                    (const int *) &A12RowCnt, (const int *) A12_inds,
                            (const double *) A12_vals);
         A11RowCnt++;
         A12RowCnt++;
      }
      else if ( A22MaxRowLeng > 0 ) // A(2,2) block
      {
         A22NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = hypre_BinarySearch(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) 
            {
               A22_inds[A22NewSize] = searchInd;
               A22_vals[A22NewSize++] = vals[j];
            }
         }
         if ( A22NewSize == 0 )
         {
            A22_inds[0] = P22Offsets_[mypid];
            A22_vals[0] = 0.0;
            A22NewSize  = 1;
         }
         HYPRE_IJMatrixSetValues(A22mat_, 1, &A22NewSize, 
	                    (const int *) &A22RowCnt, (const int *) A22_inds,
                            (const double *) A22_vals);
         A22RowCnt++;
      }
      HYPRE_ParCSRMatrixRestoreRow(Amat_csr, irow, &rowSize, &inds, &vals);
   }
   delete [] A11_inds;
   delete [] A11_vals;
   delete [] A12_inds;
   delete [] A12_vals;
   delete [] A22_inds;
   delete [] A22_vals;

   //------------------------------------------------------------------
   // finally assemble the matrix 
   //------------------------------------------------------------------

   ierr =  HYPRE_IJMatrixAssemble(A11mat_);
   ierr += HYPRE_IJMatrixGetObject(A11mat_, (void **) &A11mat_csr);
   assert( !ierr );
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A11mat_csr);
   ierr =  HYPRE_IJMatrixAssemble(A12mat_);
   ierr += HYPRE_IJMatrixGetObject(A12mat_, (void **) &A12mat_csr);
   assert( !ierr );
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A12mat_csr);
   if ( A22mat_ != NULL )
   {
      ierr =  HYPRE_IJMatrixAssemble(A22mat_);
      ierr += HYPRE_IJMatrixGetObject(A22mat_, (void **) &A22mat_csr);
      assert( !ierr );
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A22mat_csr);
   }
   else A22mat_csr = NULL;

   if ( outputLevel_ >= 3 )
   {
      sprintf( fname, "A11.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = A11StartRow; irow < A11StartRow+A11NRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A11mat_csr,irow,&rowSize,&inds,&vals);
         for ( j = 0; j < rowSize; j++ )
            printf(" %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
         HYPRE_ParCSRMatrixRestoreRow(A11mat_csr,irow,&rowSize,&inds,&vals);
      }
      fclose(fp);
      sprintf( fname, "A12.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = A12StartRow; irow < A12StartRow+A12NRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A12mat_csr,irow,&rowSize,&inds,&vals);
         for ( j = 0; j < rowSize; j++ )
            printf(" %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
         HYPRE_ParCSRMatrixRestoreRow(A12mat_csr,irow,&rowSize,&inds,&vals);
      }
      fclose(fp);
      if ( A22mat_csr != NULL )
      {
         sprintf( fname, "A22.%d", mypid);
         fp = fopen( fname, "w" );
         for ( irow = A22StartRow; irow < A22StartRow+A22NRows; irow++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(A22mat_csr,irow,&rowSize,&inds,&vals);
            for ( j = 0; j < rowSize; j++ )
               printf(" %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
            HYPRE_ParCSRMatrixRestoreRow(A22mat_csr,irow,&rowSize,&inds,&vals);
         }
         fclose(fp);
      }
   }
   return 0;
}

//******************************************************************************
// set up routine
//------------------------------------------------------------------------------

int HYPRE_LSI_Schur::setup(HYPRE_IJMatrix Amat,  HYPRE_IJVector sol,
                           HYPRE_IJVector rhs,   HYPRE_IJMatrix *rAmat,
                           HYPRE_IJVector *rsol, HYPRE_IJVector *rrhs,
                           HYPRE_IJVector *rres)
{
   int                j, irow, mypid, nprocs, rowSize, *colInd, one=1;
   int                maxRowSize, *colInd2, newRowSize, count, *newColInd;
   int                rowSize2, SNRows, SStartRow, *SRowLengs;
   int                V2Leng, V2Start, ierr;
   double             *colVal, *colVal2, *newColVal;
   HYPRE_IJVector     X2vec, R2vec;
   HYPRE_IJMatrix     Smat;
   HYPRE_ParCSRMatrix Amat_csr;
   HYPRE_ParCSRMatrix Cmat_csr, Mmat_csr, Smat_csr, A22mat_csr, RAP_csr;

   //------------------------------------------------------------------
   // build the blocks A11, A12, and the A22 block, if any
   //------------------------------------------------------------------

   if ( lookup_ == NULL )
   {
      printf("HYPRE_LSI_Schur ERROR : need lookup object.\n");
      exit(1);
   }
   if ( A11mat_        != NULL ) HYPRE_IJMatrixDestroy(A11mat_);
   if ( A12mat_        != NULL ) HYPRE_IJMatrixDestroy(A12mat_);
   if ( A22mat_        != NULL ) HYPRE_IJMatrixDestroy(A22mat_);
   if ( F1vec_         != NULL ) HYPRE_IJVectorDestroy(F1vec_);
   if ( APartition_    != NULL ) free( APartition_ );
   if ( P22LocalInds_  != NULL ) delete [] P22LocalInds_;
   if ( P22GlobalInds_ != NULL ) delete [] P22GlobalInds_;
   if ( P22Offsets_    != NULL ) delete [] P22Offsets_;
   F1vec_         = NULL;
   P22LocalInds_  = NULL;
   P22GlobalInds_ = NULL;
   P22Offsets_    = NULL;
   assembled_     = 0;
   HYPRE_IJMatrixGetObject( Amat, (void**) &Amat_csr );
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_csr, &APartition_ );
   HYPRE_ParCSRMatrixGetComm( Amat_csr, &mpiComm_ );
   ierr = computeBlockInfo();
   if ( ierr ) return ierr;
   buildBlocks(Amat);
   MPI_Comm_rank(mpiComm_, &mypid);

   //------------------------------------------------------------------
   // create Pressure Poisson matrix (T = C^T M^{-1} C)
   //------------------------------------------------------------------
   
   if (outputLevel_ >= 1) 
      printf("%4d : HYPRE_LSI_Schur setup : C^T M^{-1} C begins\n", mypid);

   HYPRE_IJMatrixGetObject(A11mat_, (void **) &Mmat_csr);
   HYPRE_IJMatrixGetObject(A12mat_, (void **) &Cmat_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Mmat_csr);
   hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) Cmat_csr,
                                       (hypre_ParCSRMatrix *) Mmat_csr,
                                       (hypre_ParCSRMatrix *) Cmat_csr,
                                       (hypre_ParCSRMatrix **) &RAP_csr);

   if (outputLevel_ >= 1) 
      printf("%4d : HYPRE_LSI_Schur setup : C^T M^{-1} C ends\n", mypid);

   //------------------------------------------------------------------
   // construct new S = A22 - RAP (A22 may be null)
   //------------------------------------------------------------------

   SNRows    = P22Size_;
   SStartRow = P22Offsets_[mypid];
   ierr  = HYPRE_IJMatrixCreate(mpiComm_, SStartRow, SStartRow+SNRows-1,
			        SStartRow, SStartRow+SNRows-1, &Smat);
   ierr += HYPRE_IJMatrixSetObjectType(Smat, HYPRE_PARCSR);
   assert(!ierr);
   if ( A22mat_ != NULL )
      HYPRE_IJMatrixGetObject(A22mat_, (void **) &A22mat_csr);

   SRowLengs = new int[SNRows];
   maxRowSize = 0;
   for ( irow = SStartRow; irow < SStartRow+SNRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(RAP_csr,irow,&rowSize,&colInd,NULL);
      newRowSize = rowSize;
      if ( A22mat_csr != NULL )
      {
         HYPRE_ParCSRMatrixGetRow(A22mat_csr,irow,&rowSize2,&colInd2,NULL);
         newRowSize += rowSize2;
         newColInd = new int[newRowSize];
         for (j = 0; j < rowSize;  j++) newColInd[j] = colInd[j];
         for (j = 0; j < rowSize2; j++) newColInd[j+rowSize] = colInd2[j];
         qsort0(newColInd, 0, newRowSize-1);
         count = 0;
         for ( j = 1; j < newRowSize; j++ )
         {
            if ( newColInd[j] != newColInd[count] )
            {
               count++;
               newColInd[count] = newColInd[j];
            }
         }
         if ( newRowSize > 0 ) count++;
         newRowSize = count;
         HYPRE_ParCSRMatrixRestoreRow(A22mat_csr,irow,&rowSize2,&colInd2,NULL);
         delete [] newColInd;
      }
      SRowLengs[irow-SStartRow] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      HYPRE_ParCSRMatrixRestoreRow(RAP_csr,irow,&rowSize,&colInd,NULL);
   }
   ierr  = HYPRE_IJMatrixSetRowSizes(Smat, SRowLengs);
   ierr += HYPRE_IJMatrixInitialize(Smat);
   assert(!ierr);
   delete [] SRowLengs;

   for ( irow = SStartRow; irow < SStartRow+SNRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(RAP_csr,irow,&rowSize,&colInd,&colVal);
      if ( A22mat_csr == NULL )
      {
         newRowSize = rowSize;
         newColInd  = new int[newRowSize];
         newColVal  = new double[newRowSize];
         for (j = 0; j < rowSize; j++) 
         {
            newColInd[j] = colInd[j];
            newColVal[j] = - colVal[j];
         }
      }
      else
      {
         HYPRE_ParCSRMatrixGetRow(A22mat_csr,irow,&rowSize2,&colInd2,&colVal2);
         newRowSize = rowSize + rowSize2;
         newColInd = new int[newRowSize];
         newColVal = new double[newRowSize];
         for (j = 0; j < rowSize; j++) 
         {
            newColInd[j] = colInd[j];
            newColVal[j] = - colVal[j];
         }
         for (j = 0; j < rowSize2; j++) 
         {
            newColInd[j+rowSize] = colInd2[j];
            newColVal[j+rowSize] = colVal2[j];
         }
         qsort1(newColInd, newColVal, 0, newRowSize-1);
         count = 0;
         for ( j = 1; j < newRowSize; j++ )
         {
            if ( newColInd[j] != newColInd[count] )
            {
               count++;
               newColInd[count] = newColInd[j];
               newColVal[count] = newColVal[j];
            }
            else newColVal[count] += newColVal[j];
         }
         if ( newRowSize > 0 ) count++;
         newRowSize = count;
         HYPRE_ParCSRMatrixRestoreRow(A22mat_csr,irow,&rowSize2,
                                      &colInd2,&colVal2);
      }
      HYPRE_IJMatrixSetValues(Smat, 1, &newRowSize, (const int *) &irow,
	                  (const int *) newColInd, (const double *) newColVal);
      HYPRE_ParCSRMatrixRestoreRow(RAP_csr,irow,&rowSize,&colInd,&colVal);
      delete [] newColInd;
      delete [] newColVal;
   }
   HYPRE_IJMatrixAssemble(Smat);
   HYPRE_IJMatrixGetObject(Smat, (void **) &Smat_csr);
   (*rAmat) = Smat;
   Smat_ = Smat;
   assembled_ = 1;

   //------------------------------------------------------------------
   // build new right hand side vectors
   //------------------------------------------------------------------

   computeRHS(rhs, rrhs);
   Svec_ = (*rrhs);

   //------------------------------------------------------------------
   // construct the new solution and residual vectors 
   //------------------------------------------------------------------

   V2Leng  = P22Size_;
   V2Start = P22Offsets_[mypid];
   HYPRE_IJVectorCreate(mpiComm_, V2Start, V2Start+V2Leng-1, &X2vec);
   HYPRE_IJVectorSetObjectType(X2vec, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(X2vec);
   ierr += HYPRE_IJVectorAssemble(X2vec);
   assert(!ierr);
   HYPRE_IJVectorCreate(mpiComm_, V2Start, V2Start+V2Leng-1, &R2vec);
   HYPRE_IJVectorSetObjectType(R2vec, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(R2vec);
   ierr += HYPRE_IJVectorAssemble(R2vec);
   assert(!ierr);
   (*rsol) = X2vec;
   (*rres) = R2vec;
   return 0;
}

//******************************************************************************
// set up rhs routine
//------------------------------------------------------------------------------

int HYPRE_LSI_Schur::computeRHS(HYPRE_IJVector rhs, HYPRE_IJVector *rrhs)
{
   int                mypid, nprocs, AStart, AEnd, ANRows, V1Leng, V1Start;
   int                V2Leng, V2Start, ierr, irow, f1Ind, f2Ind, searchInd;
   int                rowSize, *colInd;
   double             *colVal, ddata;
   HYPRE_ParVector    F1_csr, F2_csr;
   HYPRE_IJVector     F2vec, R2vec, X2vec; 
   HYPRE_ParCSRMatrix A11_csr, C_csr; 

   //------------------------------------------------------------------
   // error checking
   //------------------------------------------------------------------

   if ( assembled_ == 0 ) return 1;

   //------------------------------------------------------------------
   // get machine and matrix information 
   //------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   AStart = APartition_[mypid];
   AEnd   = APartition_[mypid+1] - 1;
   ANRows = AEnd - AStart + 1;
   HYPRE_IJMatrixGetObject(A11mat_, (void**) &A11_csr);
    
   //------------------------------------------------------------------
   // construct the reduced right hand side
   //------------------------------------------------------------------

   V1Leng  = ANRows - P22Size_;
   V1Start = AStart - P22Offsets_[mypid];
   HYPRE_IJVectorCreate(mpiComm_, V1Start, V1Start+V1Leng-1, &F1vec_);
   HYPRE_IJVectorSetObjectType(F1vec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(F1vec_);
   ierr += HYPRE_IJVectorAssemble(F1vec_);
   assert(!ierr);
   V2Leng  = P22Size_;
   V2Start = P22Offsets_[mypid];
   HYPRE_IJVectorCreate(mpiComm_, V2Start, V2Start+V2Leng-1, &F2vec);
   HYPRE_IJVectorSetObjectType(F2vec, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(F2vec);
   ierr += HYPRE_IJVectorAssemble(F2vec);
   assert(!ierr);

   f1Ind = V1Start;
   f2Ind = V2Start;
   for ( irow = AStart; irow <= AEnd; irow++ )
   {
      searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
      HYPRE_IJVectorGetValues(rhs, 1, &irow, &ddata);
      if ( searchInd < 0 )
      {
         HYPRE_ParCSRMatrixGetRow(A11_csr,f1Ind,&rowSize,&colInd,&colVal);
         ddata *= colVal[0];
         ierr = HYPRE_IJVectorSetValues(F1vec_, 1, (const int *) &f1Ind,
                        (const double *) &ddata);
         HYPRE_ParCSRMatrixRestoreRow(A11_csr,f1Ind,&rowSize,&colInd,&colVal);
         assert( !ierr );
         f1Ind++;
      }
      else
      {
         ierr = HYPRE_IJVectorSetValues(F2vec, 1, (const int *) &f2Ind,
                        (const double *) &ddata);
         assert( !ierr );
         f2Ind++;
      }
   }
   HYPRE_IJVectorGetObject(F1vec_, (void**) F1_csr);
   HYPRE_IJVectorGetObject(F2vec, (void**) F2_csr);
   HYPRE_IJMatrixGetObject(A12mat_, (void**) C_csr);
   HYPRE_ParCSRMatrixMatvecT( -1.0, C_csr, F1_csr, 1.0, F2_csr );
   (*rrhs) = F2vec;

   return 0;
}

//******************************************************************************
// compute the long solution 
//------------------------------------------------------------------------------

int HYPRE_LSI_Schur::computeSol(HYPRE_IJVector X2vec, HYPRE_IJVector Xvec)
{
   int                AStart, ANRows, AEnd, irow, searchInd, ierr;
   int                mypid, nprocs, V1Leng, V1Start, V1Cnt, V2Cnt;
   double             *xvals;
   HYPRE_IJVector     X1vec;
   HYPRE_ParVector    X1_csr, X2_csr, F1_csr;
   HYPRE_ParCSRMatrix C_csr, M_csr;

   //------------------------------------------------------------------
   // check for errors
   //------------------------------------------------------------------

   if ( assembled_ != 1 ) return 1;

   //------------------------------------------------------------------
   // extract matrix and machine information
   //------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   AStart  = APartition_[mypid];
   AEnd    = APartition_[mypid+1];
   ANRows  = AEnd - AStart;
   HYPRE_IJVectorGetObject(X2vec,   (void**) X2_csr);
   HYPRE_IJVectorGetObject(F1vec_,  (void**) F1_csr);
   HYPRE_IJMatrixGetObject(A12mat_, (void**) C_csr);
   HYPRE_IJMatrixGetObject(A11mat_, (void**) M_csr);

   //------------------------------------------------------------------
   // construct temporary vector
   //------------------------------------------------------------------

   V1Leng  = ANRows - P22Size_;
   V1Start = AStart - P22Offsets_[mypid];
   HYPRE_IJVectorCreate(mpiComm_, V1Start, V1Start+V1Leng-1, &X1vec);
   HYPRE_IJVectorSetObjectType(X1vec, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(X1vec);
   ierr += HYPRE_IJVectorAssemble(X1vec);
   assert(!ierr);

   //------------------------------------------------------------------
   // recover X1 
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixMatvec( -1.0, C_csr, X2_csr, 1.0, F1_csr );
   HYPRE_ParCSRMatrixMatvec(  1.0, M_csr, F1_csr, 0.0, X1_csr );

   //------------------------------------------------------------------
   // merge X1 and X2 to the unreduced solution vector
   //------------------------------------------------------------------

   V1Cnt = AStart - P22Offsets_[mypid];
   V2Cnt = P22Offsets_[mypid];
   xvals = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector*)Xvec));
   for ( irow = AStart; irow < AEnd; irow++ ) 
   {
      searchInd = hypre_BinarySearch( P22LocalInds_, irow, P22Size_);
      if ( searchInd >= 0 )
      {
         ierr = HYPRE_IJVectorGetValues(X2vec, 1, &V2Cnt, &xvals[irow-AStart]);
         assert( !ierr );
         V2Cnt++;
      }
      else
      {
         ierr = HYPRE_IJVectorGetValues(X1vec, 1, &V1Cnt, &xvals[irow-AStart]);
         assert( !ierr );
         V1Cnt++;
      }
   } 

   //------------------------------------------------------------------
   // clean up and return
   //------------------------------------------------------------------

   HYPRE_IJVectorDestroy(X1vec);
   return 0;
}

//******************************************************************************
// print the matrix and right hand side
//------------------------------------------------------------------------------

int HYPRE_LSI_Schur::print()
{
   int      mypid, irow, j, nnz, V2Leng, V2Start, rowSize, *colInd; 
   double   *colVal, ddata;
   FILE     *fp;
   char     fname[100];
   HYPRE_ParCSRMatrix S_csr;

   if ( ! assembled_ ) return 1;
   MPI_Comm_rank( mpiComm_, &mypid );
   sprintf(fname, "hypre_mat.out.%d", mypid);
   fp      = fopen( fname, "w");
   nnz     = 0;
   V2Leng  = P22Size_;
   V2Start = P22Offsets_[mypid];
   HYPRE_IJMatrixGetObject( Smat_, (void**) S_csr );
   for ( irow = V2Start; irow < V2Start+V2Leng; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(S_csr,irow,&rowSize,&colInd,&colVal);
      for ( j = 0; j < rowSize; j++ ) if ( colVal[j] != 0.0 ) nnz++;
      HYPRE_ParCSRMatrixRestoreRow(S_csr,irow,&rowSize,&colInd,&colVal);
   }
   fprintf(fp, "%6d  %7d \n", V2Leng, nnz);
   for ( irow = V2Start; irow < V2Start+V2Leng; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(S_csr,irow,&rowSize,&colInd,&colVal);
      for ( j = 0; j < rowSize; j++ )
      {
         if ( colVal[j] != 0.0 )
            fprintf(fp, "%6d  %6d  %25.16e \n",irow+1,colInd[j]+1,colVal[j]);
      }
      HYPRE_ParCSRMatrixRestoreRow(S_csr,irow,&rowSize,&colInd,&colVal);
   }
   fclose(fp);
   sprintf(fname, "hypre_rhs.out.%d", mypid);
   fp = fopen( fname, "w");
   fprintf(fp, "%6d \n", V2Leng);
   for ( irow = V2Start; irow < V2Start+V2Leng; irow++ )
   {
      HYPRE_IJVectorGetValues(Svec_, 1, &irow, &ddata);
      fprintf(fp, "%6d  %25.16e \n", irow+1, ddata);
   }
   fclose(fp);
   MPI_Barrier(mpiComm_);
   return 0;
}

