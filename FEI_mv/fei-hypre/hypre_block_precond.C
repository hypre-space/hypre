/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>
#include <assert.h>

#include "HYPRE.h"
#include "utilities/utilities.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../parcsr_mv/HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/HYPRE_parcsr_ls.h"

#define dabs(x) ((x > 0) ? x : -(x))

//---------------------------------------------------------------------------
// These are external functions needed internally here
//---------------------------------------------------------------------------

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

HYPRE_LSI_BlockPrecond::HYPRE_LSI_BlockPrecond(HYPRE_ParCSRMatrix Amat)
{
   Amat_          = Amat;
   APartition_    = NULL;
   P22LocalInds_  = NULL;
   P22GlobalInds_ = NULL;
   P22Offsets_    = NULL;
   P22Size_       = -1;
   P22GSize_      = -1;
   assembled_     = 0;
   outputLevel_   = 0;
   A11mat_        = NULL;
   A12mat_        = NULL;
   A22mat_        = NULL;
}

//******************************************************************************
// destructor
//------------------------------------------------------------------------------

HYPRE_LSI_BlockPrecond::~HYPRE_LSI_BlockPrecond()
{
   if ( APartition_    != NULL ) free( APartition_ );
   if ( P22LocalInds_  != NULL ) delete [] P22LocalInds_;
   if ( P22GlobalInds_ != NULL ) delete [] P22GlobalInds_;
   if ( P22Offsets_    != NULL ) delete [] P22Offsets_;
   if ( A11mat_        != NULL ) HYPRE_ParCSRMatrixDestroy(Amat11_);
   if ( A12mat_        != NULL ) HYPRE_ParCSRMatrixDestroy(Amat12_);
   if ( A22mat_        != NULL ) HYPRE_ParCSRMatrixDestroy(Amat22_);
}

//******************************************************************************
// Given a matrix A, compute the sizes and indices of the 2 x 2 blocks
// (P22Size_, P22GSize_, P22LocalInds_, P22GlobalInds_, P22Offsets_, APartition_)
//------------------------------------------------------------------------------

void HYPRE_LSI_BlockPrecond::computeBlockInfo()
{
   int      mypid, nprocs, start_row, end_row, local_nrows, irow;
   int      j, row_size, *col_ind, *disp_array;
   double   *col_val;
   MPI_Comm mpi_comm;

   //------------------------------------------------------------------
   // check that the system matrix has been set, clean up previous
   // allocations, and extract matrix information
   //------------------------------------------------------------------

   if ( Amat_ == NULL )
   {
      printf("BlockPrecond ERROR : Amat not initialized.\n");
      exit(1);
   }
   if ( APartition_    != NULL ) free( APartition_ );
   if ( P22LocalInds_  != NULL ) delete [] P22LocalInds_;
   if ( P22GlobalInds_ != NULL ) delete [] P22GlobalInds_;
   if ( P22Offsets_    != NULL ) delete [] P22Offsets_;
   APartition_    = NULL;
   P22LocalInds_  = NULL;
   P22GlobalInds_ = NULL;
   P22Offsets_    = NULL;
   assembled_     = 0;
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_, &Apartition_ );
   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   start_row   = APartition_[mypid];
   end_row     = APartition_[mypid+1] - 1;
   local_nrows = end_row - start_row + 1;
    
   //------------------------------------------------------------------
   // find the local size of the (2,2) block
   //------------------------------------------------------------------

   P22Size_ = 0;
   for ( irow = start_row; irow <= end_row; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Amat_, irow, &row_size, &col_ind, &col_val);
      for ( j = 0; j < row_size; j++ ) 
      {
         index = colInd[j];
         if ( index == irow ) break;
      }
      if ( j == row_size ) P22Size_++;
      HYPRE_ParCSRMatrixRestoreRow(Amat_, irow, &row_size, &col_ind, &col_val);
   }

   if ( outputLevel_ > 0 )
   {
      printf("%4d computeBlockInfo : P22_size = %d\n", mypid, P22Size_);
   }

   //------------------------------------------------------------------
   // allocate array for storing indices of (2,2) block variables 
   //------------------------------------------------------------------

   if ( P22Size_ > 0 ) P22LocalInds_ = new int[P22Size_];
   else                P22LocalInds_ = NULL; 

   //------------------------------------------------------------------
   // compose a local list of rows for the (2,2) block
   //------------------------------------------------------------------

   P22Size_ = 0;
   for ( irow = start_row; irow <= end_row; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Amat, irow, &row_size, &col_ind, &col_val);
      for ( j = 0; j < row_size; j++ ) 
      {
         index = colInd[j];
         if ( index == irow ) break;
      }
      if ( j == row_size ) P22LocalInds_[P22Size_++] = irow;
      HYPRE_ParCSRMatrixRestoreRow(Amat, irow, &row_size, &col_ind, &col_val);
   }

   //------------------------------------------------------------------
   // compose a global list of rows for the (2,2) block
   //------------------------------------------------------------------

    MPI_Allreduce(&P22Size_, &P22GSize_, 1, MPI_INT, MPI_SUM, mpi_comm);

    if ( outputLevel_ > 0 )
    {
       if ( P22GSize_ == 0 && mypid == 0 )
          printf("computeBlockInfo WARNING : P22Size = 0 on all processors.\n");
    }
    if ( P22GSize_ == 0 )
    {
       delete [] APartition_;
       APartition_ = NULL;
       return;
    }

    if ( P22GSize_ > 0 ) P22GlobalInds_ = new int[P22GSize_];
    else                 P22GlobalInds_ = NULL;
    disp_array     = new int[nprocs];
    MPI_Allgather(&P22Size_, 1, MPI_INT, P22Offsets_, 1, MPI_INT, mpi_comm);
    disp_array[0] = 0;
    for ( j = 1; j < nprocs; j++ ) 
       disp_array[i] = disp_array[i-1] + P22Offsets_[i-1];
    MPI_Allgatherv(P22LocalInds_, P22Size_, MPI_INT, P22GlobalInds_,
                   P22Offsets_, disp_array, MPI_INT, comm);
    delete [] disp_array;

    if ( outputLevel_ > 1 )
    {
       for ( j = 0; j < P22Size_; j++ )
          printf("%4d computeBlockInfo : P22Inds %8d = %d\n", mypid,
                 j, P22LocalInds_[j]);
    }
    return;
} 

//******************************************************************************
// Given a matrix A, build the 2 x 2 blocks
// (This function is to be called after computeBlockInfo
//------------------------------------------------------------------------------

void HYPRE_LSI_BlockPrecond::buildBlocks()
{
   int    mypid, nprocs, *partition, AStartRow, ANRows;
   int    ANRows, ANCols, AGNRows, AGNcols, AStartRow, AStartCol;
   int    A11NRows, A11NCols, A11GNRows, A11GNcols, A11StartRow, A11StartCol;
   int    A12NRows, A12NCols, A12GNRows, A12GNcols, A12StartRow, A12StartCol;
   int    A22NRows, A22NCols, A22GNRows, A22GNcols, A22StartRow, A22StartCol;
   int    *A11RowLengs, A11MaxRowLeng, A11RowCnt, A11NewSize, *A11_inds;
   int    *A12RowLengs, A12MaxRowLeng, A12RowCnt, A12NewSize, *A12_inds;
   int    *A22RowLengs, A22MaxRowLeng, A22RowCnt, A22NewSize, *A22_inds;
   int    irow, j, rowSize, *inds;
   double *vals, *A11_vals, *A12_vals, *A22_vals;

   //------------------------------------------------------------------
   // extract information about the system matrix
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_, &partition );
   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   AStartRow = partition[mypid];
   AStartCol = AStartRow;
   ANRows    = partition[mypid+1] - AStartRow;
   ANCols    = ANRows;
   AGNRows   = partition[nprocs];
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
   A12GNRows   = AGNrows - P22GSize_;
   A12GNCols   = P22GSize_;
   A12StartRow = AStartRow - P22Offsets_[mypid];
   A12StartCol = P22Offsets_[mypid];

   A22NRows    = P22Size_;
   A22NCols    = P22Size_;
   A22GNRows   = P22GSize_;
   A22GNCols   = P22GSize_;
   A22StartRow = P22Offsets_[mypid];
   A22StartCol = P22Offsets_[mypid];

   if ( outputLevel > 1 )
   {
      printf("%4d buildBlock (1,1) : StartRow  = %d\n", mypid, A11StartRow);
      printf("%4d buildBlock (1,1) : GlobalDim = %d %d\n", mypid, A11GNRows, 
                                                           A11GNCols);
      printf("%4d buildBlock (1,1) : LocalDim  = %d %d\n", mypid, A11NRows, 
                                                           A11NCols);
      printf("%4d buildBlock (1,2) : StartRow  = %d\n", mypid, A12StartRow);
      printf("%4d buildBlock (1,2) : GlobalDim = %d %d\n", mypid, A12GNRows, 
                                                           A12GNCols);
      printf("%4d buildBlock (1,2) : LocalDim  = %d %d\n", mypid, A12NRows, 
                                                           A12NCols);
      printf("%4d buildBlock (2,2) : StartRow  = %d\n", mypid, A22StartRow);
      printf("%4d buildBlock (2,2) : GlobalDim = %d %d\n", mypid, A22GNRows, 
                                                           A22GNCols);
      printf("%4d buildBlock (2,2) : LocalDim  = %d %d\n", mypid, A22NRows, 
                                                           A22NCols);
   }

   //------------------------------------------------------------------
   // create matrix contexts for the blocks
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpi_comm, A11StartRow, A11StartRow+A11NRows-1,
                                A11StartCol, A11StartCol+A11NCols-1, &IJA11mat);
   ierr += HYPRE_IJMatrixSetObjectType(IJA11mat, HYPRE_PARCSR);
   assert(!ierr);
   ierr  = HYPRE_IJMatrixCreate(mpi_comm, A12StartRow, A12StartRow+A12NRows-1,
                                A12StartCol, A12StartCol+A12NCols-1, &IJA12mat);
   ierr += HYPRE_IJMatrixSetObjectType(IJA12mat, HYPRE_PARCSR);
   assert(!ierr);
   ierr  = HYPRE_IJMatrixCreate(mpi_comm, A22StartRow, A22StartRow+A22NRows-1,
                                A22StartCol, A22StartCol+A22NCols-1, &IJA22mat);
   ierr += HYPRE_IJMatrixSetObjectType(IJA22mat, HYPRE_PARCSR);
   assert(!ierr);

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

   for ( irow = AStartRow; irow < AStartRow+ANRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Amat_, irow, &rowSize, &inds, &vals);
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
         if ( A11NewSize <= 0 ) A11NewSize = 1;
         if ( A12NewSize <= 0 ) A12NewSize = 1;
         A11RowLengs[A11RowCnt++] = A11NewSize;
         A12RowLengs[A12RowCnt++] = A12NewSize;
         A11MaxRowLeng = (A11NewSize > A11MaxRowLeng) ? A11NewSize : A11MaxRowLeng;
         A12MaxRowLeng = (A12NewSize > A12MaxRowLeng) ? A12NewSize : A12MaxRowLeng;
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
         if ( A22NewSize <= 0 ) A22NewSize = 1;
         A22RowLengs[A22RowCnt++] = A22NewSize;
         A22MaxRowLeng = (A22NewSize > A22MaxRowLeng) ? A22NewSize : A22MaxRowLeng;
      }
      HYPRE_ParCSRMatrixRestoreRow(Amat_, irow, &rowSize, &inds, &vals);
   }
   ierr  = HYPRE_IJMatrixSetRowSizes(IJA11mat, A11RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A11mat);
   assert(!ierr);
   ierr  = HYPRE_IJMatrixSetRowSizes(IJA12mat, A12RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A12mat);
   assert(!ierr);
   ierr  = HYPRE_IJMatrixSetRowSizes(IJA22mat, A22RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A22mat);
   assert(!ierr);
   delete [] A11RowLengs;
   delete [] A12RowLengs;
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

   for ( irow = AStartRow; irow < AStartRow+ANRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Amat_, irow, &rowSize, &inds, &vals);
      searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
      if ( searchInd < 0 )   // A(1,1) or A(1,2) block
      {
         A11NewSize = A12NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = hypre_BinarySearch(P22GlobalInds_,index,P22GSize_);
            if (search_ind >= 0) // A(1,2) block 
            {
               A12_inds[A12NewSize] = searchInd;
               A12_vals[A12NewSize++] = vals[j];
            }
            else
            {
               A11_inds[A11NewSize] = searchInd;
               A11_vals[A11NewSize++] = vals[j];
            }
         }
         if ( A11NewSize == 0 )
         {
            A11_inds[0] = AStartRow - P22Offsets_[mypid];
            A11_vals[0] = 0.0;
            A11NewSize  = 1;
         }
         if ( A12NewSize == 0 )
         {
            A12_inds[0] = P22Offsets_[mypid];
            A12_vals[0] = 0.0;
            A12NewSize  = 1;
         }
         HYPRE_IJMatrixSetValues(IJA11mat, 1, &A11NewSize, 
	                    (const int *) &A11RowCnt, (const int *) A11_inds,
                            (const double *) A11_vals);
         HYPRE_IJMatrixSetValues(IJA12mat, 1, &A12NewSize, 
	                    (const int *) &A12RowCnt, (const int *) A12_inds,
                            (const double *) A12_vals);
         A11RowCnt++;
         A12RowCnt++;
      }
      else // A(2,2) block
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
         HYPRE_IJMatrixSetValues(IJA22mat, 1, &A22NewSize, 
	                    (const int *) &A22RowCnt, (const int *) A22_inds,
                            (const double *) A22_vals);
      }
      HYPRE_ParCSRMatrixRestoreRow(Amat_, irow, &rowSize, &inds, &vals);
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

   ierr =  HYPRE_IJMatrixAssemble(IJA11mat);
   ierr += HYPRE_IJMatrixGetObject(IJA11mat, (void **) &A11mat_);
   assert( !ierr );
   ierr =  HYPRE_IJMatrixAssemble(IJA12mat);
   ierr += HYPRE_IJMatrixGetObject(IJA12mat, (void **) &A12mat_);
   assert( !ierr );
   ierr =  HYPRE_IJMatrixAssemble(IJA22mat);
   ierr += HYPRE_IJMatrixGetObject(IJA22mat, (void **) &A22mat_);
   assert( !ierr );
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A11mat_);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A12mat_);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A22mat_);

   if ( outputLevel_ >= 3 )
   {
      sprintf( fname, "A11.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = A11StartRow; irow < A11StartRow+A11NRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A11mat_,irow,&rowSize,&inds,&vals);
         for ( j = 0; j < rowSize; j++ )
            printf(" %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
         HYPRE_ParCSRMatrixRestoreRow(A11mat_,irow,&rowSize,&inds,&vals);
      }
      fclose(fp);
      sprintf( fname, "A12.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = A12StartRow; irow < A12StartRow+A12NRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A12mat_,irow,&rowSize,&inds,&vals);
         for ( j = 0; j < rowSize; j++ )
            printf(" %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
         HYPRE_ParCSRMatrixRestoreRow(A12mat_,irow,&rowSize,&inds,&vals);
      }
      fclose(fp);
      sprintf( fname, "A22.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = A22StartRow; irow < A22StartRow+A22NRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A22mat_,irow,&rowSize,&inds,&vals);
         for ( j = 0; j < rowSize; j++ )
            printf(" %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
         HYPRE_ParCSRMatrixRestoreRow(A22mat_,irow,&rowSize,&inds,&vals);
      }
      fclose(fp);
   }
   return;
}

//******************************************************************************
// set up routine
//------------------------------------------------------------------------------

void HYPRE_LSI_BlockPrecond::setup()
{
   computeBlockInfo();
   buildBlocks();

   //------------------------------------------------------------------
   // manipulate the A22 block to create an approximate inverse
   //------------------------------------------------------------------

   
    if ( outputLevel >= 1 )
    {
       printf("%4d BlockPrecond setup : P^T A P begins\n",mypid_);
    }
    hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix *) M_csr,
                                     (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix **) &S_csr);
    if ( outputLevel >= 1 )
    {
       printf("%4d BlockPrecond setup : P^T A P ends\n",mypid_);
    }
    return;
}

//******************************************************************************
// solve 
//------------------------------------------------------------------------------

void HYPRE_LSI_BlockPrecond::solve(HYPRE_IJVector xvec, HYPRE_IJVector fvec)
{
   int             A11Start, A11NRows, A22Start, A22NRows, AStart, ANrows;
   int             AEnd, A11Cnt, A22Cnt, *inds, irow, searchInd, ierr;
   double          *vals;
   MPI_Comm        mpi_comm;
   HYPRE_IJVector  f1, f2;
   HYPRE_ParVector f1_csr, f2_csr;

   //------------------------------------------------------------------
   // create new subvectors
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   A11Start = APartition_[mypid] - P22Offsets_[mypid];
   A11NRows = APartition_[mypid+1] - P22Offsets[mypid+1] - A11Start; 
   HYPRE_IJVectorCreate(mpi_comm, A11Start, A11Start+A11NRows-1, &f1);
   HYPRE_IJVectorSetObjectType(f1, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(f1);
   ierr += HYPRE_IJVectorAssemble(f1);
   assert(!ierr);

   A22Start = P22Offsets_[mypid];
   A22NRows = P22Offsets_[mypid+1] - A22Start;
   HYPRE_IJVectorCreate(comm_, A22Start, A22Start+A22NRows-1, &f2);
   HYPRE_IJVectorSetObjectType(f2, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(f2);
   ierr += HYPRE_IJVectorAssemble(f2hat);
   assert(!ierr);

   //------------------------------------------------------------------
   // extract the subvectors
   //------------------------------------------------------------------

   AStart = APartition_[mypid];
   AEnd   = APartition_[mypid+1];
   ANRows = AEnd - AStart;
   inds   = new int[ANRows];
   vals   = new double[ANRows];
   for ( irow = AStart; irow < Aend; irow++ ) inds[irow-AStart] = irow;
   HYPRE_IJVectorGetValues(fvec, ANRows, inds, vals);
   A11Cnt = A11Start;
   A22Cnt = A22Start;
   for ( irow = AStart; irow < Aend; irow++ ) 
   {
      searchInd = hypre_BinarySearch( P22LocalInds_, irow, P22Size_);
      if ( searchInd >= 0 )
      {
         ierr = HYPRE_IJVectorSetValues(f2, 1, (const int *) &A22Cnt,
		                        (const double *) &vals[irow]);
         assert( !ierr );
         A22Cnt++;
      }
      else
      {
         ierr = HYPRE_IJVectorSetValues(f1, 1, (const int *) &A11Cnt,
		                        (const double *) &vals[irow]);
         assert( !ierr );
         A11Cnt++;
      }
   } 
   delete [] inds;
   vals [] inds;
        
   //------------------------------------------------------------------
   // fetch parcsr subvectors
   //------------------------------------------------------------------

   HYPRE_IJVectorGetObject(f1, (void **) &f1_csr);
   HYPRE_IJVectorGetObject(f2, (void **) &f2_csr);

   //------------------------------------------------------------------
   // final clean up
   //------------------------------------------------------------------

   HYPRE_IJVectorDestroy( f1 );
   HYPRE_IJVectorDestroy( f2 );
   HYPRE_ParVectorDestroy( f1_csr );
   HYPRE_ParVectorDestroy( f2_csr );
}

