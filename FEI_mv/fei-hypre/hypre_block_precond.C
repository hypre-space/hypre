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
#include "hypre_block_precond.h"

#define dabs(x) ((x > 0) ? x : -(x))

//---------------------------------------------------------------------------
// These are external functions needed internally here
//---------------------------------------------------------------------------

extern "C" {
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix**);
}

//******************************************************************************
// Constructor
//------------------------------------------------------------------------------

HYPRE_IncFlow_BlockPrecond::HYPRE_IncFlow_BlockPrecond(HYPRE_IJMatrix Amat)
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
   diffusionCoef_ = 0.0;
   timeStep_      = 0.0;
   M22Diag_       = NULL;
   M22Length_     = 0;
   scheme_        = HYPRE_INCFLOW_BDIAG;
   A11Solver_     = NULL;
   A11Precond_    = NULL;
   A22Solver_     = NULL;
   A22Precond_    = NULL;
}

//******************************************************************************
// destructor
//------------------------------------------------------------------------------

HYPRE_IncFlow_BlockPrecond::~HYPRE_IncFlow_BlockPrecond()
{
   if ( APartition_    != NULL ) free( APartition_ );
   if ( P22LocalInds_  != NULL ) delete [] P22LocalInds_;
   if ( P22GlobalInds_ != NULL ) delete [] P22GlobalInds_;
   if ( P22Offsets_    != NULL ) delete [] P22Offsets_;
   if ( A11mat_        != NULL ) HYPRE_IJMatrixDestroy(A11mat_);
   if ( A12mat_        != NULL ) HYPRE_IJMatrixDestroy(A12mat_);
   if ( A22mat_        != NULL ) HYPRE_IJMatrixDestroy(A22mat_);
   if ( M22Diag_       != NULL ) delete [] M22Diag_;
   if ( A11Solver_     != NULL ) HYPRE_ParCSRPCGDestroy(A11Solver_);
   if ( A11Precond_    != NULL ) HYPRE_BoomerAMGDestroy(A11Precond_);
   if ( A22Solver_     != NULL ) HYPRE_ParCSRPCGDestroy(A22Solver_);
   if ( A22Precond_    != NULL ) HYPRE_BoomerAMGDestroy(A22Precond_);
}

//******************************************************************************
// load time step and diffusion coefficient
//------------------------------------------------------------------------------

int HYPRE_IncFlow_BlockPrecond::setScalarParams(double diffusion, double timeStep)
{
   diffusionCoef_ = diffusion;
   if ( timeStep > 0.0 ) timeStep_ = timeStep;
   else                  return 1;
   return 0;
}

//******************************************************************************
// load mass matrix for pressure
//------------------------------------------------------------------------------

int HYPRE_IncFlow_BlockPrecond::setVectorParams(int length, double *Mdata)
{
   if ( length <= 0 )
   {
      printf("HYPRE_IncFlow_BlockPrecond ERROR : Mdiag has <= 0 length.\n");
      exit(1);
   }
   M22Length_ = length;
   if ( M22Diag_ != NULL ) delete [] M22Diag_;
   M22Diag_ = new double[length];
   for ( int i = 0; i < length; i++ ) M22Diag_[i] = Mdata[i];
   return 0;
}

//******************************************************************************
// Given a matrix A, compute the sizes and indices of the 2 x 2 blocks
// (P22Size_, P22GSize_, P22LocalInds_, P22GlobalInds_, P22Offsets_, APartition_)
//------------------------------------------------------------------------------

int HYPRE_IncFlow_BlockPrecond::computeBlockInfo()
{
   int      mypid, nprocs, start_row, end_row, local_nrows, irow;
   int      j, row_size, *col_ind, *disp_array, index;
   double   *col_val;
   HYPRE_ParCSRMatrix Amat_csr;
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
   HYPRE_IJMatrixGetObject(Amat_, (void **) &Amat_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_csr, &APartition_ );
   HYPRE_ParCSRMatrixGetComm( Amat_csr, &mpi_comm );
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
      HYPRE_ParCSRMatrixGetRow(Amat_csr, irow, &row_size, &col_ind, &col_val);
      for ( j = 0; j < row_size; j++ ) 
      {
         index = col_ind[j];
         if ( index == irow ) break;
      }
      if ( j == row_size ) P22Size_++;
      HYPRE_ParCSRMatrixRestoreRow(Amat_csr, irow, &row_size, &col_ind, &col_val);
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
      HYPRE_ParCSRMatrixGetRow(Amat_csr, irow, &row_size, &col_ind, &col_val);
      for ( j = 0; j < row_size; j++ ) 
      {
         index = col_ind[j];
         if ( index == irow ) break;
      }
      if ( j == row_size ) P22LocalInds_[P22Size_++] = irow;
      HYPRE_ParCSRMatrixRestoreRow(Amat_csr, irow, &row_size, &col_ind, &col_val);
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
       return 1;
    }

    if ( P22GSize_ > 0 ) P22GlobalInds_ = new int[P22GSize_];
    else                 P22GlobalInds_ = NULL;
    disp_array     = new int[nprocs];
    MPI_Allgather(&P22Size_, 1, MPI_INT, P22Offsets_, 1, MPI_INT, mpi_comm);
    disp_array[0] = 0;
    for ( j = 1; j < nprocs; j++ ) 
       disp_array[j] = disp_array[j-1] + P22Offsets_[j-1];
    MPI_Allgatherv(P22LocalInds_, P22Size_, MPI_INT, P22GlobalInds_,
                   P22Offsets_, disp_array, MPI_INT, mpi_comm);
    delete [] disp_array;

    if ( outputLevel_ > 1 )
    {
       for ( j = 0; j < P22Size_; j++ )
          printf("%4d computeBlockInfo : P22Inds %8d = %d\n", mypid,
                 j, P22LocalInds_[j]);
    }
    return 0;
} 

//******************************************************************************
// Given a matrix A, build the 2 x 2 blocks
// (This function is to be called after computeBlockInfo
//------------------------------------------------------------------------------

int HYPRE_IncFlow_BlockPrecond::buildBlocks()
{
   int    mypid, nprocs, *partition, index, searchInd;
   int    ANRows, ANCols, AGNRows, AGNCols, AStartRow, AStartCol;
   int    A11NRows, A11NCols, A11GNRows, A11GNCols, A11StartRow, A11StartCol;
   int    A12NRows, A12NCols, A12GNRows, A12GNCols, A12StartRow, A12StartCol;
   int    A22NRows, A22NCols, A22GNRows, A22GNCols, A22StartRow, A22StartCol;
   int    *A11RowLengs, A11MaxRowLeng, A11RowCnt, A11NewSize, *A11_inds;
   int    *A12RowLengs, A12MaxRowLeng, A12RowCnt, A12NewSize, *A12_inds;
   int    *A22RowLengs, A22MaxRowLeng, A22RowCnt, A22NewSize, *A22_inds;
   int    irow, j, rowSize, *inds, ierr;
   double *vals, *A11_vals, *A12_vals, *A22_vals;
   char   fname[200];
   FILE   *fp;
   MPI_Comm mpi_comm;
   HYPRE_ParCSRMatrix Amat_csr, A11mat_csr, A22mat_csr, A12mat_csr;

   //------------------------------------------------------------------
   // extract information about the system matrix
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(Amat_, (void **) &Amat_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_csr, &partition );
   HYPRE_ParCSRMatrixGetComm( Amat_csr, &mpi_comm );
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

   if ( outputLevel_ > 1 )
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
                                A11StartCol, A11StartCol+A11NCols-1, &A11mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A11mat_, HYPRE_PARCSR);
   assert(!ierr);
   ierr  = HYPRE_IJMatrixCreate(mpi_comm, A12StartRow, A12StartRow+A12NRows-1,
                                A12StartCol, A12StartCol+A12NCols-1, &A12mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A12mat_, HYPRE_PARCSR);
   assert(!ierr);
   ierr  = HYPRE_IJMatrixCreate(mpi_comm, A22StartRow, A22StartRow+A22NRows-1,
                                A22StartCol, A22StartCol+A22NCols-1, &A22mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A22mat_, HYPRE_PARCSR);
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
      HYPRE_ParCSRMatrixRestoreRow(Amat_csr, irow, &rowSize, &inds, &vals);
   }
   ierr  = HYPRE_IJMatrixSetRowSizes(A11mat_, A11RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A11mat_);
   assert(!ierr);
   ierr  = HYPRE_IJMatrixSetRowSizes(A12mat_, A12RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A12mat_);
   assert(!ierr);
   ierr  = HYPRE_IJMatrixSetRowSizes(A22mat_, A22RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A22mat_);
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
      HYPRE_ParCSRMatrixGetRow(Amat_csr, irow, &rowSize, &inds, &vals);
      searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
      if ( searchInd < 0 )   // A(1,1) or A(1,2) block
      {
         A11NewSize = A12NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = hypre_BinarySearch(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) // A(1,2) block 
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
         HYPRE_IJMatrixSetValues(A11mat_, 1, &A11NewSize, 
	                    (const int *) &A11RowCnt, (const int *) A11_inds,
                            (const double *) A11_vals);
         HYPRE_IJMatrixSetValues(A12mat_, 1, &A12NewSize, 
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
         HYPRE_IJMatrixSetValues(A22mat_, 1, &A22NewSize, 
	                    (const int *) &A22RowCnt, (const int *) A22_inds,
                            (const double *) A22_vals);
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
   ierr =  HYPRE_IJMatrixAssemble(A12mat_);
   ierr += HYPRE_IJMatrixGetObject(A12mat_, (void **) &A12mat_csr);
   assert( !ierr );
   ierr =  HYPRE_IJMatrixAssemble(A22mat_);
   ierr += HYPRE_IJMatrixGetObject(A22mat_, (void **) &A22mat_csr);
   assert( !ierr );
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A11mat_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A12mat_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A22mat_csr);

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
   return 0;
}

//******************************************************************************
// set up routine
//------------------------------------------------------------------------------

int HYPRE_IncFlow_BlockPrecond::setup()
{
   computeBlockInfo();
   buildBlocks();

   //------------------------------------------------------------------
   // manipulate the A22 block to create an approximate inverse
   //------------------------------------------------------------------

   
    if ( outputLevel_ >= 1 )
    {
       printf("BlockPrecond setup : P^T A P begins\n");
    }
/*
    hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix *) M_csr,
                                     (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix **) &S_csr);
*/
    if ( outputLevel_ >= 1 )
    {
       printf("BlockPrecond setup : P^T A P ends\n");
    }
    return 0;
}

//******************************************************************************
// solve 
//------------------------------------------------------------------------------

int HYPRE_IncFlow_BlockPrecond::solve(HYPRE_IJVector xvec, HYPRE_IJVector fvec)
{
   int             A11Start, A11NRows, A22Start, A22NRows, AStart, ANRows;
   int             AEnd, A11Cnt, A22Cnt, *inds, irow, searchInd, ierr;
   int             mypid, nprocs;
   double          *vals;
   MPI_Comm        mpi_comm;
   HYPRE_IJVector  f1, f2, x1, x2;
   HYPRE_ParCSRMatrix Amat_csr;

   //------------------------------------------------------------------
   // create new subvectors
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( Amat_, (void **) &Amat_csr );
   HYPRE_ParCSRMatrixGetComm( Amat_csr, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   A11Start = APartition_[mypid] - P22Offsets_[mypid];
   A11NRows = APartition_[mypid+1] - P22Offsets_[mypid+1] - A11Start; 
   HYPRE_IJVectorCreate(mpi_comm, A11Start, A11Start+A11NRows-1, &f1);
   HYPRE_IJVectorSetObjectType(f1, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(f1);
   ierr += HYPRE_IJVectorAssemble(f1);
   assert(!ierr);

   HYPRE_IJVectorCreate(mpi_comm, A11Start, A11Start+A11NRows-1, &x1);
   HYPRE_IJVectorSetObjectType(x1, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(x1);
   ierr += HYPRE_IJVectorAssemble(x1);
   assert(!ierr);

   A22Start = P22Offsets_[mypid];
   A22NRows = P22Offsets_[mypid+1] - A22Start;
   HYPRE_IJVectorCreate(mpi_comm, A22Start, A22Start+A22NRows-1, &f2);
   HYPRE_IJVectorSetObjectType(f2, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(f2);
   ierr += HYPRE_IJVectorAssemble(f2);
   assert(!ierr);

   HYPRE_IJVectorCreate(mpi_comm, A22Start, A22Start+A22NRows-1, &x2);
   HYPRE_IJVectorSetObjectType(x2, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(x2);
   ierr += HYPRE_IJVectorAssemble(x2);
   assert(!ierr);

   //------------------------------------------------------------------
   // extract the subvectors
   //------------------------------------------------------------------

   AStart = APartition_[mypid];
   AEnd   = APartition_[mypid+1];
   ANRows = AEnd - AStart;
   inds   = new int[ANRows];
   vals   = new double[ANRows];
   for ( irow = AStart; irow < AEnd; irow++ ) inds[irow-AStart] = irow;
   HYPRE_IJVectorGetValues(fvec, ANRows, inds, vals);
   A11Cnt = A11Start;
   A22Cnt = A22Start;
   for ( irow = AStart; irow < AEnd; irow++ ) 
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
   delete [] vals;
        
   //------------------------------------------------------------------
   // solve them according to the requested scheme 
   //------------------------------------------------------------------

   switch (scheme_)
   {
      case HYPRE_INCFLOW_BDIAG : solveBSolve(x1, x2, f1, f2);
                                 break;

      case HYPRE_INCFLOW_BTRI :  solveBSolve(x1, x2, f1, f2);
                                 break;

      case HYPRE_INCFLOW_BAI  :  solveBAI(x1, x2, f1, f2);
                                 break;

      default :
           printf("HYPRE_IncFlow_BlockPrecond ERROR : scheme not recognized.\n");
           exit(1);
   }

   //------------------------------------------------------------------
   // put the solution back to xvec
   //------------------------------------------------------------------

   AStart = APartition_[mypid];
   AEnd   = APartition_[mypid+1];
   ANRows = AEnd - AStart;
   inds   = new int[ANRows];
   vals   = new double[ANRows];
   for ( irow = AStart; irow < AEnd; irow++ ) inds[irow-AStart] = irow;
   A11Cnt = A11Start;
   A22Cnt = A22Start;
   for ( irow = AStart; irow < AEnd; irow++ ) 
   {
      searchInd = hypre_BinarySearch( P22LocalInds_, irow, P22Size_);
      if ( searchInd >= 0 )
      {
         ierr = HYPRE_IJVectorGetValues(x2, 1, &A22Cnt, &vals[irow]);
         assert( !ierr );
         A22Cnt++;
      }
      else
      {
         ierr = HYPRE_IJVectorGetValues(x1, 1, &A11Cnt, &vals[irow]);
         assert( !ierr );
         A11Cnt++;
      }
   } 
   ierr = HYPRE_IJVectorSetValues(xvec, ANRows, inds, vals);
   delete [] inds;
   delete [] vals;
        
   //------------------------------------------------------------------
   // final clean up
   //------------------------------------------------------------------

   HYPRE_IJVectorDestroy( f1 );
   HYPRE_IJVectorDestroy( f2 );
   HYPRE_IJVectorDestroy( x1 );
   HYPRE_IJVectorDestroy( x2 );
   return 0;
}

//******************************************************************************
// solve with block diagonal or block triangular preconditioner
//------------------------------------------------------------------------------

int HYPRE_IncFlow_BlockPrecond::solveBSolve(HYPRE_IJVector x1,HYPRE_IJVector x2,
                                            HYPRE_IJVector f1,HYPRE_IJVector f2)
{
   int                irow, ierr, max_iter=1000, mypid, A22Start, A22NRows, *inds;
   double             tol=1.0e-6, *vals, alpha;
   MPI_Comm           mpi_comm;
   HYPRE_ParCSRMatrix Amat_csr, A11mat_csr, A22mat_csr, A12mat_csr;
   HYPRE_ParVector    x1_csr, x2_csr, f1_csr, f2_csr;

   //------------------------------------------------------------------
   // fetch machine paramters and matrix and vector pointers
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( Amat_, (void **) &Amat_csr );
   HYPRE_IJMatrixGetObject( A11mat_, (void **) &A11mat_csr );
   HYPRE_IJMatrixGetObject( A22mat_, (void **) &A22mat_csr );
   HYPRE_IJMatrixGetObject( A12mat_, (void **) &A12mat_csr );
   HYPRE_IJVectorGetObject( f1, (void **) &f1_csr );
   HYPRE_IJVectorGetObject( f2, (void **) &f2_csr );
   HYPRE_IJVectorGetObject( x1, (void **) &x1_csr );
   HYPRE_IJVectorGetObject( x2, (void **) &x2_csr );
   HYPRE_ParCSRMatrixGetComm( Amat_csr , &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );

   //------------------------------------------------------------------
   // A22 solve (A^{-1} part of A^{-1} - timeStep * diffCoef * M^{-1})
   //------------------------------------------------------------------

   if ( A22Solver_ == NULL )
   {
      HYPRE_ParCSRPCGCreate(mpi_comm, &A22Solver_);
      HYPRE_ParCSRPCGSetMaxIter(A22Solver_, max_iter );
      HYPRE_ParCSRPCGSetTol(A22Solver_, tol);
      HYPRE_BoomerAMGCreate(&A22Precond_);
      HYPRE_BoomerAMGSetMaxIter(A22Precond_, 1);
      HYPRE_BoomerAMGSetCycleType(A22Precond_, 1);
      HYPRE_BoomerAMGSetMaxLevels(A22Precond_, 0.25);
      HYPRE_BoomerAMGSetMeasureType(A22Precond_, 0);
      HYPRE_ParCSRPCGSetPrecond(A22Solver_,
                       HYPRE_BoomerAMGSolve,
                       HYPRE_BoomerAMGSetup, A22Precond_);
      HYPRE_ParCSRPCGSetup(A22Solver_, Amat_csr, f2_csr, x2_csr);
   }
   HYPRE_ParCSRPCGSolve(A22Solver_, Amat_csr, f2_csr, x2_csr);

   //------------------------------------------------------------------
   // compute x2 = x2 - timeStep * diffCoef * M^{-1} f2
   //------------------------------------------------------------------

   A22Start = P22Offsets_[mypid];
   A22NRows = P22Offsets_[mypid+1] - A22Start;
   inds     = new int[A22NRows];
   vals     = new double[A22NRows];
   for ( irow = A22Start; irow < A22Start+A22NRows; irow++ ) 
      inds[irow-A22Start] = irow;
   ierr = HYPRE_IJVectorGetValues(f2, A22NRows, inds, vals);
   for ( irow = 0; irow < A22NRows; irow++ ) vals[irow] /= M22Diag_[irow];
   ierr = HYPRE_IJVectorSetValues(f2, A22NRows, inds, vals);
   delete [] inds;
   delete [] vals;
   alpha =  - timeStep_ * diffusionCoef_;
   hypre_ParVectorAxpy(alpha,(hypre_ParVector *)f2_csr,(hypre_ParVector *)x2_csr);

   //------------------------------------------------------------------
   // f1 = f1 - C * x2 if block triangular solve 
   //------------------------------------------------------------------

   if ( scheme_ == HYPRE_INCFLOW_BTRI )
      HYPRE_ParCSRMatrixMatvec(-1.0, A12mat_csr, x2_csr, 1.0, f1_csr);

   //------------------------------------------------------------------
   // A11 solve
   //------------------------------------------------------------------

   if ( A11Solver_ == NULL )
   {
      HYPRE_ParCSRPCGCreate(mpi_comm, &A11Solver_);
      HYPRE_ParCSRPCGSetMaxIter(A11Solver_, max_iter );
      HYPRE_ParCSRPCGSetTol(A11Solver_, tol);
      HYPRE_BoomerAMGCreate(&A11Precond_);
      HYPRE_BoomerAMGSetMaxIter(A11Precond_, 1);
      HYPRE_BoomerAMGSetCycleType(A11Precond_, 1);
      HYPRE_BoomerAMGSetMaxLevels(A11Precond_, 0.25);
      HYPRE_BoomerAMGSetMeasureType(A11Precond_, 0);
      HYPRE_ParCSRPCGSetPrecond(A11Solver_,
                       HYPRE_BoomerAMGSolve,
                       HYPRE_BoomerAMGSetup, A11Precond_);
      HYPRE_ParCSRPCGSetup(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   }
   HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);

   return 0;
}

//******************************************************************************
// solve with block approximate inverse preconditioner
//------------------------------------------------------------------------------

int HYPRE_IncFlow_BlockPrecond::solveBAI(HYPRE_IJVector x1,HYPRE_IJVector x2,
                                         HYPRE_IJVector f1,HYPRE_IJVector f2)
{
   (void) x1;
   (void) x2;
   (void) f1;
   (void) f2;
   return 1;
}

