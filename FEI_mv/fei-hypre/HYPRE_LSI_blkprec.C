/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

//******************************************************************************
//******************************************************************************
// This module supports the solution of linear systems arising from the finite
// element discretization of the incompressible Navier Stokes equations.
// The steps in using this module are :
//
//    (1)  precond = new HYPRE_LSI_BlockP(HYPRE_IJMatrix Amat)
//    (2a) precond->setSchemeBlockDiag(), or
//    (2b) precond->setSchemeBlockTriangular(), or
//    (2c) precond->setSchemeBlockInverse()
//    (3)  If lumped mass matrix is to be loaded, do the following :
//         -- call directly to HYPRE : beginCreateMapFromSoln 
//         -- use FEI function to load initial guess with map
//         -- call directly to HYPRE : endCreateMapFromSoln 
//    (4)  precond->setup(mapFromSolnList_,mapFromSolnList2_,mapFromSolnLeng_)
//    (5)  precond->solve( HYPRE_IJVector x, HYPRE_IJVector f )
// 
//******************************************************************************
//******************************************************************************

//******************************************************************************
// system include files
//------------------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>
#include <assert.h>

//******************************************************************************
// HYPRE include files
//------------------------------------------------------------------------------

#include "HYPRE.h"
#include "utilities/utilities.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../parcsr_mv/HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_LSI_blkprec.h"

#ifdef SUPERLU
#include "dsp_defs.h"
#include "util.h"
#endif

//******************************************************************************
// external functions needed here and local defines
//------------------------------------------------------------------------------

extern "C" {
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix**);
   int HYPRE_LSI_Search(int *, int, int);
   int qsort0(int *, int, int);
   int qsort1(int *, double *, int, int);
}
#define dabs(x) ((x > 0) ? x : -(x))

//******************************************************************************
//******************************************************************************
// C-Interface data structure 
//------------------------------------------------------------------------------

typedef struct HYPRE_LSI_BlockPrecond_Struct
{
   void *precon;
} 
HYPRE_LSI_BlockPrecond;

//******************************************************************************
//******************************************************************************
// C-Interface functions to solver
//------------------------------------------------------------------------------

extern "C" 
int HYPRE_LSI_BlockPrecondCreate(MPI_Comm mpi_comm, HYPRE_Solver *solver)
{
   (void) mpi_comm;
   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *)
                                     calloc(1, sizeof(HYPRE_LSI_BlockPrecond));
   HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) new HYPRE_LSI_BlockP();
   cprecon->precon = (void *) precon;
   (*solver) = (HYPRE_Solver) cprecon;
   return 0;
}

//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondDestroy(HYPRE_Solver solver)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      if ( precon != NULL ) delete precon;
      else                  err = 1;
      free( cprecon );
   }
   return err; 
}

//------------------------------------------------------------------------------

extern "C"
int HYPRE_LSI_BlockPrecondSetLumpedMasses(HYPRE_Solver solver, int length, 
                                          double *mass_v)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setLumpedMasses(length, mass_v);
   }
   return err;
}
   
//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondSetSchemeBDiag(HYPRE_Solver solver)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setSchemeBlockDiagonal();
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondSetSchemeBTri(HYPRE_Solver solver)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setSchemeBlockTriangular();
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondSetSchemeBInv(HYPRE_Solver solver)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setSchemeBlockInverse();
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondSetLookup(HYPRE_Solver solver, 
                                               HYPRE_Lookup *lookup)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setLookup((Lookup *)lookup->object);
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" 
int HYPRE_LSI_BlockPrecondSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                                HYPRE_ParVector b, HYPRE_ParVector x)
{
   int err=0;

   (void) b;
   (void) x;
   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setup(Amat);
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" 
int HYPRE_LSI_BlockPrecondSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                                HYPRE_ParVector b, HYPRE_ParVector x)
{
   int err=0;

   (void) Amat;
   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->solve(b, x);
   }
   return err;
}

//******************************************************************************
//******************************************************************************
//******************************************************************************
// Constructor
//------------------------------------------------------------------------------

HYPRE_LSI_BlockP::HYPRE_LSI_BlockP()
{
   Amat_             = NULL;
   A11mat_           = NULL;
   A12mat_           = NULL;
   A22mat_           = NULL;
   F1vec_            = NULL;
   F2vec_            = NULL;
   X1vec_            = NULL;
   X2vec_            = NULL;
   X1aux_            = NULL;
   APartition_       = NULL;
   P22LocalInds_     = NULL;
   P22GlobalInds_    = NULL;
   P22Offsets_       = NULL;
   P22Size_          = -1;
   P22GSize_         = -1;
   assembled_        = 0;
   outputLevel_      = 1;
   lumpedMassLength_ = 0;
   lumpedMassDiag_   = NULL;
   scheme_           = HYPRE_INCFLOW_BDIAG;
   A11Solver_        = NULL;
   A11Precond_       = NULL;
   A22Solver_        = NULL;
   A22Precond_       = NULL;
}

//******************************************************************************
// destructor
//------------------------------------------------------------------------------

HYPRE_LSI_BlockP::~HYPRE_LSI_BlockP()
{
   if ( A11mat_         != NULL ) HYPRE_IJMatrixDestroy(A11mat_);
   if ( A12mat_         != NULL ) HYPRE_IJMatrixDestroy(A12mat_);
   if ( A22mat_         != NULL ) HYPRE_IJMatrixDestroy(A22mat_);
   if ( APartition_     != NULL ) free( APartition_ );
   if ( P22LocalInds_   != NULL ) delete [] P22LocalInds_;
   if ( P22GlobalInds_  != NULL ) delete [] P22GlobalInds_;
   if ( P22Offsets_     != NULL ) delete [] P22Offsets_;
   if ( lumpedMassDiag_ != NULL ) delete [] lumpedMassDiag_;
   if ( A11Solver_      != NULL )
   {
      if (scheme_ == HYPRE_INCFLOW_BTRI) HYPRE_ParCSRGMRESDestroy(A11Solver_);
      else                               HYPRE_ParCSRPCGDestroy(A11Solver_);
   }
   if ( A11Precond_     != NULL ) HYPRE_BoomerAMGDestroy(A11Precond_);
   if ( A22Solver_      != NULL ) HYPRE_ParCSRPCGDestroy(A22Solver_);
   if ( A22Precond_     != NULL ) HYPRE_BoomerAMGDestroy(A22Precond_);
   if ( F1vec_          != NULL ) HYPRE_IJVectorDestroy( F1vec_ );
   if ( F2vec_          != NULL ) HYPRE_IJVectorDestroy( F2vec_ );
   if ( X1vec_          != NULL ) HYPRE_IJVectorDestroy( X1vec_ );
   if ( X2vec_          != NULL ) HYPRE_IJVectorDestroy( X2vec_ );
   if ( X1aux_          != NULL ) HYPRE_IJVectorDestroy( X1aux_ );
}

//******************************************************************************
// load mass matrix for pressure
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::setLumpedMasses(int length, double *Mdata)
{
   if ( length <= 0 )
   {
      printf("BlockP setLumpedMasses ERROR : Mdiag has <= 0 length.\n");
      exit(1);
   }
   lumpedMassLength_ = length;
   if ( lumpedMassDiag_ != NULL ) delete [] lumpedMassDiag_;
   lumpedMassDiag_ = new double[length];
   for ( int i = 0; i < length; i++ ) lumpedMassDiag_[i] = Mdata[i];
   return 0;
}

//******************************************************************************
// set lookup object
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::setLookup(Lookup *object)
{
   lookup_ = object;
   return 0;
}

//******************************************************************************
// Given a matrix A, compute the sizes and indices of the 2 x 2 blocks
//******************************************************************************
// Given a matrix A, compute the sizes and indices of the 2 x 2 blocks
// (P22Size_,P22GSize_,P22LocalInds_,P22GlobalInds_,P22Offsets_,APartition_)
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::computeBlockInfo()
{
   int      mypid, nprocs, start_row, end_row, local_nrows, irow, last_node_num;
   int      j, row_size, *col_ind, *disp_array, index, global_nrows, count;
   int      node_num;
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
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_, &APartition_ );
   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   start_row    = APartition_[mypid];
   end_row      = APartition_[mypid+1] - 1;
   local_nrows  = end_row - start_row + 1;
   global_nrows = APartition_[nprocs];
    
   //------------------------------------------------------------------
   // find the local size of the (2,2) block
   //------------------------------------------------------------------

   P22Size_      = count = 0;
   last_node_num = -1;
   for ( irow = start_row; irow <= end_row; irow++ )
   {
      node_num = lookup_->getAssociatedNodeNumber(irow);
      if ( node_num != last_node_num ) 
      {
         if (count == 1) break; 
         last_node_num = node_num; 
         count = 1;
      }
      else count++;
   }
   index = irow - 1;
   for ( irow = index; irow <= end_row; irow++ ) P22Size_++;

   //for ( irow = start_row; irow <= end_row; irow++ ) 
   //{
   //   HYPRE_ParCSRMatrixGetRow(Amat_, irow, &row_size, &col_ind, &col_val);
   //   for ( j = 0; j < row_size; j++ ) 
   //   {
   //      index = col_ind[j];
   //      if ( index == irow ) break;
   //   }
   //   if ( j == row_size ) P22Size_++;
   //   HYPRE_ParCSRMatrixRestoreRow(Amat_, irow, &row_size, &col_ind, &col_val);
   //}

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

   P22Size_      = count = 0;
   last_node_num = -1;
   for ( irow = start_row; irow <= end_row; irow++ )
   {
      node_num = lookup_->getAssociatedNodeNumber(irow);
      if ( node_num != last_node_num ) 
      {
         if (count == 1) break; 
         last_node_num = node_num; 
         count = 1;
      }
      else count++;
   }
   index = irow - 1;
   for ( irow = index; irow <= end_row; irow++ ) 
      P22LocalInds_[P22Size_++] = irow;

   //for ( irow = start_row; irow <= end_row; irow++ ) 
   //{
   //   HYPRE_ParCSRMatrixGetRow(Amat_, irow, &row_size, &col_ind, &col_val);
   //   for ( j = 0; j < row_size; j++ ) 
   //   {
   //      index = col_ind[j];
   //      if ( index == irow ) break;
   //   }
   //   if ( j == row_size ) P22LocalInds_[P22Size_++] = irow;
   //   HYPRE_ParCSRMatrixRestoreRow(Amat_, irow, &row_size, &col_ind, &col_val);
   //}

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
      if ( APartition_ != NULL ) free( APartition_ );
      APartition_ = NULL;
      return 1;
   }

   if ( P22GSize_ > 0 ) P22GlobalInds_ = new int[P22GSize_];
   else                 P22GlobalInds_ = NULL;
   disp_array     = new int[nprocs];
   P22Offsets_    = new int[nprocs];
   MPI_Allgather(&P22Size_, 1, MPI_INT, P22Offsets_, 1, MPI_INT, mpi_comm);
   disp_array[0] = 0;
   for ( j = 1; j < nprocs; j++ ) 
      disp_array[j] = disp_array[j-1] + P22Offsets_[j-1];
   MPI_Allgatherv(P22LocalInds_, P22Size_, MPI_INT, P22GlobalInds_,
                  P22Offsets_, disp_array, MPI_INT, mpi_comm);
   delete [] P22Offsets_;
   P22Offsets_ = disp_array;

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

int HYPRE_LSI_BlockP::buildBlocks()
{
   int    mypid, nprocs, *partition, index, searchInd;
   int    ANRows, ANCols, AGNRows, AGNCols, AStartRow, AStartCol;
   int    A11NRows, A11NCols, A11GNRows, A11GNCols, A11StartRow, A11StartCol;
   int    A12NRows, A12NCols, A12GNRows, A12GNCols, A12StartRow, A12StartCol;
   int    A22NRows, A22NCols, A22GNRows, A22GNCols, A22StartRow, A22StartCol;
   int    *A11RowLengs, A11MaxRowLeng, A11RowCnt, A11NewSize, *A11_inds;
   int    *A12RowLengs, A12MaxRowLeng, A12RowCnt, A12NewSize, *A12_inds;
   int    *A22RowLengs, A22MaxRowLeng, A22RowCnt, A22NewSize, *A22_inds;
   int    irow, j, k, rowSize, *inds, ierr;
   double *vals, *A11_vals, *A12_vals, *A22_vals;
   char   fname[200];
   FILE   *fp;
   MPI_Comm mpi_comm;
   HYPRE_ParCSRMatrix A11mat_csr, A22mat_csr, A12mat_csr;

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
         A11MaxRowLeng = (A11NewSize > A11MaxRowLeng) ? 
                          A11NewSize : A11MaxRowLeng;
         A12MaxRowLeng = (A12NewSize > A12MaxRowLeng) ? 
                          A12NewSize : A12MaxRowLeng;
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
      HYPRE_ParCSRMatrixRestoreRow(Amat_, irow, &rowSize, &inds, &vals);
   }

   //------------------------------------------------------------------
   // create matrix contexts for the blocks
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpi_comm, A11StartRow, A11StartRow+A11NRows-1,
                                A11StartCol, A11StartCol+A11NCols-1, &A11mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A11mat_, HYPRE_PARCSR);
   ierr  = HYPRE_IJMatrixSetRowSizes(A11mat_, A11RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A11mat_);
   assert(!ierr);
   delete [] A11RowLengs;
   ierr  = HYPRE_IJMatrixCreate(mpi_comm, A12StartRow, A12StartRow+A12NRows-1,
                                A12StartCol, A12StartCol+A12NCols-1, &A12mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A12mat_, HYPRE_PARCSR);
   ierr  = HYPRE_IJMatrixSetRowSizes(A12mat_, A12RowLengs);
   ierr += HYPRE_IJMatrixInitialize(A12mat_);
   assert(!ierr);
   delete [] A12RowLengs;
   if ( A22MaxRowLeng > 0 )
   {
      ierr = HYPRE_IJMatrixCreate(mpi_comm,A22StartRow,A22StartRow+A22NRows-1,
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
      HYPRE_ParCSRMatrixGetRow(Amat_, irow, &rowSize, &inds, &vals);
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
               A11_inds[A11NewSize] = index - searchInd;
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

   free( partition );

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

int HYPRE_LSI_BlockP::setup(HYPRE_ParCSRMatrix Amat)
{
   int      i, j, irow, checkZeros, mypid, nprocs, AStart, AEnd, ANRows; 
   int      rowSize, *colInd, searchInd, newRow, one=1, maxRowSize;
   int      *colInd2, newRowSize, count, *newColInd, rowSize2;
   int      MNRows, MStartRow, *MRowLengs, SNRows, SStartRow, *SRowLengs;
   int      V1Leng, V1Start, V2Leng, V2Start, ierr;
   double   dtemp, *colVal, *colVal2, *newColVal;
   MPI_Comm mpi_comm;
   HYPRE_IJMatrix     Mmat, B22mat;
   HYPRE_ParCSRMatrix Cmat_csr, Mmat_csr, Smat_csr, A22mat_csr, B22mat_csr;
   char     fname[100];
   FILE     *fp;

   //------------------------------------------------------------------
   // build the blocks A11, A12, and the A22 block, if any
   //------------------------------------------------------------------

   Amat_ = Amat;
   computeBlockInfo();
   buildBlocks();

   //------------------------------------------------------------------
   // Extract the velocity mass matrix in HYPRE_ParCSRMatrix format :
   // the mass matrix comes either from user (lumpedMassDiag_) or 
   // extracted from the diagonal of the A(1,1) matrix => mass_v
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   AStart = APartition_[mypid];
   AEnd   = APartition_[mypid+1] - 1;
   ANRows = AEnd - AStart + 1;

   if ( lumpedMassDiag_ != NULL )
   {
      checkZeros = 1;
      for ( i = 0; i < lumpedMassLength_; i++ )
         if ( lumpedMassDiag_[i] == 0.0 ) {checkZeros = 0; break;}
   } 
   else checkZeros = 0;
   
   MNRows    = ANRows - P22Size_;
   MStartRow = AStart - P22Offsets_[mypid];
   MRowLengs = new int[MNRows];
   for ( irow = 0; irow < MNRows; irow++ ) MRowLengs[irow] = 1;
   ierr  = HYPRE_IJMatrixCreate(mpi_comm, MStartRow, MStartRow+MNRows-1,
                                MStartRow, MStartRow+MNRows-1, &Mmat);
   ierr += HYPRE_IJMatrixSetObjectType(Mmat, HYPRE_PARCSR);
   ierr  = HYPRE_IJMatrixSetRowSizes(Mmat, MRowLengs);
   ierr += HYPRE_IJMatrixInitialize(Mmat);
   assert(!ierr);
   delete [] MRowLengs;
   newRow = MStartRow;
   for ( irow = AStart; irow <= AEnd; irow++ ) 
   {
      searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
      if ( searchInd < 0 )
      {
         if ( checkZeros ) dtemp = lumpedMassDiag_[irow-AStart];
         else
         {
            HYPRE_ParCSRMatrixGetRow(Amat_,irow,&rowSize,&colInd,&colVal);
            for ( j = 0; j < rowSize; j++ ) 
               if ( colInd[j] == irow ) { dtemp = colVal[j]; break;}
            HYPRE_ParCSRMatrixRestoreRow(Amat_,irow,&rowSize,&colInd,&colVal);
         }
         dtemp = 1.0 / dtemp;
         HYPRE_IJMatrixSetValues(Mmat, 1, &one, (const int *) &newRow, 
                       (const int *) &newRow, (const double *) &dtemp);
         newRow++;
      }
   }
   ierr =  HYPRE_IJMatrixAssemble(Mmat);
   ierr += HYPRE_IJMatrixGetObject(Mmat, (void **) &Mmat_csr);
   assert( !ierr );
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Mmat_csr);

   //------------------------------------------------------------------
   // create Pressure Poisson matrix (S = C^T M^{-1} C)
   //------------------------------------------------------------------
   
   if (outputLevel_ >= 1) printf("BlockPrecond setup : C^T M^{-1} C begins\n");

   HYPRE_IJMatrixGetObject(A12mat_, (void **) &Cmat_csr);
   hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) Cmat_csr,
                                       (hypre_ParCSRMatrix *) Mmat_csr,
                                       (hypre_ParCSRMatrix *) Cmat_csr,
                                       (hypre_ParCSRMatrix **) &Smat_csr);

   if (outputLevel_ >= 1) printf("BlockPrecond setup : C^T M^{-1} C ends\n");

   //------------------------------------------------------------------
   // construct new A22 = A22 - S
   //------------------------------------------------------------------

   if ( A22mat_ != NULL )
   {
      B22mat = A22mat_;
      HYPRE_IJMatrixGetObject(B22mat, (void **) &B22mat_csr);
   } 
   else B22mat = NULL;
      
   SNRows    = P22Size_;
   SStartRow = P22Offsets_[mypid];
   ierr  = HYPRE_IJMatrixCreate(mpi_comm, SStartRow, SStartRow+SNRows-1,
			 SStartRow, SStartRow+SNRows-1, &A22mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A22mat_, HYPRE_PARCSR);
   assert(!ierr);

   SRowLengs = new int[SNRows];
   maxRowSize = 0;
   for ( irow = SStartRow; irow < SStartRow+SNRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Smat_csr,irow,&rowSize,&colInd,NULL);
      newRowSize = rowSize;
      if ( B22mat != NULL )
      {
         HYPRE_ParCSRMatrixGetRow(B22mat_csr,irow,&rowSize2,&colInd2,NULL);
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
         HYPRE_ParCSRMatrixRestoreRow(B22mat_csr,irow,&rowSize2,&colInd2,NULL);
         delete [] newColInd;
      }
      SRowLengs[irow-SStartRow] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      HYPRE_ParCSRMatrixRestoreRow(Smat_csr,irow,&rowSize,&colInd,NULL);
   }
   ierr  = HYPRE_IJMatrixSetRowSizes(A22mat_, SRowLengs);
   ierr += HYPRE_IJMatrixInitialize(A22mat_);
   assert(!ierr);
   delete [] SRowLengs;

   for ( irow = SStartRow; irow < SStartRow+SNRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Smat_csr,irow,&rowSize,&colInd,&colVal);
      if ( B22mat == NULL )
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
         HYPRE_ParCSRMatrixGetRow(B22mat_csr,irow,&rowSize2,&colInd2,&colVal2);
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
         HYPRE_ParCSRMatrixRestoreRow(B22mat_csr,irow,&rowSize2,
                                      &colInd2,&colVal2);
      }
      HYPRE_IJMatrixSetValues(A22mat_, 1, &newRowSize, (const int *) &irow,
	                  (const int *) newColInd, (const double *) newColVal);
      HYPRE_ParCSRMatrixRestoreRow(Smat_csr,irow,&rowSize,&colInd,&colVal);
      delete [] newColInd;
      delete [] newColVal;
   }
   HYPRE_IJMatrixAssemble(A22mat_);
   HYPRE_IJMatrixGetObject(A22mat_, (void **) &A22mat_csr);
   if ( B22mat != NULL ) HYPRE_IJMatrixDestroy(B22mat);

   if ( outputLevel_ > 1 && A22mat_csr != NULL )
   {
      sprintf( fname, "A22.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = SStartRow; irow < SStartRow+SNRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A22mat_csr,irow,&rowSize,&colInd,&colVal);
         for ( j = 0; j < rowSize; j++ )
            printf(" %9d %9d %25.16e\n", irow+1, colInd[j]+1, colVal[j]);
         HYPRE_ParCSRMatrixRestoreRow(A22mat_csr,irow,&rowSize,&colInd,&colVal);
      }
      fclose(fp);
   }

   //------------------------------------------------------------------
   // build temporary vectors for solution steps
   //------------------------------------------------------------------

   V1Leng  = ANRows - P22Size_;
   V1Start = AStart - P22Offsets_[mypid];
   HYPRE_IJVectorCreate(mpi_comm, V1Start, V1Start+V1Leng-1, &F1vec_);
   HYPRE_IJVectorSetObjectType(F1vec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(F1vec_);
   ierr += HYPRE_IJVectorAssemble(F1vec_);
   assert(!ierr);

   HYPRE_IJVectorCreate(mpi_comm, V1Start, V1Start+V1Leng-1, &X1vec_);
   HYPRE_IJVectorSetObjectType(X1vec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(X1vec_);
   ierr += HYPRE_IJVectorAssemble(X1vec_);
   assert(!ierr);

   if ( scheme_ == HYPRE_INCFLOW_BAI )
   {
      HYPRE_IJVectorCreate(mpi_comm, V1Start, V1Start+V1Leng-1, &X1aux_);
      HYPRE_IJVectorSetObjectType(X1aux_, HYPRE_PARCSR);
      ierr += HYPRE_IJVectorInitialize(X1aux_);
      ierr += HYPRE_IJVectorAssemble(X1aux_);
      assert(!ierr);
   }

   V2Leng  = P22Size_;
   V2Start = P22Offsets_[mypid];
   HYPRE_IJVectorCreate(mpi_comm, V2Start, V2Start+V2Leng-1, &F2vec_);
   HYPRE_IJVectorSetObjectType(F2vec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(F2vec_);
   ierr += HYPRE_IJVectorAssemble(F2vec_);
   assert(!ierr);

   HYPRE_IJVectorCreate(mpi_comm, V2Start, V2Start+V2Leng-1, &X2vec_);
   HYPRE_IJVectorSetObjectType(X2vec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(X2vec_);
   ierr += HYPRE_IJVectorAssemble(X2vec_);
   assert(!ierr);

   assembled_ = 1;
   return 0;
}

//******************************************************************************
// solve 
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::solve(HYPRE_ParVector fvec, HYPRE_ParVector xvec)
{
   int       AStart, ANRows, AEnd, irow, searchInd, ierr;
   int       mypid, nprocs, V1Leng, V1Start, V2Leng, V2Start, V1Cnt, V2Cnt;
   double    *fvals, *xvals;
   MPI_Comm  mpi_comm;

   //------------------------------------------------------------------
   // check for errors
   //------------------------------------------------------------------

   if ( assembled_ != 1 )
   {
      printf("BlockPrecond Solve ERROR : not assembled yet.\n");
      exit(1);
   }

   //------------------------------------------------------------------
   // extract matrix and machine information
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   AStart  = APartition_[mypid];
   AEnd    = APartition_[mypid+1];
   ANRows  = AEnd - AStart;

   //------------------------------------------------------------------
   // extract subvectors for the right hand side
   //------------------------------------------------------------------

   V1Leng  = ANRows - P22Size_;
   V1Start = AStart - P22Offsets_[mypid];
   V2Leng  = P22Size_;
   V2Start = P22Offsets_[mypid];
   fvals = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector*)fvec));
   V1Cnt   = V1Start;
   V2Cnt   = V2Start;
   for ( irow = AStart; irow < AEnd; irow++ ) 
   {
      searchInd = hypre_BinarySearch( P22LocalInds_, irow, P22Size_);
      if ( searchInd >= 0 )
      {
         ierr = HYPRE_IJVectorSetValues(F2vec_, 1, (const int *) &V2Cnt,
		                        (const double *) &fvals[irow-AStart]);
         assert( !ierr );
         V2Cnt++;
      }
      else
      {
         ierr = HYPRE_IJVectorSetValues(F1vec_, 1, (const int *) &V1Cnt,
		                        (const double *) &fvals[irow-AStart]);
         assert( !ierr );
         V1Cnt++;
      }
   } 
        
   //------------------------------------------------------------------
   // solve them according to the requested scheme 
   //------------------------------------------------------------------

   switch (scheme_)
   {
      case HYPRE_INCFLOW_BDIAG : solveBSolve(X1vec_, X2vec_, F1vec_, F2vec_);
                                 break;

      case HYPRE_INCFLOW_BTRI :  solveBSolve(X1vec_, X2vec_, F1vec_, F2vec_);
                                 break;

      case HYPRE_INCFLOW_BAI  :  solveBISolve(X1vec_, X2vec_, F1vec_, F2vec_);
                                 break;

      default :
           printf("HYPRE_LSI_BlockP ERROR : scheme not recognized.\n");
           exit(1);
   }

   //------------------------------------------------------------------
   // put the solution back to xvec
   //------------------------------------------------------------------

   V1Cnt = V1Start;
   V2Cnt = V2Start;
   xvals = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector*)xvec));
   for ( irow = AStart; irow < AEnd; irow++ ) 
   {
      searchInd = hypre_BinarySearch( P22LocalInds_, irow, P22Size_);
      if ( searchInd >= 0 )
      {
         ierr = HYPRE_IJVectorGetValues(X2vec_, 1, &V2Cnt, &xvals[irow-AStart]);
         assert( !ierr );
         V2Cnt++;
      }
      else
      {
         ierr = HYPRE_IJVectorGetValues(X1vec_, 1, &V1Cnt, &xvals[irow-AStart]);
         assert( !ierr );
         V1Cnt++;
      }
   } 
   return 0;
}

//******************************************************************************
// solve with block diagonal or block triangular preconditioner
// (1) for diagonal block solve :
//     (a) A11 solve
//     (b) A22 solve (A_p^{-1} - delta t \hat{M}_p^{-1}) or
//     (c) A22 solve (A22^{-1})
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::solveBSolve(HYPRE_IJVector x1,HYPRE_IJVector x2,
                                  HYPRE_IJVector f1,HYPRE_IJVector f2)
{
   int                irow, ierr, max_iter=3, mypid, A22Start, A22NRows;
   int                i, *nsweeps, *relaxType, *inds;
   double             tol=1.0e-1, *vals, alpha, *relaxWt;
   MPI_Comm           mpi_comm;
   HYPRE_ParCSRMatrix A11mat_csr, A22mat_csr, A12mat_csr;
   HYPRE_ParVector    x1_csr, x2_csr, f1_csr, f2_csr;

   //------------------------------------------------------------------
   // fetch machine paramters and matrix and vector pointers
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( A11mat_, (void **) &A11mat_csr );
   HYPRE_IJMatrixGetObject( A22mat_, (void **) &A22mat_csr );
   HYPRE_IJMatrixGetObject( A12mat_, (void **) &A12mat_csr );
   HYPRE_IJVectorGetObject( F1vec_, (void **) &f1_csr );
   HYPRE_IJVectorGetObject( F2vec_, (void **) &f2_csr );
   HYPRE_IJVectorGetObject( X1vec_, (void **) &x1_csr );
   HYPRE_IJVectorGetObject( X2vec_, (void **) &x2_csr );
   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );

   //------------------------------------------------------------------
   // set up the solvers and preconditioners
   //------------------------------------------------------------------

   if ( A11Solver_ == NULL )
   {
      HYPRE_BoomerAMGCreate(&A11Precond_);
      HYPRE_BoomerAMGSetMaxIter(A11Precond_, 3);
      HYPRE_BoomerAMGSetCycleType(A11Precond_, 1);
      HYPRE_BoomerAMGSetMaxLevels(A11Precond_, 25);
      HYPRE_BoomerAMGSetMeasureType(A11Precond_, 0);
      //HYPRE_BoomerAMGSetIOutDat(A11Precond_, 2);
      HYPRE_BoomerAMGSetCoarsenType(A11Precond_, 0);
      HYPRE_BoomerAMGSetStrongThreshold(A11Precond_, 0.5);
      nsweeps = (int *) malloc( 4 * sizeof(int) );
      for ( i = 0; i < 4; i++ ) nsweeps[i] = 2;
      HYPRE_BoomerAMGSetNumGridSweeps(A11Precond_, nsweeps);
      relaxType = (int *) malloc( 4 * sizeof(int) );
      for ( i = 0; i < 3; i++ ) relaxType[i] = 3;
      relaxType[3] = 9;
      HYPRE_BoomerAMGSetGridRelaxType(A11Precond_, relaxType);
      relaxWt = (double *) malloc( 25 * sizeof(double) );
      for ( i = 0; i < 25; i++ ) relaxWt[i] = 1.0;
      HYPRE_BoomerAMGSetRelaxWeight(A11Precond_, relaxWt);
      if ( scheme_ == HYPRE_INCFLOW_BTRI )
      {
         HYPRE_ParCSRGMRESCreate(mpi_comm, &A11Solver_);
         HYPRE_ParCSRGMRESSetMaxIter(A11Solver_, max_iter );
         HYPRE_ParCSRGMRESSetTol(A11Solver_, tol);
         HYPRE_ParCSRGMRESSetLogging(A11Solver_, 1);
         HYPRE_ParCSRGMRESSetKDim(A11Solver_, 50);
         HYPRE_ParCSRGMRESSetPrecond(A11Solver_, HYPRE_BoomerAMGSolve,
                                     HYPRE_BoomerAMGSetup, A11Precond_);
         HYPRE_ParCSRGMRESSetup(A11Solver_, A11mat_csr, f1_csr, x1_csr);

      }
      else
      {
         HYPRE_ParCSRPCGCreate(mpi_comm, &A11Solver_);
         HYPRE_ParCSRPCGSetMaxIter(A11Solver_, max_iter );
         HYPRE_ParCSRPCGSetTol(A11Solver_, tol);
         HYPRE_ParCSRPCGSetLogging(A11Solver_, 1);
         HYPRE_ParCSRPCGSetRelChange(A11Solver_, 0);
         HYPRE_ParCSRPCGSetTwoNorm(A11Solver_, 1);
         HYPRE_ParCSRPCGSetPrecond(A11Solver_, HYPRE_BoomerAMGSolve,
                                   HYPRE_BoomerAMGSetup, A11Precond_);
         HYPRE_ParCSRPCGSetup(A11Solver_, A11mat_csr, f1_csr, x1_csr);
      }
   }
   if ( A22Solver_ == NULL )
   {
      HYPRE_ParCSRPCGCreate(mpi_comm, &A22Solver_);
      HYPRE_ParCSRPCGSetMaxIter(A22Solver_, max_iter );
      HYPRE_ParCSRPCGSetTol(A22Solver_, tol);
      HYPRE_ParCSRPCGSetLogging(A22Solver_, 1);
      HYPRE_ParCSRPCGSetRelChange(A22Solver_, 0);
      HYPRE_ParCSRPCGSetTwoNorm(A22Solver_, 1);
      HYPRE_BoomerAMGCreate(&A22Precond_);
      HYPRE_BoomerAMGSetMaxLevels(A22Precond_, 25);
      HYPRE_BoomerAMGSetCycleType(A22Precond_, 1);
      HYPRE_BoomerAMGSetMaxIter(A22Precond_, 3);
      HYPRE_BoomerAMGSetMeasureType(A22Precond_, 0);
      //HYPRE_BoomerAMGSetIOutDat(A22Precond_, 2);
      HYPRE_BoomerAMGSetCoarsenType(A22Precond_, 0);
      HYPRE_BoomerAMGSetStrongThreshold(A22Precond_, 0.5);
      nsweeps = (int *) malloc( 4 * sizeof(int) );
      for ( i = 0; i < 4; i++ ) nsweeps[i] = 3;
      HYPRE_BoomerAMGSetNumGridSweeps(A22Precond_, nsweeps);
      relaxType = (int *) malloc( 4 * sizeof(int) );
      for ( i = 0; i < 3; i++ ) relaxType[i] = 3;
      relaxType[3] = 9;
      HYPRE_BoomerAMGSetGridRelaxType(A22Precond_, relaxType);
      relaxWt = (double *) malloc( 25 * sizeof(double) );
      for ( i = 0; i < 25; i++ ) relaxWt[i] = 1.0;
      HYPRE_BoomerAMGSetRelaxWeight(A22Precond_, relaxWt);
      HYPRE_ParCSRPCGSetPrecond(A22Solver_,
                       HYPRE_BoomerAMGSolve,
                       HYPRE_BoomerAMGSetup, A22Precond_);
      HYPRE_ParCSRPCGSetup(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   }

   //------------------------------------------------------------------
   // (1)  A22 solve
   // (1a) compute f1 = f1 - C x2 (if triangular scheme)
   // (2)  A11 solve
   //------------------------------------------------------------------

   HYPRE_ParCSRPCGSolve(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   if ( scheme_ == HYPRE_INCFLOW_BTRI )
   {
      HYPRE_ParCSRMatrixMatvec(-1.0, A12mat_csr, x2_csr, 1.0, f1_csr);
   }
   if ( scheme_ == HYPRE_INCFLOW_BTRI )
      HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   else
      HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   //solveUsingSuperLU(A11mat_, F1vec_, X1vec_);
   return 0;
}

//******************************************************************************
// solve with block approximate inverse preconditioner
// y1 = A11 \ f1
// x2 = A22 \ (C' * y1 - f2)
// x1 = y1 - A11 \ (C x2 )
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::solveBISolve(HYPRE_IJVector x1,HYPRE_IJVector x2,
                                   HYPRE_IJVector f1,HYPRE_IJVector f2)
{
   int                irow, ierr, max_iter=10, mypid, A22Start, A22NRows;
   int                i, *nsweeps, *relaxType, *inds;
   double             tol=1.0e-3, *vals, alpha, *relaxWt;
   MPI_Comm           mpi_comm;
   HYPRE_ParCSRMatrix A11mat_csr, A22mat_csr, A12mat_csr;
   HYPRE_ParVector    x1_csr, x2_csr, f1_csr, f2_csr, y1_csr;

   //------------------------------------------------------------------
   // fetch machine paramters and matrix and vector pointers
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( A11mat_, (void **) &A11mat_csr );
   HYPRE_IJMatrixGetObject( A22mat_, (void **) &A22mat_csr );
   HYPRE_IJMatrixGetObject( A12mat_, (void **) &A12mat_csr );
   HYPRE_IJVectorGetObject( f1, (void **) &f1_csr );
   HYPRE_IJVectorGetObject( f2, (void **) &f2_csr );
   HYPRE_IJVectorGetObject( x1, (void **) &x1_csr );
   HYPRE_IJVectorGetObject( x2, (void **) &x2_csr );
   HYPRE_IJVectorGetObject( X1aux_, (void **) &y1_csr );
   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );

   //------------------------------------------------------------------
   // set up solver and preconditioners, if not already done
   //------------------------------------------------------------------

   if ( A11Solver_ == NULL )
   {
      HYPRE_ParCSRPCGCreate(mpi_comm, &A11Solver_);
      HYPRE_ParCSRPCGSetMaxIter(A11Solver_, max_iter );
      HYPRE_ParCSRPCGSetTol(A11Solver_, tol);
      HYPRE_ParCSRPCGSetLogging(A11Solver_, 1);
      HYPRE_ParCSRPCGSetRelChange(A11Solver_, 0);
      HYPRE_ParCSRPCGSetTwoNorm(A11Solver_, 1);
      HYPRE_BoomerAMGCreate(&A11Precond_);
      HYPRE_BoomerAMGSetMaxIter(A11Precond_, 10);
      HYPRE_BoomerAMGSetCycleType(A11Precond_, 1);
      HYPRE_BoomerAMGSetMaxLevels(A11Precond_, 25);
      HYPRE_BoomerAMGSetMeasureType(A11Precond_, 0);
      HYPRE_BoomerAMGSetCoarsenType(A11Precond_, 0);
      HYPRE_BoomerAMGSetStrongThreshold(A11Precond_, 0.95);
      nsweeps = (int *) malloc( 4 * sizeof(int) );
      for ( i = 0; i < 4; i++ ) nsweeps[i] = 2;
      HYPRE_BoomerAMGSetNumGridSweeps(A11Precond_, nsweeps);
      relaxType = (int *) malloc( 4 * sizeof(int) );
      for ( i = 0; i < 3; i++ ) relaxType[i] = 3;
      relaxType[3] = 9;
      HYPRE_BoomerAMGSetGridRelaxType(A11Precond_, relaxType);
      relaxWt = (double *) malloc( 25 * sizeof(double) );
      for ( i = 0; i < 25; i++ ) relaxWt[i] = 1.0;
      HYPRE_BoomerAMGSetRelaxWeight(A11Precond_, relaxWt);
      if ( scheme_ == HYPRE_INCFLOW_BTRI )
      {
         HYPRE_ParCSRGMRESCreate(mpi_comm, &A11Solver_);
         HYPRE_ParCSRGMRESSetMaxIter(A11Solver_, max_iter );
         HYPRE_ParCSRGMRESSetTol(A11Solver_, tol);
         HYPRE_ParCSRGMRESSetLogging(A11Solver_, 1);
         HYPRE_ParCSRGMRESSetKDim(A11Solver_, 50);
         HYPRE_ParCSRGMRESSetPrecond(A11Solver_, HYPRE_BoomerAMGSolve,
                                     HYPRE_BoomerAMGSetup, A11Precond_);
         HYPRE_ParCSRGMRESSetup(A11Solver_, A11mat_csr, f1_csr, x1_csr);
      }
      else
      {
         HYPRE_ParCSRPCGCreate(mpi_comm, &A11Solver_);
         HYPRE_ParCSRPCGSetMaxIter(A11Solver_, max_iter );
         HYPRE_ParCSRPCGSetTol(A11Solver_, tol);
         HYPRE_ParCSRPCGSetLogging(A11Solver_, 1);
         HYPRE_ParCSRPCGSetRelChange(A11Solver_, 0);
         HYPRE_ParCSRPCGSetTwoNorm(A11Solver_, 1);
         HYPRE_ParCSRPCGSetPrecond(A11Solver_, HYPRE_BoomerAMGSolve,
                                   HYPRE_BoomerAMGSetup, A11Precond_);
         HYPRE_ParCSRPCGSetup(A11Solver_, A11mat_csr, f1_csr, x1_csr);
      }
   }
   if ( A22Solver_ == NULL )
   {
      HYPRE_ParCSRPCGCreate(mpi_comm, &A22Solver_);
      HYPRE_ParCSRPCGSetMaxIter(A22Solver_, max_iter );
      HYPRE_ParCSRPCGSetTol(A22Solver_, tol);
      HYPRE_ParCSRPCGSetLogging(A22Solver_, 1);
      HYPRE_ParCSRPCGSetRelChange(A22Solver_, 0);
      HYPRE_ParCSRPCGSetTwoNorm(A22Solver_, 1);
      HYPRE_BoomerAMGCreate(&A22Precond_);
      HYPRE_BoomerAMGSetMaxLevels(A22Precond_, 25);
      HYPRE_BoomerAMGSetCycleType(A22Precond_, 1);
      HYPRE_BoomerAMGSetMaxIter(A22Precond_, 10);
      HYPRE_BoomerAMGSetMeasureType(A22Precond_, 0);
      //HYPRE_BoomerAMGSetIOutDat(A22Precond_, 3);
      HYPRE_BoomerAMGSetCoarsenType(A22Precond_, 0);
      HYPRE_BoomerAMGSetStrongThreshold(A22Precond_, 0.95);
      nsweeps = (int *) malloc( 4 * sizeof(int) );
      for ( i = 0; i < 4; i++ ) nsweeps[i] = 3;
      HYPRE_BoomerAMGSetNumGridSweeps(A22Precond_, nsweeps);
      relaxType = (int *) malloc( 4 * sizeof(int) );
      for ( i = 0; i < 3; i++ ) relaxType[i] = 3;
      relaxType[3] = 9;
      HYPRE_BoomerAMGSetGridRelaxType(A22Precond_, relaxType);
      relaxWt = (double *) malloc( 25 * sizeof(double) );
      for ( i = 0; i < 25; i++ ) relaxWt[i] = 1.0;
      HYPRE_BoomerAMGSetRelaxWeight(A22Precond_, relaxWt);
      HYPRE_ParCSRPCGSetPrecond(A22Solver_,
                       HYPRE_BoomerAMGSolve,
                       HYPRE_BoomerAMGSetup, A22Precond_);
      HYPRE_ParCSRPCGSetup(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   }

   //------------------------------------------------------------------
   // (1) y1 = A11 \ f1
   // (2) x2 = A22 \ ( C' * y1 - f2 )
   // (3) x1 = y1 - A11 \ ( C * x2 )
   //------------------------------------------------------------------

   if ( scheme_ == HYPRE_INCFLOW_BTRI )
      HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_csr, f1_csr, y1_csr);
   else
      HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_csr, f1_csr, y1_csr);
   //solveUsingSuperLU(A11mat_, F1vec_, X1aux_);
   HYPRE_ParCSRMatrixMatvecT(1.0, A12mat_csr, y1_csr, -1.0, f2_csr);
   HYPRE_ParCSRPCGSolve(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   HYPRE_ParCSRMatrixMatvec(-1.0, A12mat_csr, x2_csr, 0.0, f1_csr);
   if ( scheme_ == HYPRE_INCFLOW_BTRI )
      HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   else
      HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   //solveUsingSuperLU(A11mat_, F1vec_, X1vec_);
   hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) y1_csr, 
                                     (hypre_ParVector *) x1_csr);
   return 0;
}

