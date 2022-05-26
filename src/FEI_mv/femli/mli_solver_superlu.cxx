/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* **************************************************************************** 
 * -- SuperLU routine (version 1.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center, 
 * and Lawrence Berkeley National Lab.
 * ************************************************************************* */

#ifdef MLI_SUPERLU

#include <string.h>
#include "mli_solver_superlu.h"

/* ****************************************************************************
 * constructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_SuperLU::MLI_Solver_SuperLU(char *name) : MLI_Solver(name)
{
   permR_      = NULL;
   permC_      = NULL;
   mliAmat_    = NULL;
   factorized_ = 0;
}

/* ****************************************************************************
 * destructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_SuperLU::~MLI_Solver_SuperLU()
{
   if ( permR_ != NULL ) 
   {
      Destroy_SuperNode_Matrix(&superLU_Lmat);
      Destroy_CompCol_Matrix(&superLU_Umat);
   }
   if ( permR_ != NULL ) delete [] permR_;
   if ( permC_ != NULL ) delete [] permC_;
}

/* ****************************************************************************
 * setup 
 * --------------------------------------------------------------------------*/

int MLI_Solver_SuperLU::setup( MLI_Matrix *Amat )
{
   int      globalNRows, localNRows, startRow, localNnz, globalNnz;
   int      *csrIA, *csrJA, *gcsrIA, *gcsrJA, *gcscJA, *gcscIA;
   int      nnz, row_num, irow, i, j, rowSize, *cols, *recvCntArray;
   int      *dispArray, itemp, *cntArray, icol, colNum, index;
   int      *etree, permcSpec, lwork, panel_size, relax, info, mypid, nprocs;
   double   *vals, *csrAA, *gcsrAA, *gcscAA, diagPivotThresh;
   MPI_Comm mpiComm;
   hypre_ParCSRMatrix *hypreA;
   SuperMatrix        AC;
   superlu_options_t  slu_options;
   SuperLUStat_t      slu_stat;
   GlobalLU_t         Glu;

   /* ---------------------------------------------------------------
    * fetch matrix
    * -------------------------------------------------------------*/

   mliAmat_ = Amat;
   if ( strcmp( mliAmat_->getName(), "HYPRE_ParCSR" ) )
   {
      printf("MLI_Solver_SuperLU::setup ERROR - not HYPRE_ParCSR.\n");
      exit(1);
   }
   hypreA = (hypre_ParCSRMatrix *) mliAmat_->getMatrix();

   /* ---------------------------------------------------------------
    * fetch matrix
    * -------------------------------------------------------------*/
 
   mpiComm     = hypre_ParCSRMatrixComm( hypreA );
   MPI_Comm_rank( mpiComm, &mypid );
   MPI_Comm_size( mpiComm, &nprocs );
   globalNRows = hypre_ParCSRMatrixGlobalNumRows( hypreA );
   localNRows  = hypre_ParCSRMatrixNumRows( hypreA );
   startRow    = hypre_ParCSRMatrixFirstRowIndex( hypreA );
   localNnz    = 0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      row_num = startRow + irow;
      hypre_ParCSRMatrixGetRow(hypreA, row_num, &rowSize, &cols, NULL);
      localNnz += rowSize;
      hypre_ParCSRMatrixRestoreRow(hypreA, row_num, &rowSize, &cols, NULL);
   }
   MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_INT, MPI_SUM, mpiComm );
   csrIA    = new int[localNRows+1];
   if ( localNnz > 0 ) csrJA = new int[localNnz];
   else                csrJA = NULL;
   if ( localNnz > 0 ) csrAA = new double[localNnz];
   else                csrAA = NULL;
   nnz      = 0;
   csrIA[0] = nnz;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      row_num = startRow + irow;
      hypre_ParCSRMatrixGetRow(hypreA, row_num, &rowSize, &cols, &vals);
      for ( i = 0; i < rowSize; i++ )
      {
         csrJA[nnz] = cols[i];
         csrAA[nnz++] = vals[i];
      }
      hypre_ParCSRMatrixRestoreRow(hypreA, row_num, &rowSize, &cols, &vals);
      csrIA[irow+1] = nnz;
   }

   /* ---------------------------------------------------------------
    * collect the whole matrix
    * -------------------------------------------------------------*/

   gcsrIA = new int[globalNRows+1];
   gcsrJA = new int[globalNnz];
   gcsrAA = new double[globalNnz];
   recvCntArray = new int[nprocs];
   dispArray    = new int[nprocs];

   MPI_Allgather(&localNRows,1,MPI_INT,recvCntArray,1,MPI_INT,mpiComm);
   dispArray[0] = 0;
   for ( i = 1; i < nprocs; i++ )
       dispArray[i] = dispArray[i-1] + recvCntArray[i-1];
   csrIA[0] = csrIA[localNRows];
   MPI_Allgatherv(csrIA, localNRows, MPI_INT, gcsrIA, recvCntArray, 
                  dispArray, MPI_INT, mpiComm);
   nnz = 0;
   row_num = 0;
   for ( i = 0; i < nprocs; i++ )
   {
      if ( recvCntArray[i] > 0 )
      {
         itemp = gcsrIA[row_num];
         gcsrIA[row_num] = 0;
         for ( j = 0; j < recvCntArray[i]; j++ )
            gcsrIA[row_num+j] += nnz;
         nnz += itemp;
         row_num += recvCntArray[i];
      }
   }
   gcsrIA[globalNRows] = nnz;

   MPI_Allgather(&localNnz, 1, MPI_INT, recvCntArray, 1, MPI_INT, mpiComm);
   dispArray[0] = 0;
   for ( i = 1; i < nprocs; i++ )
      dispArray[i] = dispArray[i-1] + recvCntArray[i-1];
   MPI_Allgatherv(csrJA, localNnz, MPI_INT, gcsrJA, recvCntArray, 
                  dispArray, MPI_INT, mpiComm);

   MPI_Allgatherv(csrAA, localNnz, MPI_DOUBLE, gcsrAA, recvCntArray, 
                  dispArray, MPI_DOUBLE, mpiComm);

   delete [] recvCntArray;
   delete [] dispArray;
   delete [] csrIA;
   if ( csrJA != NULL ) delete [] csrJA;
   if ( csrAA != NULL ) delete [] csrAA;

   /* ---------------------------------------------------------------
    * conversion from CSR to CSC 
    * -------------------------------------------------------------*/

   cntArray = new int[globalNRows];
   for ( irow = 0; irow < globalNRows; irow++ ) cntArray[irow] = 0;
   for ( irow = 0; irow < globalNRows; irow++ ) 
   {
      for ( i = gcsrIA[irow]; i < gcsrIA[irow+1]; i++ ) 
         if ( gcsrJA[i] >= 0 && gcsrJA[i] < globalNRows )
            cntArray[gcsrJA[i]]++;
         else
         {
            printf("%d : MLI_Solver_SuperLU ERROR : gcsrJA %d %d = %d(%d)\n",
                   mypid, irow, i, gcsrJA[i], globalNRows);
            exit(1);
         }
   }
   gcscJA = hypre_TAlloc(int,  (globalNRows+1) , HYPRE_MEMORY_HOST);
   gcscIA = hypre_TAlloc(int,  globalNnz , HYPRE_MEMORY_HOST);
   gcscAA = hypre_TAlloc(double,  globalNnz , HYPRE_MEMORY_HOST);
   gcscJA[0] = 0;
   nnz = 0;
   for ( icol = 1; icol <= globalNRows; icol++ ) 
   {
      nnz += cntArray[icol-1]; 
      gcscJA[icol] = nnz;
   }
   for ( irow = 0; irow < globalNRows; irow++ )
   {
      for ( i = gcsrIA[irow]; i < gcsrIA[irow+1]; i++ ) 
      {
         colNum = gcsrJA[i];
         index   = gcscJA[colNum]++;
         gcscIA[index] = irow;
         gcscAA[index] = gcsrAA[i];
      }
   }
   gcscJA[0] = 0;
   nnz = 0;
   for ( icol = 1; icol <= globalNRows; icol++ ) 
   {
      nnz += cntArray[icol-1]; 
      gcscJA[icol] = nnz;
   }
   delete [] cntArray;
   delete [] gcsrIA;
   delete [] gcsrJA;
   delete [] gcsrAA;

   /* ---------------------------------------------------------------
    * make SuperMatrix 
    * -------------------------------------------------------------*/
   
   dCreate_CompCol_Matrix(&superLU_Amat, globalNRows, globalNRows, 
                          gcscJA[globalNRows], gcscAA, gcscIA, gcscJA,
                          SLU_NC, SLU_D, SLU_GE);
   etree   = new int[globalNRows];
   permC_  = new int[globalNRows];
   permR_  = new int[globalNRows];
   permcSpec = 0;
   get_perm_c(permcSpec, &superLU_Amat, permC_);
   slu_options.Fact = DOFACT;
   slu_options.SymmetricMode = NO;
   sp_preorder(&slu_options, &superLU_Amat, permC_, etree, &AC);
   diagPivotThresh = 1.0;
   panel_size = sp_ienv(1);
   relax = sp_ienv(2);
   StatInit(&slu_stat);
   lwork = 0;
   slu_options.ColPerm = MY_PERMC;
   slu_options.DiagPivotThresh = diagPivotThresh;

//   dgstrf(&slu_options, &AC, dropTol, relax, panel_size,
//          etree, NULL, lwork, permC_, permR_, &superLU_Lmat,
//          &superLU_Umat, &slu_stat, &info);
   dgstrf(&slu_options, &AC, relax, panel_size,
          etree, NULL, lwork, permC_, permR_, &superLU_Lmat,
          &superLU_Umat, &Glu, &slu_stat, &info);
   Destroy_CompCol_Permuted(&AC);
   Destroy_CompCol_Matrix(&superLU_Amat);
   delete [] etree;
   factorized_ = 1;
   StatFree(&slu_stat);
   return 0;
}

/* ****************************************************************************
 * This subroutine calls the SuperLU subroutine to perform LU 
 * backward substitution 
 * --------------------------------------------------------------------------*/

int MLI_Solver_SuperLU::solve( MLI_Vector *f_in, MLI_Vector *u_in )
{
   int             globalNRows, localNRows, startRow, *recvCntArray;
   int             i, irow, nprocs, *dispArray, info;
   double          *fGlobal;
   hypre_ParVector *f, *u;
   double          *uData, *fData;
   SuperMatrix     B;
   MPI_Comm        mpiComm;
   hypre_ParCSRMatrix *hypreA;
   SuperLUStat_t      slu_stat;
   trans_t            trans;

   /* -------------------------------------------------------------
    * check that the factorization has been called
    * -----------------------------------------------------------*/

   if ( ! factorized_ )
   {
      printf("MLI_Solver_SuperLU::Solve ERROR - not factorized yet.\n");
      exit(1);
   }

   /* -------------------------------------------------------------
    * fetch matrix and vector parameters
    * -----------------------------------------------------------*/

   hypreA      = (hypre_ParCSRMatrix *) mliAmat_->getMatrix();
   mpiComm     = hypre_ParCSRMatrixComm( hypreA );
   globalNRows = hypre_ParCSRMatrixGlobalNumRows( hypreA );
   localNRows  = hypre_ParCSRMatrixNumRows( hypreA );
   startRow    = hypre_ParCSRMatrixFirstRowIndex( hypreA );
   u           = (hypre_ParVector *) u_in->getVector();
   uData       = hypre_VectorData(hypre_ParVectorLocalVector(u));
   f           = (hypre_ParVector *) f_in->getVector();
   fData       = hypre_VectorData(hypre_ParVectorLocalVector(f));

   /* -------------------------------------------------------------
    * collect global vector and create a SuperLU dense matrix
    * -----------------------------------------------------------*/

   MPI_Comm_size( mpiComm, &nprocs );
   recvCntArray = new int[nprocs];
   dispArray    = new int[nprocs];
   fGlobal      = new double[globalNRows];

   MPI_Allgather(&localNRows,1,MPI_INT,recvCntArray,1,MPI_INT,mpiComm);
   dispArray[0] = 0;
   for ( i = 1; i < nprocs; i++ )
       dispArray[i] = dispArray[i-1] + recvCntArray[i-1];
   MPI_Allgatherv(fData, localNRows, MPI_DOUBLE, fGlobal, recvCntArray, 
                  dispArray, MPI_DOUBLE, mpiComm);
   dCreate_Dense_Matrix(&B, globalNRows,1,fGlobal,globalNRows,SLU_DN,
                        SLU_D,SLU_GE);

   /* -------------------------------------------------------------
    * solve the problem
    * -----------------------------------------------------------*/

   trans = NOTRANS;
   StatInit(&slu_stat);
   dgstrs (trans, &superLU_Lmat, &superLU_Umat, permC_, permR_, &B, 
           &slu_stat, &info);

   /* -------------------------------------------------------------
    * fetch the solution
    * -----------------------------------------------------------*/

   for ( irow = 0; irow < localNRows; irow++ )
      uData[irow] = fGlobal[startRow+irow];

   /* -------------------------------------------------------------
    * clean up 
    * -----------------------------------------------------------*/

   delete [] fGlobal;
   delete [] recvCntArray;
   delete [] dispArray;
   Destroy_SuperMatrix_Store(&B);
   StatFree(&slu_stat);

   return info;
}

#endif

