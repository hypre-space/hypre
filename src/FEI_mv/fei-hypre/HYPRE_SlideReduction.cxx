/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

//***************************************************************************
// Date : Apr 26, 2002 (This version works sequentially for up to 10000 elems)
//***************************************************************************
// system includes
//---------------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define HYPRE_SLIDEMAX 100
#define HYPRE_BITMASK2 3

//***************************************************************************
// local includes
//---------------------------------------------------------------------------

#include "HYPRE.h"
#include "HYPRE_SlideReduction.h"
#include "HYPRE_LSI_mli.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "seq_mv/seq_mv.h"
#include "HYPRE_FEI.h"

//***************************************************************************
// local defines and external functions
//---------------------------------------------------------------------------

#define habs(x) (((x) > 0.0) ? x : -(x))

extern "C"
{
	// int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
    //         hypre_ParCSRMatrix*, hypre_ParCSRMatrix*, hypre_ParCSRMatrix**);
	//void hypre_qsort0(int *, int, int);
	//void hypre_qsort1(int *, double *, int, int);
	//int  HYPRE_LSI_Search(int*, int, int);
	//int  HYPRE_LSI_qsort1a(int *, int *, int, int);
	//int  HYPRE_LSI_MatrixInverse(double **, int, double ***);
}

//***************************************************************************
// Constructor
//---------------------------------------------------------------------------

HYPRE_SlideReduction::HYPRE_SlideReduction(MPI_Comm comm)
{
   Amat_             = NULL;
   A21mat_           = NULL;
   invA22mat_        = NULL;
   reducedAmat_      = NULL;
   reducedBvec_      = NULL;
   reducedXvec_      = NULL;
   reducedRvec_      = NULL;
   mpiComm_          = comm;
   outputLevel_      = 0;
   procNConstr_      = NULL;
   slaveEqnList_     = NULL;
   slaveEqnListAux_  = NULL;
   gSlaveEqnList_    = NULL;
   gSlaveEqnListAux_ = NULL;
   constrBlkInfo_    = NULL;
   constrBlkSizes_   = NULL;
   eqnStatuses_      = NULL;
   blockMinNorm_     = 1.0e-4;
   hypreRAP_         = NULL;
   truncTol_         = 1.0e-20;
   scaleMatrixFlag_  = 0;
   ADiagISqrts_      = NULL;
   useSimpleScheme_  = 0;
}

//***************************************************************************
// destructor
//---------------------------------------------------------------------------

HYPRE_SlideReduction::~HYPRE_SlideReduction()
{
   Amat_    = NULL;
   mpiComm_ = 0;
   if ( procNConstr_      != NULL ) delete [] procNConstr_;
   if ( slaveEqnList_     != NULL ) delete [] slaveEqnList_;
   if ( slaveEqnListAux_  != NULL ) delete [] slaveEqnListAux_;
   if ( eqnStatuses_      != NULL ) delete [] eqnStatuses_;
   if ( gSlaveEqnList_    != NULL ) delete [] gSlaveEqnList_;
   if ( gSlaveEqnListAux_ != NULL ) delete [] gSlaveEqnListAux_;
   if ( constrBlkInfo_    != NULL ) delete [] constrBlkInfo_;
   if ( constrBlkSizes_   != NULL ) delete [] constrBlkSizes_;
   if ( A21mat_           != NULL ) HYPRE_IJMatrixDestroy(A21mat_);
   if ( invA22mat_        != NULL ) HYPRE_IJMatrixDestroy(invA22mat_);
   if ( reducedAmat_      != NULL ) HYPRE_IJMatrixDestroy(reducedAmat_);
   if ( reducedBvec_      != NULL ) HYPRE_IJVectorDestroy(reducedBvec_);
   if ( reducedXvec_      != NULL ) HYPRE_IJVectorDestroy(reducedXvec_);
   if ( reducedRvec_      != NULL ) HYPRE_IJVectorDestroy(reducedRvec_);
   if ( hypreRAP_         != NULL ) HYPRE_ParCSRMatrixDestroy(hypreRAP_);
   if ( ADiagISqrts_      != NULL ) delete [] ADiagISqrts_;
}

//***************************************************************************
// set output level
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::setOutputLevel( int level )
{
   if ( level == 1 ) outputLevel_ |= 1;
   if ( level == 2 ) outputLevel_ |= 2;
   if ( level == 3 ) outputLevel_ |= 4;
   return 0;
}

//***************************************************************************
// set use simple scheme
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::setUseSimpleScheme()
{
   useSimpleScheme_ = 1;
   return 0;
}

//***************************************************************************
// set truncation threshold for matrix
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::setTruncationThreshold(double trunc)
{
   truncTol_ = trunc;
   return 0;
}

//***************************************************************************
// enable scaling the matrix
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::setScaleMatrix()
{
   scaleMatrixFlag_ = 1;
   return 0;
}

//***************************************************************************
// set the minimum norm for stable blocks
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::setBlockMinNorm(double norm)
{
   blockMinNorm_ = norm;
   return 0;
}

//***************************************************************************
// get matrix number of rows
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::getMatrixNumRows()
{
   int mypid, nprocs, *procNRows, localNRows, nConstraints;
   HYPRE_ParCSRMatrix A_csr;

   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   localNRows   = procNRows[mypid+1] - procNRows[mypid];
   nConstraints = procNConstr_[mypid+1] - procNConstr_[mypid];
   hypre_TFree( procNRows,HYPRE_MEMORY_HOST );
   return (localNRows-nConstraints);
}

//***************************************************************************
// get matrix diagonal
//---------------------------------------------------------------------------

double *HYPRE_SlideReduction::getMatrixDiagonal()
{
   return ADiagISqrts_;
}

//***************************************************************************
// get reduced matrix
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::getReducedMatrix(HYPRE_IJMatrix *mat)
{
   (*mat) = reducedAmat_;
   return 0;
}

//***************************************************************************
// get reduced rhs
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::getReducedRHSVector(HYPRE_IJVector *rhs)
{
   (*rhs) = reducedBvec_;
   return 0;
}

//***************************************************************************
// get reduced solution vector
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::getReducedSolnVector(HYPRE_IJVector *sol)
{
   (*sol) = reducedXvec_;
   return 0;
}

//***************************************************************************
// get auxiliary (temporary) vector
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::getReducedAuxVector(HYPRE_IJVector *auxV )
{
   (*auxV) = reducedRvec_;
   return 0;
}

//***************************************************************************
// get processor to constraint map
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::getProcConstraintMap(int **map)
{
   (*map) = procNConstr_;
   return 0;
}

//***************************************************************************
// get slave equation list
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::getSlaveEqnList(int **slist)
{
   (*slist) = slaveEqnList_;
   return 0;
}

//***************************************************************************
// get perturbation matrix (reduced = sub(A) - perturb(A))
// (for oorrecting the null space)
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::getPerturbationMatrix(HYPRE_ParCSRMatrix *matrix)
{
   (*matrix) = hypreRAP_;
   hypreRAP_ = NULL;
   return 0;
}

//***************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//
// Additional assumptions are :
//
//    - a given slave equation and the corresponding constraint equation
//      reside in the same processor
//    - constraint equations are given at the end of the local matrix
//      (hence given by endRow-nConstr to endRow)
//    - each processor gets a contiguous block of equations, and processor
//      i+1 has equation numbers higher than those of processor i
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::setup(HYPRE_IJMatrix A, HYPRE_IJVector x,
                                HYPRE_IJVector b)
{
   int   mypid, nprocs, ierr, maxBSize=HYPRE_SLIDEMAX, bSize=2;
   int   *procNRows, nrows1, nrows2, reduceAFlag;
   HYPRE_ParCSRMatrix  A_csr;
   HYPRE_ParVector     b_csr;

   //------------------------------------------------------------------
   // initial set up
   //------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   if ( mypid == 0 && (outputLevel_ & HYPRE_BITMASK2) >= 1 )
      printf("%4d : HYPRE_SlideReduction begins....\n", mypid);


   //------------------------------------------------------------------
   // check matrix and vector compatibility
   //------------------------------------------------------------------

   reduceAFlag = 1;
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &procNRows);
   nrows1 = procNRows[nprocs] - procNRows[0];
   free(procNRows);
   HYPRE_IJMatrixGetObject(A, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &procNRows);
   nrows2 = procNRows[nprocs] - procNRows[0];
   free(procNRows);
   if (nrows1 != nrows2) reduceAFlag = 0;
   if (reduceAFlag == 0)
   {
      HYPRE_IJVectorGetObject(b, (void **) &b_csr);
      procNRows = hypre_ParVectorPartitioning((hypre_ParVector *) b_csr);
      nrows2 = procNRows[nprocs] - procNRows[0];
      if (nrows1 != nrows2)
      {
         if (mypid == 0)
            printf("HYPRE_SlideReduction ERROR - A,b dim mismatch (reuse)!\n");
         exit(1);
      }
   }

   //------------------------------------------------------------------
   // clean up first
   //------------------------------------------------------------------

   if (reduceAFlag == 1)
   {
      Amat_ = A;
      if ( procNConstr_      != NULL ) delete [] procNConstr_;
      if ( slaveEqnList_     != NULL ) delete [] slaveEqnList_;
      if ( slaveEqnListAux_  != NULL ) delete [] slaveEqnListAux_;
      if ( gSlaveEqnList_    != NULL ) delete [] gSlaveEqnList_;
      if ( gSlaveEqnListAux_ != NULL ) delete [] gSlaveEqnListAux_;
      if ( constrBlkInfo_    != NULL ) delete [] constrBlkInfo_;
      if ( constrBlkSizes_   != NULL ) delete [] constrBlkSizes_;
      if ( eqnStatuses_      != NULL ) delete [] eqnStatuses_;
      if ( invA22mat_        != NULL ) HYPRE_IJMatrixDestroy(invA22mat_);
      if ( A21mat_           != NULL ) HYPRE_IJMatrixDestroy(A21mat_);
      if ( reducedAmat_      != NULL ) HYPRE_IJMatrixDestroy(reducedAmat_);
      if ( reducedBvec_      != NULL ) HYPRE_IJVectorDestroy(reducedBvec_);
      if ( reducedXvec_      != NULL ) HYPRE_IJVectorDestroy(reducedXvec_);
      if ( reducedRvec_      != NULL ) HYPRE_IJVectorDestroy(reducedRvec_);
      procNConstr_      = NULL;
      slaveEqnList_     = NULL;
      slaveEqnListAux_  = NULL;
      gSlaveEqnList_    = NULL;
      gSlaveEqnListAux_ = NULL;
      eqnStatuses_      = NULL;
      constrBlkInfo_    = NULL;
      constrBlkSizes_   = NULL;
      reducedAmat_      = NULL;
      invA22mat_        = NULL;
      A21mat_           = NULL;
      reducedBvec_      = NULL;
      reducedXvec_      = NULL;
      reducedRvec_      = NULL;
   }
   else
   {
      if ( reducedBvec_      != NULL ) HYPRE_IJVectorDestroy(reducedBvec_);
      if ( reducedXvec_      != NULL ) HYPRE_IJVectorDestroy(reducedXvec_);
      if ( reducedRvec_      != NULL ) HYPRE_IJVectorDestroy(reducedRvec_);
      reducedBvec_      = NULL;
      reducedXvec_      = NULL;
      reducedRvec_      = NULL;
   }

   //------------------------------------------------------------------
   // find the number of constraints in the local processor
   //------------------------------------------------------------------

   if (reduceAFlag == 1)
   {
      if ( findConstraints() == 0 ) return 0;
   }

   //------------------------------------------------------------------
   // see if we can find a set of slave nodes for the constraint eqns
   // If not, search for block size of 2 or higher.
   //------------------------------------------------------------------

   if (reduceAFlag == 1)
   {
      if ( useSimpleScheme_ == 0 )
      {
         ierr = findSlaveEqns1();
         while (ierr < 0 && bSize <= maxBSize)
            ierr = findSlaveEqnsBlock(bSize++);
         if ( ierr < 0 )
         {
            printf("%4d : HYPRE_SlideReduction ERROR - fail !\n", mypid);
            exit(1);
         }
         composeGlobalList();
      }
   }

   //------------------------------------------------------------------
   // build the reduced matrix
   //------------------------------------------------------------------

   if (reduceAFlag == 1)
   {
      if (useSimpleScheme_ == 0) buildReducedMatrix();
      else                       buildSubMatrices();
   }

   //------------------------------------------------------------------
   // build the reduced right hand side vector
   //------------------------------------------------------------------

   if (useSimpleScheme_ == 0) buildReducedRHSVector(b);
   else                       buildModifiedRHSVector(x,b);

   //------------------------------------------------------------------
   // if scale matrix is request, scale matrix and vector
   //------------------------------------------------------------------

   if ( scaleMatrixFlag_ == 1 )
   {
      if (reduceAFlag == 1) scaleMatrixVector();
      else
      {
         if (mypid == 0)
            printf("HYPRE_SlideReduction ERROR - reuse & scale don't match!\n");
         exit(1);
      }
   }

   //------------------------------------------------------------------
   // clean up and return
   //------------------------------------------------------------------

   if ( mypid == 0 && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("%4d : HYPRE_SlideReduction ends.\n", mypid);
   return 0;
}

//***************************************************************************
// search for local constraints (end of the matrix block)
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::findConstraints()
{
   int    mypid, nprocs, *procNRows, startRow, endRow;
   int    nConstraints, irow, ncnt, isAConstr, jcol, rowSize, *colInd;
   int    *iTempList, ip, globalNConstr;
   double *colVal;
   HYPRE_ParCSRMatrix A_csr;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow     = procNRows[mypid];
   endRow       = procNRows[mypid+1] - 1;
   free( procNRows );

   //------------------------------------------------------------------
   // search for number of local constraints
   // (==> nConstraints)
   //------------------------------------------------------------------

//#define PRINTC
#ifdef PRINTC
   int  localNRows = endRow - startRow + 1;
   char filename[100];
   FILE *fp;
   sprintf( filename, "Constr.%d", localNRows);
   fp = fopen( filename, "w" );
#endif
   nConstraints = 0;
   for ( irow = endRow; irow >= startRow; irow-- )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      isAConstr = 1;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         if ( colInd[jcol] == irow && colVal[jcol] != 0.0 )
         {
            isAConstr = 0;
            break;
         }
      }
#ifdef PRINTC
      if ( isAConstr )
      {
         for ( jcol = 0;  jcol < rowSize;  jcol++ )
            fprintf(fp,"%8d %8d %e\n",nConstraints+1,colInd[jcol]+1,
                    colVal[jcol]);
      }
#endif
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
      if ( isAConstr ) nConstraints++;
      else             break;
   }
#ifdef PRINTC
   fclose(fp);
#endif
   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("%4d : findConstraints - number of constraints = %d\n",
             mypid, nConstraints);

   //------------------------------------------------------------------
   // compute the base nConstraints on each processor
   // (==> globalNConstr, procNConstr)
   //------------------------------------------------------------------

   iTempList = new int[nprocs];
   if ( procNConstr_ != NULL ) delete [] procNConstr_;
   procNConstr_ = new int[nprocs+1];
   for ( ip = 0; ip < nprocs; ip++ ) iTempList[ip] = 0;
   iTempList[mypid] = nConstraints;
   MPI_Allreduce(iTempList,procNConstr_,nprocs,MPI_INT,MPI_SUM,mpiComm_);
   delete [] iTempList;
   globalNConstr = 0;
   ncnt = 0;
   for ( ip = 0; ip < nprocs; ip++ )
   {
      ncnt = procNConstr_[ip];
      procNConstr_[ip] = globalNConstr;
      globalNConstr += ncnt;
   }
   procNConstr_[nprocs] = globalNConstr;
   if ( slaveEqnList_ != NULL ) delete [] slaveEqnList_;
   if ( nConstraints > 0 ) slaveEqnList_ = new int[nConstraints];
   else                    slaveEqnList_ = NULL;
   for ( irow = 0; irow < nConstraints; irow++ ) slaveEqnList_[irow] = -1;
   if ( constrBlkInfo_ != NULL ) delete [] constrBlkInfo_;
   if ( nConstraints > 0 ) constrBlkInfo_ = new int[nConstraints];
   else                    constrBlkInfo_ = NULL;
   for ( irow = 0; irow < nConstraints; irow++ ) constrBlkInfo_[irow] = -1;
   if ( constrBlkSizes_ != NULL ) delete [] constrBlkSizes_;
   if ( nConstraints > 0 ) constrBlkSizes_ = new int[nConstraints];
   else                    constrBlkSizes_ = NULL;
   for ( irow = 0; irow < nConstraints; irow++ ) constrBlkSizes_[irow] = 0;
   if ( nConstraints > 0 )
   {
      eqnStatuses_ = new int[endRow-nConstraints-startRow+1];
      for (irow = 0; irow < endRow-nConstraints-startRow+1; irow++ )
         eqnStatuses_[irow] = 0;
   }
   else eqnStatuses_ = NULL;
   return globalNConstr;
}

//***************************************************************************
// search for a slave equation list (block size = 1)
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::findSlaveEqns1()
{
   int    mypid, nprocs, *procNRows, startRow, endRow;
   int    nConstraints, irow, jcol, rowSize, ncnt, *colInd, index;
   int    nCandidates, *candidateList;
   int    *constrListAux, colIndex, searchIndex, procIndex, uBound;
   int    nSum, newEndRow;
   double *colVal, searchValue;
   HYPRE_ParCSRMatrix A_csr;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow      = procNRows[mypid];
   endRow        = procNRows[mypid+1] - 1;
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];
   newEndRow     = endRow - nConstraints;

   //------------------------------------------------------------------
   // compose candidate slave list (slaves in candidateList, corresponding
   // constraint equation in constrListAux)
   //------------------------------------------------------------------

   nCandidates   = 0;
   candidateList = NULL;
   constrListAux = NULL;
   if ( nConstraints > 0 )
   {
      candidateList = new int[newEndRow-startRow+1];
      constrListAux = new int[newEndRow-startRow+1];

      //------------------------------------------------------------------
      // candidates are those with 1 link to the constraint list
      //------------------------------------------------------------------

      for ( irow = startRow; irow <= endRow-nConstraints; irow++ )
      {
         HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
         ncnt = 0;
         constrListAux[irow-startRow] = -1;
         for ( jcol = 0;  jcol < rowSize;  jcol++ )
         {
            colIndex = colInd[jcol];
            for ( procIndex = 1; procIndex <= nprocs; procIndex++ )
               if ( colIndex < procNRows[procIndex] ) break;
            uBound = procNRows[procIndex] - (procNConstr_[procIndex] -
                                             procNConstr_[procIndex-1]);
            if ( colIndex >= uBound && procIndex == (mypid+1) )
            {
               ncnt++;
               searchIndex = colIndex;
            }
            else if ( colIndex >= uBound && procIndex != (mypid+1) ) ncnt = 2;
            if ( ncnt > 1 ) break;
         }
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
         if (ncnt == 1 && searchIndex > newEndRow && searchIndex <= endRow)
         {
            constrListAux[nCandidates]   = searchIndex;
            candidateList[nCandidates++] = irow;
            if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 3 )
               printf("%4d : findSlaveEqns1 - candidate %d = %d(%d)\n",
                      mypid, nCandidates-1, irow, searchIndex);
         }
      }
      if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
         printf("%4d : findSlaveEqns1 - nCandidates, nConstr = %d %d\n",
                mypid, nCandidates, nConstraints);
   }

   //---------------------------------------------------------------------
   // search the constraint equations for the selected slave equations
   // (search for candidates column index with maximum magnitude)
   // ==> slaveEqnList_
   //---------------------------------------------------------------------

   searchIndex = 0;
   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      searchIndex = -1;
      searchValue = 1.0E-6;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         if (colVal[jcol] != 0.0 && colInd[jcol] >= startRow &&
             colInd[jcol] <= (endRow-nConstraints) &&
             eqnStatuses_[colInd[jcol]-startRow] == 0)
         {
            colIndex = hypre_BinarySearch(candidateList, colInd[jcol],
                                          nCandidates);
            if ( colIndex >= 0 && habs(colVal[jcol]) > searchValue )
            {
               if ( irow != constrListAux[colIndex] ) break;
               searchValue = habs(colVal[jcol]);
               searchIndex = colInd[jcol];
            }
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
      if ( searchIndex >= 0 )
      {
         index = irow - endRow + nConstraints - 1;
         slaveEqnList_[index]   = searchIndex;
         constrBlkInfo_[index]  = index;
         constrBlkSizes_[index] = 1;
         eqnStatuses_[searchIndex-startRow] = 1;
         if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
            printf("%4d : findSlaveEqns1 - constr %7d <=> slave %d\n",
                   mypid, irow, searchIndex);
      }
      else
      {
         slaveEqnList_[irow-endRow+nConstraints-1] = -1;
         if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
         {
            printf("%4d : findSlaveEqns1 - constraint %4d fails",mypid,irow);
            printf(" to find a slave.\n");
         }
      }
   }
   if ( nConstraints > 0 )
   {
      delete [] constrListAux;
      delete [] candidateList;
   }
   free( procNRows );

   //---------------------------------------------------------------------
   // if not all constraint-slave pairs can be found, return -1
   //---------------------------------------------------------------------

   ncnt = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
      if ( slaveEqnList_[irow] == -1 ) ncnt++;
   MPI_Allreduce(&ncnt, &nSum, 1, MPI_INT, MPI_SUM, mpiComm_);
   if ( nSum > 0 )
   {
      if ( mypid == 0 && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      {
         printf("%4d : findSlaveEqns1 fails - total number of unsatisfied",
                mypid);
         printf(" constraints = %d \n", nSum);
      }
      if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      {
         for ( irow = 0; irow < nConstraints; irow++ )
            if ( slaveEqnList_[irow] == -1 )
            {
               printf("%4d : findSlaveEqns1 - unsatisfied constraint",mypid);
               printf(" equation = %d\n", irow+endRow-nConstraints+1);
            }
      }
      return -1;
   } else return 0;
}

//***************************************************************************
// search for a slave equation list (block size of up to blkSizeMax)
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::findSlaveEqnsBlock(int blkSize)
{
   int    mypid, nprocs, *procNRows, startRow, endRow, localNRows;
   int    nConstraints, irow, jcol, rowSize, ncnt, *colInd;
   int    nCandidates, *candidateList, **constrListAuxs, colIndex, rowSize2;
   int    ic, ii, jj, searchIndex, searchInd2, newEndRow;
   int    blkSizeMax=HYPRE_SLIDEMAX, *colInd2, searchInd3;
   int    constrIndex, uBound, lBound, nSum, isACandidate, newBlkSize, *colTmp;
   int    constrIndex2, searchBlkSize, newIndex, oldIndex, irowLocal, ip;
   int    *blkInfo, blkInfoCnt;
   double *colVal, searchValue, retVal, *colVal2;
   HYPRE_ParCSRMatrix A_csr;

   //------------------------------------------------------------------
   // if block size too large, return error
   //------------------------------------------------------------------

   if ( blkSize > blkSizeMax ) return -1;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow      = procNRows[mypid];
   endRow        = procNRows[mypid+1] - 1;
   localNRows    = endRow - startRow + 1;
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];
   newEndRow     = endRow - nConstraints;
   if ( mypid == 0 && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("%4d : findSlaveEqnsBlock - size = %d\n", mypid, blkSize);

   //------------------------------------------------------------------
   // compose candidate slave list (slaves in candidateList, corresponding
   // constraint equation in constrListAuxs)
   //------------------------------------------------------------------

   nCandidates = 0;
   if ( nConstraints > 0 )
   {
      candidateList  = new int[localNRows-nConstraints];
      constrListAuxs = new int*[localNRows-nConstraints];
      for ( ic = 0; ic < localNRows-nConstraints; ic++ )
      {
         constrListAuxs[ic] = new int[blkSize];
         for (jcol = 0; jcol < blkSize; jcol++) constrListAuxs[ic][jcol] = -1;
      }

      //---------------------------------------------------------------
      // candidates are those with <input> links to the constraint list
      //---------------------------------------------------------------

      for ( irow = startRow; irow <= endRow-nConstraints; irow++ )
      {
         if ( eqnStatuses_[irow-startRow] == 1 ) continue;
         HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
         ncnt = 0;
         for ( jcol = 0;  jcol < rowSize;  jcol++ )
         {
            colIndex = colInd[jcol];
            for ( ip = 0;  ip < nprocs;  ip++ )
            {
               uBound = procNRows[ip+1];
               lBound = uBound - (procNConstr_[ip+1] - procNConstr_[ip]);
               if ( colIndex >= lBound && colIndex < uBound && ip == mypid )
               {
                  ncnt++;
                  if ( ncnt <= blkSize )
                     constrListAuxs[nCandidates][ncnt-1] = colIndex;
               }
               else if (colIndex >= lBound && colIndex < uBound && ip != mypid)
               {
                  ncnt = blkSize + 1;
                  break;
               }
            }
            if ( ncnt > blkSize ) break;
         }
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
         if ( ncnt >= 1 && ncnt <= blkSize )
         {
            isACandidate = 1;
            for ( ic = 0; ic < ncnt; ic++ )
            {
               if ( constrListAuxs[nCandidates][ic] <= newEndRow ||
                    constrListAuxs[nCandidates][ic] > endRow )
               {
                  isACandidate = 0;
                  break;
               }
            }
            if ( isACandidate )
            {
               candidateList[nCandidates++] = irow;
               if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 3 )
                  printf("%4d : findSlaveEqnsBlock - candidate %d = %d\n",
                         mypid, nCandidates-1, irow);
            }
         }
      }
      if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
         printf("%4d : findSlaveEqnsBlock - nCandidates, nConstr = %d %d\n",
                   mypid, nCandidates, nConstraints);
   }

   //---------------------------------------------------------------------
   // revise candidates taking into consideration nonsymmetry
   //---------------------------------------------------------------------

   int *tempSlaveList, *tempSlaveListAux;
   if ( nConstraints > 0 ) tempSlaveList = new int[nConstraints];
   if ( nConstraints > 0 ) tempSlaveListAux = new int[nConstraints];
   for (irow = 0; irow < nConstraints; irow++)
   {
      tempSlaveList[irow] = slaveEqnList_[irow];
      tempSlaveListAux[irow] = irow;
   }
   HYPRE_LSI_qsort1a(tempSlaveList, tempSlaveListAux, 0, nConstraints-1);

   /* for each of the candidates, examine all associated constraints dof */

   for ( irow = 0; irow < nCandidates; irow++ )
   {
      for ( ic = 0; ic < blkSize; ic++ )
      {
         constrIndex = constrListAuxs[irow][ic];
         /* if valid constraint number */
         if ( constrIndex >= 0 )
         {
            /* get the constraint row */
            HYPRE_ParCSRMatrixGetRow(A_csr,constrIndex,&rowSize,&colInd,NULL);

            /* for each nonzero entry of the constraint row */
            /* - see if the column number is an already selected slave */
            /* - if so, find the corresponding constraint no. of that slave */
            /* - add that constraint to my list */
            for ( jcol = 0; jcol < rowSize; jcol++ )
            {
               colIndex = colInd[jcol];
               searchIndex = hypre_BinarySearch(tempSlaveList,colIndex,
                                                nConstraints);
               if ( searchIndex >= 0 )
               {
                  searchInd2 = tempSlaveListAux[searchIndex] + newEndRow + 1;
                  for ( ip = 0; ip < blkSize; ip++ )
                     if ( constrListAuxs[irow][ip] == searchInd2 ||
                          constrListAuxs[irow][ip] == -1 ) break;
                  if ( ip == blkSize ) constrListAuxs[irow][0] = -5;
                  else if ( constrListAuxs[irow][ip] == -1 )
                  {
                     constrListAuxs[irow][ip] = searchInd2;
                     if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
                        printf("Slave candidate %d adds new constr %d\n",
                               candidateList[irow], searchInd2);
                  }
               }
            }
            HYPRE_ParCSRMatrixRestoreRow(A_csr,constrIndex,&rowSize,&colInd,
                                         NULL);
         }
      }
   }

   /* delete candidates that gives larger than expected blocksize */

   ncnt = 0;
   for ( irow = 0; irow < nCandidates; irow++ )
   {
      if ( constrListAuxs[irow][0] != -5 )
      {
         if ( irow != ncnt )
         {
            if ( constrListAuxs[ncnt] != NULL ) delete [] constrListAuxs[ncnt];
            constrListAuxs[ncnt] = constrListAuxs[irow];
            constrListAuxs[irow] = NULL;
            candidateList[ncnt++] = candidateList[irow];
         }
         else ncnt++;
      }
   }
   nCandidates = ncnt;
   if ( nConstraints > 0 ) delete [] tempSlaveList;
   if ( nConstraints > 0 ) delete [] tempSlaveListAux;

   //---------------------------------------------------------------------
   // search the constraint equations for the selected slave equations
   // (search for candidates column index with maximum magnitude)
   // ==> slaveEqnList_
   //---------------------------------------------------------------------

   searchIndex = 0;

   blkInfo = new int[blkSize+HYPRE_SLIDEMAX];
   for ( irow = newEndRow+1; irow <= endRow; irow++ )
   {
      /* -- if slave variable has not been picked for constraint irow -- */

      irowLocal = irow - endRow + nConstraints - 1;

      if ( slaveEqnList_[irowLocal] == -1 )
      {
         /* -- get the constraint row, and search for nonzero entries -- */

         HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize2,&colInd2,&colVal2);
         rowSize = rowSize2;
         colInd = new int[rowSize];
         colVal = new double[rowSize];
         for ( jcol = 0;  jcol < rowSize;  jcol++ )
         {
            colInd[jcol] = colInd2[jcol];
            colVal[jcol] = colVal2[jcol];
         }
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize2,&colInd2,&colVal2);
         searchIndex = -1;
         searchValue = blockMinNorm_;
         for ( jcol = 0;  jcol < rowSize;  jcol++ )
         {
            colIndex = colInd[jcol];
            if (colVal[jcol] != 0.0 && colIndex >= startRow
                                    && colIndex <= newEndRow)
            {
               /* -- see if the nonzero entry is a potential candidate -- */

               searchInd2 = hypre_BinarySearch(candidateList, colIndex,
                                               nCandidates);

               /* -- if the nonzero entry is a potential candidate, see -- */
               /* -- if the use of this as slave will give block size   -- */
               /* -- that is too large.                                 -- */

               if (searchInd2 >= 0 && eqnStatuses_[colIndex-startRow] != 1)
               {
                  newBlkSize = 1;
                  blkInfoCnt = 0;
                  for ( ic = 0;  ic < blkSize;  ic++ )
                  {
                     constrIndex  = constrListAuxs[searchInd2][ic];
                     if ( constrIndex != -1 )
                     {
                        constrIndex2 = constrIndex - endRow + nConstraints - 1;
                        if ( constrIndex != irow &&
                             slaveEqnList_[constrIndex2] != -1)
                        {
                           for ( ip = 0; ip < blkInfoCnt; ip++ )
                              if ( blkInfo[ip] == constrBlkInfo_[constrIndex2] )
                                 break;
                           if ( ip == blkInfoCnt )
                           {
                              newBlkSize += constrBlkSizes_[constrIndex2];
                              blkInfo[blkInfoCnt]=constrBlkInfo_[constrIndex2];
                              blkInfoCnt++;
                           }
                        }
/*
                        else if (constrIndex != irow ) newBlkSize++;
*/
                     }
                  }
                  if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
                  {
                     printf("%4d : constraint %d - candidate %d (%d) ", mypid,
                            irow, searchInd2, candidateList[searchInd2]);
                     printf("gives blksize = %d\n", newBlkSize);
                  }
/*
                  if (newBlkSize > 1 && newBlkSize <= blkSize)
*/
                  if (newBlkSize <= blkSize)
                  {
                     retVal = matrixCondEst(irow,colIndex,blkInfo,blkInfoCnt);
                     if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
                        printf("%4d : pivot = %e (%e) : %d\n", mypid, retVal,
                               searchValue,newBlkSize);
                     if ( retVal > searchValue )
                     {
                        searchValue   = habs(colVal[jcol]);
                        searchIndex   = colIndex;
                        searchBlkSize = newBlkSize;
                     }
                  }
               }
            }
         }
         delete [] colInd;
         delete [] colVal;
         if ( searchIndex >= 0 && searchValue > blockMinNorm_ )
         {
            searchInd2 = hypre_BinarySearch(candidateList,searchIndex,
                                            nCandidates);
            newIndex = -9;
            for ( ic = 0;  ic < blkSize;  ic++ )
            {
               constrIndex  = constrListAuxs[searchInd2][ic];
               if ( constrIndex != -1 )
               {
                  constrIndex2 = constrIndex - endRow + nConstraints - 1;
                  if (constrIndex != irow && slaveEqnList_[constrIndex2] != -1)
                  {
                     if (newIndex == -9) newIndex=constrBlkInfo_[constrIndex2];
                     oldIndex = constrBlkInfo_[constrIndex2];
                     for ( ii = 0;  ii < nConstraints;  ii++ )
                     {
                        if ( constrBlkInfo_[ii] == oldIndex )
                        {
                           constrBlkInfo_[ii]  = newIndex;
                           constrBlkSizes_[ii] = searchBlkSize;
                        }
                     }
                  }
               }
            }
            if (newIndex == -9) newIndex = irowLocal;
            constrBlkInfo_[irowLocal]  = newIndex;
            constrBlkSizes_[irowLocal] = searchBlkSize;
            slaveEqnList_[irowLocal]   = searchIndex;
            searchInd2 = hypre_BinarySearch(candidateList, searchIndex,
                                            nCandidates);
            eqnStatuses_[searchIndex-startRow] = 1;

            /* update the constrListAux - first get selected slave row */

            for ( ii = 0;  ii < blkSize;  ii++ )
            {
               constrIndex2 = constrListAuxs[searchInd2][ii];
               if ( constrIndex2 != -1 )
               {
                  HYPRE_ParCSRMatrixGetRow(A_csr,constrIndex2,&rowSize2,
                                           &colInd2,&colVal2);
                  for ( jj = 0;  jj < rowSize2;  jj++ )
                  {
                     searchInd3 = hypre_BinarySearch(candidateList,
                                                     colInd2[jj],nCandidates);
                     if ( searchInd3 >= 0 )
                     {
                        for ( ip = 0; ip < blkSize; ip++ )
                        {
                           if ( constrListAuxs[searchInd3][ip] == irow ||
                                constrListAuxs[searchInd3][ip] == -1 ) break;
                        }
                        if ( ip == blkSize )
                        {
                           constrListAuxs[searchInd3][0] = -5;
                           eqnStatuses_[colInd2[jj]-startRow] = 1;
                           if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 3 )
                              printf("*Slave candidate %d disabled.\n",
                                     candidateList[searchInd3]);
                        }
                        else if ( constrListAuxs[searchInd3][ip] == -1 )
                        {
                           constrListAuxs[searchInd3][ip] = irow;
                           if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 3 )
                              printf("*Slave candidate %d adds new constr %d\n",
                                     candidateList[searchInd3], irow);
                        }
                     }
                  }
                  HYPRE_ParCSRMatrixRestoreRow(A_csr,constrIndex2,&rowSize2,
                                               &colInd2,&colVal2);
               }
            }
            if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
               printf("%4d : findSlaveEqnsBlock - constr %d <=> slave %d (%d)\n",
                      mypid, irow, searchIndex, newIndex);
         }
         else
         {
            if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
            {
               if ( searchIndex < 0 && searchValue > blockMinNorm_ )
               {
                  printf("%4d : findSlaveEqnsBlock - constraint %4d fails (0)",
                         mypid, irow);
                  printf(" to find a slave.\n");
               }
               else if ( searchIndex >= 0 && searchValue <= blockMinNorm_ )
               {
                  printf("%4d : findSlaveEqnsBlock - constraint %4d fails (1)",
                         mypid, irow);
                  printf(" to find a slave.\n");
               }
               else
               {
                  printf("%4d : findSlaveEqnsBlock - constraint %4d fails (2)",
                         mypid, irow);
                  printf(" to find a slave.\n");
               }
               if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 3 )
               {
                  HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
                  colTmp = new int[rowSize];
                  rowSize2 = rowSize;
                  for ( ii = 0;  ii < rowSize;  ii++ ) colTmp[ii] = colInd[ii];
                  HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,
                                               &colVal);
                  for ( jcol = 0;  jcol < rowSize2;  jcol++ )
                  {
                     colIndex = colTmp[jcol];
                     printf("%4d : row %d has col %d (%d,%d) (%d,%d)\n",mypid,
                            irow,colIndex,jcol,rowSize,procNRows[mypid],
                            procNRows[mypid+1]);
                     if ( colIndex >= procNRows[mypid] &&
                          colIndex < procNRows[mypid+1])
                     {
                        HYPRE_ParCSRMatrixGetRow(A_csr,colIndex,&rowSize,
                                                 &colInd,NULL);
                        for ( ii = 0; ii < rowSize;  ii++ )
                           printf("%4d :     col %d has col %d (%d,%d)\n",mypid,
                                  colIndex,colInd[ii],ii,rowSize);
                        HYPRE_ParCSRMatrixRestoreRow(A_csr,colIndex,&rowSize,
                                                     &colInd,NULL);
                     }
                  }
                  delete [] colTmp;
               }
            }
         }
      }
   }
   delete [] blkInfo;
   if ( nConstraints > 0 )
   {
      for ( ic = 0; ic < localNRows-nConstraints; ic++ )
         if ( constrListAuxs[ic] != NULL ) delete [] constrListAuxs[ic];
      delete [] constrListAuxs;
      delete [] candidateList;
   }
   free( procNRows );

#if 0
   int is, *iArray1, *iArray2;
   if ( constrBlkInfo_ != NULL )
   {
      iArray1 = new int[nConstraints];
      iArray2 = new int[nConstraints];
      for ( is = 0; is < nConstraints; is++ )
      {
         iArray1[is] = constrBlkInfo_[is];
         iArray2[is] = constrBlkSizes_[is];
      }
      HYPRE_LSI_qsort1a(iArray1, iArray2, 0, nConstraints-1);
      ip = -1; ncnt = 0;
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( iArray1[is] != ip )
         {
            iArray1[ncnt] = iArray1[is];
            iArray2[ncnt] = iArray2[is];
            ncnt++;
            ip = iArray1[is];
         }
      }
      HYPRE_LSI_qsort1a(iArray2, iArray1, 0, ncnt-1);
      ip = 1;
      for ( is = 1; is < ncnt; is++ )
      {
         if ( iArray2[is] == iArray2[is-1] ) ip++;
         else
         {
            printf("%4d : number of blocks with blksize %6d = %d\n",
                   mypid, iArray2[is-1], ip);
            ip = 1;
         }
      }
      printf("%4d : number of blocks with blksize %6d = %d\n",
             mypid, iArray2[ncnt-1], ip);
      delete [] iArray1;
      delete [] iArray2;
   }
#endif

   //---------------------------------------------------------------------
   // if not all constraint-slave pairs can be found, return -1
   //---------------------------------------------------------------------

   ncnt = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
      if ( slaveEqnList_[irow] == -1 ) ncnt++;
   MPI_Allreduce(&ncnt, &nSum, 1, MPI_INT, MPI_SUM, mpiComm_);
   if ( nSum > 0 )
   {
      if ( mypid == 0 && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      {
         printf("%4d : findSlaveEqnsBlock fails - total number of unsatisfied",
                mypid);
         printf(" constraints = %d \n", nSum);
      }
      if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      {
         for ( irow = 0; irow < nConstraints; irow++ )
            if ( slaveEqnList_[irow] == -1 )
            {
               printf("%4d : findSlaveEqnsBlock - unsatisfied constraint",mypid);
               printf(" equation = %d\n", irow+endRow-nConstraints+1);
            }
      }
      return -1;
   }
   else return 0;
}

//***************************************************************************
// compose global slave equation list
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::composeGlobalList()
{
   int mypid, nprocs, nConstraints, is, ip, *recvCntArray, *displArray;
   int globalNConstr, ierr, ncnt, *iArray1, *iArray2;

   //------------------------------------------------------------------
   // fetch machine and constraint parameters
   //------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];
   globalNConstr = procNConstr_[nprocs];

   //------------------------------------------------------------------
   // sort the local selected node list and its auxiliary list, then
   // form a global list of slave nodes on each processor
   // form the corresponding auxiliary list for later pruning
   // ==> slaveEqnListAux_, gSlaveEqnList_, gSlaveEqnListAux_
   //------------------------------------------------------------------

   if ( slaveEqnListAux_  != NULL ) delete [] slaveEqnListAux_;
   if ( gSlaveEqnList_    != NULL ) delete [] gSlaveEqnList_;
   if ( gSlaveEqnListAux_ != NULL ) delete [] gSlaveEqnListAux_;
   slaveEqnListAux_ = NULL;
   if ( nConstraints > 0 )
   {
      slaveEqnListAux_ = new int[nConstraints];
      for ( is = 0; is < nConstraints; is++ ) slaveEqnListAux_[is] = is;
      HYPRE_LSI_qsort1a(slaveEqnList_, slaveEqnListAux_, 0, nConstraints-1);
      ierr = 0;
      for ( is = 1;  is < nConstraints;  is++ )
      {
         if ( slaveEqnList_[is] == slaveEqnList_[is-1] )
         {
            printf("%4d : HYPRE_SlideReduction ERROR - repeated slave",mypid);
            printf(" equation %d\n", slaveEqnList_[is]);
            ierr = 1;
            break;
         }
      }
      if ( ierr )
      {
         for ( is = 0;  is < nConstraints;  is++ )
         {
            printf("%4d : HYPRE_SlideReduction slave %d = %d \n",mypid,is,
                   slaveEqnList_[is]);
         }
         exit(1);
      }
   }
   gSlaveEqnList_    = new int[globalNConstr];
   gSlaveEqnListAux_ = new int[globalNConstr];

   //------------------------------------------------------------------
   // compose global slave equation list
   //------------------------------------------------------------------

   recvCntArray = new int[nprocs];
   displArray   = new int[nprocs];
   MPI_Allgather(&nConstraints,1,MPI_INT,recvCntArray,1,MPI_INT,mpiComm_);
   displArray[0] = 0;
   for ( ip = 1; ip < nprocs; ip++ )
      displArray[ip] = displArray[ip-1] + recvCntArray[ip-1];
   for ( ip = 0; ip < nConstraints; ip++ )
      slaveEqnListAux_[ip] += displArray[mypid];
   MPI_Allgatherv(slaveEqnList_, nConstraints, MPI_INT, gSlaveEqnList_,
                  recvCntArray, displArray, MPI_INT, mpiComm_);
   MPI_Allgatherv(slaveEqnListAux_, nConstraints, MPI_INT, gSlaveEqnListAux_,
                  recvCntArray, displArray, MPI_INT, mpiComm_);
   for ( is = 0; is < nConstraints; is++ )
      slaveEqnListAux_[is] -= displArray[mypid];
   delete [] recvCntArray;
   delete [] displArray;

   if ( constrBlkInfo_ != NULL && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
   {
      iArray1 = new int[nConstraints];
      iArray2 = new int[nConstraints];
      for ( is = 0; is < nConstraints; is++ )
      {
         iArray1[is] = constrBlkInfo_[is];
         iArray2[is] = constrBlkSizes_[is];
      }
      HYPRE_LSI_qsort1a(iArray1, iArray2, 0, nConstraints-1);
      ip = -1; ncnt = 0;
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( iArray1[is] != ip )
         {
            iArray1[ncnt] = iArray1[is];
            iArray2[ncnt] = iArray2[is];
            ncnt++;
            ip = iArray1[is];
         }
      }
      HYPRE_LSI_qsort1a(iArray2, iArray1, 0, ncnt-1);
      ip = 1;
      for ( is = 1; is < ncnt; is++ )
      {
         if ( iArray2[is] == iArray2[is-1] ) ip++;
         else
         {
            printf("%4d : number of blocks with blksize %6d = %d\n",
                   mypid, iArray2[is-1], ip);
            ip = 1;
         }
      }
      printf("%4d : number of blocks with blksize %6d = %d\n",
             mypid, iArray2[ncnt-1], ip);
      delete [] iArray1;
      delete [] iArray2;
   }

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) > 1 )
      for ( is = 0; is < nConstraints; is++ )
         printf("%4d : HYPRE_SlideReduction - slaveEqnList %d = %d(%d)\n",
                mypid, is, slaveEqnList_[is], slaveEqnListAux_[is]);

   return 0;
}

//****************************************************************************
// build the submatrix matrix
//----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildSubMatrices()
{
   int    mypid, nprocs, *procNRows, startRow, endRow, localNRows;
   int    globalNConstr, globalNRows, nConstraints, ncnt;
   int    A21NRows, A21NCols, A21GlobalNRows, A21GlobalNCols;
   int    A21StartRow, A21StartCol, ierr, maxRowSize, *A21MatSize;
   int    newEndRow, irow, rowIndex, newRowSize, nnzA21, rowCount;
   int    *colInd, *newColInd, rowSize, *reducedAMatSize, uBound;
   int    reducedANRows, reducedANCols, reducedAStartRow, reducedAStartCol;
   int    reducedAGlobalNRows, reducedAGlobalNCols, jcol;
   int    procIndex, colIndex, totalNNZ, newColIndex;
   double *colVal, *newColVal;
   HYPRE_ParCSRMatrix A_csr, A21_csr, reducedA_csr;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow      = procNRows[mypid];
   endRow        = procNRows[mypid+1] - 1;
   localNRows    = endRow - startRow + 1;
   globalNConstr = procNConstr_[nprocs];
   globalNRows   = procNRows[nprocs];
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];

   //******************************************************************
   // extract A21 from A
   //------------------------------------------------------------------
   // calculate the dimension of A21
   //------------------------------------------------------------------

   A21NRows       = nConstraints;
   A21NCols       = localNRows - nConstraints;
   A21GlobalNRows = globalNConstr;
   A21GlobalNCols = globalNRows - globalNConstr;
   A21StartRow    = procNConstr_[mypid];
   A21StartCol    = procNRows[mypid] - procNConstr_[mypid];

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
   {
      printf("%4d : buildA21Mat(2) - A21StartRow  = %d\n", mypid, A21StartRow);
      printf("%4d : buildA21Mat(2) - A21GlobalDim = %d %d\n", mypid,
                                  A21GlobalNRows, A21GlobalNCols);
      printf("%4d : buildA21Mat(2) - A21LocalDim  = %d %d\n",mypid,
                                  A21NRows, A21NCols);
   }

   //------------------------------------------------------------------
   // create a matrix context for A21
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpiComm_,A21StartRow,A21StartRow+A21NRows-1,
                                A21StartCol,A21StartCol+A21NCols-1,&A21mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A21mat_, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute the number of nonzeros in the nConstraint row of A21
   //------------------------------------------------------------------

   rowCount  = maxRowSize = 0;
   newEndRow = endRow - nConstraints;
   if ( A21NRows > 0 ) A21MatSize = new int[A21NRows];
   else                A21MatSize = NULL;
   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         colIndex = colInd[jcol];
         if (colVal[jcol] != 0.0 &&
             (colIndex <= newEndRow || colIndex > endRow)) newRowSize++;
      }
      A21MatSize[irow-newEndRow-1] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
   }
   nnzA21 = 0;
   for ( irow = 0; irow < nConstraints; irow++ ) nnzA21 += A21MatSize[irow];
   MPI_Allreduce(&nnzA21,&ncnt,1,MPI_INT,MPI_SUM,mpiComm_);
   if ( mypid == 0 && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("   0 : buildSubMatrices : NNZ of A21 = %d\n", ncnt);

   //------------------------------------------------------------------
   // after fetching the row sizes, set up A21 with such sizes
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixSetRowSizes(A21mat_, A21MatSize);
   ierr += HYPRE_IJMatrixInitialize(A21mat_);
   hypre_assert(!ierr);
   if ( A21NRows > 0 ) delete [] A21MatSize;

   //------------------------------------------------------------------
   // next load the first nConstraint row to A21 extracted from A
   // (at the same time, the D block is saved for future use)
   //------------------------------------------------------------------

   rowCount  = A21StartRow;
   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];

   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         colIndex = colInd[jcol];
         if (colVal[jcol] != 0.0 &&
             (colIndex <= newEndRow || colIndex > endRow))
         {
            for ( procIndex = 0; procIndex < nprocs; procIndex++ )
               if ( procNRows[procIndex] > colIndex ) break;
            procIndex--;
            newColIndex = colInd[jcol] - procNConstr_[procIndex];
            newColInd[newRowSize]   = newColIndex;
            newColVal[newRowSize++] = colVal[jcol];
         }
      }
      HYPRE_IJMatrixSetValues(A21mat_, 1, &newRowSize, (const int *) &rowCount,
                (const int *) newColInd, (const double *) newColVal);
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
      rowCount++;
   }
   delete [] newColInd;
   delete [] newColVal;

   HYPRE_IJMatrixAssemble(A21mat_);
   HYPRE_IJMatrixGetObject(A21mat_, (void **) &A21_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A21_csr);

   //******************************************************************
   // extract submatrix of A not corresponding to constraints
   //------------------------------------------------------------------

   reducedANRows       = localNRows - nConstraints;
   reducedANCols       = reducedANRows;
   reducedAStartRow    = procNRows[mypid] - procNConstr_[mypid];
   reducedAStartCol    = reducedAStartRow;
   reducedAGlobalNRows = globalNRows - globalNConstr;
   reducedAGlobalNCols = reducedAGlobalNRows;

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
   {
      printf("%4d : buildReducedMatrix - reduceAGlobalDim = %d %d\n", mypid,
                       reducedAGlobalNRows, reducedAGlobalNCols);
      printf("%4d : buildReducedMatrix - reducedALocalDim  = %d %d\n", mypid,
                       reducedANRows, reducedANCols);
   }

   //------------------------------------------------------------------
   // create a matrix context for reducedA
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpiComm_,reducedAStartRow,
                 reducedAStartRow+reducedANRows-1, reducedAStartCol,
                 reducedAStartCol+reducedANCols-1,&reducedAmat_);
   ierr += HYPRE_IJMatrixSetObjectType(reducedAmat_, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute row sizes for reducedA
   //------------------------------------------------------------------

   reducedAMatSize = new int[reducedANRows];
   rowCount = maxRowSize = totalNNZ = 0;
   for ( irow = startRow; irow <= newEndRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         for ( procIndex = 0; procIndex < nprocs; procIndex++ )
            if ( procNRows[procIndex] > colIndex ) break;
         uBound = procNRows[procIndex] -
                  (procNConstr_[procIndex]-procNConstr_[procIndex-1]);
         procIndex--;
         if (colIndex < uBound) newRowSize++;
      }
      rowIndex = reducedAStartRow + rowCount;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      reducedAMatSize[rowCount++] = newRowSize;
      totalNNZ += newRowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
   }
   ierr  = HYPRE_IJMatrixSetRowSizes(reducedAmat_, reducedAMatSize);
   ierr += HYPRE_IJMatrixInitialize(reducedAmat_);
   hypre_assert(!ierr);
   delete [] reducedAMatSize;

   //------------------------------------------------------------------
   // load the reducedA matrix
   //------------------------------------------------------------------

   rowCount  = 0;
   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];
   for ( irow = startRow; irow <= newEndRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr, irow, &rowSize, &colInd, &colVal);
      newRowSize = 0;
      for (jcol = 0; jcol < rowSize; jcol++)
      {
         colIndex = colInd[jcol];
         for ( procIndex = 0; procIndex < nprocs; procIndex++ )
            if ( procNRows[procIndex] > colIndex ) break;
         uBound = procNRows[procIndex] -
                  (procNConstr_[procIndex]-procNConstr_[procIndex-1]);
         procIndex--;
         if ( colIndex < uBound )
         {
            newColInd[newRowSize] = colIndex - procNConstr_[procIndex];
            newColVal[newRowSize++] = colVal[jcol];
         }
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
      }
      rowIndex = reducedAStartRow + rowCount;
      ierr = HYPRE_IJMatrixSetValues(reducedAmat_, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert(!ierr);
      rowCount++;
   }
   delete [] newColInd;
   delete [] newColVal;

   free( procNRows );

   //------------------------------------------------------------------
   // assemble the reduced matrix
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(reducedAmat_);
   HYPRE_IJMatrixGetObject(reducedAmat_, (void **) &reducedA_csr);

   return 0;
}

//****************************************************************************
// build reduced rhs vector (form red_f1 = f1 - A12*x2)
//----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildModifiedRHSVector(HYPRE_IJVector x,
                                                 HYPRE_IJVector b)
{
   int    nprocs, mypid, *procNRows, startRow, endRow, localNRows;
   int    nConstraints, redBNRows, redBStart, ierr, x2NRows, x2Start;
   int    vecIndex, irow;
   double *b_data, *rb_data, *x_data, *x2_data;
   HYPRE_ParCSRMatrix A_csr, A21_csr;
   HYPRE_IJVector     x2;
   HYPRE_ParVector    b_csr, rb_csr, x_csr, x2_csr;
   hypre_Vector       *b_local, *rb_local, *x_local, *x2_local;

   //------------------------------------------------------------------
   // sanitize
   //------------------------------------------------------------------

   if (reducedBvec_ != NULL ) HYPRE_IJVectorDestroy(reducedBvec_);
   if (reducedXvec_ != NULL ) HYPRE_IJVectorDestroy(reducedXvec_);
   if (reducedRvec_ != NULL ) HYPRE_IJVectorDestroy(reducedRvec_);
   reducedBvec_ = NULL;
   reducedXvec_ = NULL;
   reducedRvec_ = NULL;

   //------------------------------------------------------------------
   // get machine and matrix information
   //------------------------------------------------------------------

   if (reducedAmat_ == NULL) return 0;
   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow     = procNRows[mypid];
   endRow       = procNRows[mypid+1] - 1;
   localNRows   = endRow - startRow + 1;
   if (procNConstr_ == NULL || procNConstr_[nprocs] == 0)
   {
      printf("%4d : buildModifiedRHSVector WARNING - no local data.\n",mypid);
      free(procNRows);
      return 1;
   }

   //------------------------------------------------------------------
   // create reducedB
   //------------------------------------------------------------------

   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];
   redBNRows = localNRows - nConstraints;
   redBStart = procNRows[mypid] - procNConstr_[mypid];
   ierr  = HYPRE_IJVectorCreate(mpiComm_, redBStart,
                        redBStart+redBNRows-1, &reducedBvec_);
   ierr += HYPRE_IJVectorSetObjectType(reducedBvec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(reducedBvec_);
   ierr += HYPRE_IJVectorAssemble(reducedBvec_);
   hypre_assert( !ierr );
   HYPRE_IJVectorGetObject(reducedBvec_, (void **) &rb_csr);
   HYPRE_IJVectorGetObject(b, (void **) &b_csr);
   b_local  = hypre_ParVectorLocalVector((hypre_ParVector *) b_csr);
   b_data   = (double *) hypre_VectorData(b_local);
   rb_local = hypre_ParVectorLocalVector((hypre_ParVector *) rb_csr);
   rb_data  = (double *) hypre_VectorData(rb_local);
   for ( irow = 0; irow < localNRows-nConstraints; irow++ )
      rb_data[irow] = b_data[irow];

   //------------------------------------------------------------------
   // create x_2
   //------------------------------------------------------------------

   x2NRows   = nConstraints;
   x2Start   = procNConstr_[mypid];
   HYPRE_IJVectorCreate(mpiComm_, x2Start, x2Start+x2NRows-1, &x2);
   HYPRE_IJVectorSetObjectType(x2, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(x2);
   ierr += HYPRE_IJVectorAssemble(x2);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(x2, (void **) &x2_csr);
   HYPRE_IJVectorGetObject(x, (void **) &x_csr);
   x_local  = hypre_ParVectorLocalVector((hypre_ParVector *) x_csr);
   x_data   = (double *) hypre_VectorData(x_local);
   x2_local = hypre_ParVectorLocalVector((hypre_ParVector *) x2_csr);
   x2_data  = (double *) hypre_VectorData(x2_local);
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      vecIndex = localNRows - nConstraints + irow;
      x2_data[irow] = x_data[vecIndex];
   }

   //------------------------------------------------------------------
   // form reducedB = A21^T * u_2
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(A21mat_, (void **) &A21_csr);
   HYPRE_ParCSRMatrixMatvecT(-1.0, A21_csr, x2_csr, 1.0, rb_csr);
   HYPRE_IJVectorDestroy(x2);

   //------------------------------------------------------------------
   // create a few more vectors
   //------------------------------------------------------------------

   ierr  = HYPRE_IJVectorCreate(mpiComm_, redBStart,
                        redBStart+redBNRows-1, &reducedXvec_);
   ierr += HYPRE_IJVectorSetObjectType(reducedXvec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(reducedXvec_);
   ierr += HYPRE_IJVectorAssemble(reducedXvec_);
   hypre_assert( !ierr );

   ierr  = HYPRE_IJVectorCreate(mpiComm_, redBStart,
                        redBStart+redBNRows-1, &reducedRvec_);
   ierr += HYPRE_IJVectorSetObjectType(reducedRvec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(reducedRvec_);
   ierr += HYPRE_IJVectorAssemble(reducedRvec_);
   hypre_assert( !ierr );
   free( procNRows );

   return 0;
}

//*****************************************************************************
// given the solution vector, copy the actual solution
//-----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildModifiedSolnVector(HYPRE_IJVector x)
{
   int    mypid, nprocs, *procNRows, startRow, endRow, localNRows;
   int    nConstraints, irow;
   double *x_data, *rx_data;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    x_csr, rx_csr;
   hypre_Vector       *x_local, *rx_local;

   //------------------------------------------------------------------
   // get machine and matrix information
   //------------------------------------------------------------------

   if ( reducedXvec_ == NULL ) return -1;
   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow     = procNRows[mypid];
   endRow       = procNRows[mypid+1] - 1;
   localNRows   = endRow - startRow + 1;
   nConstraints = procNConstr_[mypid+1] - procNConstr_[mypid];
   free( procNRows );
   if (( outputLevel_ & HYPRE_BITMASK2 ) >= 1 &&
       (procNConstr_==NULL || procNConstr_[nprocs]==0))
   {
      printf("%4d : buildModifiedSolnVector WARNING - no local entry.\n",
             mypid);
      return 1;
   }

   //------------------------------------------------------------------
   // compute b2 - A21 * sol  (v1 = b2 + v1)
   //------------------------------------------------------------------

   HYPRE_IJVectorGetObject(x, (void **) &x_csr);
   x_local  = hypre_ParVectorLocalVector((hypre_ParVector *) x_csr);
   x_data   = (double *) hypre_VectorData(x_local);
   HYPRE_IJVectorGetObject(reducedXvec_, (void **) &rx_csr);
   rx_local = hypre_ParVectorLocalVector((hypre_ParVector *) rx_csr);
   rx_data  = (double *) hypre_VectorData(rx_local);
   for ( irow = 0; irow < localNRows-nConstraints; irow++ )
      x_data[irow] = rx_data[irow];

   return 0;
}

//****************************************************************************
// build the reduced matrix
//----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildReducedMatrix()
{
   int    mypid, nprocs, *procNRows, startRow, endRow, localNRows;
   int    globalNConstr, globalNRows, nConstraints, newEndRow;
   int    reducedANRows, reducedANCols, reducedAStartRow, reducedAStartCol;
   int    reducedAGlobalNRows, reducedAGlobalNCols, ncnt, irow, jcol;
   int    rowSize, *colInd, *reducedAMatSize, rowCount, maxRowSize;
   int    rowSize2, *colInd2, newRowSize, rowIndex, searchIndex, uBound;
   int    procIndex, colIndex, ierr, *newColInd, totalNNZ;
   double *colVal, *colVal2, *newColVal;
   char   fname[40];
   HYPRE_ParCSRMatrix A_csr, A21_csr, invA22_csr, RAP_csr, reducedA_csr;

   //------------------------------------------------------------------
   // first compute A21 and invA22
   //------------------------------------------------------------------

   buildA21Mat();
   buildInvA22Mat();

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow      = procNRows[mypid];
   endRow        = procNRows[mypid+1] - 1;
   localNRows    = endRow - startRow + 1;
   globalNConstr = procNConstr_[nprocs];
   globalNRows   = procNRows[nprocs];
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];
   newEndRow     = endRow - nConstraints;

   reducedANRows       = localNRows - nConstraints;
   reducedANCols       = reducedANRows;
   reducedAStartRow    = procNRows[mypid] - procNConstr_[mypid];
   reducedAStartCol    = reducedAStartRow;
   reducedAGlobalNRows = globalNRows - globalNConstr;
   reducedAGlobalNCols = reducedAGlobalNRows;

   //------------------------------------------------------------------
   // perform the triple matrix product A12 * invA22 * A21
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(A21mat_, (void **) &A21_csr);
   HYPRE_IJMatrixGetObject(invA22mat_, (void **) &invA22_csr);
   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("%4d : buildReducedMatrix - Triple matrix product starts\n",
             mypid);

   hypre_BoomerAMGBuildCoarseOperator((hypre_ParCSRMatrix *) A21_csr,
                                      (hypre_ParCSRMatrix *) invA22_csr,
                                      (hypre_ParCSRMatrix *) A21_csr,
                                      (hypre_ParCSRMatrix **) &RAP_csr);

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("%4d : buildReducedMatrix - Triple matrix product ends\n",
             mypid);

   if ( outputLevel_ >= 4 )
   {
      sprintf(fname, "rap.%d", mypid);
      FILE *fp = fopen(fname, "w");

      if ( mypid == 0 )
      {
         printf("====================================================\n");
         printf("%4d : Printing RAP matrix... \n", mypid);
         fflush(stdout);
      }
      for (irow=reducedAStartRow; irow<reducedAStartRow+reducedANRows;irow++)
      {
         HYPRE_ParCSRMatrixGetRow(RAP_csr,irow,&rowSize,&colInd,&colVal);
         for ( jcol = 0; jcol < rowSize; jcol++ )
            if ( colVal[jcol] != 0.0 )
               fprintf(fp,"%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                      colVal[jcol]);
         HYPRE_ParCSRMatrixRestoreRow(RAP_csr,irow,&rowSize,&colInd,&colVal);
      }
      fclose(fp);
      if ( mypid == 0 )
         printf("====================================================\n");
   }

   //******************************************************************
   // form reduceA = A11 - A12 * invA22 * A21
   //------------------------------------------------------------------

   //------------------------------------------------------------------
   // compute row sizes
   //------------------------------------------------------------------

   reducedAMatSize = new int[reducedANRows];

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
   {
      printf("%4d : buildReducedMatrix - reduceAGlobalDim = %d %d\n", mypid,
                       reducedAGlobalNRows, reducedAGlobalNCols);
      printf("%4d : buildReducedMatrix - reducedALocalDim  = %d %d\n", mypid,
                       reducedANRows, reducedANCols);
   }

   //------------------------------------------------------------------
   // create a matrix context for reducedA
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpiComm_,reducedAStartRow,
                 reducedAStartRow+reducedANRows-1, reducedAStartCol,
                 reducedAStartCol+reducedANCols-1,&reducedAmat_);
   ierr += HYPRE_IJMatrixSetObjectType(reducedAmat_, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute row sizes for reducedA
   //------------------------------------------------------------------

   rowCount = maxRowSize = totalNNZ = 0;
   for ( irow = startRow; irow <= newEndRow; irow++ )
   {
      searchIndex = hypre_BinarySearch(slaveEqnList_, irow, nConstraints);
      if ( searchIndex >= 0 )  reducedAMatSize[rowCount++] = 1;
      else
      {
         HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
         rowIndex = reducedAStartRow + rowCount;
         ierr = HYPRE_ParCSRMatrixGetRow(RAP_csr,rowIndex,&rowSize2,
                                         &colInd2, &colVal2);
         hypre_assert( !ierr );
         newRowSize = rowSize + rowSize2;
         maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
         newColInd = new int[newRowSize];
         for (jcol = 0; jcol < rowSize; jcol++)
            newColInd[jcol] = colInd[jcol];
         for (jcol = 0; jcol < rowSize2; jcol++)
            newColInd[rowSize+jcol] = colInd2[jcol];
         hypre_qsort0(newColInd, 0, newRowSize-1);
         ncnt = 0;
         for ( jcol = 1; jcol < newRowSize; jcol++ )
            if (newColInd[jcol] != newColInd[ncnt])
               newColInd[++ncnt] = newColInd[jcol];
         if ( newRowSize > 0 ) ncnt++;
         reducedAMatSize[rowCount++] = ncnt;
         totalNNZ += ncnt;
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
         ierr = HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowIndex,&rowSize2,
                                             &colInd2,&colVal2);
         delete [] newColInd;
         hypre_assert( !ierr );
      }
   }
   ierr  = HYPRE_IJMatrixSetRowSizes(reducedAmat_, reducedAMatSize);
   ierr += HYPRE_IJMatrixInitialize(reducedAmat_);
   hypre_assert(!ierr);
   delete [] reducedAMatSize;

   int totalNNZA = 0;
   for ( irow = startRow; irow <= endRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,NULL,NULL);
      totalNNZA += rowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,NULL,NULL);
   }
   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
   {
      printf("%4d : buildReducedMatrix - NNZ of reducedA = %d %d %e\n", mypid,
             totalNNZ, totalNNZA, 1.0*totalNNZ/totalNNZA);
   }

   //------------------------------------------------------------------
   // load the reducedA matrix
   //------------------------------------------------------------------

   rowCount  = 0;
   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];
   for ( irow = startRow; irow <= newEndRow; irow++ )
   {
      searchIndex = hypre_BinarySearch(slaveEqnList_, irow, nConstraints);
      rowIndex    = reducedAStartRow + rowCount;
      if ( searchIndex >= 0 )
      {
         newRowSize   = 1;
         newColInd[0] = reducedAStartRow + rowCount;
         newColVal[0] = 1.0;
      }
      else
      {
         HYPRE_ParCSRMatrixGetRow(A_csr, irow, &rowSize, &colInd, &colVal);
         HYPRE_ParCSRMatrixGetRow(RAP_csr,rowIndex,&rowSize2,&colInd2,
                                  &colVal2);
         newRowSize = rowSize + rowSize2;
         ncnt       = 0;
         for ( jcol = 0; jcol < rowSize; jcol++ )
         {
            colIndex = colInd[jcol];
            for ( procIndex = 0; procIndex < nprocs; procIndex++ )
               if ( procNRows[procIndex] > colIndex ) break;
            uBound = procNRows[procIndex] -
                     (procNConstr_[procIndex]-procNConstr_[procIndex-1]);
            procIndex--;
            if ( colIndex < uBound )
            {
               searchIndex = hypre_BinarySearch(gSlaveEqnList_, colIndex,
                                                globalNConstr);
               if ( searchIndex < 0 )
               {
                  newColInd[ncnt] = colIndex - procNConstr_[procIndex];
                  newColVal[ncnt++] = colVal[jcol];
               }
            }
         }
         for ( jcol = 0; jcol < rowSize2; jcol++ )
         {
            newColInd[ncnt+jcol] = colInd2[jcol];
            newColVal[ncnt+jcol] = - colVal2[jcol];
         }
         newRowSize = ncnt + rowSize2;
         hypre_qsort1(newColInd, newColVal, 0, newRowSize-1);
         ncnt = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( jcol != ncnt && newColInd[jcol] == newColInd[ncnt] )
               newColVal[ncnt] += newColVal[jcol];
            else if ( newColInd[jcol] != newColInd[ncnt] )
            {
               ncnt++;
               newColVal[ncnt] = newColVal[jcol];
               newColInd[ncnt] = newColInd[jcol];
            }
         }
         newRowSize = ncnt + 1;
         ncnt = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( habs(newColVal[jcol]) >= truncTol_ )
            {
               newColInd[ncnt] = newColInd[jcol];
               newColVal[ncnt++] = newColVal[jcol];
            }
         }
         newRowSize = ncnt;
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
         HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowIndex,&rowSize2,&colInd2,
                                      &colVal2);
      }
      ierr = HYPRE_IJMatrixSetValues(reducedAmat_, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert(!ierr);
      rowCount++;
   }
   delete [] newColInd;
   delete [] newColVal;
   hypreRAP_ = RAP_csr;
   free( procNRows );

   //------------------------------------------------------------------
   // assemble the reduced matrix
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(reducedAmat_);
   HYPRE_IJMatrixGetObject(reducedAmat_, (void **) &reducedA_csr);

   if ( outputLevel_ >= 5 )
   {
      sprintf(fname, "reducedA.%d", mypid);
      FILE *fp = fopen(fname, "w");

      if ( mypid == 0 )
      {
         printf("====================================================\n");
         printf("%4d : Printing reducedA matrix... \n", mypid);
         fflush(stdout);
      }
      for ( irow = reducedAStartRow;
             irow < reducedAStartRow+localNRows-nConstraints; irow++ )
      {
         //printf("%d : reducedA ROW %d\n", mypid, irow);
         ierr = HYPRE_ParCSRMatrixGetRow(reducedA_csr,irow,&rowSize,
                                         &colInd, &colVal);
         //hypre_qsort1(colInd, colVal, 0, rowSize-1);
         for ( jcol = 0; jcol < rowSize; jcol++ )
            if ( colVal[jcol] != 0.0 )
               fprintf(fp,"%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                      colVal[jcol]);
         HYPRE_ParCSRMatrixRestoreRow(reducedA_csr,irow,&rowSize,&colInd,
                                      &colVal);
      }
      fclose(fp);
      if ( mypid == 0 )
         printf("====================================================\n");
   }
   return 0;
}

//****************************************************************************
// build reduced rhs vector (form red_f1 = f1 - A12*invA22*f2)
//----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildReducedRHSVector(HYPRE_IJVector b)
{
   int    nprocs, mypid, *procNRows, startRow, endRow, localNRows;
   int    nConstraints, newEndRow, f2LocalLength, f2Start, ierr;
   int    irow, jcol, vecIndex, redBLocalLength, redBStart, rowIndex;
   double *b_data, *f2_data, ddata;
   HYPRE_ParCSRMatrix A_csr, A21_csr, invA22_csr;
   HYPRE_IJVector     f2, f2hat;
   HYPRE_ParVector    b_csr, f2_csr, f2hat_csr, rb_csr;
   hypre_Vector       *b_local, *f2_local;

   //------------------------------------------------------------------
   // get machine and matrix information
   //------------------------------------------------------------------

   if ( reducedAmat_ == NULL ) return 0;
   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow     = procNRows[mypid];
   endRow       = procNRows[mypid+1] - 1;
   localNRows   = endRow - startRow + 1;
   if ( procNConstr_ == NULL || procNConstr_[nprocs] == 0 )
   {
      printf("%4d : buildReducedRHSVector WARNING - no local entries.\n",mypid);
      free(procNRows);
      return 1;
   }

   //------------------------------------------------------------------
   // form f2hat = invA22 * f2
   //------------------------------------------------------------------

   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];
   newEndRow     = endRow - nConstraints;
   f2LocalLength = 2 * nConstraints;
   f2Start       = 2 * procNConstr_[mypid];

   HYPRE_IJVectorCreate(mpiComm_, f2Start, f2Start+f2LocalLength-1, &f2);
   HYPRE_IJVectorSetObjectType(f2, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(f2);
   ierr += HYPRE_IJVectorAssemble(f2);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(f2, (void **) &f2_csr);

   HYPRE_IJVectorCreate(mpiComm_, f2Start, f2Start+f2LocalLength-1, &f2hat);
   HYPRE_IJVectorSetObjectType(f2hat, HYPRE_PARCSR);
   ierr  = HYPRE_IJVectorInitialize(f2hat);
   ierr += HYPRE_IJVectorAssemble(f2hat);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(f2hat, (void **) &f2hat_csr);

   HYPRE_IJVectorGetObject(b, (void **) &b_csr);
   b_local  = hypre_ParVectorLocalVector((hypre_ParVector *) b_csr);
   b_data   = (double *) hypre_VectorData(b_local);
   f2_local = hypre_ParVectorLocalVector((hypre_ParVector *) f2_csr);
   f2_data  = (double *) hypre_VectorData(f2_local);

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      vecIndex = -1;
      for ( jcol = 0; jcol < nConstraints; jcol++ )
      {
         if ( slaveEqnListAux_[jcol] == irow )
         {
            vecIndex = slaveEqnList_[jcol];
            break;
         }
      }
      hypre_assert( vecIndex >= startRow );
      hypre_assert( vecIndex <= endRow );
      f2_data[irow] = b_data[vecIndex-startRow];
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      vecIndex = localNRows - nConstraints + irow;
      f2_data[irow+nConstraints] = b_data[vecIndex];
   }

   HYPRE_IJMatrixGetObject(invA22mat_, (void **) &invA22_csr);
   HYPRE_ParCSRMatrixMatvec( 1.0, invA22_csr, f2_csr, 0.0, f2hat_csr );
   HYPRE_IJVectorDestroy(f2);

   //------------------------------------------------------------------
   // form reducedB = A21^T * f2hat
   //------------------------------------------------------------------

   redBLocalLength = localNRows - nConstraints;
   redBStart       = procNRows[mypid] - procNConstr_[mypid];

   ierr  = HYPRE_IJVectorCreate(mpiComm_, redBStart,
			redBStart+redBLocalLength-1, &reducedBvec_);
   ierr += HYPRE_IJVectorSetObjectType(reducedBvec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(reducedBvec_);
   ierr += HYPRE_IJVectorAssemble(reducedBvec_);
   hypre_assert( !ierr );

   HYPRE_IJVectorGetObject(reducedBvec_, (void **) &rb_csr);
   HYPRE_IJMatrixGetObject(A21mat_, (void **) &A21_csr);
   HYPRE_ParCSRMatrixMatvecT(-1.0, A21_csr, f2hat_csr, 0.0, rb_csr);
   HYPRE_IJVectorDestroy(f2hat);

   //------------------------------------------------------------------
   // finally form reducedB = f1 - f2til
   //------------------------------------------------------------------

   rowIndex = redBStart;
   for ( irow = startRow; irow <= newEndRow; irow++ )
   {
      if ( hypre_BinarySearch(slaveEqnList_, irow, nConstraints) < 0 )
      {
         ddata = b_data[irow-startRow];
         HYPRE_IJVectorAddToValues(reducedBvec_, 1, (const int *) &rowIndex,
			           (const double *) &ddata);
      }
      else
      {
         ddata = 0.0;
         HYPRE_IJVectorSetValues(reducedBvec_, 1, (const int *) &rowIndex,
			         (const double *) &ddata);
      }
      rowIndex++;
   }
   HYPRE_IJVectorGetObject(reducedBvec_, (void **) &rb_csr);

   //------------------------------------------------------------------
   // create a few more vectors
   //------------------------------------------------------------------

   ierr  = HYPRE_IJVectorCreate(mpiComm_, redBStart,
			redBStart+redBLocalLength-1, &reducedXvec_);
   ierr += HYPRE_IJVectorSetObjectType(reducedXvec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(reducedXvec_);
   ierr += HYPRE_IJVectorAssemble(reducedXvec_);
   hypre_assert( !ierr );

   ierr  = HYPRE_IJVectorCreate(mpiComm_, redBStart,
			redBStart+redBLocalLength-1, &reducedRvec_);
   ierr += HYPRE_IJVectorSetObjectType(reducedRvec_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(reducedRvec_);
   ierr += HYPRE_IJVectorAssemble(reducedRvec_);
   hypre_assert( !ierr );
   free( procNRows );

   return 0;
}

//*****************************************************************************
// given the submatrices and the solution vector, restore the actual solution
//  A21 x_1 + A22 x_2 = b2
//  x_2 = invA22 * ( b2 - A21 x_1 )
//-----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildReducedSolnVector(HYPRE_IJVector x,
                                                 HYPRE_IJVector b)
{
   int    mypid, nprocs, *procNRows, startRow, endRow, localNRows;
   int    nConstraints, newEndRow, vecStart, vecLocalLength, ierr;
   int    irow, jcol, rowIndex, searchIndex, length;
   double *b_data, *v1_data, *rx_data, *x_data, *x2_data;
   HYPRE_ParCSRMatrix A_csr, A21_csr, invA22_csr;
   HYPRE_ParVector    x_csr, x2_csr, v1_csr, b_csr, rx_csr;
   HYPRE_IJVector     v1, x2;
   hypre_Vector       *b_local, *v1_local, *rx_local, *x_local, *x2_local;

   //------------------------------------------------------------------
   // get machine and matrix information
   //------------------------------------------------------------------

   if ( reducedAmat_ == NULL ) return 0;
   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow     = procNRows[mypid];
   endRow       = procNRows[mypid+1] - 1;
   localNRows   = endRow - startRow + 1;
   nConstraints = procNConstr_[mypid+1] - procNConstr_[mypid];
   newEndRow    = endRow - nConstraints;
   if (( outputLevel_ & HYPRE_BITMASK2 ) >= 1 &&
       (procNConstr_==NULL || procNConstr_[nprocs]==0))
   {
      printf("%4d : buildReducedSolnVector WARNING - no local entry.\n",mypid);
      return 1;
   }

   //------------------------------------------------------------------
   // compute v1 = - A21 * sol
   //------------------------------------------------------------------

   vecStart       = 2 * procNConstr_[mypid];
   vecLocalLength = 2 * nConstraints;
   ierr  = HYPRE_IJVectorCreate(mpiComm_, vecStart,
                               vecStart+vecLocalLength-1, &v1);
   ierr += HYPRE_IJVectorSetObjectType(v1, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(v1);
   ierr += HYPRE_IJVectorAssemble(v1);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(v1, (void **) &v1_csr);
   HYPRE_IJMatrixGetObject(A21mat_, (void **) &A21_csr);
   HYPRE_IJVectorGetObject(reducedXvec_, (void **) &rx_csr);
   if ( scaleMatrixFlag_ == 1 && ADiagISqrts_ != NULL )
   {
      rx_local = hypre_ParVectorLocalVector((hypre_ParVector *) rx_csr);
      rx_data  = (double *) hypre_VectorData(rx_local);
      length   = hypre_VectorSize(rx_local);
      for ( irow = 0; irow < length; irow++ )
         rx_data[irow] *= ADiagISqrts_[irow];
   }
   HYPRE_ParCSRMatrixMatvec( -1.0, A21_csr, rx_csr, 0.0, v1_csr );

   //------------------------------------------------------------------
   // compute b2 - A21 * sol  (v1 = b2 + v1)
   //------------------------------------------------------------------

   HYPRE_IJVectorGetObject(b, (void **) &b_csr);
   b_local  = hypre_ParVectorLocalVector((hypre_ParVector *) b_csr);
   b_data   = (double *) hypre_VectorData(b_local);
   v1_local = hypre_ParVectorLocalVector((hypre_ParVector *) v1_csr);
   v1_data  = (double *) hypre_VectorData(v1_local);

   rowIndex = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      searchIndex = -1;
      for ( jcol = 0; jcol < nConstraints; jcol++ )
      {
         if ( slaveEqnListAux_[jcol] == irow )
         {
            searchIndex = slaveEqnList_[jcol];
            break;
         }
      }
      hypre_assert( searchIndex >= startRow );
      hypre_assert( searchIndex <= newEndRow );
      v1_data[rowIndex++] += b_data[searchIndex-startRow];
   }
   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
      v1_data[rowIndex++] += b_data[irow-startRow];

   //-------------------------------------------------------------
   // compute inv(A22) * (f2 - A21 * sol) --> x2 = invA22 * v1
   //-------------------------------------------------------------

   ierr  = HYPRE_IJVectorCreate(mpiComm_, vecStart,
                               vecStart+vecLocalLength-1, &x2);
   ierr += HYPRE_IJVectorSetObjectType(x2, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(x2);
   ierr += HYPRE_IJVectorAssemble(x2);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(x2, (void **) &x2_csr );
   HYPRE_IJMatrixGetObject(invA22mat_, (void **) &invA22_csr );
   HYPRE_ParCSRMatrixMatvec(1.0, invA22_csr, v1_csr, 0.0, x2_csr);

#if 0
   FILE *fp = fopen("rhs.m", "w");
   for ( irow = 0; irow < 2*nConstraints; irow++ )
      fprintf(fp, " %6d %25.16e \n", irow+1, v1_data[irow]);
   fclose(fp);
#endif

   //-------------------------------------------------------------
   // inject final solution to the solution vector x
   //-------------------------------------------------------------

   HYPRE_IJVectorGetObject(x, (void **) &x_csr );
   rx_local = hypre_ParVectorLocalVector((hypre_ParVector *) rx_csr);
   rx_data  = (double *) hypre_VectorData(rx_local);
   x_local  = hypre_ParVectorLocalVector((hypre_ParVector *) x_csr);
   x_data   = (double *) hypre_VectorData(x_local);
   x2_local = hypre_ParVectorLocalVector((hypre_ParVector *) x2_csr);
   x2_data  = (double *) hypre_VectorData(x2_local);

   for ( irow = 0; irow < localNRows-nConstraints; irow++ )
      x_data[irow] = rx_data[irow];

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( jcol = 0; jcol < nConstraints; jcol++ )
      {
         if ( slaveEqnListAux_[jcol] == irow )
         {
            searchIndex = slaveEqnList_[jcol];
            break;
         }
      }
      x_data[searchIndex-startRow] = x2_data[irow];
   }
   for ( irow = nConstraints; irow < 2*nConstraints; irow++ )
      x_data[localNRows-2*nConstraints+irow] = x2_data[irow];

   //------------------------------------------------------------------
   // compute true residual
   //------------------------------------------------------------------

#if 0
   double          rnorm;
   HYPRE_IJVector  R;
   HYPRE_ParVector R_csr;

   ierr  = HYPRE_IJVectorCreate(mpiComm_, procNRows[mypid],
                                procNRows[mypid+1]-1, &R);
   ierr += HYPRE_IJVectorSetObjectType(R, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(R);
   ierr += HYPRE_IJVectorAssemble(R);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(R, (void **) &R_csr);
   HYPRE_ParVectorCopy( b_csr, R_csr );
   HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, R_csr );
   HYPRE_ParVectorInnerProd( R_csr, R_csr, &rnorm);
   hypre_Vector *R_local=hypre_ParVectorLocalVector((hypre_ParVector*) R_csr);
   double *R_data  = (double *) hypre_VectorData(R_local);
   HYPRE_ParVectorInnerProd( R_csr, R_csr, &rnorm);
   double rnorm2 = 0.0;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      searchIndex = -1;
      for ( jcol = 0; jcol < nConstraints; jcol++ )
      {
         if ( slaveEqnListAux_[jcol] == irow )
         {
            searchIndex = slaveEqnList_[jcol];
            break;
         }
      }
      rnorm2 += (R_data[searchIndex-startRow] * R_data[searchIndex-startRow]);
   }
   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
      rnorm2 += (R_data[irow-startRow] * R_data[irow-startRow]);
   HYPRE_IJVectorDestroy(R);
   if ( mypid == 0 )
      printf("HYPRE_SlideRedction norm check = %e %e %e\n", sqrt(rnorm),
             sqrt(rnorm-rnorm2), sqrt(rnorm2));
#endif

   //----------------------------------------------------------------
   // clean up
   //----------------------------------------------------------------

   HYPRE_IJVectorDestroy(v1);
   HYPRE_IJVectorDestroy(x2);
   free( procNRows );
   return 0;
}

//****************************************************************************
// build A21 matrix
//----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildA21Mat()
{
   int    mypid, nprocs, *procNRows, startRow, endRow, localNRows;
   int    globalNConstr, globalNRows, nConstraints, A21NRows, A21NCols;
   int    A21GlobalNRows, A21GlobalNCols, A21StartRow, A21StartCol, ierr;
   int    rowCount, maxRowSize, newEndRow, *A21MatSize, irow, is, rowIndex;
   int    rowSize, *colInd, newRowSize, jcol, colIndex, searchIndex;
   int    nnzA21, *newColInd, procIndex, ncnt, newColIndex;
   double *colVal, *newColVal;
   char   fname[40];
   HYPRE_ParCSRMatrix A_csr, A21_csr;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow      = procNRows[mypid];
   endRow        = procNRows[mypid+1] - 1;
   localNRows    = endRow - startRow + 1;
   globalNConstr = procNConstr_[nprocs];
   globalNRows   = procNRows[nprocs];
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];

   //******************************************************************
   // extract A21 from A
   //------------------------------------------------------------------
   // calculate the dimension of A21
   //------------------------------------------------------------------

   A21NRows       = 2 * nConstraints;
   A21NCols       = localNRows - nConstraints;
   A21GlobalNRows = 2 * globalNConstr;
   A21GlobalNCols = globalNRows - globalNConstr;
   A21StartRow    = 2 * procNConstr_[mypid];
   A21StartCol    = procNRows[mypid] - procNConstr_[mypid];

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
   {
      printf("%4d : buildA21Mat - A21StartRow  = %d\n", mypid, A21StartRow);
      printf("%4d : buildA21Mat - A21GlobalDim = %d %d\n", mypid,
                                  A21GlobalNRows, A21GlobalNCols);
      printf("%4d : buildA21Mat - A21LocalDim  = %d %d\n",mypid,
                                  A21NRows, A21NCols);
   }

   //------------------------------------------------------------------
   // create a matrix context for A21
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpiComm_,A21StartRow,A21StartRow+A21NRows-1,
                                A21StartCol,A21StartCol+A21NCols-1,&A21mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A21mat_, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute the number of nonzeros in the first nConstraint row of A21
   // (which consists of the rows in selectedList), the nnz will
   // be reduced by excluding the constraint and selected slave columns
   //------------------------------------------------------------------

   rowCount   = maxRowSize = 0;
   newEndRow  = endRow - nConstraints;
   if ( A21NRows > 0 ) A21MatSize = new int[A21NRows];
   else                A21MatSize = NULL;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         colIndex = colInd[jcol];
         if ( colVal[jcol] != 0.0 )
         {
            if ( colIndex <= newEndRow || colIndex > endRow )
            {
               searchIndex = hypre_BinarySearch(gSlaveEqnList_,colIndex,
                                                globalNConstr);
               if ( searchIndex < 0 ) newRowSize++;
            }
         }
      }
      A21MatSize[irow] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
   }

   //------------------------------------------------------------------
   // compute the number of nonzeros in the second nConstraint row of A21
   // (which consists of the rows in constraint equations)
   //------------------------------------------------------------------

   rowCount = nConstraints;
   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         if ( colVal[jcol] != 0.0 )
         {
            colIndex = colInd[jcol];
            if ( colIndex <= newEndRow || colIndex > endRow )
            {
               searchIndex = hypre_BinarySearch(gSlaveEqnList_,colIndex,
                                                globalNConstr);
               if ( searchIndex < 0 ) newRowSize++;
            }
         }
      }
      A21MatSize[rowCount] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
      rowCount++;
   }
   nnzA21 = 0;
   for ( irow = 0; irow < 2*nConstraints; irow++ ) nnzA21 += A21MatSize[irow];
   MPI_Allreduce(&nnzA21,&ncnt,1,MPI_INT,MPI_SUM,mpiComm_);
   if ( mypid == 0 && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("   0 : buildA21Mat : NNZ of A21 = %d\n", ncnt);

   //------------------------------------------------------------------
   // after fetching the row sizes, set up A21 with such sizes
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixSetRowSizes(A21mat_, A21MatSize);
   ierr += HYPRE_IJMatrixInitialize(A21mat_);
   hypre_assert(!ierr);
   if ( A21NRows > 0 ) delete [] A21MatSize;

   //------------------------------------------------------------------
   // next load the first nConstraint row to A21 extracted from A
   // (at the same time, the D block is saved for future use)
   //------------------------------------------------------------------

   rowCount  = A21StartRow;
   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         if ( colVal[jcol] != 0.0 )
         {
            colIndex = colInd[jcol];
            if ( colIndex <= newEndRow || colIndex > endRow )
            {
               searchIndex = HYPRE_LSI_Search(gSlaveEqnList_,colIndex,
                                              globalNConstr);
               if ( searchIndex < 0 )
               {
                  for ( procIndex = 0; procIndex < nprocs; procIndex++ )
                     if ( procNRows[procIndex] > colIndex ) break;
                  procIndex--;
                  newColIndex = colIndex - procNConstr_[procIndex];
                  newColInd[newRowSize]   = newColIndex;
                  newColVal[newRowSize++] = colVal[jcol];
                  if ( newColIndex < 0 || newColIndex >= A21GlobalNCols )
                  {
                     printf("%4d : buildA21Mat ERROR - ",mypid);
                     printf(" out of range (%d,%d (%d))\n", rowCount,
                            colIndex, A21GlobalNCols);
                     for ( is = 0; is < rowSize; is++ )
                        printf("%4d : row %7d has col = %7d\n",mypid,rowIndex,
                               colInd[is]);
                     exit(1);
                  }
                  if ( newRowSize > maxRowSize+1 )
                  {
                     if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
                     {
                        printf("%4d : buildA21Mat WARNING - ",mypid);
                        printf("passing array boundary(1).\n");
                     }
                  }
               }
            }
         }
      }
      HYPRE_IJMatrixSetValues(A21mat_, 1, &newRowSize, (const int *) &rowCount,
                     (const int *) newColInd, (const double *) newColVal);
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      rowCount++;
   }

   //------------------------------------------------------------------
   // next load the second nConstraint rows to A21 extracted from A
   //------------------------------------------------------------------

   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         colIndex = colInd[jcol];
         if (colVal[jcol] != 0.0 &&
             (colIndex <= newEndRow || colIndex > endRow))
         {
            searchIndex = hypre_BinarySearch(gSlaveEqnList_, colIndex,
                                             globalNConstr);
            if ( searchIndex < 0 )
            {
               for ( procIndex = 0; procIndex < nprocs; procIndex++ )
                  if ( procNRows[procIndex] > colIndex ) break;
               procIndex--;
               newColIndex = colInd[jcol] - procNConstr_[procIndex];
               newColInd[newRowSize]   = newColIndex;
               newColVal[newRowSize++] = colVal[jcol];
             }
          }
      }
      HYPRE_IJMatrixSetValues(A21mat_, 1, &newRowSize, (const int *) &rowCount,
		(const int *) newColInd, (const double *) newColVal);
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
      rowCount++;
   }
   delete [] newColInd;
   delete [] newColVal;
   free( procNRows );

   //------------------------------------------------------------------
   // finally assemble the matrix and sanitize
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(A21mat_);
   HYPRE_IJMatrixGetObject(A21mat_, (void **) &A21_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A21_csr);

   if ( outputLevel_ >= 5 )
   {
      sprintf(fname, "A21.%d", mypid);
      FILE *fp = fopen(fname, "w");

      if ( mypid == 0 )
      {
         printf("====================================================\n");
         printf("%4d : Printing A21 matrix... \n", mypid);
         fflush(stdout);
      }
      for (irow = A21StartRow;irow < A21StartRow+2*nConstraints;irow++)
      {
         HYPRE_ParCSRMatrixGetRow(A21_csr,irow,&rowSize,&colInd,&colVal);
         for ( jcol = 0; jcol < rowSize; jcol++ )
            if ( colVal[jcol] != 0.0 )
               fprintf(fp, "%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                      colVal[jcol]);
         HYPRE_ParCSRMatrixRestoreRow(A21_csr, irow, &rowSize, &colInd,
                                      &colVal);
      }
      fclose(fp);
      if ( mypid == 0 )
         printf("====================================================\n");
   }
   return 0;
}

//****************************************************************************
// build invA22 matrix
// - given A22 = | B    C |, compute | 0       C^{-T}          |
//               | C^T  0 |          | C^{-1} -C^{-1} B C^{-T} |
//----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildInvA22Mat()
{
   int    mypid, nprocs, *procNRows, endRow;
   int    is, globalNConstr, nConstraints, irow, jcol, rowSize;
   int    *colInd, newEndRow, rowIndex, colIndex, searchIndex;
   int    ig, ir, ic, ierr, index, offset, newRowSize, *newColInd;
   int    maxBlkSize=HYPRE_SLIDEMAX, procIndex;
   int    nGroups, *groupIDs, **groupRowNums, *groupSizes, *iTempList;
   int     rowCount, *colInd2, rowSize2;
   double *colVal, *colVal2, **Imat, **Imat2, *newColVal;
   char   fname[40];
   HYPRE_ParCSRMatrix A_csr, invA22_csr;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   endRow        = procNRows[mypid+1] - 1;
   globalNConstr = procNConstr_[nprocs];
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];
   newEndRow     = endRow - nConstraints;

   //------------------------------------------------------------------
   // construct the group information
   //------------------------------------------------------------------

   nGroups = 0;
   if ( nConstraints > 0 )
   {
      iTempList = new int[nConstraints];
      for ( irow = 0; irow < nConstraints; irow++ )
         iTempList[irow] = constrBlkInfo_[irow];
      hypre_qsort0( iTempList, 0, nConstraints-1 );
      nGroups = 1;
      for ( irow = 1; irow < nConstraints; irow++ )
         if ( iTempList[irow] != iTempList[irow-1] ) nGroups++;
      groupIDs = new int[nGroups];
      groupSizes = new int[nGroups];
      groupIDs[0] = iTempList[0];
      groupSizes[0] = 1;
      nGroups = 1;
      for ( irow = 1; irow < nConstraints; irow++ )
      {
         if ( iTempList[irow] != iTempList[irow-1] )
         {
            groupSizes[nGroups] = 1;
            groupIDs[nGroups++] = iTempList[irow];
         }
         else groupSizes[nGroups-1]++;
      }
      groupRowNums = new int*[nGroups];
      for ( ig = 0; ig < nGroups; ig++ )
      {
         if ( groupSizes[ig] > maxBlkSize )
         {
            printf("%4d : buildInvA22 ERROR - block Size %d >= %d\n", mypid,
                   groupSizes[ig], maxBlkSize);
            printf("%4d : buildInvA22 ERROR - group ID = %d\n", mypid,
                   groupIDs[ig]);
            exit(1);
         }
      }
      for ( ig = 0; ig < nGroups; ig++ )
      {
         groupRowNums[ig] = new int[groupSizes[ig]];
         groupSizes[ig] = 0;
      }
      for ( irow = 0; irow < nConstraints; irow++ )
      {
         index = constrBlkInfo_[irow];
         searchIndex = hypre_BinarySearch(groupIDs, index, nGroups);
         groupRowNums[searchIndex][groupSizes[searchIndex]++] = irow;
      }
      delete [] iTempList;
   }

   //------------------------------------------------------------------
   // first extract the (2,1) block of A22
   // ( constraints-to-local slaves )
   //------------------------------------------------------------------

   int    *CT_JA = NULL, CTRowSize, CTOffset;
   double *CT_AA = NULL;
   if ( nConstraints > 0 )
   {
      CT_JA    = new int[nConstraints*maxBlkSize];
      CT_AA    = new double[nConstraints*maxBlkSize];
      for ( irow = 0; irow < nConstraints*maxBlkSize; irow++ ) CT_JA[irow] = -1;
   }
#if 0
FILE *fp = fopen("CT.m","w");
#endif
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = newEndRow + 1 + irow;
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      CTOffset  = maxBlkSize * irow;
      CTRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         searchIndex = hypre_BinarySearch(slaveEqnList_,colIndex,nConstraints);
         if ( searchIndex >= 0 )
         {
            CT_JA[CTOffset+CTRowSize] = slaveEqnListAux_[searchIndex];
            CT_AA[CTOffset+CTRowSize] = colVal[jcol];
            CTRowSize++;
#if 0
fprintf(fp,"%d %d %25.16e\n",irow+1,CT_JA[CTOffset+CTRowSize-1]+1,colVal[jcol]);
#endif
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
   }
#if 0
fclose(fp);
#endif

   //------------------------------------------------------------------
   // invert the (2,1) block of A22
   //------------------------------------------------------------------

#if 0
FILE *fp2 = fopen("invCT.m","w");
#endif
   Imat = hypre_TAlloc(double*,  maxBlkSize , HYPRE_MEMORY_HOST);
   for ( ir = 0; ir < maxBlkSize; ir++ )
      Imat[ir] = hypre_TAlloc(double,  maxBlkSize , HYPRE_MEMORY_HOST);

   for ( ig = 0; ig < nGroups; ig++ )
   {
      for ( ir = 0; ir < groupSizes[ig]; ir++ )
         for ( ic = 0; ic < groupSizes[ig]; ic++ ) Imat[ir][ic] = 0.0;
      for ( ir = 0; ir < groupSizes[ig]; ir++ )
      {
         rowIndex = groupRowNums[ig][ir];
         offset   = rowIndex * maxBlkSize;
         for ( ic = 0; ic < maxBlkSize; ic++ )
         {
            colIndex = CT_JA[offset+ic];
            if ( colIndex != -1 )
            {
               for ( is = 0; is < groupSizes[ig]; is++ )
                  if ( colIndex == groupRowNums[ig][is] ) break;
               Imat[ir][is] = CT_AA[offset+ic];
            }
         }
      }
      ierr = HYPRE_LSI_MatrixInverse((double**) Imat, groupSizes[ig], &Imat2);
      if ( ierr )
      {
         printf("Failed Block %d has indices (%d) : ", ig, groupSizes[ig]);
         for ( ir = 0; ir < groupSizes[ig]; ir++ )
            printf(" %d ", groupRowNums[ig][ir]);
         printf("\n");
         for ( ir = 0; ir < groupSizes[ig]; ir++ )
         {
            for ( ic = 0; ic < groupSizes[ig]; ic++ )
               printf(" %e ", Imat[ir][ic]);
            printf("\n");
         }
         printf("\n");
         for ( ir = 0; ir < groupSizes[ig]; ir++ )
         {
            for ( ic = 0; ic < groupSizes[ig]; ic++ )
               printf(" %e ", Imat2[ir][ic]);
            printf("\n");
         }
         printf("\n");
         for ( ir = 0; ir < groupSizes[ig]; ir++ )
         {
            rowIndex = groupRowNums[ig][ir];
            offset   = rowIndex * maxBlkSize;
            for ( ic = 0; ic < maxBlkSize; ic++ )
            {
               colIndex = CT_JA[offset+ic];
               if ( colIndex != -1 )
               {
                  printf("  rowIndex,colIndex,val = %d %d %e\n",rowIndex,
                         colIndex,CT_AA[offset+ic]);
               }
            }
         }
      }
      hypre_assert( !ierr );
      for ( ir = 0; ir < groupSizes[ig]; ir++ )
      {
         rowIndex = groupRowNums[ig][ir];
         offset   = rowIndex * maxBlkSize;
         for (ic = 0; ic < maxBlkSize; ic++) CT_JA[offset+ic] = -1;
         for ( ic = 0; ic < groupSizes[ig]; ic++ )
         {
            if ( Imat2[ir][ic] != 0.0 )
            {
               CT_JA[offset+ic] = groupRowNums[ig][ic];
               CT_AA[offset+ic] = Imat2[ir][ic];
#if 0
fprintf(fp2,"%d %d %25.16e\n",rowIndex+1,CT_JA[offset+ic]+1,CT_AA[offset+ic]);
#endif
            }
         }
         free( Imat2[ir] );
      }
      free( Imat2 );
   }
#if 0
fclose(fp2);
#endif
   for ( ir = 0; ir < maxBlkSize; ir++ ) free( Imat[ir] );
   free( Imat );

   //------------------------------------------------------------------
   // form ParCSRMatrix of the (2,1) block of A22
   //------------------------------------------------------------------

   int                *hypreCTMatSize, maxRowSize;
   hypre_ParCSRMatrix *hypreCT;
   HYPRE_IJMatrix     IJCT;

   ierr = HYPRE_IJMatrixCreate(mpiComm_, procNConstr_[mypid],
                    procNConstr_[mypid]+nConstraints-1, procNConstr_[mypid],
                    procNConstr_[mypid]+nConstraints-1, &IJCT);
   ierr += HYPRE_IJMatrixSetObjectType(IJCT, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) hypreCTMatSize = new int[nConstraints];
   else                    hypreCTMatSize = NULL;
   maxRowSize = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      newRowSize = 0;
      offset     = irow * maxBlkSize;
      for ( ic = 0; ic < maxBlkSize; ic++ )
         if ( CT_JA[offset+ic] != -1 ) newRowSize++;
      hypreCTMatSize[irow] = newRowSize;
      maxRowSize = (newRowSize > maxRowSize) ? newRowSize : maxRowSize;
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJCT, hypreCTMatSize);
   ierr = HYPRE_IJMatrixInitialize(IJCT);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) delete [] hypreCTMatSize;
   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex   = procNConstr_[mypid] + irow;
      offset     = irow * maxBlkSize;
      newRowSize = 0;
      for ( ic = 0; ic < maxBlkSize; ic++ )
      {
         if ( CT_JA[offset+ic] != -1 )
         {
            newColInd[newRowSize]   = CT_JA[offset+ic] + procNConstr_[mypid];
            newColVal[newRowSize++] = CT_AA[offset+ic];
         }
      }
      ierr = HYPRE_IJMatrixSetValues(IJCT, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert( !ierr );
   }
   delete [] newColInd;
   delete [] newColVal;
   if ( nConstraints > 0 )
   {
      delete [] CT_JA;
      delete [] CT_AA;
   }
   HYPRE_IJMatrixAssemble(IJCT);
   HYPRE_IJMatrixGetObject(IJCT, (void **) &hypreCT);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreCT);

   //------------------------------------------------------------------
   // next extract the (1,2) block of A22
   // ( local slaves-to-constraints )
   //------------------------------------------------------------------

   int    *C_JA = NULL, CRowSize, COffset;
   double *C_AA = NULL;
   if ( nConstraints > 0 )
   {
      C_JA    = new int[nConstraints*maxBlkSize];
      C_AA    = new double[nConstraints*maxBlkSize];
      for ( irow = 0; irow < nConstraints*maxBlkSize; irow++ ) C_JA[irow] = -1;
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      CRowSize = 0;
      COffset  = maxBlkSize * irow;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         if ( colIndex > newEndRow && colIndex <= endRow )
         {
            C_JA[COffset+CRowSize] = colIndex - newEndRow - 1;
            C_AA[COffset+CRowSize] = colVal[jcol];
            CRowSize++;
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
   }

   //------------------------------------------------------------------
   // invert the (2,1) block of A22
   //------------------------------------------------------------------

   Imat = hypre_TAlloc(double*,  maxBlkSize , HYPRE_MEMORY_HOST);
   for ( ir = 0; ir < maxBlkSize; ir++ )
      Imat[ir] = hypre_TAlloc(double,  maxBlkSize , HYPRE_MEMORY_HOST);

   for ( ig = 0; ig < nGroups; ig++ )
   {
      for ( ir = 0; ir < groupSizes[ig]; ir++ )
         for ( ic = 0; ic < groupSizes[ig]; ic++ ) Imat[ir][ic] = 0.0;
      for ( ir = 0; ir < groupSizes[ig]; ir++ )
      {
         rowIndex = groupRowNums[ig][ir];
         offset   = rowIndex * maxBlkSize;
         for ( ic = 0; ic < maxBlkSize; ic++ )
         {
            colIndex = C_JA[offset+ic];
            if ( colIndex != -1 )
            {
               for ( is = 0; is < groupSizes[ig]; is++ )
                  if ( colIndex == groupRowNums[ig][is] ) break;
               Imat[ir][is] = C_AA[offset+ic];
            }
         }
      }
      ierr = HYPRE_LSI_MatrixInverse((double**) Imat, groupSizes[ig], &Imat2);
      hypre_assert( !ierr );
      for ( ir = 0; ir < groupSizes[ig]; ir++ )
      {
         rowIndex = groupRowNums[ig][ir];
         offset   = rowIndex * maxBlkSize;
         for (ic = 0; ic < maxBlkSize; ic++) C_JA[offset+ic] = -1;
         for ( ic = 0; ic < groupSizes[ig]; ic++ )
         {
            if ( Imat2[ir][ic] != 0.0 )
            {
               C_JA[offset+ic] = groupRowNums[ig][ic];
               C_AA[offset+ic] = Imat2[ir][ic];
            }
         }
         free( Imat2[ir] );
      }
      free( Imat2 );
   }
   for ( ir = 0; ir < maxBlkSize; ir++ ) free( Imat[ir] );
   free( Imat );

   //------------------------------------------------------------------
   // form ParCSRMatrix of the (1,2) block of A22
   //------------------------------------------------------------------

   int                *hypreCMatSize;
   hypre_ParCSRMatrix *hypreC;
   HYPRE_IJMatrix     IJC;

   ierr = HYPRE_IJMatrixCreate(mpiComm_, procNConstr_[mypid],
                    procNConstr_[mypid]+nConstraints-1, procNConstr_[mypid],
                    procNConstr_[mypid]+nConstraints-1, &IJC);
   ierr += HYPRE_IJMatrixSetObjectType(IJC, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) hypreCMatSize = new int[nConstraints];
   else                    hypreCMatSize = NULL;
   maxRowSize = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      newRowSize = 0;
      offset     = irow * maxBlkSize;
      for ( ic = 0; ic < maxBlkSize; ic++ )
         if ( C_JA[offset+ic] != -1 ) newRowSize++;
      hypreCMatSize[irow] = newRowSize;
      maxRowSize = (newRowSize > maxRowSize) ? newRowSize : maxRowSize;
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJC, hypreCMatSize);
   ierr = HYPRE_IJMatrixInitialize(IJC);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) delete [] hypreCMatSize;
   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      offset     = irow * maxBlkSize;
      newRowSize = 0;
      for ( ic = 0; ic < maxBlkSize; ic++ )
      {
         if ( C_JA[offset+ic] != -1 )
         {
            newColInd[newRowSize]   = C_JA[offset+ic] + procNConstr_[mypid];
            newColVal[newRowSize++] = C_AA[offset+ic];
         }
      }
      ierr = HYPRE_IJMatrixSetValues(IJC, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert( !ierr );
   }
   delete [] newColInd;
   delete [] newColVal;
   if ( nConstraints > 0 )
   {
      delete [] C_JA;
      delete [] C_AA;
   }
   HYPRE_IJMatrixAssemble(IJC);
   HYPRE_IJMatrixGetObject(IJC, (void **) &hypreC);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreC);
   if ( nConstraints > 0 )
   {
      delete [] groupIDs;
      delete [] groupSizes;
      for ( ig = 0; ig < nGroups; ig++ ) delete [] groupRowNums[ig];
      delete [] groupRowNums;
   }

   //------------------------------------------------------------------
   // form ParCSRMatrix of the (2,2) block of the invA22 matrix
   //------------------------------------------------------------------

   int                *hypreBMatSize=NULL;
   hypre_ParCSRMatrix *hypreB;
   HYPRE_IJMatrix     IJB;

   ierr = HYPRE_IJMatrixCreate(mpiComm_, procNConstr_[mypid],
                procNConstr_[mypid]+nConstraints-1, procNConstr_[mypid],
                procNConstr_[mypid]+nConstraints-1, &IJB);
   ierr = HYPRE_IJMatrixSetObjectType(IJB, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) hypreBMatSize = new int[nConstraints];
   maxRowSize = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         searchIndex = hypre_BinarySearch(gSlaveEqnList_, colIndex,
                                          globalNConstr);
         if ( searchIndex >= 0 ) newRowSize++;
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      hypreBMatSize[irow] = newRowSize;
      maxRowSize = (newRowSize > maxRowSize) ? newRowSize : maxRowSize;
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJB, hypreBMatSize);
   ierr = HYPRE_IJMatrixInitialize(IJB);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) delete [] hypreBMatSize;

   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         searchIndex = hypre_BinarySearch(gSlaveEqnList_, colIndex,
                                          globalNConstr);
         if ( searchIndex >= 0 )
         {
            newColInd[newRowSize] = gSlaveEqnListAux_[searchIndex];
            newColVal[newRowSize++] = - colVal[jcol];
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      rowIndex = procNConstr_[mypid] + irow;
      ierr = HYPRE_IJMatrixSetValues(IJB, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert( !ierr );
   }
   HYPRE_IJMatrixAssemble(IJB);
   HYPRE_IJMatrixGetObject(IJB, (void **) &hypreB);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreB);
   delete [] newColInd;
   delete [] newColVal;

   //------------------------------------------------------------------
   // perform triple matrix product - C^{-1} B C^{-T}
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrix hypreCBC;

#if 0
   strcpy( fname, "hypreCT" );
   HYPRE_ParCSRMatrixPrint((HYPRE_ParCSRMatrix) hypreCT, fname);
   strcpy( fname, "hypreB" );
   HYPRE_ParCSRMatrixPrint((HYPRE_ParCSRMatrix) hypreB, fname);
#endif

   hypre_BoomerAMGBuildCoarseOperator(hypreCT, hypreB, hypreCT,
                                      (hypre_ParCSRMatrix **) &hypreCBC);
#if 0
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCT,irow,&rowSize,
                              &colInd,&colVal);
      for (jcol = 0; jcol < rowSize; jcol++ )
         printf("CT : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCT,irow,&rowSize,
                                    &colInd,&colVal);
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreB,irow,&rowSize,
                              &colInd,&colVal);
      for (jcol = 0; jcol < rowSize; jcol++ )
         printf("B : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreB,irow,&rowSize,
                                    &colInd,&colVal);
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                              &colInd,&colVal);
      for (jcol = 0; jcol < rowSize; jcol++ )
         printf("CBC : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                                    &colInd,&colVal);
   }
#endif

   HYPRE_IJMatrixDestroy( IJB );

   //------------------------------------------------------------------
   // calculate the dimension of invA22
   //------------------------------------------------------------------

   int invA22NRows       = 2 * nConstraints;
   int invA22NCols       = invA22NRows;
   int invA22StartRow    = 2 * procNConstr_[mypid];
   int invA22StartCol    = invA22StartRow;
   int *invA22MatSize=NULL;

   //------------------------------------------------------------------
   // create a matrix context for A22
   //------------------------------------------------------------------

   ierr = HYPRE_IJMatrixCreate(mpiComm_, invA22StartRow,
                    invA22StartRow+invA22NRows-1, invA22StartCol,
                    invA22StartCol+invA22NCols-1, &invA22mat_);
   ierr += HYPRE_IJMatrixSetObjectType(invA22mat_, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute the no. of nonzeros in the first nConstraint row of invA22
   //------------------------------------------------------------------

   maxRowSize  = 0;
   if ( invA22NRows > 0 ) invA22MatSize = new int[invA22NRows];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCT,rowIndex,
                                       &rowSize,NULL,NULL);
      hypre_assert( !ierr );
      invA22MatSize[irow] = rowSize;
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCT,rowIndex,
                                   &rowSize,NULL,NULL);
      maxRowSize = ( rowSize > maxRowSize ) ? rowSize : maxRowSize;
   }

   //------------------------------------------------------------------
   // compute the number of nonzeros in the second nConstraints row of
   // invA22 (consisting of [D and A22 block])
   //------------------------------------------------------------------

#if 0
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                              &colInd,&colVal);
      for (jcol = 0; jcol < rowSize; jcol++ )
         printf("CBC1 : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                                    &colInd,&colVal);
   }
#endif
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreC,rowIndex,
                               &rowSize,&colInd,&colVal);
      hypre_assert( !ierr );
      newRowSize = rowSize;
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreC,rowIndex,
                                   &rowSize,&colInd,&colVal);
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,rowIndex,
                                      &rowSize,&colInd,&colVal);
      hypre_assert( !ierr );
      newRowSize += rowSize;
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,rowIndex,
                                   &rowSize,&colInd,&colVal);
      invA22MatSize[nConstraints+irow] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
   }

   //------------------------------------------------------------------
   // after fetching the row sizes, set up invA22 with such sizes
   //------------------------------------------------------------------

#if 0
   for ( irow = 0; irow < nConstraints; irow++ )
   {
   HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                           &colInd,&colVal);
   for (jcol = 0; jcol < rowSize; jcol++ )
      printf("CBC2 : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
   HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                                 &colInd,&colVal);
}
#endif
   ierr  = HYPRE_IJMatrixSetRowSizes(invA22mat_, invA22MatSize);
   ierr += HYPRE_IJMatrixInitialize(invA22mat_);
   hypre_assert(!ierr);
   if ( invA22NRows > 0 ) delete [] invA22MatSize;

   //------------------------------------------------------------------
   // next load the first nConstraints row to invA22 extracted from A
   // (that is, the D block)
   //------------------------------------------------------------------

   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCT,rowIndex,
                                     &rowSize,&colInd,&colVal);
      hypre_assert( !ierr );
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         newColInd[newRowSize] = colInd[jcol] + procNConstr_[mypid] +
                                 nConstraints;
         newColVal[newRowSize++] = colVal[jcol];
      }
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCT,rowIndex,
                                   &rowSize,&colInd,&colVal);
      rowCount = invA22StartRow + irow;
      ierr = HYPRE_IJMatrixSetValues(invA22mat_, 1, &rowSize,
                (const int *) &rowCount, (const int *) newColInd,
                (const double *) newColVal);
      hypre_assert(!ierr);
   }

   //------------------------------------------------------------------
   // next load the second nConstraints rows to A22 extracted from A
   //------------------------------------------------------------------

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex   = procNConstr_[mypid] + irow;
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreC,rowIndex,
                               &rowSize,&colInd,&colVal);
      hypre_assert( !ierr );
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         newColInd[newRowSize] = colInd[jcol] + procNConstr_[mypid];
         newColVal[newRowSize++] = colVal[jcol];
      }
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreC,rowIndex,
                                   &rowSize,&colInd,&colVal);
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,rowIndex,
                                      &rowSize2,&colInd2,&colVal2);
      hypre_assert( !ierr );
      for ( jcol = 0; jcol < rowSize2; jcol++ )
      {
         colIndex = colInd2[jcol];
         for ( procIndex = 0; procIndex <= nprocs; procIndex++ )
            if ( procNConstr_[procIndex] > colIndex ) break;
         newColInd[newRowSize] = colIndex + procNConstr_[procIndex];
         newColVal[newRowSize++] = colVal2[jcol];
      }
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,rowIndex,
                                   &rowSize2,&colInd2,&colVal2);
      rowCount = invA22StartRow + nConstraints + irow;
      ierr = HYPRE_IJMatrixSetValues(invA22mat_, 1, &newRowSize,
		(const int *) &rowCount, (const int *) newColInd,
		(const double *) newColVal);
      hypre_assert(!ierr);
   }
   delete [] newColInd;
   delete [] newColVal;
   free( procNRows );
   HYPRE_IJMatrixDestroy( IJC );
   HYPRE_IJMatrixDestroy( IJCT );
   HYPRE_ParCSRMatrixDestroy( (HYPRE_ParCSRMatrix) hypreCBC );

   //------------------------------------------------------------------
   // finally assemble the matrix and sanitize
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(invA22mat_);
   HYPRE_IJMatrixGetObject(invA22mat_, (void **) &invA22_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) invA22_csr);

   if ( outputLevel_ >= 5 )
   {
      sprintf( fname, "invA.%d", mypid );
      FILE *fp = fopen( fname, "w");

      if ( mypid == 0 )
      {
         printf("====================================================\n");
         printf("%4d : Printing invA22 matrix... \n", mypid);
         fflush(stdout);
      }
      for (irow=invA22StartRow; irow < invA22StartRow+invA22NRows;irow++)
      {
         HYPRE_ParCSRMatrixGetRow(invA22_csr,irow,&rowSize,&colInd,&colVal);
         for ( jcol = 0; jcol < rowSize; jcol++ )
            if ( colVal[jcol] != 0.0 )
               fprintf(fp,"%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                      colVal[jcol]);
         HYPRE_ParCSRMatrixRestoreRow(invA22_csr,irow,&rowSize,&colInd,
                                      &colVal);
      }
      fclose(fp);
      if ( mypid == 0 )
         printf("====================================================\n");
   }
   return 0;
}

//***************************************************************************
// scale matrix
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::scaleMatrixVector()
{
   int                *partition, startRow, localNRows, index, offset;
   int                 irow, jcol, iP, rowSize, *colInd, *rowLengs;
   int                 mypid, nprocs;
   int                 nSends, *sendStarts, *sendMap, *offdMap, ierr;
   int                 nRecvs, *recvStarts, pstart, pend, maxRowLeng;
   int                 *ADiagI, *ADiagJ, *AOffdI, *AOffdJ, rowInd;
   double              *ADiagA, *AOffdA, *bData, *b2Data;
   double              *scaleVec, *extScaleVec, *colVal, *sBuffer;
   HYPRE_IJMatrix      newA;
   HYPRE_IJVector      newB;
   hypre_ParCSRMatrix  *A_csr;
   hypre_CSRMatrix     *ADiag, *AOffd;
   hypre_ParVector     *b_csr, *b2_csr;
   hypre_ParCSRCommPkg *commPkg;
   hypre_ParCSRCommHandle *commHandle;

   //-----------------------------------------------------------------------
   // fetch matrix and parameters
   //-----------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   MPI_Comm_size( mpiComm_, &nprocs );
   HYPRE_IJMatrixGetObject(reducedAmat_, (void **) &A_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A_csr,&partition);
   startRow    = partition[mypid];
   localNRows  = partition[mypid+1] - startRow;
   free( partition );
   ADiag  = hypre_ParCSRMatrixDiag(A_csr);
   ADiagI = hypre_CSRMatrixI(ADiag);
   ADiagJ = hypre_CSRMatrixJ(ADiag);
   ADiagA = hypre_CSRMatrixData(ADiag);
   AOffd  = hypre_ParCSRMatrixOffd(A_csr);
   AOffdI = hypre_CSRMatrixI(AOffd);
   AOffdJ = hypre_CSRMatrixJ(AOffd);
   AOffdA = hypre_CSRMatrixData(AOffd);
   HYPRE_IJVectorGetObject(reducedBvec_, (void **) &b_csr);
   bData  = hypre_VectorData(hypre_ParVectorLocalVector(b_csr));

   offdMap = hypre_ParCSRMatrixColMapOffd(A_csr);
   commPkg = hypre_ParCSRMatrixCommPkg((hypre_ParCSRMatrix *) A_csr);
   nSends  = hypre_ParCSRCommPkgNumSends(commPkg);
   nRecvs  = hypre_ParCSRCommPkgNumRecvs(commPkg);
   recvStarts = hypre_ParCSRCommPkgRecvVecStarts(commPkg);
   sendStarts = hypre_ParCSRCommPkgSendMapStarts(commPkg);
   sendMap    = hypre_ParCSRCommPkgSendMapElmts(commPkg);

   //-----------------------------------------------------------------------
   // fetch diagonal of A
   //-----------------------------------------------------------------------

   scaleVec  = new double[localNRows];
   rowLengs  = new int[localNRows];
   extScaleVec = NULL;
   if ( nRecvs > 0 ) extScaleVec = new double[recvStarts[nRecvs]];

   maxRowLeng = 0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      scaleVec[irow] = 0.0;
      rowLengs[irow] = ADiagI[irow+1] - ADiagI[irow] +
                       AOffdI[irow+1] - AOffdI[irow];
      if ( rowLengs[irow] > maxRowLeng ) maxRowLeng = rowLengs[irow];
      for ( jcol = ADiagI[irow]; jcol < ADiagI[irow+1];  jcol++ )
         if ( ADiagJ[jcol] == irow ) scaleVec[irow] = ADiagA[jcol];
   }
   for ( irow = 0; irow < localNRows; irow++ )
   {
      if ( habs( scaleVec[irow] ) == 0.0 )
      {
         printf("%d : scaleMatrixVector - diag %d = %e <= 0 \n",mypid,irow,
                scaleVec[irow]);
         exit(1);
      }
      scaleVec[irow] = 1.0/sqrt(scaleVec[irow]);
   }

   //-----------------------------------------------------------------------
   // exchange diagonal of A
   //-----------------------------------------------------------------------

   if ( nSends > 0 )
   {
      sBuffer = new double[sendStarts[nSends]];
      offset = 0;
      for ( iP = 0; iP < nSends; iP++ )
      {
         pstart = sendStarts[iP];
         pend   = sendStarts[iP+1];
         for ( jcol = pstart; jcol < pend; jcol++ )
         {
            index = sendMap[jcol];
            sBuffer[offset++] = scaleVec[index];
         }
      }
   }
   else sBuffer = NULL;

   commHandle = hypre_ParCSRCommHandleCreate(1,commPkg,sBuffer,extScaleVec);
   hypre_ParCSRCommHandleDestroy(commHandle);

   if ( nSends > 0 ) delete [] sBuffer;

   //-----------------------------------------------------------------------
   // construct new matrix
   //-----------------------------------------------------------------------

   HYPRE_IJMatrixCreate(mpiComm_, startRow, startRow+localNRows-1,
                        startRow, startRow+localNRows-1, &newA);
   HYPRE_IJMatrixSetObjectType(newA, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(newA, rowLengs);
   HYPRE_IJMatrixInitialize(newA);
   delete [] rowLengs;
   colInd = new int[maxRowLeng];
   colVal = new double[maxRowLeng];
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rowSize = 0;
      for ( jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++ )
      {
         index = ADiagJ[jcol];
         colInd[rowSize] = index + startRow;
         colVal[rowSize++] = scaleVec[irow]*scaleVec[index]*ADiagA[jcol];
      }
      for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
      {
         index = AOffdJ[jcol];
         colInd[rowSize] = offdMap[index];
         colVal[rowSize++] = scaleVec[irow]*extScaleVec[index]*AOffdA[jcol];
      }
      rowInd = irow + startRow;
      HYPRE_IJMatrixSetValues(newA, 1, &rowSize, (const int *) &rowInd,
                  (const int *) colInd, (const double *) colVal);
   }
   HYPRE_IJMatrixAssemble(newA);
   delete [] colInd;
   delete [] colVal;
   delete [] extScaleVec;

   //-----------------------------------------------------------------------
   // construct new vector
   //-----------------------------------------------------------------------

   ierr  = HYPRE_IJVectorCreate(mpiComm_,startRow,startRow+localNRows-1,&newB);
   ierr += HYPRE_IJVectorSetObjectType(newB, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(newB);
   ierr += HYPRE_IJVectorAssemble(newB);
   ierr += HYPRE_IJVectorGetObject(newB, (void **) &b2_csr);
   b2Data = hypre_VectorData(hypre_ParVectorLocalVector(b2_csr));
   hypre_assert( !ierr );
   for ( irow = 0; irow < localNRows; irow++ )
      b2Data[irow] = bData[irow] * scaleVec[irow];

   ADiagISqrts_ = scaleVec;
   reducedAmat_ = newA;
   reducedBvec_ = newB;
   return 0;
}

//****************************************************************************
// estimate conditioning of a small block
//----------------------------------------------------------------------------

double HYPRE_SlideReduction::matrixCondEst(int globalRowID, int globalColID,
                                           int *blkInfo,int blkCnt)
{
   int    mypid, nprocs, *procNRows, endRow, nConstraints;
   int    localBlkCnt, *localBlkInfo, irow, jcol, matDim, searchIndex;
   int    *rowIndices, rowSize, *colInd, rowIndex, index2, ierr;
   int    *localSlaveEqns, *localSlaveAuxs;
   double *colVal, **matrix, **matrix2, retVal, value;
   HYPRE_ParCSRMatrix A_csr;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   endRow       = procNRows[mypid+1] - 1;
   nConstraints = procNConstr_[mypid+1] - procNConstr_[mypid];
   free( procNRows );

   //------------------------------------------------------------------
   // collect all row indices
   //------------------------------------------------------------------

   localBlkCnt  = blkCnt;
   localBlkInfo = new int[blkCnt];
   for (irow = 0; irow < blkCnt; irow++) localBlkInfo[irow] = blkInfo[irow];

   hypre_qsort0(localBlkInfo, 0, localBlkCnt-1);
   matDim = 1;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      searchIndex = hypre_BinarySearch(localBlkInfo, constrBlkInfo_[irow],
                                       localBlkCnt);
      if ( searchIndex >= 0 ) matDim++;
   }
   rowIndices = new int[matDim];
   matDim = 0;
   rowIndices[matDim++] = globalRowID;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      searchIndex = hypre_BinarySearch(localBlkInfo, constrBlkInfo_[irow],
                                       localBlkCnt);
      if ( searchIndex >= 0 )
         rowIndices[matDim++] = endRow - nConstraints + irow + 1;
   }
   hypre_qsort0(rowIndices, 0, matDim-1);
   matrix = hypre_TAlloc(double*,  matDim , HYPRE_MEMORY_HOST);
   localSlaveEqns = new int[nConstraints];
   localSlaveAuxs = new int[nConstraints];
   for ( irow = 0; irow < nConstraints; irow++ )
      localSlaveEqns[irow] = slaveEqnList_[irow];
   localSlaveEqns[globalRowID-(endRow+1-nConstraints)] = globalColID;
   for ( irow = 0; irow < nConstraints; irow++ )
      localSlaveAuxs[irow] = irow;
   HYPRE_LSI_qsort1a(localSlaveEqns, localSlaveAuxs, 0, nConstraints-1);

   for ( irow = 0; irow < matDim; irow++ )
   {
      matrix[irow] = hypre_TAlloc(double,  matDim , HYPRE_MEMORY_HOST);
      for ( jcol = 0; jcol < matDim; jcol++ ) matrix[irow][jcol] = 0.0;
   }
   for ( irow = 0; irow < matDim; irow++ )
   {
      rowIndex = rowIndices[irow];
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         searchIndex = hypre_BinarySearch(localSlaveEqns,colInd[jcol],
                                          nConstraints);
         if ( searchIndex >= 0 )
         {
            index2 = localSlaveAuxs[searchIndex] + endRow - nConstraints + 1;
            searchIndex = hypre_BinarySearch(rowIndices,index2,matDim);
            if ( searchIndex >= 0 ) matrix[irow][searchIndex] = colVal[jcol];
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
   }
#if 0
   if ( matDim <= 4 )
      for ( irow = 0; irow < matDim; irow++ )
      {
         for ( jcol = 0; jcol < matDim; jcol++ )
            printf(" %e ", matrix[irow][jcol]);
         printf("\n");
      }
#endif
   ierr = HYPRE_LSI_MatrixInverse((double**) matrix,matDim,&matrix2);
   if ( ierr ) retVal = 1.0e-10;
   else
   {
      retVal = 0.0;
      for ( irow = 0; irow < matDim; irow++ )
      {
         for ( jcol = 0; jcol < matDim; jcol++ )
         {
            value  = habs(matrix2[irow][jcol]);
            retVal = ( value > retVal ) ? value : retVal;
         }
      }
      retVal = 1.0 / retVal;
      for ( irow = 0; irow < matDim; irow++ ) free(matrix2[irow]);
      free( matrix2 );
   }
   for ( irow = 0; irow < matDim; irow++ ) free(matrix[irow]);
   free( matrix );
   delete [] localBlkInfo;
   delete [] rowIndices;
   delete [] localSlaveEqns;
   delete [] localSlaveAuxs;
   return retVal;
}

//***************************************************************************
//***************************************************************************
//   Obsolete, but keep it for now
//***************************************************************************
// search for a slave equation list (block size = 2)
//---------------------------------------------------------------------------

int HYPRE_SlideReduction::findSlaveEqns2(int **couplings)
{
   int    mypid, nprocs, *procNRows, startRow, endRow;
   int    nConstraints, irow, jcol, rowSize, ncnt, *colInd;
   int    nCandidates, *candidateList;
   int    *constrListAux, colIndex, searchIndex, newEndRow;
   int    *constrListAux2;
   int    constrIndex, uBound, lBound, nSum, nPairs, index;
   double *colVal, searchValue;
   HYPRE_ParCSRMatrix A_csr;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow      = procNRows[mypid];
   endRow        = procNRows[mypid+1] - 1;
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];
   newEndRow     = endRow - nConstraints;
   nPairs        = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
      if ( slaveEqnList_[irow] == -1 ) nPairs++;
   (*couplings) = new int[nPairs*2+1];
   (*couplings)[0] = nPairs;

   //------------------------------------------------------------------
   // compose candidate slave list (slaves in candidateList, corresponding
   // constraint equation in constrListAux and constrListAux2)
   //------------------------------------------------------------------

   nCandidates    = 0;
   candidateList  = NULL;
   constrListAux  = NULL;
   constrListAux2 = NULL;
   if ( nConstraints > 0 )
   {
      candidateList  = new int[endRow-nConstraints-startRow+1];
      constrListAux  = new int[endRow-nConstraints-startRow+1];
      constrListAux2 = new int[endRow-nConstraints-startRow+1];

      //---------------------------------------------------------------
      // candidates are those with 2 links to the constraint list
      //---------------------------------------------------------------

      uBound = procNRows[mypid+1];
      lBound = uBound - nConstraints;

      for ( irow = startRow; irow <= endRow-nConstraints; irow++ )
      {
         HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
         ncnt = 0;
         constrListAux[nCandidates]  = -1;
         constrListAux2[nCandidates] = -1;
         for ( jcol = 0;  jcol < rowSize;  jcol++ )
         {
            colIndex = colInd[jcol];
            if ( colIndex >= lBound && colIndex < uBound )
            {
               ncnt++;

               if ( ncnt == 1 && constrListAux[nCandidates] == -1 )
                  constrListAux[nCandidates] = colIndex;
               else if ( ncnt == 2 && constrListAux2[nCandidates] == -1 )
                  constrListAux2[nCandidates] = colIndex;
            }
            if ( ncnt > 2 ) break;
         }
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
         if ( ncnt == 2 )
         {
            if ( constrListAux[nCandidates] > newEndRow &&
                 constrListAux[nCandidates] <= endRow &&
                 constrListAux2[nCandidates] > newEndRow &&
                 constrListAux2[nCandidates] <= endRow )
            {
               candidateList[nCandidates++] = irow;
               if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
                  printf("%4d : findSlaveEqns2 - candidate %d = %d\n",
                         mypid, nCandidates-1, irow);
            }
         }
      }
      if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
         printf("%4d : findSlaveEqns2 - nCandidates, nConstr = %d %d\n",
                   mypid, nCandidates, nConstraints);
   }

   //---------------------------------------------------------------------
   // search the constraint equations for the selected slave equations
   // (search for candidates column index with maximum magnitude)
   // ==> slaveEqnList_
   //---------------------------------------------------------------------

   nPairs      = 0;
   searchIndex = 0;

   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
   {
      if ( slaveEqnList_[irow-endRow+nConstraints-1] == -1 )
      {
         HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
         searchIndex = -1;
         searchValue = -1.0E10;
         for ( jcol = 0;  jcol < rowSize;  jcol++ )
         {
            if (colVal[jcol] != 0.0 && colInd[jcol] >= startRow
                                    && colInd[jcol] <= (endRow-nConstraints))
            {
               colIndex = hypre_BinarySearch(candidateList, colInd[jcol],
                                             nCandidates);

               /* -- if the column corresponds to a candidate, then    -- */
               /* -- see if that candidate has a constraint connection -- */
               /* -- that has been satisfied (slaveEqnList_[irow])     -- */

               if ( colIndex >= 0 )
               {
                  constrIndex = constrListAux[colIndex];
                  if ( constrIndex == irow )
                     constrIndex = constrListAux2[colIndex];
                  if (slaveEqnList_[constrIndex-endRow+nConstraints-1] != -1)
                  {
                     if ( habs(colVal[jcol]) > searchValue )
                     {
                        searchValue = habs(colVal[jcol]);
                        searchIndex = colInd[jcol];
                     }
                  }
               }
            }
         }
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
         if ( searchIndex >= 0 )
         {
            slaveEqnList_[irow-endRow+nConstraints-1] = searchIndex;
            index = hypre_BinarySearch(candidateList,searchIndex,nCandidates);
            (*couplings)[nPairs*2+1] = constrListAux[index];
            (*couplings)[nPairs*2+2] = constrListAux2[index];
            nPairs++;
            if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
               printf("%4d : findSlaveEqns2 - constr %d <=> slave %d\n",
                      mypid, irow, searchIndex);
         }
         else
         {
            if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
            {
               printf("%4d : findSlaveEqns2 - constraint %4d fails", mypid,
                      irow);
               printf(" to find a slave.\n");
            }
            break;
         }
      }
   }
   if ( nConstraints > 0 )
   {
      delete [] constrListAux;
      delete [] constrListAux2;
      delete [] candidateList;
   }
   free( procNRows );

   //---------------------------------------------------------------------
   // if not all constraint-slave pairs can be found, return -1
   //---------------------------------------------------------------------

   ncnt = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
      if ( slaveEqnList_[irow] == -1 ) ncnt++;
   MPI_Allreduce(&ncnt, &nSum, 1, MPI_INT, MPI_SUM, mpiComm_);
   if ( nSum > 0 )
   {
      if ( mypid == 0 && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      {
         printf("%4d : findSlaveEqns2 fails - total number of unsatisfied",
                mypid);
         printf(" constraints = %d \n", nSum);
      }
      if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      {
         for ( irow = 0; irow < nConstraints; irow++ )
            if ( slaveEqnList_[irow] == -1 )
            {
               printf("%4d : findSlaveEqns2 - unsatisfied constraint",mypid);
               printf(" equation = %d\n", irow+endRow-nConstraints+1);
            }
      }
      return -1;
   }
   else return 0;
}

//****************************************************************************
// build reduced matrix
//----------------------------------------------------------------------------

int HYPRE_SlideReduction::buildReducedMatrix2()
{
   int    mypid, nprocs, *procNRows, startRow, endRow, localNRows;
   int    globalNConstr, globalNRows, nConstraints, A21NRows, A21NCols;
   int    A21GlobalNRows, A21GlobalNCols, A21StartRow, A21StartCol, ierr;
   int    rowCount, maxRowSize, newEndRow, *A21MatSize, irow, is, rowIndex;
   int    rowSize, *colInd, newRowSize, jcol, colIndex, searchIndex;
   int    nnzA21, *newColInd, procIndex, ncnt, uBound, *colInd2, rowSize2;
   int    newColIndex, *rowTags=NULL;
   double *colVal, *newColVal, mat2X2[4], denom, *colVal2;
   char   fname[40];
   HYPRE_ParCSRMatrix A_csr, A21_csr, invA22_csr, RAP_csr, reducedA_csr;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_IJMatrixGetObject(Amat_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &procNRows );
   startRow      = procNRows[mypid];
   endRow        = procNRows[mypid+1] - 1;
   localNRows    = endRow - startRow + 1;
   globalNConstr = procNConstr_[nprocs];
   globalNRows   = procNRows[nprocs];
   nConstraints  = procNConstr_[mypid+1] - procNConstr_[mypid];

   //******************************************************************
   // extract A21 from A
   //------------------------------------------------------------------
   // calculate the dimension of A21
   //------------------------------------------------------------------

   A21NRows       = 2 * nConstraints;
   A21NCols       = localNRows - nConstraints;
   A21GlobalNRows = 2 * globalNConstr;
   A21GlobalNCols = globalNRows - globalNConstr;
   A21StartRow    = 2 * procNConstr_[mypid];
   A21StartCol    = procNRows[mypid] - procNConstr_[mypid];

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
   {
      printf("%4d : buildReducedMatrix - A21StartRow  = %d\n", mypid,
                                         A21StartRow);
      printf("%4d : buildReducedMatrix - A21GlobalDim = %d %d\n", mypid,
                                         A21GlobalNRows, A21GlobalNCols);
      printf("%4d : buildReducedMatrix - A21LocalDim  = %d %d\n",mypid,
                                         A21NRows, A21NCols);
   }

   //------------------------------------------------------------------
   // create a matrix context for A21
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpiComm_,A21StartRow,A21StartRow+A21NRows-1,
                               A21StartCol,A21StartCol+A21NCols-1,&A21mat_);
   ierr += HYPRE_IJMatrixSetObjectType(A21mat_, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute the number of nonzeros in the first nConstraint row of A21
   // (which consists of the rows in selectedList), the nnz will
   // be reduced by excluding the constraint and selected slave columns
   //------------------------------------------------------------------

   rowCount   = maxRowSize = 0;
   newEndRow  = endRow - nConstraints;
   A21MatSize = new int[A21NRows];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         colIndex = colInd[jcol];
         if ( colVal[jcol] != 0.0 )
         {
            if ( colIndex <= newEndRow || colIndex > endRow )
            {
               searchIndex = hypre_BinarySearch(gSlaveEqnList_,colIndex,
                                                globalNConstr);
               if ( searchIndex < 0 ) newRowSize++;
            }
         }
      }
      A21MatSize[irow] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
   }

   //------------------------------------------------------------------
   // compute the number of nonzeros in the second nConstraint row of A21
   // (which consists of the rows in constraint equations)
   //------------------------------------------------------------------

   rowCount = nConstraints;
   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         if ( colVal[jcol] != 0.0 )
         {
            colIndex = colInd[jcol];
            if ( colIndex <= newEndRow || colIndex > endRow )
            {
               searchIndex = hypre_BinarySearch(gSlaveEqnList_,colIndex,
                                                globalNConstr);
               if ( searchIndex < 0 ) newRowSize++;
            }
         }
      }
      A21MatSize[rowCount] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
      rowCount++;
   }
   nnzA21 = 0;
   for ( irow = 0; irow < 2*nConstraints; irow++ ) nnzA21 += A21MatSize[irow];
   MPI_Allreduce(&nnzA21,&ncnt,1,MPI_INT,MPI_SUM,mpiComm_);
   if ( mypid == 0 && ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("   0 : buildReducedMatrix : NNZ of A21 = %d\n", ncnt);

   //------------------------------------------------------------------
   // after fetching the row sizes, set up A21 with such sizes
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixSetRowSizes(A21mat_, A21MatSize);
   ierr += HYPRE_IJMatrixInitialize(A21mat_);
   hypre_assert(!ierr);
   delete [] A21MatSize;

   //------------------------------------------------------------------
   // next load the first nConstraint row to A21 extracted from A
   // (at the same time, the D block is saved for future use)
   //------------------------------------------------------------------

   rowCount  = A21StartRow;
   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         if ( colVal[jcol] != 0.0 )
         {
            colIndex = colInd[jcol];
            if ( colIndex <= newEndRow || colIndex > endRow )
            {
               searchIndex = HYPRE_LSI_Search(gSlaveEqnList_,colIndex,
                                              globalNConstr);
               if ( searchIndex < 0 )
               {
                  for ( procIndex = 0; procIndex < nprocs; procIndex++ )
                     if ( procNRows[procIndex] > colIndex ) break;
                  procIndex--;
                  newColIndex = colIndex - procNConstr_[procIndex];
                  newColInd[newRowSize]   = newColIndex;
                  newColVal[newRowSize++] = colVal[jcol];
                  if ( newColIndex < 0 || newColIndex >= A21GlobalNCols )
                  {
                     printf("%4d : buildReducedMatrix ERROR - A21",mypid);
                     printf(" out of range (%d,%d (%d))\n", rowCount,
                            colIndex, A21GlobalNCols);
                     for ( is = 0; is < rowSize; is++ )
                        printf("%4d : row %7d has col = %7d\n",mypid,rowIndex,
                               colInd[is]);
                     exit(1);
                  }
                  if ( newRowSize > maxRowSize+1 )
                  {
                     if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 2 )
                     {
                        printf("%4d : buildReducedMatrix WARNING - ",mypid);
                        printf("passing array boundary(1).\n");
                     }
                  }
               }
            }
         }
      }
      HYPRE_IJMatrixSetValues(A21mat_,1,&newRowSize,(const int *) &rowCount,
                     (const int *) newColInd, (const double *) newColVal);
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      rowCount++;
   }

   //------------------------------------------------------------------
   // next load the second nConstraint rows to A21 extracted from A
   //------------------------------------------------------------------

   for ( irow = endRow-nConstraints+1; irow <= endRow; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         colIndex = colInd[jcol];
         if (colVal[jcol] != 0.0 &&
             (colIndex <= newEndRow || colIndex > endRow))
         {
            searchIndex = hypre_BinarySearch(gSlaveEqnList_, colIndex,
                                             globalNConstr);
            if ( searchIndex < 0 )
            {
               for ( procIndex = 0; procIndex < nprocs; procIndex++ )
                  if ( procNRows[procIndex] > colIndex ) break;
               procIndex--;
               newColIndex = colInd[jcol] - procNConstr_[procIndex];
               newColInd[newRowSize]   = newColIndex;
               newColVal[newRowSize++] = colVal[jcol];
             }
          }
      }
      HYPRE_IJMatrixSetValues(A21mat_,1,&newRowSize,(const int *) &rowCount,
		(const int *) newColInd, (const double *) newColVal);
      HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
      rowCount++;
   }
   delete [] newColInd;
   delete [] newColVal;

   //------------------------------------------------------------------
   // finally assemble the matrix and sanitize
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(A21mat_);
   HYPRE_IJMatrixGetObject(A21mat_, (void **) &A21_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A21_csr);

   if ( outputLevel_ >= 5 )
   {
      sprintf(fname, "A21.%d", mypid);
      FILE *fp = fopen(fname, "w");

      if ( mypid == 0 )
      {
         printf("====================================================\n");
         printf("%4d : Printing A21 matrix... \n", mypid);
         fflush(stdout);
      }
      for (irow = A21StartRow;irow < A21StartRow+2*nConstraints;irow++)
      {
         HYPRE_ParCSRMatrixGetRow(A21_csr,irow,&rowSize,&colInd,&colVal);
         for ( jcol = 0; jcol < rowSize; jcol++ )
            if ( colVal[jcol] != 0.0 )
               fprintf(fp,"%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                       colVal[jcol]);
         HYPRE_ParCSRMatrixRestoreRow(A21_csr, irow, &rowSize,
                                      &colInd, &colVal);
      }
      if ( mypid == 0 )
         printf("====================================================\n");
      fclose(fp);
   }

   //******************************************************************
   // construct invA22
   // - given A22 = | B    C |, compute | 0       C^{-T}          |
   //               | C^T  0 |          | C^{-1} -C^{-1} B C^{-T} |
   //------------------------------------------------------------------

   //------------------------------------------------------------------
   // first extract the (2,1) block of A22
   // ( constraints-to-local slaves )
   //------------------------------------------------------------------

   int    *CT_JA = NULL;
   double *CT_AA = NULL;
   if ( nConstraints > 0 )
   {
      CT_JA = new int[nConstraints*2];
      CT_AA = new double[nConstraints*2];
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = newEndRow + 1 + irow;
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      CT_JA[irow*2] = CT_JA[irow*2+1] = -1;
      CT_AA[irow*2] = CT_AA[irow*2+1] = 0.0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         searchIndex = hypre_BinarySearch(slaveEqnList_,colIndex,nConstraints);
         if ( searchIndex >= 0 )
         {
            if ( CT_JA[irow*2] == -1 )
            {
               CT_JA[irow*2] = slaveEqnListAux_[searchIndex];
               CT_AA[irow*2] = colVal[jcol];
            }
            else
            {
               CT_JA[irow*2+1] = slaveEqnListAux_[searchIndex];
               CT_AA[irow*2+1] = colVal[jcol];
            }
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
   }

   //------------------------------------------------------------------
   // invert the (2,1) block of A22
   //------------------------------------------------------------------

   if ( nConstraints > 0 ) rowTags = new int[nConstraints];
   for ( irow = 0; irow < nConstraints; irow++ ) rowTags[irow] = -1;

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      if ( rowTags[irow] == -1 )
      {
         if ( CT_JA[irow*2+1] == -1 )
            CT_AA[irow*2] = 1.0 / CT_AA[irow*2];
         else
         {
            if ( CT_JA[2*irow] == irow )
            {
               mat2X2[0] = CT_AA[2*irow];
               mat2X2[2] = CT_AA[2*irow+1];
               rowIndex = CT_JA[2*irow+1];
            }
            else
            {
               mat2X2[0] = CT_AA[2*irow+1];
               mat2X2[2] = CT_AA[2*irow];
               rowIndex  = CT_JA[2*irow];
            }
            if ( rowTags[rowIndex] != -1 )
               CT_AA[rowIndex*2] = 1.0 / CT_AA[rowIndex*2];
            if ( CT_JA[2*rowIndex] == rowIndex )
            {
               mat2X2[3] = CT_AA[2*rowIndex];
               mat2X2[1] = CT_AA[2*rowIndex+1];
            }
            else
            {
               mat2X2[3] = CT_AA[2*rowIndex+1];
               mat2X2[1] = CT_AA[2*rowIndex];
            }
            rowTags[rowIndex] = 0;
            denom = mat2X2[0] * mat2X2[3] - mat2X2[1] * mat2X2[2];
            denom = 1.0 / denom;
            CT_JA[irow*2] = irow;
            CT_AA[irow*2] = mat2X2[3] * denom;
            CT_JA[irow*2+1] = rowIndex;
            CT_AA[irow*2+1] = - mat2X2[2] * denom;
            CT_JA[rowIndex*2] = rowIndex;
            CT_AA[rowIndex*2] = mat2X2[0] * denom;
            CT_JA[rowIndex*2+1] = irow;
            CT_AA[rowIndex*2+1] = - mat2X2[1] * denom;
         }
         rowTags[irow] = 0;
      }
   }
   if ( nConstraints > 0 ) delete [] rowTags;

   //------------------------------------------------------------------
   // form ParCSRMatrix of the (2,1) block of A22
   //------------------------------------------------------------------

   int                *hypreCTMatSize;
   hypre_ParCSRMatrix *hypreCT;
   HYPRE_IJMatrix     IJCT;

   ierr = HYPRE_IJMatrixCreate(mpiComm_, procNConstr_[mypid],
                    procNConstr_[mypid]+nConstraints-1, procNConstr_[mypid],
                    procNConstr_[mypid]+nConstraints-1, &IJCT);
   ierr += HYPRE_IJMatrixSetObjectType(IJCT, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) hypreCTMatSize = new int[nConstraints];
   maxRowSize = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      hypreCTMatSize[irow] = 1;
      if ( CT_JA[irow*2+1] != -1 && CT_AA[irow*2+1] != 0.0 )
         hypreCTMatSize[irow]++;
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJCT, hypreCTMatSize);
   ierr = HYPRE_IJMatrixInitialize(IJCT);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) delete [] hypreCTMatSize;
   newColInd = new int[2];
   newColVal = new double[2];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      newColInd[0] = CT_JA[irow*2] + procNConstr_[mypid];
      newColVal[0] = CT_AA[irow*2];
      newRowSize = 1;
      if ( CT_JA[irow*2+1] != -1 && CT_AA[irow*2+1] != 0.0 )
      {
         newColInd[1] = CT_JA[irow*2+1] + procNConstr_[mypid];
         newColVal[1] = CT_AA[irow*2+1];
         newRowSize++;
      }
      ierr = HYPRE_IJMatrixSetValues(IJCT, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert( !ierr );
   }
   delete [] newColInd;
   delete [] newColVal;
   if ( nConstraints > 0 )
   {
      delete [] CT_JA;
      delete [] CT_AA;
   }
   HYPRE_IJMatrixAssemble(IJCT);
   HYPRE_IJMatrixGetObject(IJCT, (void **) &hypreCT);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreCT);

   //------------------------------------------------------------------
   // next extract the (1,2) block of A22
   // ( local slaves-to-constraints )
   //------------------------------------------------------------------

   int    *C_JA = NULL;
   double *C_AA = NULL;
   if ( nConstraints > 0 )
   {
      C_JA = new int[nConstraints*2];
      C_AA = new double[nConstraints*2];
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      C_JA[irow*2] = C_JA[irow*2+1] = -1;
      C_AA[irow*2] = C_AA[irow*2+1] = 0.0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         if ( colIndex > newEndRow && colIndex <= endRow )
         {
            if ( C_JA[irow*2] == -1 )
            {
               C_JA[irow*2] = colIndex - newEndRow - 1;
               C_AA[irow*2] = colVal[jcol];
            }
            else
            {
               C_JA[irow*2+1] = colIndex - newEndRow - 1;
               C_AA[irow*2+1] = colVal[jcol];
            }
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
   }

   //------------------------------------------------------------------
   // invert the (1,2) block of A22
   //------------------------------------------------------------------

   if ( nConstraints > 0 ) rowTags = new int[nConstraints];
   for ( irow = 0; irow < nConstraints; irow++ ) rowTags[irow] = -1;

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      if ( rowTags[irow] == -1 )
      {
         if ( C_JA[irow*2+1] == -1 )
            C_AA[irow*2] = 1.0 / C_AA[irow*2];
         else
         {
            if ( C_JA[2*irow] == irow )
            {
               mat2X2[0] = C_AA[2*irow];
               mat2X2[2] = C_AA[2*irow+1];
               rowIndex  = C_JA[2*irow+1];
            }
            else
            {
               mat2X2[0] = C_AA[2*irow+1];
               mat2X2[2] = C_AA[2*irow];
               rowIndex  = C_JA[2*irow];
            }
            if ( rowTags[rowIndex] != -1 )
               C_AA[rowIndex*2] = 1.0 / C_AA[rowIndex*2];
            if ( C_JA[2*rowIndex] == rowIndex )
            {
               mat2X2[3] = C_AA[2*rowIndex];
               mat2X2[1] = C_AA[2*rowIndex+1];
            }
            else
            {
               mat2X2[3] = C_AA[2*rowIndex+1];
               mat2X2[1] = C_AA[2*rowIndex];
            }
            rowTags[rowIndex] = 0;
            denom = mat2X2[0] * mat2X2[3] - mat2X2[1] * mat2X2[3];
            denom = 1.0 / denom;
            C_JA[irow*2] = irow;
            C_AA[irow*2] = mat2X2[3] * denom;
            C_JA[irow*2+1] = rowIndex;
            C_AA[irow*2+1] = - mat2X2[2] * denom;
            C_JA[rowIndex*2] = rowIndex;
            C_AA[rowIndex*2] = mat2X2[0] * denom;
            C_JA[rowIndex*2+1] = irow;
            C_AA[rowIndex*2+1] = - mat2X2[1] * denom;
         }
         rowTags[irow] = 0;
      }
   }
   if ( nConstraints > 0 ) delete [] rowTags;

   //------------------------------------------------------------------
   // form ParCSRMatrix of the (1,2) block of A22
   //------------------------------------------------------------------

   int                *hypreCMatSize;
   hypre_ParCSRMatrix *hypreC;
   HYPRE_IJMatrix     IJC;

   ierr = HYPRE_IJMatrixCreate(mpiComm_, procNConstr_[mypid],
                    procNConstr_[mypid]+nConstraints-1, procNConstr_[mypid],
                    procNConstr_[mypid]+nConstraints-1, &IJC);
   ierr += HYPRE_IJMatrixSetObjectType(IJC, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) hypreCMatSize = new int[nConstraints];
   maxRowSize = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      hypreCMatSize[irow] = 1;
      if ( C_JA[irow*2+1] != -1 && C_AA[irow*2+1] != 0.0 )
         hypreCMatSize[irow]++;
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJC, hypreCMatSize);
   ierr = HYPRE_IJMatrixInitialize(IJC);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) delete [] hypreCMatSize;
   newColInd = new int[2];
   newColVal = new double[2];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      newColInd[0] = C_JA[irow*2] + procNConstr_[mypid];
      newColVal[0] = C_AA[irow*2];
      newRowSize = 1;
      if ( C_JA[irow*2+1] != -1 && C_AA[irow*2+1] != 0.0 )
      {
         newColInd[1] = C_JA[irow*2+1] + procNConstr_[mypid];
         newColVal[1] = C_AA[irow*2+1];
         newRowSize++;
      }
      ierr = HYPRE_IJMatrixSetValues(IJC, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert( !ierr );
   }
   delete [] newColInd;
   delete [] newColVal;
   if ( nConstraints > 0 )
   {
      delete [] C_JA;
      delete [] C_AA;
   }
   HYPRE_IJMatrixAssemble(IJC);
   HYPRE_IJMatrixGetObject(IJC, (void **) &hypreC);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hypreC);

   //------------------------------------------------------------------
   // form ParCSRMatrix of the (2,2) block of the invA22 matrix
   //------------------------------------------------------------------

   int                *hypreBMatSize;
   hypre_ParCSRMatrix *hypreB;
   HYPRE_IJMatrix     IJB;

   ierr = HYPRE_IJMatrixCreate(mpiComm_, procNConstr_[mypid],
                procNConstr_[mypid]+nConstraints-1, procNConstr_[mypid],
                procNConstr_[mypid]+nConstraints-1, &IJB);
   ierr = HYPRE_IJMatrixSetObjectType(IJB, HYPRE_PARCSR);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) hypreBMatSize = new int[nConstraints];
   maxRowSize = 0;
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         searchIndex = hypre_BinarySearch(gSlaveEqnList_, colIndex,
                                          globalNConstr);
         if ( searchIndex >= 0 ) newRowSize++;
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      hypreBMatSize[irow] = newRowSize;
      maxRowSize = (newRowSize > maxRowSize) ? newRowSize : maxRowSize;
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJB, hypreBMatSize);
   ierr = HYPRE_IJMatrixInitialize(IJB);
   hypre_assert(!ierr);
   if ( nConstraints > 0 ) delete [] hypreBMatSize;

   if ( maxRowSize > 0 )
   {
      newColInd = new int[maxRowSize];
      newColVal = new double[maxRowSize];
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      for ( is = 0; is < nConstraints; is++ )
      {
         if ( slaveEqnListAux_[is] == irow )
         {
            rowIndex = slaveEqnList_[is];
            break;
         }
      }
      HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         colIndex = colInd[jcol];
         searchIndex = hypre_BinarySearch(gSlaveEqnList_, colIndex,
                                          globalNConstr);
         if ( searchIndex >= 0 )
         {
            newColInd[newRowSize] = gSlaveEqnListAux_[searchIndex];
            newColVal[newRowSize++] = - colVal[jcol];
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
      rowIndex = procNConstr_[mypid] + irow;
      ierr = HYPRE_IJMatrixSetValues(IJB, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert( !ierr );
   }
   HYPRE_IJMatrixAssemble(IJB);
   HYPRE_IJMatrixGetObject(IJB, (void **) &hypreB);
   if ( maxRowSize > 0 )
   {
      delete [] newColInd;
      delete [] newColVal;
   }

   //------------------------------------------------------------------
   // perform triple matrix product - C^{-1} B C^{-T}
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrix hypreCBC;

   strcpy( fname, "hypreCT" );
   HYPRE_ParCSRMatrixPrint((HYPRE_ParCSRMatrix) hypreCT, fname);
   strcpy( fname, "hypreB" );
   HYPRE_ParCSRMatrixPrint((HYPRE_ParCSRMatrix) hypreB, fname);
   hypre_BoomerAMGBuildCoarseOperator(hypreCT, hypreB, hypreCT,
                                      (hypre_ParCSRMatrix **) &hypreCBC);
#if 0
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCT,irow,&rowSize,
                              &colInd,&colVal);
      for (jcol = 0; jcol < rowSize; jcol++ )
         printf("CT : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCT,irow,&rowSize,
                                    &colInd,&colVal);
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreB,irow,&rowSize,
                              &colInd,&colVal);
      for (jcol = 0; jcol < rowSize; jcol++ )
         printf("B : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreB,irow,&rowSize,
                                    &colInd,&colVal);
   }
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                              &colInd,&colVal);
      for (jcol = 0; jcol < rowSize; jcol++ )
         printf("CBC : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                                    &colInd,&colVal);
   }
#endif

   HYPRE_IJMatrixDestroy( IJB );

   //------------------------------------------------------------------
   // calculate the dimension of invA22
   //------------------------------------------------------------------

   int invA22NRows       = A21NRows;
   int invA22NCols       = invA22NRows;
   int invA22StartRow    = A21StartRow;
   int invA22StartCol    = invA22StartRow;
   int *invA22MatSize;

   //------------------------------------------------------------------
   // create a matrix context for A22
   //------------------------------------------------------------------

   ierr = HYPRE_IJMatrixCreate(mpiComm_, invA22StartRow,
                    invA22StartRow+invA22NRows-1, invA22StartCol,
                    invA22StartCol+invA22NCols-1, &invA22mat_);
   ierr += HYPRE_IJMatrixSetObjectType(invA22mat_, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute the no. of nonzeros in the first nConstraint row of invA22
   //------------------------------------------------------------------

   maxRowSize  = 0;
   invA22MatSize = new int[invA22NRows];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCT,rowIndex,
                                       &rowSize,NULL,NULL);
      hypre_assert( !ierr );
      invA22MatSize[irow] = rowSize;
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCT,rowIndex,
                                   &rowSize,NULL,NULL);
      maxRowSize = ( rowSize > maxRowSize ) ? rowSize : maxRowSize;
   }

   //------------------------------------------------------------------
   // compute the number of nonzeros in the second nConstraints row of
   // invA22 (consisting of [D and A22 block])
   //------------------------------------------------------------------

#if 0
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                              &colInd,&colVal);
      for (jcol = 0; jcol < rowSize; jcol++ )
         printf("CBC1 : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                                    &colInd,&colVal);
   }
#endif
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreC,rowIndex,
                               &rowSize,&colInd,&colVal);
      hypre_assert( !ierr );
      newRowSize = rowSize;
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreC,rowIndex,
                                   &rowSize,&colInd,&colVal);
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,rowIndex,
                                      &rowSize,&colInd,&colVal);
      hypre_assert( !ierr );
      newRowSize += rowSize;
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,rowIndex,
                                   &rowSize,&colInd,&colVal);
      invA22MatSize[nConstraints+irow] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
   }

   //------------------------------------------------------------------
   // after fetching the row sizes, set up invA22 with such sizes
   //------------------------------------------------------------------

#if 0
   for ( irow = 0; irow < nConstraints; irow++ )
   {
   HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                           &colInd,&colVal);
   for (jcol = 0; jcol < rowSize; jcol++ )
      printf("CBC2 : %5d %5d %25.16e\n",irow+1,colInd[jcol]+1,colVal[jcol]);
   HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,irow,&rowSize,
                                 &colInd,&colVal);
}
#endif
   ierr  = HYPRE_IJMatrixSetRowSizes(invA22mat_, invA22MatSize);
   ierr += HYPRE_IJMatrixInitialize(invA22mat_);
   hypre_assert(!ierr);
   delete [] invA22MatSize;

   //------------------------------------------------------------------
   // next load the first nConstraints row to invA22 extracted from A
   // (that is, the D block)
   //------------------------------------------------------------------

   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];
   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex = procNConstr_[mypid] + irow;
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCT,rowIndex,
                                     &rowSize,&colInd,&colVal);
      hypre_assert( !ierr );
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         newColInd[newRowSize] = colInd[jcol] + procNConstr_[mypid] +
                                 nConstraints;
         newColVal[newRowSize++] = colVal[jcol];
      }
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCT,rowIndex,
                                   &rowSize,&colInd,&colVal);
      rowCount = invA22StartRow + irow;
      ierr = HYPRE_IJMatrixSetValues(invA22mat_, 1, &rowSize,
                (const int *) &rowCount, (const int *) newColInd,
                (const double *) newColVal);
      hypre_assert(!ierr);

   }

   //------------------------------------------------------------------
   // next load the second nConstraints rows to A22 extracted from A
   //------------------------------------------------------------------

   for ( irow = 0; irow < nConstraints; irow++ )
   {
      rowIndex   = procNConstr_[mypid] + irow;
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreC,rowIndex,
                               &rowSize,&colInd,&colVal);
      hypre_assert( !ierr );
      newRowSize = 0;
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         newColInd[newRowSize] = colInd[jcol] + procNConstr_[mypid];
         newColVal[newRowSize++] = colVal[jcol];
      }
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreC,rowIndex,
                                   &rowSize,&colInd,&colVal);
      ierr = HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreCBC,rowIndex,
                                      &rowSize2,&colInd2,&colVal2);
      hypre_assert( !ierr );
      for ( jcol = 0; jcol < rowSize2; jcol++ )
      {
         newColInd[newRowSize] = colInd2[jcol] + procNConstr_[mypid] +
                                 nConstraints;
         newColVal[newRowSize++] = colVal2[jcol];
      }
      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreCBC,rowIndex,
                                   &rowSize2,&colInd2,&colVal2);
      rowCount = invA22StartRow + nConstraints + irow;
      ierr = HYPRE_IJMatrixSetValues(invA22mat_, 1, &newRowSize,
		(const int *) &rowCount, (const int *) newColInd,
		(const double *) newColVal);
      hypre_assert(!ierr);
   }
   delete [] newColInd;
   delete [] newColVal;
   HYPRE_IJMatrixDestroy( IJC );
   HYPRE_IJMatrixDestroy( IJCT );
   HYPRE_ParCSRMatrixDestroy( (HYPRE_ParCSRMatrix) hypreCBC );

   //------------------------------------------------------------------
   // finally assemble the matrix and sanitize
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(invA22mat_);
   HYPRE_IJMatrixGetObject(invA22mat_, (void **) &invA22_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) invA22_csr);

   if ( outputLevel_ >= 5 )
   {
      sprintf(fname, "invA22.%d", mypid);
      FILE *fp = fopen(fname, "w");

      if ( mypid == ncnt )
      {
         printf("====================================================\n");
         printf("%4d : Printing invA22 matrix... \n", mypid);
         fflush(stdout);
      }
      for (irow=invA22StartRow; irow < invA22StartRow+invA22NRows;irow++)
      {
         HYPRE_ParCSRMatrixGetRow(invA22_csr,irow,&rowSize,&colInd,&colVal);
         for ( jcol = 0; jcol < rowSize; jcol++ )
            if ( colVal[jcol] != 0.0 )
               fprintf(fp,"%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                      colVal[jcol]);
         HYPRE_ParCSRMatrixRestoreRow(invA22_csr,irow,&rowSize,&colInd,
                                      &colVal);
      }
      if ( mypid == ncnt )
            printf("====================================================\n");
      fclose(fp);
   }

   //******************************************************************
   // perform the triple matrix product A12 * invA22 * A21
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(A21mat_, (void **) &A21_csr);
   HYPRE_IJMatrixGetObject(invA22mat_, (void **) &invA22_csr);
   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("%4d : buildReducedMatrix - Triple matrix product starts\n",
             mypid);

   hypre_BoomerAMGBuildCoarseOperator((hypre_ParCSRMatrix *) A21_csr,
                                      (hypre_ParCSRMatrix *) invA22_csr,
                                      (hypre_ParCSRMatrix *) A21_csr,
                                      (hypre_ParCSRMatrix **) &RAP_csr);

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
      printf("%4d : buildReducedMatrix - Triple matrix product ends\n",
             mypid);

   if ( outputLevel_ >= 4 )
   {
      sprintf(fname, "rap.%d", mypid);
      FILE *fp = fopen(fname, "w");

      if ( mypid == 0 )
      {
         printf("====================================================\n");
         printf("%4d : Printing RAP matrix... \n", mypid);
         fflush(stdout);
      }
      for ( irow = A21StartRow; irow < A21StartRow+A21NCols; irow++ )
      {
         HYPRE_ParCSRMatrixGetRow(RAP_csr,irow,&rowSize,&colInd,&colVal);
         for ( jcol = 0; jcol < rowSize; jcol++ )
            if ( colVal[jcol] != 0.0 )
               printf("%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                      colVal[jcol]);
         HYPRE_ParCSRMatrixRestoreRow(RAP_csr,irow,&rowSize,&colInd,
                                      &colVal);
      }
      fclose(fp);
      if ( mypid == 0 )
         printf("====================================================\n");
   }

   //******************************************************************
   // finally form reduceA = A11 - A12 * invA22 * A21
   //------------------------------------------------------------------

   //------------------------------------------------------------------
   // compute row sizes
   //------------------------------------------------------------------

   int reducedANRows       = localNRows - nConstraints;
   int reducedANCols       = reducedANRows;
   int reducedAStartRow    = procNRows[mypid] - procNConstr_[mypid];
   int reducedAStartCol    = reducedAStartRow;
   int reducedAGlobalNRows = globalNRows - globalNConstr;
   int reducedAGlobalNCols = reducedAGlobalNRows;
   int *reducedAMatSize    = new int[reducedANRows];

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 1 )
   {
      printf("%4d : buildReducedMatrix - reduceAGlobalDim = %d %d\n", mypid,
                       reducedAGlobalNRows, reducedAGlobalNCols);
      printf("%4d : buildReducedMatrix - reducedALocalDim  = %d %d\n", mypid,
                       reducedANRows, reducedANCols);
   }

   //------------------------------------------------------------------
   // create a matrix context for reducedA
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpiComm_,reducedAStartRow,
                 reducedAStartRow+reducedANRows-1, reducedAStartCol,
                 reducedAStartCol+reducedANCols-1,&reducedAmat_);
   ierr += HYPRE_IJMatrixSetObjectType(reducedAmat_, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute row sizes for reducedA
   //------------------------------------------------------------------

   rowCount = maxRowSize = 0;
   for ( irow = startRow; irow <= newEndRow; irow++ )
   {
      searchIndex = hypre_BinarySearch(slaveEqnList_, irow, nConstraints);
      if ( searchIndex >= 0 )  reducedAMatSize[rowCount++] = 1;
      else
      {
         HYPRE_ParCSRMatrixGetRow(A_csr,irow,&rowSize,&colInd,NULL);
         rowIndex = reducedAStartRow + rowCount;
         ierr = HYPRE_ParCSRMatrixGetRow(RAP_csr,rowIndex,&rowSize2,
                                         &colInd2, NULL);
         hypre_assert( !ierr );
         newRowSize = rowSize + rowSize2;
         maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
         newColInd = new int[newRowSize];
         for (jcol = 0; jcol < rowSize; jcol++) newColInd[jcol] = colInd[jcol];
         for (jcol = 0; jcol < rowSize2; jcol++)
            newColInd[rowSize+jcol] = colInd2[jcol];
         hypre_qsort0(newColInd, 0, newRowSize-1);
         ncnt = 0;
         for ( jcol = 1; jcol < newRowSize; jcol++ )
            if (newColInd[jcol] != newColInd[ncnt])
               newColInd[++ncnt] = newColInd[jcol];
         if ( newRowSize > 0 ) ncnt++;
         reducedAMatSize[rowIndex++] = ncnt;
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,NULL);
         ierr = HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowIndex,&rowSize2,
                                             &colInd2,NULL);
         delete [] newColInd;
         hypre_assert( !ierr );
         rowCount++;
      }
   }
   ierr  = HYPRE_IJMatrixSetRowSizes(reducedAmat_, reducedAMatSize);
   ierr += HYPRE_IJMatrixInitialize(reducedAmat_);
   hypre_assert(!ierr);
   delete [] reducedAMatSize;

   //------------------------------------------------------------------
   // load the reducedA matrix
   //------------------------------------------------------------------

   rowCount  = 0;
   newColInd = new int[maxRowSize+1];
   newColVal = new double[maxRowSize+1];
   for ( irow = startRow; irow <= newEndRow; irow++ )
   {
      searchIndex = hypre_BinarySearch(slaveEqnList_, irow, nConstraints);
      rowIndex    = reducedAStartRow + rowCount;
      if ( searchIndex >= 0 )
      {
         newRowSize   = 1;
         newColInd[0] = reducedAStartRow + rowCount;
         newColVal[0] = 1.0;
      }
      else
      {
         HYPRE_ParCSRMatrixGetRow(A_csr, irow, &rowSize, &colInd, &colVal);
         HYPRE_ParCSRMatrixGetRow(RAP_csr,rowIndex,&rowSize2,&colInd2,
                                  &colVal2);
         newRowSize = rowSize + rowSize2;
         ncnt       = 0;
         for ( jcol = 0; jcol < rowSize; jcol++ )
         {
            colIndex = colInd[jcol];
            for ( procIndex = 0; procIndex < nprocs; procIndex++ )
               if ( procNRows[procIndex] > colIndex ) break;
            uBound = procNRows[procIndex] -
                     (procNConstr_[procIndex]-procNConstr_[procIndex-1]);
            procIndex--;
            if ( colIndex < uBound )
            {
               searchIndex = hypre_BinarySearch(gSlaveEqnList_, colIndex,
                                                globalNConstr);
               if ( searchIndex < 0 )
               {
                  newColInd[ncnt] = colIndex - procNConstr_[procIndex];
                  newColVal[ncnt++] = colVal[jcol];
               }
            }
         }
         for ( jcol = 0; jcol < rowSize2; jcol++ )
         {
            newColInd[ncnt+jcol] = colInd2[jcol];
            newColVal[ncnt+jcol] = - colVal2[jcol];
         }
         newRowSize = ncnt + rowSize2;
         hypre_qsort1(newColInd, newColVal, 0, newRowSize-1);
         ncnt = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( jcol != ncnt && newColInd[jcol] == newColInd[ncnt] )
               newColVal[ncnt] += newColVal[jcol];
            else if ( newColInd[jcol] != newColInd[ncnt] )
            {
               ncnt++;
               newColVal[ncnt] = newColVal[jcol];
               newColInd[ncnt] = newColInd[jcol];
            }
         }
         newRowSize = ncnt + 1;
         HYPRE_ParCSRMatrixRestoreRow(A_csr,irow,&rowSize,&colInd,&colVal);
         HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowIndex,&rowSize2,&colInd2,
                                      &colVal2);
      }
      ierr = HYPRE_IJMatrixSetValues(reducedAmat_, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      hypre_assert(!ierr);
      rowCount++;
   }
   delete [] newColInd;
   delete [] newColVal;

   //------------------------------------------------------------------
   // assemble the reduced matrix
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(reducedAmat_);
   HYPRE_IJMatrixGetObject(reducedAmat_, (void **) &reducedA_csr);

   if ( ( outputLevel_ & HYPRE_BITMASK2 ) >= 5 )
   {
      MPI_Barrier(mpiComm_);
      ncnt = 0;
      while ( ncnt < nprocs )
      {
         if ( mypid == ncnt )
         {
            printf("====================================================\n");
            printf("%4d : Printing reducedA matrix... \n", mypid);
            fflush(stdout);
            for ( irow = reducedAStartRow;
                   irow < reducedAStartRow+localNRows-nConstraints; irow++ )
            {
               //printf("%d : reducedA ROW %d\n", mypid, irow);
               ierr = HYPRE_ParCSRMatrixGetRow(reducedA_csr,irow,&rowSize,
                                               &colInd, &colVal);
               //hypre_qsort1(colInd, colVal, 0, rowSize-1);
               for ( jcol = 0; jcol < rowSize; jcol++ )
                  if ( colVal[jcol] != 0.0 )
                     printf("%6d  %6d  %25.8e \n",irow+1,colInd[jcol]+1,
                            colVal[jcol]);
               HYPRE_ParCSRMatrixRestoreRow(reducedA_csr,irow,&rowSize,
                                            &colInd, &colVal);
            }
            printf("====================================================\n");
         }
         MPI_Barrier(mpiComm_);
         ncnt++;
      }
   }

   //------------------------------------------------------------------
   // store away matrix and clean up
   //------------------------------------------------------------------

   free( procNRows );
   HYPRE_ParCSRMatrixDestroy(RAP_csr);
   return 0;
}

