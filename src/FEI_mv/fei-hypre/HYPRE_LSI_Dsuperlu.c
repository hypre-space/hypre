/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.15 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_LSI_DSuperLU interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"

/*---------------------------------------------------------------------------
 * Distributed SUPERLU include files
 *-------------------------------------------------------------------------*/

#include "dsuperlu_include.h"

#ifdef HAVE_DSUPERLU
#include "../DSuperLU/SRC/superlu_ddefs.h"

typedef struct HYPRE_LSI_DSuperLU_Struct
{
   MPI_Comm           comm_;
   HYPRE_ParCSRMatrix Amat_;
   superlu_options_t  options_;
   SuperMatrix        sluAmat_;
   ScalePermstruct_t  ScalePermstruct_;
   SuperLUStat_t      stat_;
   LUstruct_t         LUstruct_;
   SOLVEstruct_t      SOLVEstruct_;
   int                globalNRows_;
   int                localNRows_;
   int                startRow_;
   int                outputLevel_;
   double             *berr_;
   gridinfo_t         sluGrid_;
   int                setupFlag_;
}
HYPRE_LSI_DSuperLU;

int HYPRE_LSI_DSuperLUGenMatrix(HYPRE_Solver solver);

/***************************************************************************
 * HYPRE_LSI_DSuperLUCreate - Return a DSuperLU object "solver".  
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_LSI_DSuperLU *sluPtr;
   sluPtr = (HYPRE_LSI_DSuperLU *) malloc(sizeof(HYPRE_LSI_DSuperLU));
   assert ( sluPtr != NULL );
   sluPtr->comm_        = comm;
   sluPtr->Amat_        = NULL;
   sluPtr->localNRows_  = 0;
   sluPtr->globalNRows_ = 0;
   sluPtr->startRow_    = 0;
   sluPtr->outputLevel_ = 0;
   sluPtr->setupFlag_   = 0;
   sluPtr->berr_ = (double *) malloc(sizeof(double));
   *solver = (HYPRE_Solver) sluPtr;
   return 0;
}

/***************************************************************************
 * HYPRE_LSI_DSuperLUDestroy - Destroy a DSuperLU object.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUDestroy( HYPRE_Solver solver )
{
   HYPRE_LSI_DSuperLU *sluPtr;
   sluPtr = (HYPRE_LSI_DSuperLU *) solver;
   sluPtr->Amat_ = NULL;
   if (sluPtr->setupFlag_ == 1)
   {
      PStatFree(&(sluPtr->stat_));
      Destroy_CompRowLoc_Matrix_dist(&(sluPtr->sluAmat_));
      ScalePermstructFree(&(sluPtr->ScalePermstruct_));
      Destroy_LU(sluPtr->globalNRows_, &(sluPtr->sluGrid_), &(sluPtr->LUstruct_));
      LUstructFree(&(sluPtr->LUstruct_));
      if (sluPtr->options_.SolveInitialized)
         dSolveFinalize(&(sluPtr->options_), &(sluPtr->SOLVEstruct_));
      superlu_gridexit(&(sluPtr->sluGrid_));
   }
   free(sluPtr->berr_);
   free(sluPtr);
   return 0;
}

/***************************************************************************
 * HYPRE_LSI_DSuperLUSetOutputLevel - Set debug level 
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUSetOutputLevel(HYPRE_Solver solver, int level)
{
   HYPRE_LSI_DSuperLU *sluPtr = (HYPRE_LSI_DSuperLU *) solver;
   sluPtr->outputLevel_ = level;
   return 0;
}

/***************************************************************************
 * HYPRE_LSI_DSuperLUSetup - Set up function for LSI_DSuperLU.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_csr,
                            HYPRE_ParVector b, HYPRE_ParVector x )
{
   int                nprocs, mypid, nprow, npcol, info, iZero=0;
   HYPRE_LSI_DSuperLU *sluPtr = (HYPRE_LSI_DSuperLU *) solver;
   MPI_Comm           mpiComm;

   /* ---------------------------------------------------------------- */
   /* get machine information                                          */
   /* ---------------------------------------------------------------- */

   mpiComm = sluPtr->comm_;
   MPI_Comm_size(mpiComm, &nprocs);
   MPI_Comm_rank(mpiComm, &mypid);

   /* ---------------------------------------------------------------- */
   /* compute grid information                                         */
   /* ---------------------------------------------------------------- */

   nprow = sluPtr->sluGrid_.nprow = 1;
   npcol = sluPtr->sluGrid_.npcol = nprocs;
   superlu_gridinit(mpiComm, nprow, npcol, &(sluPtr->sluGrid_));
   if (mypid != sluPtr->sluGrid_.iam)
   {
      printf("DSuperLU ERROR: mismatched mypid and SuperLU iam.\n");
      exit(1);
   }

   /* ---------------------------------------------------------------- */
   /* get whole matrix and compose SuperLU matrix                      */
   /* ---------------------------------------------------------------- */

   sluPtr->Amat_ = A_csr;
   HYPRE_LSI_DSuperLUGenMatrix(solver);

   /* ---------------------------------------------------------------- */
   /* set solver options                                               */
   /* ---------------------------------------------------------------- */

   set_default_options_dist(&(sluPtr->options_));
   /* options->Fact              = DOFACT (SamePattern,FACTORED}
      options->Equil             = YES (NO, ROW, COL, BOTH)
                                   (YES not robust)
      options->ParSymbFact       = NO;
      options->ColPerm           = MMD_AT_PLUS_A (NATURAL, MMD_ATA, 
                                   METIS_AT_PLUS_A, PARMETIS, MY_PERMC}
                                   (MMD_AT_PLUS_A the fastest, a factor
                                    of 3+ better than MMD_ATA, which in
                                    turn is 25% better than NATURAL)
      options->RowPerm           = LargeDiag (NOROWPERM, MY_PERMR)
      options->ReplaceTinyPivot  = YES (NO)
      options->IterRefine        = DOUBLE (NOREFINE, SINGLE, EXTRA)
                                   (EXTRA not supported, DOUBLE more
                                    accurate)
      options->Trans             = NOTRANS (TRANS, CONJ)
      options->SolveInitialized  = NO;
      options->RefineInitialized = NO;
      options->PrintStat         = YES;
   */
   sluPtr->options_.Fact = DOFACT;
   sluPtr->options_.Equil = NO;
   sluPtr->options_.IterRefine = DOUBLE;
   if (sluPtr->outputLevel_ < 2) sluPtr->options_.PrintStat = NO;
   ScalePermstructInit(sluPtr->globalNRows_, sluPtr->globalNRows_,
                       &(sluPtr->ScalePermstruct_));
   LUstructInit(sluPtr->globalNRows_, sluPtr->globalNRows_, 
                &(sluPtr->LUstruct_));
   sluPtr->berr_[0] = 0.0;
   PStatInit(&(sluPtr->stat_));
   pdgssvx(&(sluPtr->options_), &(sluPtr->sluAmat_), 
           &(sluPtr->ScalePermstruct_), NULL, sluPtr->localNRows_, iZero, 
           &(sluPtr->sluGrid_), &(sluPtr->LUstruct_), 
           &(sluPtr->SOLVEstruct_), sluPtr->berr_, &(sluPtr->stat_), &info);
   sluPtr->options_.Fact = FACTORED; 
   if (sluPtr->outputLevel_ >= 2)
      PStatPrint(&(sluPtr->options_),&(sluPtr->stat_),&(sluPtr->sluGrid_));

   sluPtr->setupFlag_ = 1;

   if (mypid == 0 && sluPtr->outputLevel_ >=2)
   {
      printf("DSuperLUSetup: diagScale = %d\n",
             sluPtr->ScalePermstruct_.DiagScale);
      printf("DSuperLUSetup: berr = %e\n", sluPtr->berr_[0]);
      printf("DSuperLUSetup: info = %d\n", info);
   }
   return 0;
}

/***************************************************************************
 * HYPRE_LSI_DSuperLUSolve - Solve function for DSuperLU.
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b, HYPRE_ParVector x )
{
   int                localNRows, irow, iOne=1, info, mypid;
   double             *rhs, *soln;
   HYPRE_LSI_DSuperLU *sluPtr = (HYPRE_LSI_DSuperLU *) solver;

   /* ---------------------------------------------------------------- */
   /* get machine, matrix, and vector information                      */
   /* ---------------------------------------------------------------- */

   MPI_Comm_rank(sluPtr->comm_, &mypid);
   localNRows  = sluPtr->localNRows_;
   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));
   for (irow = 0; irow < localNRows; irow++) soln[irow] = rhs[irow];

   /* ---------------------------------------------------------------- */
   /* solve                                                            */
   /* ---------------------------------------------------------------- */

   pdgssvx(&(sluPtr->options_), &(sluPtr->sluAmat_), 
           &(sluPtr->ScalePermstruct_), soln, localNRows, iOne, 
           &(sluPtr->sluGrid_), &(sluPtr->LUstruct_), 
           &(sluPtr->SOLVEstruct_), sluPtr->berr_, &(sluPtr->stat_), &info);

   /* ---------------------------------------------------------------- */
   /* diagnostics message                                              */
   /* ---------------------------------------------------------------- */

   if (mypid == 0 && sluPtr->outputLevel_ >=2)
   {
      printf("DSuperLUSolve: info = %d\n", info);
      printf("DSuperLUSolve: diagScale = %d\n",
             sluPtr->ScalePermstruct_.DiagScale);
   }
   return 0;
}

/****************************************************************************
 * Create SuperLU matrix in CSR
 *--------------------------------------------------------------------------*/

int HYPRE_LSI_DSuperLUGenMatrix(HYPRE_Solver solver)
{
   int        nprocs, mypid, *csrIA, *csrJA, *procNRows, localNNZ;
   int        startRow, localNRows, rowSize, *colInd, irow, jcol;
   double     *csrAA, *colVal;
   HYPRE_LSI_DSuperLU *sluPtr = (HYPRE_LSI_DSuperLU *) solver;
   HYPRE_ParCSRMatrix Amat;
   MPI_Comm   mpiComm;

   /* ---------------------------------------------------------------- */
   /* fetch parallel machine parameters                                */
   /* ---------------------------------------------------------------- */

   mpiComm = sluPtr->comm_;
   MPI_Comm_rank(mpiComm, &mypid);
   MPI_Comm_size(mpiComm, &nprocs);

   /* ---------------------------------------------------------------- */
   /* fetch matrix information                                         */
   /* ---------------------------------------------------------------- */

   Amat = sluPtr->Amat_;
   HYPRE_ParCSRMatrixGetRowPartitioning(Amat, &procNRows);
   startRow = procNRows[mypid];
   sluPtr->startRow_ = startRow;
   localNNZ = 0;
   for (irow = startRow; irow < procNRows[mypid+1]; irow++)
   {
      HYPRE_ParCSRMatrixGetRow(Amat,irow,&rowSize,&colInd,&colVal);
      localNNZ += rowSize;
      HYPRE_ParCSRMatrixRestoreRow(Amat,irow,&rowSize,&colInd,&colVal);
   }
   localNRows = procNRows[mypid+1] - procNRows[mypid];
   sluPtr->localNRows_ = localNRows;
   sluPtr->globalNRows_ = procNRows[nprocs];
   csrIA = (int *) intMalloc_dist(localNRows+1); 
   csrJA = (int *) intMalloc_dist(localNNZ); 
   csrAA = (double *) doubleMalloc_dist(localNNZ); 
   localNNZ = 0;

   csrIA[0] = localNNZ;
   for (irow = startRow; irow < procNRows[mypid+1]; irow++)
   {
      HYPRE_ParCSRMatrixGetRow(Amat,irow,&rowSize,&colInd,&colVal);
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         csrJA[localNNZ] = colInd[jcol];
         csrAA[localNNZ++] = colVal[jcol];
      }
      csrIA[irow-startRow+1] = localNNZ;
      HYPRE_ParCSRMatrixRestoreRow(Amat,irow,&rowSize,&colInd,&colVal);
   }
   /*for (irow = startRow; irow < procNRows[mypid+1]; irow++)
    *   qsort1(csrJA, csrAA, csrIA[irow-startRow], csrIA[irow-startRow+1]-1);
    */

   /* ---------------------------------------------------------------- */
   /* create SuperLU matrix                                            */
   /* ---------------------------------------------------------------- */

   dCreate_CompRowLoc_Matrix_dist(&(sluPtr->sluAmat_), sluPtr->globalNRows_,
            sluPtr->globalNRows_, localNNZ, localNRows, startRow, csrAA, 
            csrJA, csrIA, SLU_NR_loc, SLU_D, SLU_GE);
   free(procNRows);
   return 0;
}
#else
   int bogus;
#endif

