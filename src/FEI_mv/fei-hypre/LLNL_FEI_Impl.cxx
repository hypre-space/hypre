/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/**************************************************************************
  Module:  LLNL_FEI_Impl.cxx
  Purpose: custom implementation of the FEI
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "LLNL_FEI_Impl.h"

/*-------------------------------------------------------------------------
 local defines
 -------------------------------------------------------------------------*/

#define SOLVERLOCK 1024

/**************************************************************************
 LLNL_FEI_Impl is the top level finite element interface.
 **************************************************************************/

/**************************************************************************
 Constructor
 -------------------------------------------------------------------------*/
LLNL_FEI_Impl::LLNL_FEI_Impl( MPI_Comm comm )
{
   mpiComm_     = comm;
   feiPtr_      = new LLNL_FEI_Fei(comm);
   solverPtr_   = NULL;
   lscPtr_      = NULL;
   matPtr_      = NULL;
   FLAG_SolverLib_ = 0;
}

/**************************************************************************
 destructor
 -------------------------------------------------------------------------*/
LLNL_FEI_Impl::~LLNL_FEI_Impl()
{
   if (feiPtr_    != NULL) delete feiPtr_;
   if (solverPtr_ != NULL) delete solverPtr_;
   if (lscPtr_    != NULL) delete lscPtr_;
}

/**************************************************************************
 parameter function
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::parameters(int numParams, char **paramString)
{
   int  i, iOne=1;
   char param1[100], param2[100], *param3;
   SolverLib_t solver;

   for ( i = 0; i < numParams; i++ )
   {
      sscanf(paramString[i], "%s", param1);
      if ( !strcmp( param1, "externalSolver" ) )
      {
         //printf("LLNL_FEI_Impl::make sure you call externalSolver before ");
         //printf("loading the matrix.\n");
         if ( (FLAG_SolverLib_ & SOLVERLOCK) == 0 )
         {
            sscanf(paramString[i], "%s %s", param1, param2);
            if ( !strcmp( param2, "HYPRE" ) ) FLAG_SolverLib_ = 1;
            else                              FLAG_SolverLib_ = 0;
         }
      }
      else if ( !strcmp( param1, "transferSolution" ) )
      {
         transferSolution();
      }
   }
   FLAG_SolverLib_ |= SOLVERLOCK;
   if ( (FLAG_SolverLib_ - SOLVERLOCK) > 0 )
   {
      if ( lscPtr_ != NULL ) delete lscPtr_;
      if ( solverPtr_ != NULL )
      {
         delete solverPtr_;
         solverPtr_ = NULL;
      }
      param3 = new char[30];
      strcpy( param3, "matrixNoOverlap" );
      feiPtr_->parameters(iOne,&param3);
      delete [] param3;
      solver = HYPRE;
      lscPtr_ = new LLNL_FEI_LSCore(solver);
   }
   else
   {
      if ( solverPtr_ != NULL ) delete solverPtr_;
      if ( lscPtr_ != NULL )
      {
         delete lscPtr_;
         lscPtr_ = NULL;
      }
      solverPtr_ = new LLNL_FEI_Solver(mpiComm_);
   }
   feiPtr_->parameters(numParams,paramString);
   if (solverPtr_ != NULL) solverPtr_->parameters(numParams,paramString);
   if (lscPtr_    != NULL) lscPtr_->parameters(numParams,paramString);
   return 0;
}

/**************************************************************************
 solve
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::solve(int *status)
{
   double *rhsVector, *solnVector;

   if ((FLAG_SolverLib_ & SOLVERLOCK) != 0) FLAG_SolverLib_ -= SOLVERLOCK;
   feiPtr_->getRHSVector(&rhsVector);
   feiPtr_->getSolnVector(&solnVector);
   feiPtr_->getMatrix(&matPtr_);
   if ( solverPtr_ != NULL )
   {
      solverPtr_->loadRHSVector(rhsVector);
      solverPtr_->loadSolnVector(solnVector);
      solverPtr_->loadMatrix(matPtr_);
      solverPtr_->solve(status);
   }
   else if (lscPtr_ != NULL)
   {
      int    localNRows, *diagIA, *diagJA, *indices, *offsets, rowSize;
      int    extNRows, *offdIA, *offdJA, *colMap, maxRowSize, *colInds;
      int    i, j, rowInd, one=1, iter, mypid;
      double *diagAA, *offdAA, *colVals;
      char   format[20];

      MPI_Comm_rank(mpiComm_, &mypid);
      strcpy( format, "HYPRE" );
      matPtr_->getLocalMatrix(&localNRows, &diagIA, &diagJA, &diagAA);
      matPtr_->getExtMatrix(&extNRows, &offdIA, &offdJA, &offdAA, &colMap);
      offsets = matPtr_->getEqnOffsets();
      lscPtr_->setGlobalOffsets(localNRows, NULL, offsets, NULL);
      maxRowSize = 0;
      for ( i = 0; i < localNRows; i++ )
      {
         rowSize = diagIA[i+1] - diagIA[i];
         if (offdIA != NULL ) rowSize += offdIA[i+1] - offdIA[i];
         if (rowSize > maxRowSize) maxRowSize = rowSize;
      }
      if ( maxRowSize > 0 )
      {
         colInds = new int[maxRowSize];
         colVals = new double[maxRowSize];
      }
      for ( i = 0; i < localNRows; i++ )
      {
         rowSize = 0;
         for ( j = diagIA[i]; j < diagIA[i+1]; j++ )
         {
            colInds[rowSize] = diagJA[j] + offsets[mypid];
            colVals[rowSize++] = diagAA[j];
         }
         if ( offdIA != NULL )
         {
            for ( j = offdIA[i]; j < offdIA[i+1]; j++ )
            {
               colInds[rowSize] = colMap[offdJA[j]-localNRows];
               colVals[rowSize++] = offdAA[j];
            }
         }
         rowInd = offsets[mypid] + i;
         lscPtr_->putIntoSystemMatrix(one, &rowInd, rowSize,
                      (const int *) colInds, (const double* const*) &colVals);
      }
      if ( maxRowSize > 0 )
      {
         delete [] colInds;
         delete [] colVals;
      }
      if ( localNRows > 0 ) indices = new int[localNRows];
      for ( i = 0; i < localNRows; i++ ) indices[i] = i + offsets[mypid];
      lscPtr_->putIntoRHSVector(localNRows, (const double *) rhsVector,
                                (const int *) indices);
      lscPtr_->putInitialGuess((const int *) indices,
                               (const double *) solnVector, localNRows);
      lscPtr_->matrixLoadComplete();
      // Charles : this status check not in application code?
      if ((*status) != -9999) lscPtr_->solve(status,&iter);
      lscPtr_->getSolution(solnVector, localNRows);
      if (localNRows > 0) delete [] indices;
   }
   feiPtr_->disassembleSolnVector(solnVector);
   return 0;
}

/**************************************************************************
 residual norm calculation
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::residualNorm(int whichNorm, int numFields, int *fieldIDs,
                              double *norms )
{
   (void) numFields;
   (void) fieldIDs;
   double *solnVec, *rhsVec;
   feiPtr_->getSolnVector(&solnVec);
   feiPtr_->getRHSVector(&rhsVec);
   matPtr_->residualNorm(whichNorm,solnVec,rhsVec,norms);
   return 0;
}

/**************************************************************************
 transfer the solution from lsc to the fei mesh so that when getSolution
 is called to fei, it will fetch the correct data
 -------------------------------------------------------------------------*/
int LLNL_FEI_Impl::transferSolution()
{
   int    localNRows, *diagIA, *diagJA;
   double *diagAA, *solnVector;
   matPtr_->getLocalMatrix(&localNRows, &diagIA, &diagJA, &diagAA);
   solnVector = new double[localNRows];
   lscPtr_->getSolution(solnVector, localNRows);
   feiPtr_->disassembleSolnVector(solnVector);
   return 0;
}

