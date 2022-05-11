/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/**************************************************************************
  Module:  LLNL_FEI_Solver.cxx
  Purpose: custom implementation of the FEI/Solver
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "LLNL_FEI_Solver.h"

#ifdef HAVE_SUPERLU_20
#include "dsp_defs.h"
#include "superlu_util.h"
#endif
#ifdef HAVE_SUPERLU
#include "slu_ddefs.h"
#include "slu_util.h"
#endif

/**************************************************************************
 Constructor
 -------------------------------------------------------------------------*/
LLNL_FEI_Solver::LLNL_FEI_Solver( MPI_Comm comm )
{
   mpiComm_     = comm;
   MPI_Comm_rank( comm, &mypid_ );
   outputLevel_ = 0;

   /* -----------------------------------------------------------------
    * solver information
    * ----------------------------------------------------------------*/

   solverID_            = 0;
   krylovMaxIterations_ = 1000;
   krylovAbsRel_        = 0;       /* 0 - relative norm */
   krylovTolerance_     = 1.0e-6;
   krylovIterations_    = 0;
   krylovResidualNorm_  = 0.0;
   gmresDim_            = 20;


   /* -----------------------------------------------------------------
    * node boundary condition information
    * ----------------------------------------------------------------*/

   matPtr_     = NULL;
   solnVector_ = NULL;
   rhsVector_  = NULL;

   /* -----------------------------------------------------------------
    * others
    * ----------------------------------------------------------------*/

   TimerSolve_        = 0.0;
   TimerSolveStart_   = 0.0;
}

/**************************************************************************
 parameters function
 -------------------------------------------------------------------------*/
int LLNL_FEI_Solver::parameters(int numParams, char **paramString)
{
   int  i, olevel;
#ifdef HAVE_SUPERLU
   int  nprocs;
#endif
   char param[256], param1[256];

   for ( i = 0; i < numParams; i++ )
   {
      sscanf(paramString[i],"%s", param1);
      if ( !strcmp(param1, "outputLevel") )
      {
         sscanf(paramString[i],"%s %d", param1, &olevel);
         outputLevel_ = olevel;
         if ( olevel < 0 ) outputLevel_ = 0;
         if ( olevel > 4 ) outputLevel_ = 4;
      }
      else if ( !strcmp(param1, "gmresDim") )
      {
         sscanf(paramString[i],"%s %d", param1, &gmresDim_);
         if ( gmresDim_ < 0 ) gmresDim_ = 10;
      }
      else if ( !strcmp(param1, "maxIterations") )
      {
         sscanf(paramString[i],"%s %d", param1, &krylovMaxIterations_);
         if ( krylovMaxIterations_ <= 0 ) krylovMaxIterations_ = 1;
      }
      else if ( !strcmp(param1, "tolerance") )
      {
         sscanf(paramString[i],"%s %lg", param1, &krylovTolerance_);
         if ( krylovTolerance_ >= 1.0 || krylovTolerance_ <= 0.0 )
            krylovTolerance_ = 1.0e-6;
      }
      else if ( !strcmp(param1, "stopCrit") )
      {
         sscanf(paramString[i],"%s %s", param1, param);
         if      ( !strcmp(param, "absolute") ) krylovAbsRel_ = 1;
         else if ( !strcmp(param, "relative") ) krylovAbsRel_ = 0;
         else                                  krylovAbsRel_ = 0;
      }
      else if ( !strcmp(param1, "solver") )
      {
         sscanf(paramString[i],"%s %s", param1, param);
         if      ( !strcmp(param, "cg") )      solverID_ = 0;
         else if ( !strcmp(param, "gmres") )   solverID_ = 1;
         else if ( !strcmp(param, "cgs") )     solverID_ = 2;
         else if ( !strcmp(param, "bicgstab")) solverID_ = 3;
         else if ( !strcmp(param, "superlu") )
         {
#ifdef HAVE_SUPERLU
            MPI_Comm_size( mpiComm_, &nprocs );
            if ( nprocs == 1 ) solverID_ = 4;
            else
            {
               printf("LLNL_FEI_Solver WARNING : SuperLU not supported on ");
               printf("more than 1 proc.  Use GMRES instead.\n");
               solverID_ = 1;
            }
#else
            printf("LLNL_FEI_Solver WARNING : SuperLU not available.\n");
            solverID_ = 1;
#endif
         }
         else solverID_ = 1;
      }
      else if ( !strcmp(param1, "preconditioner") )
      {
         sscanf(paramString[i],"%s %s", param1, param);
         if ( (! !strcmp(param, "diag")) && (! !strcmp(param, "diagonal")) )
            printf("LLNL_FEI_Solver::parameters - invalid preconditioner.\n");
      }
   }
   return 0;
}

/**************************************************************************
 solve the linear system
 -------------------------------------------------------------------------*/
int LLNL_FEI_Solver::solve(int *status)
{
   int    nprocs;
   double dArray[2], dArray2[2];

   if ( matPtr_ == NULL || solnVector_ == NULL || rhsVector_ == NULL )
   {
      printf("%4d : LLNL_FEI_Solver::solve ERROR - not initialized.\n",mypid_);
      (*status) = 1;
      return 1;
   }
   MPI_Comm_size(mpiComm_, &nprocs);
   if ( outputLevel_ >= 1 && mypid_ == 0 )
      printf("\t**************************************************\n");
   switch (solverID_)
   {
      case 0 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tLLNL_FEI CG with diagonal preconditioning\n");
                  printf("\tmaxIterations     = %d\n",krylovMaxIterations_);
                  printf("\ttolerance         = %e\n",krylovTolerance_);
               }
               (*status) = solveUsingCG();
               break;
      case 1 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tLLNL_FEI GMRES with diagonal preconditioning\n");
                  printf("\t\tGMRES dimension = %d\n", gmresDim_);
                  printf("\tmaxIterations     = %d\n",krylovMaxIterations_);
                  printf("\ttolerance         = %e\n",krylovTolerance_);
               }
               (*status) = solveUsingGMRES();
               break;
      case 2 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tLLNL_FEI CGS with diagonal preconditioning\n");
                  printf("\tmaxIterations     = %d\n",krylovMaxIterations_);
                  printf("\ttolerance         = %e\n",krylovTolerance_);
               }
               (*status) = solveUsingCGS();
               break;
      case 3 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tLLNL_FEI Bicgstab with diagonal preconditioning\n");
                  printf("\tmaxIterations     = %d\n",krylovMaxIterations_);
                  printf("\ttolerance         = %e\n",krylovTolerance_);
               }
               (*status) = solveUsingBicgstab();
               break;
      case 4 : TimerSolveStart_ = MPI_Wtime();
               if ( outputLevel_ >= 1 && mypid_ == 0 )
               {
                  printf("\tLLNL_FEI direct link to SuperLU \n");
               }
               (*status) = solveUsingSuperLU();
               break;
   }
   TimerSolve_ = MPI_Wtime() - TimerSolveStart_;
   if (outputLevel_ >= 2)
   {
      dArray[0] = TimerSolve_;
      dArray[1] = TimerSolve_;
      MPI_Allreduce(dArray,dArray2,1,MPI_DOUBLE,MPI_SUM,mpiComm_);
      MPI_Allreduce(&dArray[1],&dArray2[1],1,MPI_DOUBLE,MPI_MAX,mpiComm_);
   }
   if (outputLevel_ >= 1 && mypid_ == 0)
   {
      printf("\tLLNL_FEI local solver : number of iterations = %d\n",
             krylovIterations_);
      if (outputLevel_ >= 2)
      {
         printf("\tLLNL_FEI local solver : final residual norm  = %e\n",
                krylovResidualNorm_);
         printf("\tLLNL_FEI local solver    : average solve time   = %e\n",
                dArray2[0]/(double) nprocs);
         printf("\tLLNL_FEI local solver    : maximum solve time   = %e\n",
                dArray2[1]);
      }
      printf("\t**************************************************\n");
   }
   return (*status);
}

/**************************************************************************
 solve linear system using conjugate gradient
 -------------------------------------------------------------------------*/
int LLNL_FEI_Solver::solveUsingCG()
{
   int    irow, iter, converged=0, localNRows, extNRows, totalNRows;
   int    numTrials, innerIteration;
   double alpha, beta, rho=0.0, rhom1, rnorm0, rnorm, sigma, eps1;
   double *rVec, *pVec, *apVec, *zVec, dArray[2], dArray2[2], *diagonal;

   /* -----------------------------------------------------------------
    * compute matrix information and allocate Krylov vectors
    * -----------------------------------------------------------------*/

   localNRows = matPtr_->getNumLocalRows();
   extNRows   = matPtr_->getNumExtRows();
   diagonal   = matPtr_->getMatrixDiagonal();
   totalNRows = localNRows + extNRows;
   rVec       = new double[totalNRows];

   /* -----------------------------------------------------------------
    * compute initial residual vector and norm
    * -----------------------------------------------------------------*/

   matPtr_->matvec( solnVector_, rVec );
   for ( irow = 0; irow < localNRows; irow++ )
      rVec[irow] = rhsVector_[irow] - rVec[irow];
   rnorm0 = rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rnorm0 += (rVec[irow] * rVec[irow]);
      rnorm  += (rhsVector_[irow] * rhsVector_[irow]);
   }
   dArray[0] = rnorm0;
   dArray[1] = rnorm;
   MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
   rnorm0 = sqrt(dArray2[1]);
   rnorm  = sqrt(dArray2[0]);
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tLLNL_FEI_Solver_CG initial rnorm = %e (%e)\n",rnorm,rnorm0);
   if ( rnorm0 == 0.0 )
   {
      delete [] rVec;
      return 0;
   }

   /* -----------------------------------------------------------------
    * initialization
    * -----------------------------------------------------------------*/

   iter       = 0;
   numTrials  = 0;
   pVec       = new double[totalNRows];
   apVec      = new double[totalNRows];
   zVec       = new double[totalNRows];
   for ( irow = 0; irow < localNRows; irow++ ) pVec[irow] = 0.0;
   if ( krylovAbsRel_ == 0 ) eps1 = krylovTolerance_ * rnorm0;
   else                      eps1 = krylovTolerance_;
   if ( rnorm < eps1 ) converged = 1;

   /* -----------------------------------------------------------------
    * loop until convergence is achieved
    * -----------------------------------------------------------------*/

   while ( converged == 0 && numTrials < 2 )
   {
      innerIteration = 0;
      while ( rnorm >= eps1 && iter < krylovMaxIterations_ )
      {
         iter++;
         innerIteration++;
         if ( innerIteration == 1 )
         {
            if ( diagonal != NULL )
               for (irow = 0; irow < localNRows; irow++)
                  zVec[irow] = rVec[irow] * diagonal[irow];
            else
               for (irow = 0; irow < localNRows; irow++)
                  zVec[irow] = rVec[irow];

            rhom1 = rho;
            rho   = 0.0;
            for ( irow = 0; irow < localNRows; irow++ )
               rho += rVec[irow] * zVec[irow];
            dArray[0] = rho;
            MPI_Allreduce(dArray, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
            rho  = dArray2[0];
            beta = 0.0;
         }
         else beta = rho / rhom1;
         for ( irow = 0; irow < localNRows; irow++ )
            pVec[irow] = zVec[irow] + beta * pVec[irow];
         matPtr_->matvec( pVec, apVec );
         sigma = 0.0;
         for ( irow = 0; irow < localNRows; irow++ )
            sigma += pVec[irow] * apVec[irow];
         dArray[0] = sigma;
         MPI_Allreduce(dArray, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
         sigma  = dArray2[0];
         alpha  = rho / sigma;
         for ( irow = 0; irow < localNRows; irow++ )
         {
            solnVector_[irow] += alpha * pVec[irow];
            rVec[irow] -= alpha * apVec[irow];
         }
         rnorm = 0.0;
         for ( irow = 0; irow < localNRows; irow++ )
            rnorm += rVec[irow] * rVec[irow];
         dArray[0] = rnorm;

         if ( diagonal != NULL )
            for (irow = 0; irow < localNRows; irow++)
               zVec[irow] = rVec[irow] * diagonal[irow];
         else
            for (irow = 0; irow < localNRows; irow++) zVec[irow] = rVec[irow];

         rhom1 = rho;
         rho   = 0.0;
         for ( irow = 0; irow < localNRows; irow++ )
            rho += rVec[irow] * zVec[irow];
         dArray[1] = rho;
         MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
         rho = dArray2[1];
         rnorm = sqrt( dArray2[0] );
         if ( outputLevel_ >= 2 && iter % 1 == 0 && mypid_ == 0 )
            printf("\tLLNL_FEI_Solver_CG : iteration %d - rnorm = %e (%e)\n",
                   iter, rnorm, eps1);
      }
      matPtr_->matvec( solnVector_, rVec );
      for ( irow = 0; irow < localNRows; irow++ )
         rVec[irow] = rhsVector_[irow] - rVec[irow];
      rnorm = 0.0;
      for ( irow = 0; irow < localNRows; irow++ )
         rnorm += rVec[irow] * rVec[irow];
      dArray[0] = rnorm;
      MPI_Allreduce(dArray, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
      rnorm = sqrt( dArray2[0] );
      if ( outputLevel_ >= 2 && mypid_ == 0 )
         printf("\tLLNL_FEI_Solver_CG actual rnorm = %e \n",rnorm);
      if ( (rnorm < eps1 || rnorm < 1.0e-16) ||
            iter >= krylovMaxIterations_ ) converged = 1;
      numTrials++;
   }

   krylovIterations_   = iter;
   krylovResidualNorm_ = rnorm;

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/

   delete [] rVec;
   delete [] pVec;
   delete [] apVec;
   delete [] zVec;
   return (1-converged);
}

/**************************************************************************
 solve linear system using GMRES
 -------------------------------------------------------------------------*/
int LLNL_FEI_Solver::solveUsingGMRES()
{
   int    irow, iter, converged=0, localNRows, extNRows, totalNRows;
   int    innerIterations, iV, iV2, kStep, kp1, jV;
   double rnorm0, rnorm, eps1, epsmac=1.0e-16, gam;
   double **kVectors, **HH, *RS, *C, *S, *dArray, *dArray2;
   double *tVector, *tVector2, *v1, *v2, *diagonal, dtemp;

   /* -----------------------------------------------------------------
    * compute matrix information and allocate Krylov vectors
    * -----------------------------------------------------------------*/

   localNRows = matPtr_->getNumLocalRows();
   extNRows   = matPtr_->getNumExtRows();
   diagonal   = matPtr_->getMatrixDiagonal();
   totalNRows = localNRows + extNRows;
   kVectors   = new double*[gmresDim_+2];
   for (iV = 0; iV <= gmresDim_+1; iV++) kVectors[iV] = new double[totalNRows];
   dArray  = new double[gmresDim_+1];
   dArray2 = new double[gmresDim_+1];

   /* -----------------------------------------------------------------
    * compute initial residual vector and norm
    * -----------------------------------------------------------------*/

   tVector = kVectors[1];
   matPtr_->matvec( solnVector_, tVector );
   for ( irow = 0; irow < localNRows; irow++ )
      tVector[irow] = rhsVector_[irow] - tVector[irow];
   rnorm0 = rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rnorm0 += (tVector[irow] * tVector[irow]);
      rnorm  += (rhsVector_[irow] * rhsVector_[irow]);
   }
   dArray[0] = rnorm0;
   dArray[1] = rnorm;
   MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
   rnorm0 = sqrt(dArray2[0]);
   rnorm  = sqrt(dArray2[1]);
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tLLNL_FEI_Solver_GMRES initial rnorm = %e (%e)\n",
             rnorm, rnorm0);
   if ( rnorm0 < 1.0e-20 )
   {
      for (iV = 0; iV <= gmresDim_+1; iV++) delete [] kVectors[iV];
      delete [] kVectors;
      delete [] dArray;
      delete [] dArray2;
      return 0;
   }

   /* -----------------------------------------------------------------
    * initialization
    * -----------------------------------------------------------------*/

   if ( krylovAbsRel_ == 0 ) eps1 = krylovTolerance_ * rnorm0;
   else                      eps1 = krylovTolerance_;
   HH = new double*[gmresDim_+2];
   for (iV=1; iV<=gmresDim_+1; iV++) HH[iV] = new double[gmresDim_+2];
   RS = new double[gmresDim_+2];
   S  = new double[gmresDim_+1];
   C  = new double[gmresDim_+1];

   /* -----------------------------------------------------------------
    * loop until convergence is achieved
    * -----------------------------------------------------------------*/

   iter = 0;

   while ( rnorm >= eps1 && iter < krylovMaxIterations_ )
   {
      dtemp = 1.0 / rnorm;
      tVector = kVectors[1];
      for (irow = 0; irow < localNRows; irow++) tVector[irow] *= dtemp;
      RS[1] = rnorm;
      innerIterations = 0;

      while ( innerIterations < gmresDim_ && rnorm >= eps1 &&
              iter < krylovMaxIterations_ )
      {
         innerIterations++;
         iter++;
         kStep = innerIterations;
         kp1   = innerIterations + 1;
         v1   = kVectors[kStep];
         v2   = kVectors[0];
         if ( diagonal != NULL )
            for (irow = 0; irow < localNRows; irow++)
               v2[irow] = v1[irow] * diagonal[irow];
         else
            for (irow = 0; irow < localNRows; irow++) v2[irow] = v1[irow];

         matPtr_->matvec( kVectors[0], kVectors[kp1] );

#if 0
         tVector = kVectors[kp1];
         for ( iV = 1; iV <= kStep; iV++ )
         {
            dtemp = 0.0;
            tVector2 = kVectors[iV];
            for ( irow = 0; irow < localNRows; irow++ )
               dtemp += tVector2[irow] * tVector[irow];
            dArray[iV-1] = dtemp;
         }
         MPI_Allreduce(dArray, dArray2, kStep, MPI_DOUBLE, MPI_SUM,
                       mpiComm_);

         tVector  = kVectors[kp1];
         for ( iV = 1; iV <= kStep; iV++ )
         {
            dtemp = dArray2[iV-1];
            HH[iV][kStep] = dtemp;
            tVector2 = kVectors[iV];
            for ( irow = 0; irow < localNRows; irow++ )
               tVector[irow] -= dtemp * tVector2[irow];
         }
#else
         tVector = kVectors[kp1];
         for ( iV = 1; iV <= kStep; iV++ )
         {
            dtemp = 0.0;
            tVector2 = kVectors[iV];
            for ( irow = 0; irow < localNRows; irow++ )
               dtemp += tVector2[irow] * tVector[irow];
            dArray[0] = dtemp;
            MPI_Allreduce(dArray, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
            dtemp = dArray2[0];
            HH[iV][kStep] = dtemp;
            for ( irow = 0; irow < localNRows; irow++ )
               tVector[irow] -= dtemp * tVector2[irow];
         }
#endif
         dtemp = 0.0;
         for ( irow = 0; irow < localNRows; irow++ )
            dtemp += tVector[irow] * tVector[irow];
         MPI_Allreduce(&dtemp, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
         dtemp = sqrt(dArray2[0]);
         HH[kp1][kStep] = dtemp;
         if ( dtemp != 0.0 )
         {
            dtemp = 1.0 / dtemp;
            for (irow = 0; irow < localNRows; irow++) tVector[irow] *= dtemp;
         }
         for ( iV = 2; iV <= kStep; iV++ )
         {
            dtemp = HH[iV-1][kStep];
            HH[iV-1][kStep] =  C[iV-1] * dtemp + S[iV-1] * HH[iV][kStep];
            HH[iV][kStep]   = -S[iV-1] * dtemp + C[iV-1] * HH[iV][kStep];
         }
         gam = sqrt(HH[kStep][kStep]*HH[kStep][kStep]+
                    HH[kp1][kStep]*HH[kp1][kStep]);
         if ( gam == 0.0 ) gam = epsmac;
         C[kStep]  = HH[kStep][kStep] / gam;
         S[kStep]  = HH[kp1][kStep] / gam;
         RS[kp1]   = -S[kStep] * RS[kStep];
         RS[kStep] = C[kStep] * RS[kStep];
         HH[kStep][kStep] = C[kStep] * HH[kStep][kStep] +
                            S[kStep] * HH[kp1][kStep];
         rnorm = fabs(RS[kp1]);
         if ( outputLevel_ >= 2 && mypid_ == 0 )
            printf("\tLLNL_FEI_Solver_GMRES : iteration %d - rnorm = %e\n",
                   iter, rnorm);
      }
      RS[kStep] = RS[kStep] / HH[kStep][kStep];
      for ( iV = 2; iV <= kStep; iV++ )
      {
         iV2 = kStep - iV + 1;
         dtemp = RS[iV2];
         for ( jV = iV2+1; jV <= kStep; jV++ )
            dtemp = dtemp - HH[iV2][jV] * RS[jV];
         RS[iV2] = dtemp / HH[iV2][iV2];
      }
      tVector = kVectors[1];
      dtemp   = RS[1];
      for ( irow = 0; irow < localNRows; irow++ ) tVector[irow] *= dtemp;
      for ( iV = 2; iV <= kStep; iV++ )
      {
         dtemp = RS[iV];
         tVector2 = kVectors[iV];
         for ( irow = 0; irow < localNRows; irow++ )
            tVector[irow] += dtemp * tVector2[irow];
      }
      tVector = kVectors[1];
      if ( diagonal != NULL )
      {
         for (irow = 0; irow < localNRows; irow++)
            tVector[irow] *= diagonal[irow];
      }
      for (irow = 0; irow < localNRows; irow++)
         solnVector_[irow] += tVector[irow];
      matPtr_->matvec( solnVector_, tVector );
      for ( irow = 0; irow < localNRows; irow++ )
         tVector[irow] = rhsVector_[irow] - tVector[irow];
      rnorm = 0.0;
      for ( irow = 0; irow < localNRows; irow++ )
         rnorm += (tVector[irow] * tVector[irow]);
      MPI_Allreduce(&rnorm, dArray2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
      rnorm = sqrt(dArray2[0]);
   }
   if ( rnorm < eps1 ) converged = 1;
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tLLNL_FEI_Solver_GMRES : final rnorm = %e\n", rnorm);

   krylovIterations_   = iter;
   krylovResidualNorm_ = rnorm;

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/

   for (iV = 0; iV <= gmresDim_+1; iV++) delete [] kVectors[iV];
   delete [] kVectors;
   for ( iV =1; iV <= gmresDim_+1; iV++ ) delete [] HH[iV];
   delete [] HH;
   delete [] RS;
   delete [] S;
   delete [] C;
   delete [] dArray;
   delete [] dArray2;
   return (1-converged);
}

/**************************************************************************
 solve linear system using CGS
 -------------------------------------------------------------------------*/
int LLNL_FEI_Solver::solveUsingCGS()
{
   int    irow, iter, converged=0, localNRows, extNRows, totalNRows;
   int    numTrials, innerIteration;
   double *rVec, *rhVec, *vVec, *pVec, *qVec, *uVec, *tVec;
   double rho1, rho2, sigma, alpha, dtemp, dtemp2, rnorm, rnorm0;
   double beta, beta2, eps1, dArray[2], dArray2[2], *diagonal;

   /* -----------------------------------------------------------------
    * compute matrix information and allocate Krylov vectors
    * -----------------------------------------------------------------*/

   localNRows = matPtr_->getNumLocalRows();
   extNRows   = matPtr_->getNumExtRows();
   diagonal   = matPtr_->getMatrixDiagonal();
   totalNRows = localNRows + extNRows;
   rVec       = new double[totalNRows];

   /* -----------------------------------------------------------------
    * compute initial residual vector and norm
    * -----------------------------------------------------------------*/

   matPtr_->matvec( solnVector_, rVec );
   for ( irow = 0; irow < localNRows; irow++ )
      rVec[irow] = rhsVector_[irow] - rVec[irow];
   rnorm0 = rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rnorm0 += (rVec[irow] * rVec[irow]);
      rnorm  += (rhsVector_[irow] * rhsVector_[irow]);
   }
   dArray[0] = rnorm0;
   dArray[1] = rnorm;
   MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
   rnorm0 = sqrt(dArray2[1]);
   rnorm  = sqrt(dArray2[0]);
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tLLNL_FEI_Solver_CGS initial rnorm = %e (%e)\n",rnorm,rnorm0);
   if ( rnorm0 == 0.0 )
   {
      delete [] rVec;
      return 0;
   }

   /* -----------------------------------------------------------------
    * initialization
    * -----------------------------------------------------------------*/

   rhVec = new double[totalNRows];
   vVec  = new double[totalNRows];
   pVec  = new double[totalNRows];
   qVec  = new double[totalNRows];
   uVec  = new double[totalNRows];
   tVec  = new double[totalNRows];
   for (irow = 0; irow < localNRows; irow++) rhVec[irow] = rVec[irow];
   for (irow = 0; irow < totalNRows; irow++) pVec[irow] = qVec[irow] = 0.0;
   rho2 = rnorm * rnorm;
   beta = rho2;
   iter = 0;
   numTrials  = 0;
   if ( krylovAbsRel_ == 0 ) eps1 = krylovTolerance_ * rnorm0;
   else                      eps1 = krylovTolerance_;
   if ( rnorm < eps1 )  converged = 1;

   /* -----------------------------------------------------------------
    * loop until convergence is achieved
    * -----------------------------------------------------------------*/

   while ( converged == 0 && numTrials < 1 )
   {
      innerIteration = 0;
      while ( rnorm >= eps1 && iter < krylovMaxIterations_ )
      {
         iter++;
         innerIteration++;
         rho1 = rho2;
         beta2 = beta * beta;
         for (irow = 0; irow < totalNRows; irow++)
         {
            tVec[irow] = beta * qVec[irow];
            uVec[irow] = rVec[irow] + tVec[irow];
            pVec[irow] = uVec[irow] + tVec[irow] + beta2 * pVec[irow];
         }
         if ( diagonal != NULL )
         {
            for (irow = 0; irow < localNRows; irow++)
               tVec[irow] = pVec[irow] * diagonal[irow];
         }
         else
            for (irow = 0; irow < localNRows; irow++) tVec[irow] = pVec[irow];

         matPtr_->matvec( tVec, vVec );
         sigma = 0.0;
         for ( irow = 0; irow < localNRows; irow++ )
            sigma += (rhVec[irow] * vVec[irow]);
         MPI_Allreduce(&sigma, dArray, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
         sigma = dArray[0];
         alpha = rho1 / sigma;

         for (irow = 0; irow < totalNRows; irow++)
         {
            qVec[irow] = uVec[irow] - alpha * vVec[irow];
            uVec[irow] += qVec[irow];
         }
         if ( diagonal != NULL )
         {
            for (irow = 0; irow < localNRows; irow++)
            {
               tVec[irow] = uVec[irow] * diagonal[irow];
               solnVector_[irow] += alpha * uVec[irow] * diagonal[irow];
            }
         }
         else
         {
            for (irow = 0; irow < localNRows; irow++)
            {
               tVec[irow] = uVec[irow];
               solnVector_[irow] += alpha * uVec[irow];
            }
         }
         matPtr_->matvec( tVec, vVec );

         for (irow = 0; irow < totalNRows; irow++)
            rVec[irow] -= alpha * vVec[irow];

         dtemp = dtemp2 = 0.0;
         for ( irow = 0; irow < localNRows; irow++ )
         {
            dtemp  += (rVec[irow] * rhVec[irow]);
            dtemp2 += (rVec[irow] * rVec[irow]);
         }
         dArray[0] = dtemp;
         dArray[1] = dtemp2;
         MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
         rho2 = dArray2[0];
         beta = rho2 / rho1;
         rnorm = sqrt(dArray2[1]);
         if ( outputLevel_ >= 2 && iter % 1 == 0 && mypid_ == 0 )
            printf("\tLLNL_FEI_Solver_CGS : iteration %d - rnorm = %e (%e)\n",
                   iter, rnorm, eps1);
      }
      matPtr_->matvec( solnVector_, rVec );
      for ( irow = 0; irow < localNRows; irow++ )
         rVec[irow] = rhsVector_[irow] - rVec[irow];
      rnorm = 0.0;
      for ( irow = 0; irow < localNRows; irow++ )
         rnorm += rVec[irow] * rVec[irow];
      MPI_Allreduce(&rnorm, dArray, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
      rnorm = sqrt( dArray[0] );
      if ( outputLevel_ >= 2 && mypid_ == 0 )
         printf("\tLLNL_FEI_Solver_CGS actual rnorm = %e \n",rnorm);
      if ( rnorm < eps1 || iter >= krylovMaxIterations_ ) break;
      numTrials++;
   }
   if ( rnorm < eps1 ) converged = 1;

   krylovIterations_   = iter;
   krylovResidualNorm_ = rnorm;

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/

   delete [] rVec;
   delete [] rhVec;
   delete [] pVec;
   delete [] qVec;
   delete [] uVec;
   delete [] tVec;
   return (1-converged);
}

/**************************************************************************
 solve linear system using Bicgstab
 -------------------------------------------------------------------------*/
int LLNL_FEI_Solver::solveUsingBicgstab()
{
   int    irow, iter, converged=0, localNRows, extNRows, totalNRows;
   int    iM, jM, numTrials, innerIteration, blen=2;
   double *rVec, *rhVec, *xhVec, *tVec, **utVec, **rtVec, *ut2, *rt2;
   double rho, rho1, alpha, dtemp, dtemp2, rnorm, rnorm0, *ut1, *rt1;
   double beta, omega, gamma, eps1, dArray[2], dArray2[2], *diagonal;
   double *sigma, *gammap, *gammanp, *gammapp, **mat, **tau;

   /* -----------------------------------------------------------------
    * compute matrix information and allocate Krylov vectors
    * -----------------------------------------------------------------*/

   localNRows  = matPtr_->getNumLocalRows();
   extNRows    = matPtr_->getNumExtRows();
   diagonal    = matPtr_->getMatrixDiagonal();
   totalNRows  = localNRows + extNRows;
   rVec        = new double[totalNRows];

   /* -----------------------------------------------------------------
    * compute initial residual vector and norm
    * -----------------------------------------------------------------*/

   matPtr_->matvec( solnVector_, rVec );
   for ( irow = 0; irow < localNRows; irow++ )
      rVec[irow] = rhsVector_[irow] - rVec[irow];
   rnorm0 = rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rnorm0 += (rVec[irow] * rVec[irow]);
      rnorm  += (rhsVector_[irow] * rhsVector_[irow]);
   }
   dArray[0] = rnorm0;
   dArray[1] = rnorm;
   MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
   rnorm0 = sqrt(dArray2[1]);
   rnorm  = sqrt(dArray2[0]);
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tLLNL_FEI_Solver_Bicgstab initial rnorm = %e (%e)\n",
             rnorm,rnorm0);
   if ( rnorm0 == 0.0 )
   {
      delete [] rVec;
      return 0;
   }

   /* -----------------------------------------------------------------
    * initialization
    * -----------------------------------------------------------------*/

   if ( krylovAbsRel_ == 0 ) eps1 = krylovTolerance_ * rnorm0;
   else                      eps1 = krylovTolerance_;
   if ( rnorm < eps1 )  converged = 1;

   sigma   = new double[blen+1];
   gammap  = new double[blen+1];
   gammanp = new double[blen+1];
   gammapp = new double[blen+1];
   mat     = new double*[blen+1];
   tau     = new double*[blen+1];
   for ( iM = 1; iM <= blen; iM++ )
   {
      mat[iM] = new double[blen+1];
      tau[iM] = new double[blen+1];
   }
   rhVec = new double[totalNRows];
   xhVec = new double[totalNRows];
   tVec  = new double[totalNRows];
   utVec = new double*[blen+2];
   rtVec = new double*[blen+2];
   for ( iM = 0; iM < blen+2; iM++ )
   {
      utVec[iM] = new double[totalNRows];
      rtVec[iM] = new double[totalNRows];
   }
   iter = 0;
   numTrials  = 0;

   /* -----------------------------------------------------------------
    * loop until convergence is achieved
    * -----------------------------------------------------------------*/

   while ( converged == 0 && numTrials < 1 )
   {
      innerIteration = 0;
      for ( irow = 0; irow < localNRows; irow++ )
      {
         rhVec[irow] = rtVec[0][irow] = rVec[irow];
         xhVec[irow] = solnVector_[irow];
         utVec[0][irow] = 0.0;
      }
      omega = rho = 1.0;
      alpha = 0.0;
      while ( rnorm >= eps1 && iter < krylovMaxIterations_ )
      {
         iter += blen;
         innerIteration += blen;
         ut1 = utVec[0];
         ut2 = utVec[1];
         rt1 = rtVec[0];
         rt2 = rtVec[1];
         for ( irow = 0; irow < localNRows; irow++ )
         {
            ut2[irow] = ut1[irow];
            rt2[irow] = rt1[irow];
         }
         rho = -omega * rho;
         for ( iM = 0; iM < blen; iM++ )
         {
            dtemp = 0.0;
            for ( irow = 0; irow < localNRows; irow++ )
               dtemp += (rhVec[irow] * rtVec[iM+1][irow]);
            MPI_Allreduce(&dtemp, &rho1, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
            beta = alpha * rho1 / rho;
            rho   = rho1;
            dtemp = -beta;
            for ( jM = 0; jM <= iM; jM++ )
               for ( irow = 0; irow < localNRows; irow++ )
                  utVec[jM+1][irow] = dtemp * utVec[jM+1][irow] +
                                      rtVec[jM+1][irow];
            if (diagonal != NULL)
            {
               ut1 = utVec[iM+1];
               for (irow = 0; irow < localNRows; irow++)
                  tVec[irow] = ut1[irow] * diagonal[irow];
            }
            else
            {
               ut1 = utVec[iM+1];
               for (irow = 0; irow < localNRows; irow++)
                  tVec[irow] = ut1[irow];
            }
            matPtr_->matvec( tVec, utVec[iM+2] );
            dtemp = 0.0;
            for ( irow = 0; irow < localNRows; irow++ )
               dtemp += (rhVec[irow] * utVec[iM+2][irow]);
            MPI_Allreduce(&dtemp, &gamma, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);

            alpha = rho / gamma;
            for ( jM = 0; jM <= iM; jM++ )
               for ( irow = 0; irow < localNRows; irow++ )
                  rtVec[jM+1][irow] -= alpha * utVec[jM+2][irow];

            if ( diagonal != NULL )
            {
               for (irow = 0; irow < localNRows; irow++)
                  tVec[irow] = rtVec[iM+1][irow] * diagonal[irow];
            }
            else
            {
               rt1 = rtVec[iM+1];
               for (irow = 0; irow < localNRows; irow++)
                  tVec[irow] = rt1[irow];
            }
            matPtr_->matvec( tVec, rtVec[iM+2] );
            for (irow = 0; irow < localNRows; irow++)
               xhVec[irow] += alpha * utVec[1][irow];
         }
         for ( iM = 1; iM <= blen; iM++ )
            for ( jM = 1; jM <= blen; jM++ ) mat[iM][jM] = 0.0;
         for ( iM = 1; iM <= blen; iM++ )
         {
            for ( jM = 1; jM <= iM-1; jM++ )
            {
               dtemp = 0.0;
               for ( irow = 0; irow < localNRows; irow++ )
                  dtemp += (rtVec[jM+1][irow] * rtVec[iM+1][irow]);
               MPI_Allreduce(&dtemp, &dtemp2, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
               tau[jM][iM] = dtemp2 / sigma[jM];
               mat[jM][iM] = tau[jM][iM] * sigma[jM];
               dtemp = -tau[jM][iM];
               for (irow = 0; irow < localNRows; irow++)
                  rtVec[iM+1][irow] += dtemp * rtVec[jM+1][irow];
            }
            dtemp = 0.0;
            for ( irow = 0; irow < localNRows; irow++ )
               dtemp += (rtVec[iM+1][irow] * rtVec[iM+1][irow]);
            dArray[0] = dtemp;
            dtemp = 0.0;
            for ( irow = 0; irow < localNRows; irow++ )
               dtemp += (rtVec[1][irow] * rtVec[iM+1][irow]);
            dArray[1] = dtemp;
            MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, mpiComm_);
            sigma[iM] = dArray2[0];
            mat[iM][iM] = sigma[iM];
            gammap[iM] = dArray2[1] / sigma[iM];
         }
         gammanp[blen] = gammap[blen];
         omega = gammanp[blen];
         for ( iM = blen-1; iM >= 1; iM-- )
         {
           gammanp[iM] = gammap[iM];
           for (jM=iM+1; jM<=blen; jM++)
             gammanp[iM] = gammanp[iM] - tau[iM][jM] * gammanp[jM];
         }
         for (iM=1; iM<=blen-1; iM++)
         {
            gammapp[iM] = gammanp[iM+1];
            for (jM=iM+1; jM<=blen-1; jM++)
               gammapp[iM] = gammapp[iM] + tau[iM][jM] * gammanp[jM+1];
         }
         dtemp = gammanp[1];
         for (irow = 0; irow < localNRows; irow++)
            xhVec[irow] += dtemp * rtVec[1][irow];
         dtemp = - gammap[blen];
         for (irow = 0; irow < localNRows; irow++)
            rtVec[1][irow] += dtemp * rtVec[blen+1][irow];
         dtemp = - gammanp[blen];
         for (irow = 0; irow < localNRows; irow++)
            utVec[1][irow] += dtemp * utVec[blen+1][irow];
         for (iM=1; iM<=blen-1; iM++)
         {
            dtemp = - gammanp[iM];
            for (irow = 0; irow < localNRows; irow++)
               utVec[1][irow] += dtemp * utVec[iM+1][irow];
            dtemp = gammapp[iM];
            for (irow = 0; irow < localNRows; irow++)
               xhVec[irow] += dtemp * rtVec[iM+1][irow];
            dtemp = - gammap[iM];
            for (irow = 0; irow < localNRows; irow++)
               rtVec[1][irow] += dtemp * rtVec[iM+1][irow];
         }
         ut1 = utVec[0];
         ut2 = utVec[1];
         rt1 = rtVec[0];
         rt2 = rtVec[1];
         for ( irow = 0; irow < localNRows; irow++ )
         {
            ut1[irow] = ut2[irow];
            rt1[irow] = rt2[irow];
            solnVector_[irow] = xhVec[irow];
         }
         dtemp = 0.0;
         for ( irow = 0; irow < localNRows; irow++ )
            dtemp += (rtVec[1][irow] * rtVec[1][irow]);
         MPI_Allreduce(&dtemp, &rnorm, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
         rnorm = sqrt( rnorm );
         if ( outputLevel_ >= 2 && iter % 1 == 0 && mypid_ == 0 )
            printf("\tLLNL_FEI_Solver_Bicgstab : iteration %d - rnorm = %e (%e)\n",
                   iter, rnorm, eps1);
      }
      if ( diagonal != NULL )
      {
         for (irow = 0; irow < localNRows; irow++)
            solnVector_[irow] *= diagonal[irow];
      }
      matPtr_->matvec( solnVector_, rVec );
      for ( irow = 0; irow < localNRows; irow++ )
         rVec[irow] = rhsVector_[irow] - rVec[irow];
      rnorm = 0.0;
      for ( irow = 0; irow < localNRows; irow++ )
         rnorm += rVec[irow] * rVec[irow];
      MPI_Allreduce(&rnorm, dArray, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
      rnorm = sqrt( dArray[0] );
      if ( outputLevel_ >= 2 && mypid_ == 0 )
         printf("\tLLNL_FEI_Solver_Bicgstab actual rnorm = %e \n",rnorm);
      if ( rnorm < eps1 || iter >= krylovMaxIterations_ ) break;
      numTrials++;
   }
   if ( rnorm < eps1 ) converged = 1;

   krylovIterations_   = iter;
   krylovResidualNorm_ = rnorm;

   /* -----------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------------*/

   delete [] sigma;
   delete [] gammap;
   delete [] gammanp;
   delete [] gammapp;
   for ( iM = 1; iM <= blen; iM++ )
   {
      delete [] mat[iM];
      delete [] tau[iM];
   }
   delete [] mat;
   delete [] tau;
   delete [] rVec;
   delete [] rhVec;
   delete [] xhVec;
   delete [] tVec;
   for ( iM = 0; iM < blen+2; iM++ )
   {
      delete [] utVec[iM];
      delete [] rtVec[iM];
   }
   delete [] utVec;
   delete [] rtVec;

   return (1-converged);
}

/**************************************************************************
 solve linear system using SuperLU
 -------------------------------------------------------------------------*/
int LLNL_FEI_Solver::solveUsingSuperLU()
{
#ifdef HAVE_SUPERLU
   int    localNRows, localNnz, *countArray, irow, jcol, *cscIA, *cscJA;
   int    colNum, index, *etree, permcSpec, lwork, panelSize, relax, info;
   int    *permC, *permR, *diagIA, *diagJA;
   double *cscAA, diagPivotThresh, *rVec, rnorm;
   double *diagAA;
   trans_t           trans;
   superlu_options_t slu_options;
   SuperLUStat_t     slu_stat;
   GlobalLU_t        Glu;
   SuperMatrix superLU_Amat;
   SuperMatrix superLU_Lmat;
   SuperMatrix superLU_Umat;
   SuperMatrix AC;
   SuperMatrix B;

   /* ---------------------------------------------------------------
    * conversion from CSR to CSC
    * -------------------------------------------------------------*/

   matPtr_->getLocalMatrix(&localNRows,&diagIA,&diagJA,&diagAA);
   countArray = new int[localNRows];
   for ( irow = 0; irow < localNRows; irow++ ) countArray[irow] = 0;
   for ( irow = 0; irow < localNRows; irow++ )
      for ( jcol = diagIA[irow]; jcol < diagIA[irow+1]; jcol++ )
         countArray[diagJA[jcol]]++;
   localNnz = diagIA[localNRows];
   cscJA = hypre_TAlloc(int,  (localNRows+1) , HYPRE_MEMORY_HOST);
   cscIA = hypre_TAlloc(int,  localNnz , HYPRE_MEMORY_HOST);
   cscAA = hypre_TAlloc(double,  localNnz , HYPRE_MEMORY_HOST);
   cscJA[0] = 0;
   localNnz = 0;
   for ( jcol = 1; jcol <= localNRows; jcol++ )
   {
      localNnz += countArray[jcol-1];
      cscJA[jcol] = localNnz;
   }
   for ( irow = 0; irow < localNRows; irow++ )
   {
      for ( jcol = diagIA[irow]; jcol < diagIA[irow+1]; jcol++ )
      {
         colNum = diagJA[jcol];
         index  = cscJA[colNum]++;
         cscIA[index] = irow;
         cscAA[index] = diagAA[jcol];
      }
   }
   cscJA[0] = 0;
   localNnz = 0;
   for ( jcol = 1; jcol <= localNRows; jcol++ )
   {
      localNnz += countArray[jcol-1];
      cscJA[jcol] = localNnz;
   }
   delete [] countArray;

   /* ---------------------------------------------------------------
    * make SuperMatrix
    * -------------------------------------------------------------*/

   dCreate_CompCol_Matrix(&superLU_Amat, localNRows, localNRows,
                          cscJA[localNRows], cscAA, cscIA, cscJA, SLU_NC,
                          SLU_D, SLU_GE);
   etree     = new int[localNRows];
   permC     = new int[localNRows];
   permR     = new int[localNRows];
   permcSpec = 0;
   get_perm_c(permcSpec, &superLU_Amat, permC);
   slu_options.Fact = DOFACT;
   slu_options.SymmetricMode = NO;
   sp_preorder(&slu_options, &superLU_Amat, permC, etree, &AC);
   diagPivotThresh = 1.0;
   panelSize = sp_ienv(1);
   relax = sp_ienv(2);
   StatInit(&slu_stat);
   lwork = 0;
   slu_options.ColPerm = MY_PERMC;
   slu_options.DiagPivotThresh = diagPivotThresh;

//   dgstrf(&slu_options, &AC, dropTol, relax, panelSize,
//          etree, NULL, lwork, permC, permR, &superLU_Lmat,
//          &superLU_Umat, &slu_stat, &info);
   dgstrf(&slu_options, &AC, relax, panelSize,
          etree, NULL, lwork, permC, permR, &superLU_Lmat,
          &superLU_Umat, &Glu, &slu_stat, &info);

   Destroy_CompCol_Permuted(&AC);
   Destroy_CompCol_Matrix(&superLU_Amat);
   delete [] etree;

   /* -------------------------------------------------------------
    * create a SuperLU dense matrix from right hand side
    * -----------------------------------------------------------*/

   for ( irow = 0; irow < localNRows; irow++ )
      solnVector_[irow] = rhsVector_[irow];
   dCreate_Dense_Matrix(&B, localNRows, 1, solnVector_, localNRows,
                        SLU_DN, SLU_D, SLU_GE);

   /* -------------------------------------------------------------
    * solve the problem
    * -----------------------------------------------------------*/

   trans = NOTRANS;
   dgstrs (trans, &superLU_Lmat, &superLU_Umat, permC, permR, &B,
           &slu_stat, &info);
   rVec = new double[localNRows];
   matPtr_->matvec( solnVector_, rVec );
   for ( irow = 0; irow < localNRows; irow++ )
      rVec[irow] = rhsVector_[irow] - rVec[irow];
   rnorm = 0.0;
   for ( irow = 0; irow < localNRows; irow++ )
      rnorm += rVec[irow] * rVec[irow];
   rnorm = sqrt( rnorm );
   if ( outputLevel_ >= 2 && mypid_ == 0 )
      printf("\tLLNL_FEI_Solver_SuperLU rnorm = %e \n",rnorm);

   krylovIterations_   = 1;
   krylovResidualNorm_ = rnorm;

   /* -------------------------------------------------------------
    * clean up
    * -----------------------------------------------------------*/

   Destroy_SuperMatrix_Store(&B);
   delete [] rVec;
   if ( permR != NULL )
   {
      Destroy_SuperNode_Matrix(&superLU_Lmat);
      Destroy_CompCol_Matrix(&superLU_Umat);
   }
   delete [] permR;
   delete [] permC;
   StatFree(&slu_stat);
   return (info);
#else
   return (1);
#endif
}

