/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * functions for the top level MLI data structure
 *
 *****************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "mli.h"
#include "mli_utils.h"

/*****************************************************************************
 * constructor 
 *---------------------------------------------------------------------------*/

MLI::MLI( MPI_Comm comm )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::MLI\n");
#endif
   mpiComm_       = comm;
   maxLevels_     = 40;
   numLevels_     = 40;
   coarsestLevel_ = 0;
   outputLevel_   = 0;
   assembled_     = MLI_FALSE;
   tolerance_     = 1.0e-6;
   maxIterations_ = 20;
   currIter_      = 0;
   oneLevels_     = new MLI_OneLevel*[maxLevels_];
   for (int j = 0; j < maxLevels_; j++) oneLevels_[j] = new MLI_OneLevel(this);
   for (int i = 0; i < maxLevels_; i++)
   {
      oneLevels_[i]->setLevelNum(i);
      if ( i < (maxLevels_-1) ) oneLevels_[i]->setNextLevel(oneLevels_[i+1]);
      if ( i > 0 )              oneLevels_[i]->setPrevLevel(oneLevels_[i-1]);
   }
   coarseSolver_ = NULL;
   methodPtr_    = NULL;
   solveTime_    = 0.0;
   buildTime_    = 0.0;
} 

/*****************************************************************************
 * destructor 
 *---------------------------------------------------------------------------*/

MLI::~MLI()
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::~MLI\n");
#endif
   for ( int i = 0; i < maxLevels_; i++ ) delete oneLevels_[i];
   delete [] oneLevels_;
   if ( coarseSolver_ != NULL ) delete coarseSolver_;
   if ( methodPtr_    != NULL ) delete methodPtr_;
}

/*****************************************************************************
 * set discretization matrix
 *---------------------------------------------------------------------------*/

int MLI::setSystemMatrix( int level, MLI_Matrix *A )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setSystemMatrix, level = %d\n", level);
#endif
   if ( level >= 0 && level < maxLevels_ ) oneLevels_[level]->setAmat( A );
   else
   {
      printf("MLI::setSystemMatrix ERROR : wrong level = %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set restriction operator
 *---------------------------------------------------------------------------*/

int MLI::setRestriction( int level, MLI_Matrix *R )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setRestriction, level = %d\n", level);
#endif
   if ( level >= 0 && level < maxLevels_ ) oneLevels_[level]->setRmat( R );
   else
   {
      printf("MLI::setRestriction ERROR : wrong level = %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set prolongation operator
 *---------------------------------------------------------------------------*/

int MLI::setProlongation( int level, MLI_Matrix *P )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setProlongation, level = %d\n", level);
#endif
   if ( level >= 0 && level < maxLevels_ ) oneLevels_[level]->setPmat( P );
   else
   {
      printf("MLI::setProlongation ERROR : wrong level = %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set smoother 
 *---------------------------------------------------------------------------*/

int MLI::setSmoother( int level, int pre_post, MLI_Solver *smoother )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setSmoother, level = %d\n", level);
#endif
   if ( level >= 0 && level < maxLevels_ )
   {
      oneLevels_[level]->setSmoother( pre_post, smoother );
   }
   else
   {
      printf("MLI::setSmoother ERROR : wrong level = %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set coarse solver 
 *---------------------------------------------------------------------------*/

int MLI::setCoarseSolve( MLI_Solver *solver )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setCoarseSolve\n");
#endif
   if ( ! assembled_ ) coarseSolver_ = solver; 
   else                oneLevels_[coarsestLevel_]->setCoarseSolve(solver);
   return 0;
}

/*****************************************************************************
 * set finite element data information 
 *---------------------------------------------------------------------------*/

int MLI::setFEData( int level, MLI_FEData *fedata, MLI_Mapper *map )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setFEData\n");
#endif
   if ( level >= 0 && level < maxLevels_ )
   {
      oneLevels_[level]->setFEData( fedata, map );
   }
   else
   {
      printf("MLI::setFEData ERROR : wrong level = %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set finite element data information 
 *---------------------------------------------------------------------------*/

int MLI::setSFEI( int level, MLI_SFEI *sfei )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setSFEI\n");
#endif
   if ( level >= 0 && level < maxLevels_ )
   {
      oneLevels_[level]->setSFEI( sfei );
   }
   else
   {
      printf("MLI::setSFEI ERROR : wrong level = %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set cycle type at various levels
 *---------------------------------------------------------------------------*/

int MLI::setCyclesAtLevel( int level, int cycles )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setCyclesAtLevel at level %d, cycles = %d\n",level,cycles);
#endif
   if ( level >= 0 && level < maxLevels_ )
   {
      oneLevels_[level]->setCycles( cycles );
   }
   else if ( level == -1 )
   {
      for (int i = 0; i < maxLevels_; i++) oneLevels_[i]->setCycles( cycles );
   }
   else
   {
      printf("MLI::setCyclesAtLevel ERROR : wrong level = %d\n",level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set ML method 
 *---------------------------------------------------------------------------*/

int MLI::setMethod( MLI_Method *object )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setMethod = %s\n", object->getName());
#endif
   methodPtr_ = object;
   return 0;
}

/*****************************************************************************
 * set up the grid hierarchy
 *---------------------------------------------------------------------------*/

int MLI::setup()
{
   int  nlevels, status=0;
   char paramString[100];

   currIter_  = 0;
   buildTime_ = MLI_Utils_WTime();
   sprintf( paramString, "setOutputLevel %d", outputLevel_ );
   methodPtr_->setParams(paramString, 0, NULL);
   nlevels        = methodPtr_->setup(this);
   coarsestLevel_ = nlevels - 1;
   buildTime_ = MLI_Utils_WTime() - buildTime_;
   for (int i = 0; i < nlevels; i++) status += oneLevels_[i]->setup();
   if ( coarseSolver_ != NULL ) 
   {
      oneLevels_[coarsestLevel_]->setCoarseSolve(coarseSolver_); 
      coarseSolver_ = NULL;
   }
   assembled_ = 1;
   return status;
}

/*****************************************************************************
 * perform one cycle
 *---------------------------------------------------------------------------*/

int MLI::cycle( MLI_Vector *sol, MLI_Vector *rhs )
{
   oneLevels_[0]->setSolutionVector( sol );
   oneLevels_[0]->setRHSVector( rhs );
   int status = oneLevels_[0]->solve1Cycle();
   return status;
}

/*****************************************************************************
 * perform solve 
 *---------------------------------------------------------------------------*/

int MLI::solve( MLI_Vector *sol, MLI_Vector *rhs )
{
   int        iter=0, mypid;
   double     norm2, relTol, oldNorm2, zero=0.0;
   MLI_Matrix *Amat;
   MLI_Vector *res;
#if 0
   char       paramString[30];
   MLI_Solver *preSmoother;
#endif

   /*-------------------------------------------------------------------*/
   /* check for error                                                   */
   /*-------------------------------------------------------------------*/

   if ( ! assembled_ )
   {
      printf("MLI::solve ERROR - setup not called yet.\n");
      exit(1);
   }

   /*-------------------------------------------------------------------*/
   /* if coarse solver was set before setup, put it in the coarse level */
   /*-------------------------------------------------------------------*/

   if ( coarseSolver_ != NULL ) 
   {
      oneLevels_[coarsestLevel_]->setCoarseSolve(coarseSolver_); 
      coarseSolver_ = NULL;
   }

   /*-------------------------------------------------------------------*/
   /* compute initial residual norm and convergence tolerance           */
   /*-------------------------------------------------------------------*/

   MPI_Comm_rank(mpiComm_, &mypid);
   res        = oneLevels_[0]->getResidualVector();
   Amat       = oneLevels_[0]->getAmat();
   solveTime_ = MLI_Utils_WTime();
   if ( maxIterations_ == 1 )
   {
      norm2   = 1.0;
      relTol = 0.1;
      sol->setConstantValue(zero);
#if 0
      strcpy( paramString, "zeroInitialGuess" );
      preSmoother = oneLevels_[0]->getPreSmoother();
      if (preSmoother != NULL) preSmoother->setParams(paramString, 0, NULL);
#endif
   }
   else
   {
      Amat->apply( -1.0, sol, 1.0, rhs, res );
      norm2   = res->norm2();
      relTol = tolerance_ * norm2;
      if ( outputLevel_ > 0 && currIter_ == 0 )
         printf("\tMLI Initial norm = %16.8e (%16.8e)\n", norm2, relTol);
   }

   while ( norm2 > relTol && iter < maxIterations_ ) 
   {
      iter++;
      currIter_++;
      cycle( sol, rhs );
      if ( maxIterations_ > 1 )
      {
         Amat->apply( -1.0, sol, 1.0, rhs, res );
         oldNorm2 = norm2;
         norm2 = res->norm2();
         if ( outputLevel_ > 0 && mypid == 0 && maxIterations_ > 1 )
            printf("\tMLI iteration = %5d, rnorm = %14.6e (%14.6e)\n",
                   currIter_, norm2, norm2/oldNorm2);
      }
      if ( iter < maxIterations_ )
      {
         oneLevels_[0]->resetSolutionVector();
         oneLevels_[0]->resetRHSVector();
      }
   }
   solveTime_ = MLI_Utils_WTime() - solveTime_;
   if ( norm2 > tolerance_ || iter >= maxIterations_ ) return 1;
   else                                                return 0;
}

/*****************************************************************************
 * print 
 *---------------------------------------------------------------------------*/

int MLI::print()
{
   int mypid;
   MPI_Comm_rank(mpiComm_, &mypid);
   if ( mypid == 0 )
   {
      printf("\t***************** MLI Information *********************\n");
      printf("\t*** maxLevels         = %d\n", maxLevels_);
      printf("\t*** output level      = %d\n", outputLevel_);
      printf("\t*** max iterations    = %d\n", maxIterations_);
      printf("\t*** tolerance         = %e\n", tolerance_);
      printf("\t*******************************************************\n");
   }
   return 0;
}

/*****************************************************************************
 * output timing information 
 *---------------------------------------------------------------------------*/

int MLI::printTiming()
{
   int mypid;

   MPI_Comm_rank( mpiComm_, &mypid );
   if ( mypid == 0 )
   {
      printf("\t***************** MLI Timing Information **************\n");
      printf("\t*** MLI Build time = %e seconds\n", buildTime_);
      printf("\t*** MLI Solve time = %e seconds\n", solveTime_);
      printf("\t*******************************************************\n");
   }
   return 0;
}

/*****************************************************************************
 * get oneLevel object 
 *---------------------------------------------------------------------------*/

MLI_OneLevel* MLI::getOneLevelObject( int level )
{
   if ( level >= 0 && level < maxLevels_ ) return oneLevels_[level];
   else
   {
      printf("MLI::getOneLevelObject ERROR : wrong level = %d\n", level);
      return NULL;
   }
}

/*****************************************************************************
 * get system matrix 
 *---------------------------------------------------------------------------*/

MLI_Matrix* MLI::getSystemMatrix( int level )
{
   if ( level >= 0 && level < maxLevels_ ) 
      return oneLevels_[level]->getAmat();
   else
   {
      printf("MLI::getSystemMatrix ERROR : wrong level = %d\n", level);
      return NULL;
   }
}

/*****************************************************************************
 * get prolongation operator 
 *---------------------------------------------------------------------------*/

MLI_Matrix* MLI::getProlongation( int level )
{
   if ( level >= 0 && level < maxLevels_ ) 
      return oneLevels_[level]->getPmat();
   else
   {
      printf("MLI::getProlongation ERROR : wrong level = %d\n", level);
      return NULL;
   }
}

/*****************************************************************************
 * get restriction operator 
 *---------------------------------------------------------------------------*/

MLI_Matrix* MLI::getRestriction( int level )
{
   if ( level >= 0 && level < maxLevels_ ) 
      return oneLevels_[level]->getRmat();
   else
   {
      printf("MLI::getRestriction ERROR : wrong level = %d\n", level);
      return NULL;
   }
}

/*****************************************************************************
 * get smoother
 *---------------------------------------------------------------------------*/

MLI_Solver* MLI::getSmoother( int level, int pre_post )
{
   if ( level >= 0 && level < maxLevels_ ) 
   {
      if ( pre_post == MLI_SMOOTHER_PRE ) 
         return oneLevels_[level]->getPreSmoother();
      else if ( pre_post == MLI_SMOOTHER_POST ) 
         return oneLevels_[level]->getPostSmoother();
      else 
      {
         printf("MLI::getSmoother ERROR : pre or post ? \n");
         return ((MLI_Solver *) NULL);
      }
   }
   else
   {
      printf("MLI::getRestriction ERROR : wrong level = %d\n", level);
      return ((MLI_Solver *) NULL);
   }
}

/*****************************************************************************
 * get fedata
 *---------------------------------------------------------------------------*/

MLI_FEData* MLI::getFEData( int level )
{
   if ( level >= 0 && level < maxLevels_ )
      return oneLevels_[level]->getFEData();
   else
   {
      printf("MLI::getFEData ERROR : wrong level = %d\n", level);
      return ((MLI_FEData *) NULL);
   }
}

/*****************************************************************************
 * get sfei
 *---------------------------------------------------------------------------*/

MLI_SFEI* MLI::getSFEI( int level )
{
   if ( level >= 0 && level < maxLevels_ )
      return oneLevels_[level]->getSFEI();
   else
   {
      printf("MLI::getSFEI ERROR : wrong level = %d\n", level);
      return ((MLI_SFEI *) NULL);
   }
}

/*****************************************************************************
 * get node to equation map
 *---------------------------------------------------------------------------*/

MLI_Mapper* MLI::getNodeEqnMap( int level )
{
   if ( level >= 0 && level < maxLevels_ )
      return oneLevels_[level]->getNodeEqnMap();
   else
   {
      printf("MLI::getNodeEqnMap ERROR : wrong level = %d\n", level);
      return ((MLI_Mapper *) NULL);
   }
}

/*****************************************************************************
 * reset discretization matrix
 *---------------------------------------------------------------------------*/

int MLI::resetSystemMatrix( int level  )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::resetSystemMatrix, level = %d\n", level);
#endif
   if ( level >= 0 && level < maxLevels_ ) oneLevels_[level]->resetAmat();
   else
   {
      printf("MLI::resetSystemMatrix ERROR : wrong level = %d\n", level);
      exit(1);
   }
   return 0;
}

