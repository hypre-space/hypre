/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * functions for the top level MLI data structure
 *
 *****************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "base/mli.h"
#include "util/mli_utils.h"

/*****************************************************************************
 * constructor 
 *---------------------------------------------------------------------------*/

MLI::MLI( MPI_Comm comm )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::MLI\n");
#endif
   mpi_comm       = comm;
   max_levels     = 40;
   num_levels     = 40;
   coarsest_level = 0;
   output_level   = 0;
   assembled      = MLI_FALSE;
   tolerance      = 1.0e-6;
   max_iterations = 20;
   curr_iter      = 0;
   one_levels     = new MLI_OneLevel*[max_levels];
   for (int j = 0; j < max_levels; j++) one_levels[j] = new MLI_OneLevel(this);
   for (int i = 0; i < max_levels; i++)
   {
      one_levels[i]->setLevelNum(i);
      if ( i < (max_levels-1) ) one_levels[i]->setNextLevel(one_levels[i+1]);
      if ( i > 0 )              one_levels[i]->setPrevLevel(one_levels[i-1]);
   }
   coarse_solver = NULL;
   method_ptr    = NULL;
   solve_time    = 0.0;
   build_time    = 0.0;
} 

/*****************************************************************************
 * destructor 
 *---------------------------------------------------------------------------*/

MLI::~MLI()
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::~MLI\n");
#endif
   for ( int i = 0; i < max_levels; i++ ) delete one_levels[i];
   delete [] one_levels;
   if ( coarse_solver != NULL ) delete coarse_solver;
   if ( method_ptr    != NULL ) delete method_ptr;
}

/*****************************************************************************
 * set discretization matrix
 *---------------------------------------------------------------------------*/

int MLI::setSystemMatrix( int level, MLI_Matrix *A )
{
#ifdef MLI_DEBUG_DETAILED
   printf("MLI::setSystemMatrix, level = %d\n", level);
#endif
   if ( level >= 0 && level < max_levels ) one_levels[level]->setAmat( A );
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
   if ( level >= 0 && level < max_levels ) one_levels[level]->setRmat( R );
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
   if ( level >= 0 && level < max_levels ) one_levels[level]->setPmat( P );
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
   if ( level >= 0 && level < max_levels )
   {
      one_levels[level]->setSmoother( pre_post, smoother );
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
   if ( ! assembled ) coarse_solver = solver; 
   else               one_levels[coarsest_level]->setCoarseSolve(solver);
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
   if ( level >= 0 && level < max_levels )
   {
      one_levels[level]->setFEData( fedata, map );
   }
   else
   {
      printf("MLI::setFEData ERROR : wrong level = %d\n", level);
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
   if ( level >= 0 && level < max_levels )
   {
      one_levels[level]->setCycles( cycles );
   }
   else if ( level == -1 )
   {
      for (int i = 0; i < max_levels; i++) one_levels[i]->setCycles( cycles );
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
   method_ptr = object;
   return 0;
}

/*****************************************************************************
 * set up the grid hierarchy
 *---------------------------------------------------------------------------*/

int MLI::setup()
{
   int  nlevels, status=0;
   char param_string[100];

   curr_iter      = 0;
   build_time     = MLI_Utils_WTime();
   sprintf( param_string, "setOutputLevel %d", output_level );
   method_ptr->setParams(param_string, 0, NULL);
   nlevels        = method_ptr->setup(this);
   coarsest_level = nlevels - 1;
   build_time = MLI_Utils_WTime() - build_time;
   for (int i = 0; i < nlevels; i++) status += one_levels[i]->setup();
   if ( coarse_solver != NULL ) 
   {
      one_levels[coarsest_level]->setCoarseSolve(coarse_solver); 
      coarse_solver = NULL;
   }
   assembled = 1;
   return status;
}

/*****************************************************************************
 * perform one cycle
 *---------------------------------------------------------------------------*/

int MLI::cycle( MLI_Vector *sol, MLI_Vector *rhs )
{
   one_levels[0]->setSolutionVector( sol );
   one_levels[0]->setRHSVector( rhs );
   int status = one_levels[0]->solve1Cycle();
   return status;
}

/*****************************************************************************
 * perform solve 
 *---------------------------------------------------------------------------*/

int MLI::solve( MLI_Vector *sol, MLI_Vector *rhs )
{
   int        iter=0, mypid;
   double     norm2, rel_tol, old_norm2, zero=0.0;
   MLI_Matrix *Amat;
   MLI_Vector *res;
#if 0
   char       paramString[30];
   MLI_Solver *preSmoother;
#endif

   /*-------------------------------------------------------------------*/
   /* check for error                                                   */
   /*-------------------------------------------------------------------*/

   if ( ! assembled )
   {
      printf("MLI::solve ERROR - setup not called yet.\n");
      exit(1);
   }

   /*-------------------------------------------------------------------*/
   /* if coarse solver was set before setup, put it in the coarse level */
   /*-------------------------------------------------------------------*/

   if ( coarse_solver != NULL ) 
   {
      one_levels[coarsest_level]->setCoarseSolve(coarse_solver); 
      coarse_solver = NULL;
   }

   /*-------------------------------------------------------------------*/
   /* compute initial residual norm and convergence tolerance           */
   /*-------------------------------------------------------------------*/

   MPI_Comm_rank(mpi_comm, &mypid);
   res        = one_levels[0]->getResidualVector();
   Amat       = one_levels[0]->getAmat();
   solve_time = MLI_Utils_WTime();
   if ( max_iterations == 1 )
   {
      norm2   = 1.0;
      rel_tol = 0.1;
      sol->setConstantValue(zero);
#if 0
      strcpy( paramString, "zeroInitialGuess" );
      preSmoother = one_levels[0]->getPreSmoother();
      if (preSmoother != NULL) preSmoother->setParams(paramString, 0, NULL);
#endif
   }
   else
   {
      Amat->apply( -1.0, sol, 1.0, rhs, res );
      norm2   = res->norm2();
      rel_tol = tolerance * norm2;
      if ( output_level > 0 && curr_iter == 0 )
         printf("\tMLI Initial norm = %16.8e (%16.8e)\n", norm2, rel_tol);
   }

   while ( norm2 > rel_tol && iter < max_iterations ) 
   {
      iter++;
      curr_iter++;
      cycle( sol, rhs );
      if ( max_iterations > 1 )
      {
         Amat->apply( -1.0, sol, 1.0, rhs, res );
         old_norm2 = norm2;
         norm2 = res->norm2();
         if ( output_level > 0 && mypid == 0 && max_iterations > 1 )
            printf("\tMLI iteration = %5d, rnorm = %14.6e (%14.6e)\n",
                   curr_iter, norm2, norm2/old_norm2);
      }
      if ( iter < max_iterations )
      {
         one_levels[0]->resetSolutionVector();
         one_levels[0]->resetRHSVector();
      }
   }
   solve_time = MLI_Utils_WTime() - solve_time;
   if ( norm2 > tolerance || iter >= max_iterations ) return 1;
   else                                               return 0;
}

/*****************************************************************************
 * print 
 *---------------------------------------------------------------------------*/

int MLI::print()
{
   int mypid;
   MPI_Comm_rank(mpi_comm, &mypid);
   if ( mypid == 0 )
   {
      printf("\t***************** MLI Information *********************\n");
      printf("\t*** max_levels        = %d\n", max_levels);
      printf("\t*** output level      = %d\n", output_level);
      printf("\t*** max_iterations    = %d\n", max_iterations);
      printf("\t*** tolerance         = %e\n", tolerance);
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

   MPI_Comm_rank( mpi_comm, &mypid );
   if ( mypid == 0 )
   {
      printf("\t***************** MLI Timing Information **************\n");
      printf("\t*** MLI Build time = %e seconds\n", build_time);
      printf("\t*** MLI Solve time = %e seconds\n", solve_time);
      printf("\t*******************************************************\n");
   }
   return 0;
}

/*****************************************************************************
 * get oneLevel object 
 *---------------------------------------------------------------------------*/

MLI_OneLevel* MLI::getOneLevelObject( int level )
{
   if ( level >= 0 && level < max_levels ) return one_levels[level];
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
   if ( level >= 0 && level < max_levels ) 
      return one_levels[level]->getAmat();
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
   if ( level >= 0 && level < max_levels ) 
      return one_levels[level]->getPmat();
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
   if ( level >= 0 && level < max_levels ) 
      return one_levels[level]->getRmat();
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
   if ( level >= 0 && level < max_levels ) 
   {
      if ( pre_post == MLI_SMOOTHER_PRE ) 
         return one_levels[level]->getPreSmoother();
      else if ( pre_post == MLI_SMOOTHER_POST ) 
         return one_levels[level]->getPostSmoother();
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
   if ( level >= 0 && level < max_levels )
      return one_levels[level]->getFEData();
   else
   {
      printf("MLI::getFEData ERROR : wrong level = %d\n", level);
      return ((MLI_FEData *) NULL);
   }
}

/*****************************************************************************
 * get node to equation map
 *---------------------------------------------------------------------------*/

MLI_Mapper* MLI::getNodeEqnMap( int level )
{
   if ( level >= 0 && level < max_levels )
      return one_levels[level]->getNodeEqnMap();
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
   if ( level >= 0 && level < max_levels ) one_levels[level]->resetAmat();
   else
   {
      printf("MLI::resetSystemMatrix ERROR : wrong level = %d\n", level);
      exit(1);
   }
   return 0;
}

