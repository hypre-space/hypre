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
#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include "mli.h"
#include "../util/mli_utils.h"

/*****************************************************************************
 * constructor 
 *---------------------------------------------------------------------------*/

MLI::MLI( MPI_Comm comm )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI::MLI" << endl;
   cout.flush();
#endif
   mpi_comm       = comm;
   max_levels     = 40;
   num_levels     = 40;
   coarsest_level = 0;
   output_level   = 1;
   method         = MLI_NONE;
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
   method_data   = NULL;
   solve_time    = 0.0;
   build_time    = 0.0;
} 

/*****************************************************************************
 * destructor 
 *---------------------------------------------------------------------------*/

MLI::~MLI()
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI::~MLI" << endl;
   cout.flush();
#endif
   for ( int i = 0; i < max_levels; i++ ) delete one_levels[i];
   delete [] one_levels;
   if ( coarse_solver != NULL ) delete coarse_solver;
   if ( method_data   != NULL ) delete method_data;
}

/*****************************************************************************
 * set discretization matrix
 *---------------------------------------------------------------------------*/

int MLI::setSystemMatrix( int level, MLI_Matrix *A )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI::setSystemMatrix, level = " << level << endl;
   cout.flush();
#endif
   if ( level >= 0 && level < max_levels ) one_levels[level]->setAmat( A );
   else
   {
      cout << "MLI::setSystemMatrix ERROR : wrong level = " << level << endl;
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
   cout << "MLI::setRestriction, level = " << level << endl;
   cout.flush();
#endif
   if ( level >= 0 && level < max_levels ) one_levels[level]->setRmat( R );
   else
   {
      cout << "MLI::setRestriction ERROR : wrong level = " << level << endl;
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
   cout << "MLI::setProlongation, level = " << level << endl;
   cout.flush();
#endif
   if ( level >= 0 && level < max_levels ) one_levels[level]->setPmat( P );
   else
   {
      cout << "MLI::setProlongation ERROR : wrong level = " << level << endl;
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
   cout << "MLI::setSmoother, level = " << level << endl;
   cout.flush();
#endif
   if ( level >= 0 && level < max_levels )
   {
      one_levels[level]->setSmoother( pre_post, smoother );
   }
   else
   {
      cout << "MLI::setSmoother ERROR : wrong level = " << level << endl;
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
   cout << "MLI::setCoarseSolve" << endl;
   cout.flush();
#endif
   coarse_solver = solver; 
   return 0;
}

/*****************************************************************************
 * set finite element data information 
 *---------------------------------------------------------------------------*/

int MLI::setFEData( int level, MLI_FEData *fedata )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI::setFEData" << endl;
   cout.flush();
#endif
   if ( level >= 0 && level < max_levels )
   {
      one_levels[level]->setFEData( fedata );
   }
   else
   {
      cout << "MLI::setFEData ERROR : wrong level = " << level << endl;
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
   cout << "MLI::setCyclesAtLevel at level " << level << " cycles = " 
        << cycles << endl;
   cout.flush();
#endif
   if ( level >= 0 && level < max_levels )
   {
      one_levels[level]->setCycles( cycles );
   }
   else
   {
      cout << "MLI::setCyclesAtLevel ERROR : wrong level = " << level << endl;
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
   cout << "MLI::setMethod = " << object->getName() << endl;
   cout.flush();
#endif
   if ( !strcmp( object->getName(), "MLI_AMGSA" ) )
   {
      method      = MLI_AMGSA_ID;
      method_data = object;
      return 0;
   }
   else if ( !strcmp( object->getName(), "MLI_AMGSA_CALIB" ) )
   {
      method      = MLI_AMGSA_CALIB_ID;
      method_data = object;
      return 0;
   }
   else
   {
      cout << "MLI::setMethod : method not supported." << endl;
      method      = -1; 
      method_data = NULL; 
      return 1;
   }
}

/*****************************************************************************
 * set up the grid hierarchy
 *---------------------------------------------------------------------------*/

int MLI::setup()
{
   int nlevels, status=1;

#ifdef MLI_DEBUG_DETAILED
   cout << "MLI::setup - method = " << method << endl;
   cout.flush();
#endif

   curr_iter  = 0;
   build_time = MLI_Utils_WTime();
   switch ( method )
   {
      case MLI_AMGSA_ID : 
           nlevels = method_data->setup(this);
           status  = 0;
           break;
      case MLI_AMGSA_CALIB_ID : 
           nlevels = method_data->setup(this);
           status  = 0;
           break;
      default :        
           cout << "MLI::setup : method not supported." << endl;
           cout.flush();
           status = 1;
           break;
   }
   build_time = MLI_Utils_WTime() - build_time;
   for (int i = 0; i < nlevels; i++) status += one_levels[i]->setup();
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
   int        iter=0;
   double     norm2, rel_tol, old_norm2;
   MLI_Matrix *Amat;
   MLI_Vector *res;

   res   = one_levels[0]->getResidualVector();
   Amat  = one_levels[0]->getAmat();
   Amat->apply( -1.0, sol, 1.0, rhs, res );
   norm2   = res->norm2();
   rel_tol = tolerance * norm2;
   solve_time = MLI_Utils_WTime();
   if ( output_level > 0 && curr_iter == 0 )
   {
      printf("\tMLI Initial norm = %16.8e (%16.8e)\n", norm2, rel_tol);
   }
   while ( norm2 > rel_tol && iter < max_iterations ) 
   {
      iter++;
      curr_iter++;
      cycle( sol, rhs );
      Amat->apply( -1.0, sol, 1.0, rhs, res );
      old_norm2 = norm2;
      norm2 = res->norm2();
      if ( output_level > 0 )
         printf("\tMLI iteration = %5d, rnorm = %14.6e (%14.6e)\n",curr_iter,
                norm2, norm2/old_norm2);
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
      cout << "\t***************** MLI Information *********************" << endl;
      cout << "\t*** max_levels        = " << max_levels << endl;
      cout << "\t*** output level      = " << output_level << endl;
      cout << "\t*** max_iterations    = " << max_iterations << endl;
      cout << "\t*** tolerance         = " << tolerance << endl;
      cout << "\t*******************************************************" << endl;
      cout.flush();
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
      cout << "\t***************** MLI Timing Information **************" << endl;
      cout << "\t*** MLI Build time = " << build_time << " seconds" << endl;
      cout << "\t*** MLI Solve time = " << solve_time << " seconds" << endl;
      cout << "\t*******************************************************" << endl;
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
      cout << "MLI::getOneLevelObject ERROR : wrong level = " << level << endl;
      return NULL;
   }
}

/*****************************************************************************
 * get matrix 
 *---------------------------------------------------------------------------*/

MLI_Matrix* MLI::getSystemMatrix( int level )
{
   if ( level >= 0 && level < max_levels ) 
      return one_levels[level]->getAmat();
   else
   {
      cout << "MLI::getMatrix ERROR : wrong level = " << level << endl;
      return NULL;
   }
}

/*****************************************************************************
 * reset discretization matrix
 *---------------------------------------------------------------------------*/

int MLI::resetSystemMatrix( int level  )
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI::resetSystemMatrix, level = " << level << endl;
   cout.flush();
#endif
   if ( level >= 0 && level < max_levels ) one_levels[level]->resetAmat();
   else
   {
      cout << "MLI::resetSystemMatrix ERROR : wrong level = " << level << endl;
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * reset ML method 
 *---------------------------------------------------------------------------*/

int MLI::resetMethod()
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI::resetMethod" << endl;
   cout.flush();
#endif
   method      = -1;
   method_data = NULL; 
   return 0;
}

