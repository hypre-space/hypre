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

#include "mli.h"

/*****************************************************************************
 * constructor 
 *---------------------------------------------------------------------------*/

MLI *MLI_Create( MPI_Comm mpi_comm, int nlevels )
{
   int i, actual_nlevels;
   MLI *mli;

   mli = (MLI *) calloc( MLI, 1 );
   mli->assembled  = MLI_FALSE;
   mli->method     = MLI_NONE;
   if ( nlevels <= 0 ) mli->max_levels = 40;
   else                mli->max_levels = nlevels;
   mli->coarsest_level = -1;
   mli->mpi_comm       = mpi_comm;
   mli->debug_level    = 0;
   mli->tolerance      = 1.0e-12;
   mli->max_iterations = 1;
   actual_nlevels = mli->max_levels;
   mli->one_levels = (MLI_OneLevel *) calloc((MLI_OneLevel *), actual_nlevels);
   for ( i = 0; i < actual_nlevels; i++ )
   {
      mli->one_levels[i] = MLI_OneLevelCreate( mli );
      if ( i > 0 )                  MLI_OneLevelSetPrevLevel(mli->one_levels[i-1]);
      if ( i < (actual_nlevels-1) ) MLI_OneLevelSetNextLevel(mli->one_levels[i+1]);
   }
   mli->timing.solve_time = 0.0;
   mli->timing.build_time = 0.0;
   return mli;
} 

/*****************************************************************************
 * destructor 
 *---------------------------------------------------------------------------*/

int MLI_Destroy( MLI *mli )
{
   for (i = 0; i < mli->max_levels; i++) MLI_OneLevelDestroy(mli->one_levels[i]);
   free( mli->one_levels );
   return 0;
}

/*****************************************************************************
 * set diagnostics level 
 *---------------------------------------------------------------------------*/

int MLI_SetDebugLevel( MLI *mli, int debug_level )
{
   mli->debug_level = debug_level;
   return 0;
}

/*****************************************************************************
 * set convergence tolerance 
 *---------------------------------------------------------------------------*/

int MLI_SetTolerance( MLI *mli, double tolerance )
{
   mli->tolerance = tolerance;
   return 0;
}

/*****************************************************************************
 * set maximum iterations count
 *---------------------------------------------------------------------------*/

int MLI_SetMaxIterations( MLI *mli, int iterations )
{
   mli->max_iterations = iterations;
   return 0;
}

/*****************************************************************************
 * set maximum iterations count
 *---------------------------------------------------------------------------*/

int MLI_PrintTiming( MLI *mli )
{
   int mypid;

   MPI_Comm_rank( mli->mpi_comm, &mypid );
   if ( mypid == 0 )
   {
      printf("MLI Build time = %8.3e seconds.\n", timing->build_time);
      printf("MLI Solve time = %8.3e seconds.\n", timing->solve_time);
   }
   return 0;
}

/*****************************************************************************
 * set discretization matrix
 *---------------------------------------------------------------------------*/

int MLI_SetAmat( MLI *mli, int level, MLI_Matrix *Amat)
{
   if ( level >= 0 && level < mli->max_levels )
   {
      MLI_OneLevelSetAmat( mli->one_levels[level], Amat );
   }
   else
   {
      printf("MLI_SetAmat ERROR : wrong level - %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set restriction operator
 *---------------------------------------------------------------------------*/

int MLI_SetRmat( MLI *mli, int level, MLI_Matrix *Rmat)
{
   if ( level >= 0 && level < mli->max_levels )
   {
      MLI_OneLevelSetRmat( mli->one_levels[level], Rmat );
   }
   else
   {
      printf("MLI_SetRmat ERROR : wrong level - %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set prolongation operator
 *---------------------------------------------------------------------------*/

int MLI_SetPmat( MLI *mli, int level, MLI_Matrix *Pmat)
{
   if ( level >= 0 && level < mli->max_levels )
   {
      MLI_OneLevelSetPmat( mli->one_levels[level], Pmat );
   }
   else
   {
      printf("MLI_SetPmat ERROR : wrong level - %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set smoother 
 *---------------------------------------------------------------------------*/

int MLI_SetSmoother( MLI *mli, int level , int pre_post, MLI_Solver *smoother )
{
   if ( level >= 0 && level < mli->max_levels )
   {
      MLI_OneLevelSetSmoother( mli->one_levels[level], pre_post, smoother );
   }
   else
   {
      printf("MLI_SetSmoother ERROR : wrong level - %d\n", level);
      exit(1);
   }
   return 0;
}

/*****************************************************************************
 * set smoother 
 *---------------------------------------------------------------------------*/

int MLI_SetCoarseSolver( MLI *mli, MLI_Solver *csolver )
{
   mli->coarse_solver = csolver;
   return 0;
}

/*****************************************************************************
 * set ML method 
 *---------------------------------------------------------------------------*/

int MLI_SetMethod( MLI *mli, int method, MLI_Method *method_data )
{
   switch ( method )
   {
      case MLI_AGGRAMG : mli->method = MLI_AGGRAMG;
                         mli->method_data = method_data;
                         break;
      default :          printf("MLI_SetMethod : method not supported.\n");
                         break;
   }
   return 0;
}

/*****************************************************************************
 * set V cycle or W cycle or F-cycle
 *---------------------------------------------------------------------------*/

int MLI_SetCycleType( MLI *mli, int cycle_type )
{
   switch ( cycle_type )
   {
      case MLI_VCYCLE : mli->method = MLI_VCYCLE;
                        break;
      case MLI_WCYCLE : mli->method = MLI_WCYCLE;
                        break;
      default :         printf("MLI_SetCycleType : type not supported.\n");
                        break;
   }
   return 0;
}

/*****************************************************************************
 * set up the grid hierarchy
 *---------------------------------------------------------------------------*/

int MLI_Setup( MLI *mli )
{
   switch ( mli->method )
   {
      case MLI_AGGRAMG : MLI_AggrAMGGenMLStructure( mli->method_data, mli );
                         break;
      default :          printf("MLI_Setup : method not supported.\n");
                         exit(1);
                         break;
   }
   return 0;
}

/*****************************************************************************
 * iterate until convergence
 *---------------------------------------------------------------------------*/

int MLI_Iterate( MLI *mli, MLI_Vector *sol, MLI_Vector *rhs )
{
   return 0;
}

