/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * functions for the MLI_Solver
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <string.h>
#include <strings.h>
#include "base/mli_defs.h"
#include "solver/mli_solver.h"
#include "solver/mli_solver_jacobi.h"
#include "solver/mli_solver_gs.h"
#include "solver/mli_solver_sgs.h"
#include "solver/mli_solver_parasails.h"
#include "solver/mli_solver_bsgs.h"
#include "solver/mli_solver_mls.h"
#include "solver/mli_solver_superlu.h"
#include "solver/mli_solver_arpacksuperlu.h"

/*****************************************************************************
 * constructor 
 *--------------------------------------------------------------------------*/

MLI_Solver::MLI_Solver( int sid )
{
   switch ( sid )
   {
      case MLI_SOLVER_JACOBI_ID :
           strcpy( solver_name, "Jacobi" );
           solver_id  = MLI_SOLVER_JACOBI_ID;
           break;
      case MLI_SOLVER_GS_ID :
           strcpy( solver_name, "GS" );
           solver_id  = MLI_SOLVER_GS_ID;
           break;
      case MLI_SOLVER_SGS_ID :
           strcpy( solver_name, "SGS" );
           solver_id  = MLI_SOLVER_SGS_ID;
           break;
      case MLI_SOLVER_PARASAILS_ID :
#ifdef MLI_PARASAILS
           strcpy( solver_name, "ParaSails" );
           solver_id  = MLI_SOLVER_PARASAILS_ID;
#else
           printf("MLI_Solver::constructor ERROR - ParaSails not available\n");
           exit(1);
#endif
           break;
      case MLI_SOLVER_BSGS_ID :
           strcpy( solver_name, "BSGS" );
           solver_id  = MLI_SOLVER_BSGS_ID;
           break;
      case MLI_SOLVER_MLS_ID :
           strcpy( solver_name, "MLS" );
           solver_id  = MLI_SOLVER_MLS_ID;
           break;
      case MLI_SOLVER_SUPERLU_ID :
#ifdef MLI_SUPERLU
           strcpy( solver_name, "SuperLU" );
           solver_id  = MLI_SOLVER_SUPERLU_ID;
#else
           printf("MLI_Solver::constructor ERROR - SuperLU not available\n");
           exit(1);
#endif
           break;
      case MLI_SOLVER_ARPACKSUPERLU_ID :
#ifdef MLI_SUPERLU
           strcpy( solver_name, "ARPACKSuperLU" );
           solver_id  = MLI_SOLVER_ARPACKSUPERLU_ID;
#else
           printf("MLI_Solver::constructor ERROR - SuperLU not available\n");
           exit(1);
#endif
           break;
      default :
           printf("MLI_Solver::constructor ERROR - invalid solver.\n");
           printf("Valid ones are : Jacobi, GS, SGS, ParaSails, \n");
           printf("BSGS, MLS, SuperLU, ARPACKSuperLU.\n");
           fflush(stdout);
           exit(1);
   }
}

/*****************************************************************************
 * another constructor 
 *--------------------------------------------------------------------------*/

MLI_Solver::MLI_Solver( char *str )
{
   if ( !strcasecmp(str, "Jacobi" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_JACOBI_ID;
   }
   else if ( !strcasecmp(str, "GS" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_GS_ID;
   }
   else if ( !strcasecmp(str, "SGS" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_SGS_ID;
   }
   else if ( !strcasecmp(str, "ParaSails" ) )
   {
#ifdef MLI_PARASAILS
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_PARASAILS_ID;
#else
      printf("MLI_Solver::constructor ERROR - SuperLU not available\n");
      exit(1);
#endif
   }
   else if ( !strcasecmp(str, "BSGS" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_BSGS_ID;
   }
   else if ( !strcasecmp(str, "MLS" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_MLS_ID;
   }
   else if ( !strcasecmp(str, "SuperLU" ) )
   {
#ifdef MLI_SUPERLU
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_SUPERLU_ID;
#else
      printf("MLI_Solver::constructor ERROR - SuperLU not available\n");
      exit(1);
#endif
   }
   else if ( !strcasecmp(str, "ARPACKSuperLU" ) )
   {
#ifdef MLI_SUPERLU
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_ARPACKSUPERLU_ID;
#else
      printf("MLI_Solver::constructor ERROR - SuperLU not available\n");
      exit(1);
#endif
   }
   else
   {
      printf("MLI_Solver::constructor ERROR - solver %s undefined\n",str);
      exit(1);
   }
}

/*****************************************************************************
 * create a solver 
 *--------------------------------------------------------------------------*/

MLI_Solver *MLI_Solver_CreateFromName( char *str )
{
   MLI_Solver *solver_ptr=NULL;

   if      (!strcasecmp(str, "Jacobi")) solver_ptr = new MLI_Solver_Jacobi();
   else if (!strcasecmp(str, "GS"))     solver_ptr = new MLI_Solver_GS();
   else if (!strcasecmp(str, "SGS"))    solver_ptr = new MLI_Solver_SGS();
   else if (!strcasecmp(str, "ParaSails")) 
   {
#ifdef MLI_PARASAILS
      solver_ptr = new MLI_Solver_ParaSails();
#else
      printf("MLI_Solver_Create ERROR : ParaSails not available\n");
      exit(1);
#endif
   }
   else if (!strcasecmp(str, "BSGS")) solver_ptr = new MLI_Solver_BSGS();
   else if (!strcasecmp(str, "MLS"))     solver_ptr = new MLI_Solver_MLS();
   else if (!strcasecmp(str, "SuperLU"))   
   {
#ifdef MLI_SUPERLU
      solver_ptr = new MLI_Solver_SuperLU();
#else
      printf("MLI_Solver_Create ERROR : SuperLU not available\n");
      exit(1);
#endif
   }
   else if (!strcasecmp(str, "ARPACKSuperLU"))   
   {
#ifdef MLI_SUPERLU
      solver_ptr = new MLI_Solver_ARPACKSuperLU();
#else
      printf("MLI_Solver_Create ERROR : SuperLU not available\n");
      exit(1);
#endif
   }
   else
   {
      printf("MLI_Solver_Create ERROR : solver %s undefined.\n",str);
      fflush(stdout);
      exit(1);
   }
   return solver_ptr;
}

/*****************************************************************************
 * create a solver 
 *--------------------------------------------------------------------------*/

MLI_Solver *MLI_Solver_CreateFromID( int solver_id )
{
   MLI_Solver *solver_ptr=NULL;

   switch ( solver_id )
   {
      case MLI_SOLVER_JACOBI_ID :
           solver_ptr = new MLI_Solver_Jacobi();
           break;
      case MLI_SOLVER_GS_ID :
           solver_ptr = new MLI_Solver_GS();
           break;
      case MLI_SOLVER_SGS_ID :
           solver_ptr = new MLI_Solver_SGS();
           break;
      case MLI_SOLVER_PARASAILS_ID :
#ifdef MLI_PARASAILS
           solver_ptr = new MLI_Solver_ParaSails();
#else
           printf("MLI_Solver_Create ERROR : ParaSails not available\n");
           exit(1);
#endif
           break;
      case MLI_SOLVER_BSGS_ID :
           solver_ptr = new MLI_Solver_BSGS();
           break;
      case MLI_SOLVER_MLS_ID :
           solver_ptr = new MLI_Solver_MLS();
           break;
      case MLI_SOLVER_SUPERLU_ID :
#ifdef MLI_SUPERLU
           solver_ptr = new MLI_Solver_SuperLU();
#else
           printf("MLI_Solver_Create ERROR : SuperLU not available\n");
           exit(1);
#endif
           break;
      case MLI_SOLVER_ARPACKSUPERLU_ID :
#ifdef MLI_SUPERLU
           solver_ptr = new MLI_Solver_ARPACKSuperLU();
#else
           printf("MLI_Solver_Create ERROR : SuperLU not available\n");
           exit(1);
#endif
           break;
      default :
           printf("MLI_Solver_Create ERROR : invalid solver.\n");
           printf("Valid ones are : \n");
           printf("\t %5d (Jacobi)       \n", MLI_SOLVER_JACOBI_ID);
           printf("\t %5d (GS)           \n", MLI_SOLVER_GS_ID);
           printf("\t %5d (SGS)          \n", MLI_SOLVER_SGS_ID);
           printf("\t %5d (ParaSails)    \n", MLI_SOLVER_PARASAILS_ID);
           printf("\t %5d (BSGS)         \n", MLI_SOLVER_BSGS_ID);
           printf("\t %5d (MLS)          \n", MLI_SOLVER_MLS_ID);
           printf("\t %5d (SuperLU)      \n", MLI_SOLVER_SUPERLU_ID);
           printf("\t %5d (ARPACKSuperLU)\n", MLI_SOLVER_ARPACKSUPERLU_ID); 
           fflush(stdout);
           exit(1);
   }
   return solver_ptr;
}

