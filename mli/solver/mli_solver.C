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
#include <iostream.h>
#include "../base/mli_defs.h"
#include "mli_solver.h"
#include "mli_solver_jacobi.h"
#include "mli_solver_gs.h"
#include "mli_solver_sgs.h"
#include "mli_solver_parasails.h"
#include "mli_solver_schwarz.h"
#include "mli_solver_mls.h"
#include "mli_solver_superlu.h"

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
#ifdef MLI_PARASAILS
      case MLI_SOLVER_PARASAILS_ID :
           strcpy( solver_name, "ParaSails" );
           solver_id  = MLI_SOLVER_PARASAILS_ID;
           break;
#endif
      case MLI_SOLVER_SCHWARZ_ID :
           strcpy( solver_name, "Schwarz" );
           solver_id  = MLI_SOLVER_SCHWARZ_ID;
           break;
      case MLI_SOLVER_MLS_ID :
           strcpy( solver_name, "MLS" );
           solver_id  = MLI_SOLVER_MLS_ID;
           break;
#ifdef MLI_SUPERLU
      case MLI_SOLVER_SUPERLU_ID :
           strcpy( solver_name, "SuperLU" );
           solver_id  = MLI_SOLVER_SUPERLU_ID;
           break;
#endif
      default :
           cout << "MLI_Solver constructor ERROR : invalid solver\n";
           cout << "    valid ones are : Jacobi, GS, SGS, ParaSails, \n";
           cout << "                     Schwarz, MLS, SuperLU. \n";
           cout.flush();
           exit(1);
   }
}

/*****************************************************************************
 * another constructor 
 *--------------------------------------------------------------------------*/

MLI_Solver::MLI_Solver( char *str )
{
   if ( !strcmp(str, "Jacobi" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_JACOBI_ID;
   }
   else if ( !strcmp(str, "GS" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_GS_ID;
   }
   else if ( !strcmp(str, "SGS" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_SGS_ID;
   }
#ifdef MLI_PARASAILS
   else if ( !strcmp(str, "ParaSails" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_PARASAILS_ID;
   }
#endif
   else if ( !strcmp(str, "Schwarz" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_SCHWARZ_ID;
   }
   else if ( !strcmp(str, "MLS" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_MLS_ID;
   }
#ifdef MLI_SUPERLU
   else if ( !strcmp(str, "SuperLU" ) )
   {
      strcpy( solver_name, str );
      solver_id  = MLI_SOLVER_SUPERLU_ID;
   }
#endif
   else
   {
      cout << "MLI_Solver constructor ERROR : solver " << str << " undefined\n";
      cout.flush();
      exit(1);
   }
}

/*****************************************************************************
 * create a solver 
 *--------------------------------------------------------------------------*/

MLI_Solver *MLI_Solver_CreateFromName( char *str )
{
   MLI_Solver *solver_ptr=NULL;

   if      (!strcmp(str, "Jacobi"))    solver_ptr = new MLI_Solver_Jacobi();
   else if (!strcmp(str, "GS"))        solver_ptr = new MLI_Solver_GS();
   else if (!strcmp(str, "SGS"))       solver_ptr = new MLI_Solver_SGS();
#ifdef MLI_PARASAILS
   else if (!strcmp(str, "ParaSails")) solver_ptr = new MLI_Solver_ParaSails();
#endif
   else if (!strcmp(str, "Schwarz"))   solver_ptr = new MLI_Solver_Schwarz();
   else if (!strcmp(str, "MLS"))       solver_ptr = new MLI_Solver_MLS();
#ifdef MLI_SUPERLU
   else if (!strcmp(str, "SuperLU"))   solver_ptr = new MLI_Solver_SuperLU();
#endif
   else
   {
      cout << "MLI_Solver_Create ERROR : solver " << str << " undefined\n";
      cout.flush();
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
#ifdef MLI_PARASAILS
      case MLI_SOLVER_PARASAILS_ID :
           solver_ptr = new MLI_Solver_ParaSails();
           break;
#endif
      case MLI_SOLVER_SCHWARZ_ID :
           solver_ptr = new MLI_Solver_Schwarz();
           break;
      case MLI_SOLVER_MLS_ID :
           solver_ptr = new MLI_Solver_MLS();
           break;
#ifdef MLI_SUPERLU
      case MLI_SOLVER_SUPERLU_ID :
           solver_ptr = new MLI_Solver_SuperLU();
           break;
#endif
      default :
           cout << "MLI_Solver_Create ERROR : invalid solver\n";
           cout << "  valid ones are : \n";
           cout << "         " << MLI_SOLVER_JACOBI_ID    << " (Jacobi)\n";
           cout << "         " << MLI_SOLVER_GS_ID        << " (GS)\n";
           cout << "         " << MLI_SOLVER_SGS_ID       << " (SGS)\n";
           cout << "         " << MLI_SOLVER_PARASAILS_ID << " (ParaSails)\n";
           cout << "         " << MLI_SOLVER_SCHWARZ_ID   << " (Schwarz)\n";
           cout << "         " << MLI_SOLVER_MLS_ID       << " (MLS)\n";
           cout << "         " << MLI_SOLVER_SUPERLU_ID   << " (Jacobi)\n";
           cout.flush();
           exit(1);
   }
   return solver_ptr;
}

