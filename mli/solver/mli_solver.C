/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <iostream.h>
#include "../base/mli_defs.h"
#include "mli_jacobi.h"
#include "mli_gs.h"
#include "mli_sgs.h"
#include "mli_parasails.h"
#include "mli_schwarz.h"
#include "mli_mls.h"
#include "mli_superlu.h"

/******************************************************************************
 * construct different smoothers
 *****************************************************************************/

MLI_Solver *MLI_Solver_Construct( int solver_id )
{
   MLI_Solver *solver;

   switch ( solver_id )
   {
      case MLI_SOLVER_JACOBI_ID :
           solver = new MLI_SolverJacobi();
           break;
      case MLI_SOLVER_GS_ID :
           solver = new MLI_SolverGS();
           break;
      case MLI_SOLVER_SGS_ID :
           solver = new MLI_SolverSGS();
           break;
      case MLI_SOLVER_PARASAILS_ID :
           solver = new MLI_SolverParaSails();
           break;
      case MLI_SOLVER_SCHWARZ_ID :
           solver = new MLI_SolverSchwarz();
           break;
      case MLI_SOLVER_MLS_ID :
           solver = new MLI_SolverMLS();
           break;
#ifdef SUPERLU
      case MLI_SOLVER_SUPERLU_ID :
           solver = new MLI_SolverSuperLU();
           break;
#endif
      default :
           cout << "ERROR : Smoother not recognized = " << solver_id << "\n";
   }
   return solver;
}

