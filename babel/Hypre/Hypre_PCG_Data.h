/* *****************************************************
 *
 *	File:  Hypre_PCG_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_PCG_DataMembers_
#define Hypre_PCG_DataMembers_

#include "Hypre_Vector_IOR.h"
#include "Hypre_LinearOperator_IOR.h"
#include "Hypre_Solver_IOR.h"

struct Hypre_PCG_private_type
{
   double   tol;
   double   cf_tol;
   int      max_iter;
   int      two_norm;
   int      rel_change;

   Hypre_Vector    p;
   Hypre_Vector    s;
   Hypre_Vector    r;

   Hypre_LinearOperator matvec;
   Hypre_Solver         preconditioner;

   /* log info (always logged) */
   int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   double  *rel_norms;
}
;
#endif

