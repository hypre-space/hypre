/* *****************************************************
 *
 *	File:  Hypre_GMRES_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_GMRES_DataMembers_
#define Hypre_GMRES_DataMembers_

#include "Hypre_Vector_IOR.h"
#include "Hypre_LinearOperator_IOR.h"
#include "Hypre_Solver_IOR.h"
#include "Hypre_MPI_Com_IOR.h"

struct Hypre_GMRES_private_type
{
   int      k_dim;
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   Hypre_Vector    r;
   Hypre_Vector    w;
   Hypre_Vector   *p;   /* p is an array of k_dim+1 n-vectors */

   Hypre_LinearOperator matvec;
   Hypre_Solver         preconditioner;

   /* log info (always logged) */
   int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

   Hypre_MPI_Com comm;
}
;
#endif

