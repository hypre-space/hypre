/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParCSRParaSails interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "./HYPRE_parcsr_ls.h"

#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_protos.h"

#include "../matrix_matrix/HYPRE_matrix_matrix_protos.h"

#include "../distributed_ls/ParaSails/hypre_ParaSails.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"
#include "../parcsr_mv/par_vector.h"

/* If code is more mysterious, then it must be good */
typedef struct
{
    hypre_ParaSails obj;
    int             sym;
    double          thresh;
    int             nlevels;
    double          filter;
    double          loadbal;
    int             reuse; /* reuse pattern */
    MPI_Comm        comm;
    int		    logging;
}
Secret;

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsCreate - Return a ParaSails preconditioner object 
 * "solver".  The default parameters for the preconditioner are also set,
 * so a call to HYPRE_ParCSRParaSailsSetParams is not absolutely necessary.
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRParaSailsCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   Secret *secret;
   
   secret = (Secret *) malloc(sizeof(Secret));

   if (secret == NULL)
       return 1;

   secret->sym     = 1;
   secret->thresh  = 0.1;
   secret->nlevels = 1;
   secret->filter  = 0.1;
   secret->loadbal = 0.0;
   secret->reuse   = 0;
   secret->comm    = comm;
   secret->logging = 0;

   hypre_ParaSailsCreate(comm, &secret->obj);

   *solver = (HYPRE_Solver) secret;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsDestroy - Destroy a ParaSails object.
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRParaSailsDestroy( HYPRE_Solver solver )
{
   int ierr = 0;
   Secret *secret;

   secret = (Secret *) solver;
   ierr = hypre_ParaSailsDestroy(secret->obj);

   free(secret);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetup - Set up function for ParaSails.
 * This function is not called on subsequent times if the preconditioner is 
 * being reused.
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRParaSailsSetup( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   int ierr = 0;
   static int virgin = 1;
   HYPRE_DistributedMatrix mat;
   Secret *secret = (Secret *) solver;

   /* The following call will also create the distributed matrix */

   ierr = HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &mat );
   if (ierr) return ierr;

   if (virgin || secret->reuse == 0) /* call set up at least once */
   {
       virgin = 0;
       ierr = hypre_ParaSailsSetup(secret->obj, mat, secret->sym, 
           secret->thresh, secret->nlevels, secret->filter, secret->loadbal,
	   secret->logging);
       if (ierr) return ierr;
   }
   else /* reuse is true; this is a subsequent call */
   {
       /* reuse pattern: always use filter value of 0 and loadbal of 0 */
       ierr = hypre_ParaSailsSetupValues(secret->obj, mat,
	 0.0, 0.0, secret->logging);
       if (ierr) return ierr;
   }

   ierr = HYPRE_DistributedMatrixDestroy(mat);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSolve - Solve function for ParaSails.
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRParaSailsSolve( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x     )
{
   int ierr = 0;
   double *rhs, *soln;
   Secret *secret = (Secret *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   ierr = hypre_ParaSailsApply(secret->obj, rhs, soln);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetParams - Set the parameters "thresh" and "nlevels"
 * for a ParaSails object.
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRParaSailsSetParams(HYPRE_Solver solver, 
                               double       thresh,
                               int          nlevels )
{
   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;
   secret->nlevels = nlevels;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetFilter - Set the filter parameter.
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRParaSailsSetFilter(HYPRE_Solver solver, 
                               double       filter  )
{
   Secret *secret = (Secret *) solver;

   secret->filter = filter;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetSym - Set whether the matrix is symmetric:
 * nonzero = symmetric, 0 = nonsymmetric.
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRParaSailsSetSym(HYPRE_Solver solver, 
                            int          sym     )
{
   Secret *secret = (Secret *) solver;

   secret->sym = sym;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetLoadbal
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRParaSailsSetLoadbal(HYPRE_Solver solver, 
                                double       loadbal )
{
   Secret *secret = (Secret *) solver;

   secret->loadbal = loadbal;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetReuse - reuse pattern if "reuse" if nonzero
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRParaSailsSetReuse(HYPRE_Solver solver, 
                              int          reuse   )
{
   Secret *secret = (Secret *) solver;

   secret->reuse = reuse;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetLogging -
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRParaSailsSetLogging(HYPRE_Solver solver, 
                                int          logging )
{
   Secret *secret = (Secret *) solver;

   secret->logging = logging;

   return 0;
}

/******************************************************************************
 *
 * Some of HYPRE_ParaSails interface (rest in distributed_ls/ParaSails
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsCreate - Return a ParaSails preconditioner object 
 * "solver".  The default parameters for the preconditioner are also set,
 * so a call to HYPRE_ParaSailsSetParams is not absolutely necessary.
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParaSailsCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   Secret *secret;
   
   secret = (Secret *) malloc(sizeof(Secret));

   if (secret == NULL)
       return 1;

   secret->sym     = 1;
   secret->thresh  = 0.1;
   secret->nlevels = 1;
   secret->filter  = 0.1;
   secret->loadbal = 0.0;
   secret->reuse   = 0;
   secret->comm    = comm;
   secret->logging = 0;

   hypre_ParaSailsCreate(comm, &secret->obj);

   *solver = (HYPRE_Solver) secret;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsDestroy - Destroy a ParaSails object.
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParaSailsDestroy( HYPRE_Solver solver )
{
   int ierr = 0;
   Secret *secret;

   secret = (Secret *) solver;
   ierr = hypre_ParaSailsDestroy(secret->obj);

   free(secret);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetup - Set up function for ParaSails.
 * This function is not called on subsequent times if the preconditioner is 
 * being reused.
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParaSailsSetup( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x     )
{
   int ierr = 0;
   static int virgin = 1;
   HYPRE_DistributedMatrix mat;
   Secret *secret = (Secret *) solver;

   /* The following call will also create the distributed matrix */

   ierr = HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &mat );
   if (ierr) return ierr;

   if (virgin || secret->reuse == 0) /* call set up at least once */
   {
       virgin = 0;
       ierr = hypre_ParaSailsSetup(secret->obj, mat, secret->sym, 
           secret->thresh, secret->nlevels, secret->filter, secret->loadbal,
	   secret->logging);
       if (ierr) return ierr;
   }
   else /* reuse is true; this is a subsequent call */
   {
       /* reuse pattern: always use filter value of 0 and loadbal of 0 */
       ierr = hypre_ParaSailsSetupValues(secret->obj, mat,
	 0.0, 0.0, secret->logging);
       if (ierr) return ierr;
   }

   ierr = HYPRE_DistributedMatrixDestroy(mat);

   return ierr;
}
/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSolve - Solve function for ParaSails.
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParaSailsSolve( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x     )
{
   int ierr = 0;
   double *rhs, *soln;
   Secret *secret = (Secret *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   ierr = hypre_ParaSailsApply(secret->obj, rhs, soln);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetParams - Set the parameters "thresh" and "nlevels"
 * for a ParaSails object.
 *--------------------------------------------------------------------------*/

int
HYPRE_ParaSailsSetParams(HYPRE_Solver solver, 
                         double       thresh,
                         int          nlevels )
{
   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;
   secret->nlevels = nlevels;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetFilter - Set the filter parameter.
 *--------------------------------------------------------------------------*/

int
HYPRE_ParaSailsSetFilter(HYPRE_Solver solver, 
                         double       filter  )
{
   Secret *secret = (Secret *) solver;

   secret->filter = filter;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetSym - Set whether the matrix is symmetric:
 * nonzero = symmetric, 0 = nonsymmetric.
 *--------------------------------------------------------------------------*/

int
HYPRE_ParaSailsSetSym(HYPRE_Solver solver, 
                      int          sym     )
{
   Secret *secret = (Secret *) solver;

   secret->sym = sym;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLoadbal
 *--------------------------------------------------------------------------*/

int
HYPRE_ParaSailsSetLoadbal(HYPRE_Solver solver, 
                          double       loadbal )
{
   Secret *secret = (Secret *) solver;

   secret->loadbal = loadbal;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetReuse - reuse pattern if "reuse" if nonzero
 *--------------------------------------------------------------------------*/

int
HYPRE_ParaSailsSetReuse(HYPRE_Solver solver, 
                        int          reuse   )
{
   Secret *secret = (Secret *) solver;

   secret->reuse = reuse;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLogging -
 *--------------------------------------------------------------------------*/

int
HYPRE_ParaSailsSetLogging(HYPRE_Solver solver, 
                          int          logging )
{
   Secret *secret = (Secret *) solver;

   secret->logging = logging;

   return 0;
}
