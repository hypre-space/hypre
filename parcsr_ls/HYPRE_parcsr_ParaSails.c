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

#include "../distributed_linear_solvers/ParaSails/HYPRE_ParaSails.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_matrix_vector/vector.h"
#include "../parcsr_matrix_vector/par_vector.h"

/* If code is more mysterious, then it must be good */
typedef struct
{
    MPI_Comm        comm;
    HYPRE_ParaSails obj;
    double          thresh;
    int             nlevels;
}
Secret;

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsCreate
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRParaSailsCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   int ierr = 0;
   Secret *secret;
   
   secret = (Secret *) malloc(sizeof(Secret));

   if (secret == NULL)
       ierr = 1;

   secret->comm = comm;

   *solver = (HYPRE_Solver) secret;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRParaSailsDestroy( HYPRE_Solver solver )
{
   int ierr = 0;
   Secret *secret;

   secret = (Secret *) solver;
   ierr = HYPRE_ParaSailsDestroy(secret->obj);

   free(secret);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRParaSailsSetup( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   int ierr = 0;
   HYPRE_DistributedMatrix matrix;
   Secret *secret = (Secret *) solver;

   ierr = HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &matrix );
   if (ierr) return ierr;

   ierr = HYPRE_ParaSailsCreate(secret->comm, matrix, &secret->obj);
   if (ierr) return ierr;

   ierr = HYPRE_ParaSailsSetup(secret->obj, secret->thresh, secret->nlevels);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRParaSailsSolve( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   int ierr = 0;
   double *rhs, *soln;
   Secret *secret = (Secret *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   ierr = HYPRE_ParaSailsApply(secret->obj, soln, rhs);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetParams
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRParaSailsSetParams(HYPRE_Solver solver, 
                    double thresh,
                    int    nlevels)
{
   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;
   secret->nlevels = nlevels;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSelectThresh
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRParaSailsSelectThresh(HYPRE_Solver solver, 
                    double *threshp)
{
   Secret *secret = (Secret *) solver;

   return HYPRE_ParaSailsSelectThresh(secret->obj, threshp);
}

