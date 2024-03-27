/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRParaSails interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "./HYPRE_parcsr_ls.h"
#include "./_hypre_parcsr_ls.h"

#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_protos.h"

#include "../matrix_matrix/HYPRE_matrix_matrix_protos.h"

#include "../distributed_ls/ParaSails/hypre_ParaSails.h"

/* these includes required for HYPRE_ParaSailsBuildIJMatrix */
#include "../IJ_mv/HYPRE_IJ_mv.h"

/* Must include implementation definition for ParVector since no data access
   functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"
/* AB 8/06 - replace header file */
/* #include "../parcsr_mv/par_vector.h" */
#include "../parcsr_mv/_hypre_parcsr_mv.h"

/* If code is more mysterious, then it must be good */
typedef struct
{
   hypre_ParaSails obj;
   HYPRE_Int       sym;
   HYPRE_Real      thresh;
   HYPRE_Int       nlevels;
   HYPRE_Real      filter;
   HYPRE_Real      loadbal;
   HYPRE_Int       reuse; /* reuse pattern */
   MPI_Comm        comm;
   HYPRE_Int       logging;
}
Secret;

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsCreate - Return a ParaSails preconditioner object
 * "solver".  The default parameters for the preconditioner are also set,
 * so a call to HYPRE_ParCSRParaSailsSetParams is not absolutely necessary.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(comm);
   HYPRE_UNUSED_VAR(solver);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret;

   secret = hypre_TAlloc(Secret, 1, HYPRE_MEMORY_HOST);

   if (secret == NULL)
   {
      hypre_error(HYPRE_ERROR_MEMORY);
      return hypre_error_flag;
   }

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

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsDestroy - Destroy a ParaSails object.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsDestroy( HYPRE_Solver solver )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret;

   secret = (Secret *) solver;
   hypre_ParaSailsDestroy(secret->obj);

   hypre_TFree(secret, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetup - Set up function for ParaSails.
 * This function is not called on subsequent times if the preconditioner is
 * being reused.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsSetup( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x      )
{
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(A);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   static HYPRE_Int virgin = 1;
   HYPRE_DistributedMatrix mat;
   Secret *secret = (Secret *) solver;

   /* The following call will also create the distributed matrix */

   HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &mat );
   if (hypre_error_flag) { return hypre_error_flag; }

   if (virgin || secret->reuse == 0) /* call set up at least once */
   {
      virgin = 0;
      hypre_ParaSailsSetup(
         secret->obj, mat, secret->sym, secret->thresh, secret->nlevels,
         secret->filter, secret->loadbal, secret->logging);
      if (hypre_error_flag) { return hypre_error_flag; }
   }
   else /* reuse is true; this is a subsequent call */
   {
      /* reuse pattern: always use filter value of 0 and loadbal of 0 */
      hypre_ParaSailsSetupValues(secret->obj, mat,
                                 0.0, 0.0, secret->logging);
      if (hypre_error_flag) { return hypre_error_flag; }
   }

   HYPRE_DistributedMatrixDestroy(mat);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSolve - Solve function for ParaSails.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsSolve( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x     )
{
   HYPRE_UNUSED_VAR(A);

#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);
   HYPRE_UNUSED_VAR(solver);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_Real *rhs, *soln;
   Secret *secret = (Secret *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   hypre_ParaSailsApply(secret->obj, rhs, soln);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetParams - Set the parameters "thresh" and "nlevels"
 * for a ParaSails object.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsSetParams(HYPRE_Solver solver,
                               HYPRE_Real   thresh,
                               HYPRE_Int    nlevels )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(thresh);
   HYPRE_UNUSED_VAR(nlevels);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;
   secret->nlevels = nlevels;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetFilter - Set the filter parameter,
 * HYPRE_ParCSRParaSailsGetFilter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsSetFilter(HYPRE_Solver solver,
                               HYPRE_Real   filter  )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(filter);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->filter = filter;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParCSRParaSailsGetFilter(HYPRE_Solver solver,
                               HYPRE_Real * filter  )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(filter);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *filter = secret->filter;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetSym - Set whether the matrix is symmetric:
 * nonzero = symmetric, 0 = nonsymmetric.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsSetSym(HYPRE_Solver solver,
                            HYPRE_Int    sym     )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(sym);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->sym = sym;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetLoadbal, HYPRE_ParCSRParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsSetLoadbal(HYPRE_Solver solver,
                                HYPRE_Real   loadbal )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(loadbal);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->loadbal = loadbal;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParCSRParaSailsGetLoadbal(HYPRE_Solver solver,
                                HYPRE_Real * loadbal )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(loadbal);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *loadbal = secret->loadbal;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetReuse - reuse pattern if "reuse" if nonzero
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsSetReuse(HYPRE_Solver solver,
                              HYPRE_Int    reuse   )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(reuse);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->reuse = reuse;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetLogging -
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRParaSailsSetLogging(HYPRE_Solver solver,
                                HYPRE_Int    logging )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(logging);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->logging = logging;

   return hypre_error_flag;
#endif
}

/******************************************************************************
 *
 * HYPRE_ParaSails interface
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsCreate - Return a ParaSails preconditioner object
 * "solver".  The default parameters for the preconditioner are also set,
 * so a call to HYPRE_ParaSailsSetParams is not absolutely necessary.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(comm);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret;

   secret = hypre_TAlloc(Secret, 1, HYPRE_MEMORY_HOST);

   if (secret == NULL)
   {
      hypre_error(HYPRE_ERROR_MEMORY);
      return hypre_error_flag;
   }

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

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsDestroy - Destroy a ParaSails object.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsDestroy( HYPRE_Solver solver )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret;

   secret = (Secret *) solver;
   hypre_ParaSailsDestroy(secret->obj);

   hypre_TFree(secret, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetup - Set up function for ParaSails.
 * This function is not called on subsequent times if the preconditioner is
 * being reused.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetup( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x     )
{
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(A);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   static HYPRE_Int virgin = 1;
   HYPRE_DistributedMatrix mat;
   Secret *secret = (Secret *) solver;
   HYPRE_Int ierr;

   /* The following call will also create the distributed matrix */

   ierr = HYPRE_GetError(); HYPRE_ClearAllErrors();
   HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &mat );
   if (hypre_error_flag) { return hypre_error_flag |= ierr; }

   if (virgin || secret->reuse == 0) /* call set up at least once */
   {
      virgin = 0;
      hypre_ParaSailsSetup(
         secret->obj, mat, secret->sym, secret->thresh, secret->nlevels,
         secret->filter, secret->loadbal, secret->logging);
      if (hypre_error_flag) { return hypre_error_flag |= ierr; }
   }
   else /* reuse is true; this is a subsequent call */
   {
      /* reuse pattern: always use filter value of 0 and loadbal of 0 */
      hypre_ParaSailsSetupValues(secret->obj, mat,
                                 0.0, 0.0, secret->logging);
      if (hypre_error_flag) { return hypre_error_flag |= ierr; }
   }

   HYPRE_DistributedMatrixDestroy(mat);

   return hypre_error_flag;
#endif
}
/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSolve - Solve function for ParaSails.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSolve( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x     )
{
   HYPRE_UNUSED_VAR(A);

#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_Real *rhs, *soln;
   Secret *secret = (Secret *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   hypre_ParaSailsApply(secret->obj, rhs, soln);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetParams - Set the parameters "thresh" and "nlevels"
 * for a ParaSails object.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetParams(HYPRE_Solver solver,
                         HYPRE_Real   thresh,
                         HYPRE_Int    nlevels )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(thresh);
   HYPRE_UNUSED_VAR(nlevels);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;
   secret->nlevels = nlevels;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetThresh - Set the "thresh" parameter only
 * for a ParaSails object.
 * HYPRE_ParaSailsGetThresh
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetThresh( HYPRE_Solver solver,
                          HYPRE_Real   thresh )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(thresh);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParaSailsGetThresh( HYPRE_Solver solver,
                          HYPRE_Real * thresh )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(thresh);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *thresh = secret->thresh;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetNlevels - Set the "nlevels" parameter only
 * for a ParaSails object.
 * HYPRE_ParaSailsGetNlevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetNlevels( HYPRE_Solver solver,
                           HYPRE_Int    nlevels )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(nlevels);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->nlevels  = nlevels;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParaSailsGetNlevels( HYPRE_Solver solver,
                           HYPRE_Int  * nlevels )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(nlevels);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *nlevels = secret->nlevels;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetFilter - Set the filter parameter.
 * HYPRE_ParaSailsGetFilter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetFilter(HYPRE_Solver solver,
                         HYPRE_Real   filter  )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(filter);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->filter = filter;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParaSailsGetFilter(HYPRE_Solver solver,
                         HYPRE_Real * filter  )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(filter);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *filter = secret->filter;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetSym - Set whether the matrix is symmetric:
 * nonzero = symmetric, 0 = nonsymmetric.
 * HYPRE_ParaSailsGetSym
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetSym(HYPRE_Solver solver,
                      HYPRE_Int    sym     )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(sym);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->sym = sym;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParaSailsGetSym(HYPRE_Solver solver,
                      HYPRE_Int  * sym     )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(sym);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *sym = secret->sym;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLoadbal, HYPRE_ParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetLoadbal(HYPRE_Solver solver,
                          HYPRE_Real   loadbal )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(loadbal);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->loadbal = loadbal;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParaSailsGetLoadbal(HYPRE_Solver solver,
                          HYPRE_Real * loadbal )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(loadbal);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *loadbal = secret->loadbal;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetReuse - reuse pattern if "reuse" if nonzero
 * HYPRE_ParaSailsGetReuse
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetReuse(HYPRE_Solver solver,
                        HYPRE_Int    reuse   )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(reuse);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->reuse = reuse;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParaSailsGetReuse(HYPRE_Solver solver,
                        HYPRE_Int  * reuse   )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(reuse);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *reuse = secret->reuse;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLogging, HYPRE_ParaSailsGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParaSailsSetLogging(HYPRE_Solver solver,
                          HYPRE_Int    logging )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(logging);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->logging = logging;

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParaSailsGetLogging(HYPRE_Solver solver,
                          HYPRE_Int  * logging )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(logging);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *logging = secret->logging;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsBuildIJMatrix -
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ParaSailsBuildIJMatrix(HYPRE_Solver solver, HYPRE_IJMatrix *pij_A)
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(pij_A);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   hypre_ParaSailsBuildIJMatrix(secret->obj, pij_A);

   return hypre_error_flag;
#endif
}
