/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_parcsr_ls mixed precision functions
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

#ifdef HYPRE_MIXED_PRECISION

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetup_mp( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   hypre_ParVector *btemp = NULL;
   hypre_ParVector *xtemp = NULL;

   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A),
                             &btemp);
   HYPRE_ParVectorInitialize_flt( btemp );
   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A),
                             &xtemp);
   HYPRE_ParVectorInitialize_flt( xtemp );

   /* copy from double-precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_ParVectorCopy_mp(b, btemp);
   HYPRE_ParVectorCopy_mp(x, xtemp);

   /* call setup */
   HYPRE_BoomerAMGSetup_flt( solver, A, btemp, xtemp );

   /* copy from single precision {btemp,xtemp} to double-precision {b,x} */
   HYPRE_ParVectorCopy_mp(btemp, b);
   HYPRE_ParVectorCopy_mp(xtemp, x);

   /* free data */
   HYPRE_ParVectorDestroy_flt(btemp);
   HYPRE_ParVectorDestroy_flt(xtemp);

   return 0;

}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_BoomerAMGSolve_mp( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   hypre_ParVector *btemp = NULL;
   hypre_ParVector *xtemp = NULL;

   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A),
                             &btemp);
   HYPRE_ParVectorInitialize_flt( btemp );
   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A),
                             &xtemp);
   HYPRE_ParVectorInitialize_flt( xtemp );

   /* copy from double-precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_ParVectorCopy_mp(b, btemp);
   HYPRE_ParVectorCopy_mp(x, xtemp);

   /* call setup */
   HYPRE_BoomerAMGSolve_flt( solver, A, btemp, xtemp );

   /* copy from single precision {btemp,xtemp} to double-precision {b,x} */
   HYPRE_ParVectorCopy_mp(btemp, b);
   HYPRE_ParVectorCopy_mp(xtemp, x);

   /* free data */
   HYPRE_ParVectorDestroy_flt(btemp);
   HYPRE_ParVectorDestroy_flt(xtemp);

   return 0;
}

#endif
