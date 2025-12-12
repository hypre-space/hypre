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

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGCreate_mp( HYPRE_Solver *solver)
{
   if (!solver)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (HYPRE_Solver) hypre_MPAMGCreate_mp( ) ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGDestroy_mp( HYPRE_Solver solver )
{
   return ( hypre_MPAMGDestroy_mp( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetup_mp( HYPRE_Solver solver,
                     HYPRE_ParCSRMatrix A,
                     HYPRE_ParVector b,
                     HYPRE_ParVector x      )
{
  if (!A)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   return ( hypre_MPAMGSetup_mp( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSolve_mp( HYPRE_Solver solver,
                     HYPRE_ParCSRMatrix A,
                     HYPRE_ParVector b,
                     HYPRE_ParVector x      )
{
  if (!A)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   return ( hypre_MPAMGSolve_mp( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_MPAMGPrecSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGPrecSetup_mp( HYPRE_Solver solver,
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

/* copy from double precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_ParVectorCopy_mp(b, btemp);
   HYPRE_ParVectorCopy_mp(x, xtemp);

/* call setup */
   HYPRE_MPAMGSetup_mp( solver, A, btemp, xtemp );

/* copy from single precision {btemp,xtemp} to double precision {b,x} */
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
HYPRE_MPAMGPrecSolve_mp( HYPRE_Solver solver,
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

/* copy from double precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_ParVectorCopy_mp(b, btemp);
   HYPRE_ParVectorCopy_mp(x, xtemp);

/* call setup */
   HYPRE_MPAMGSolve_mp( solver, A, btemp, xtemp );

/* copy from single precision {btemp,xtemp} to double precision {b,x} */
   HYPRE_ParVectorCopy_mp(btemp, b);
   HYPRE_ParVectorCopy_mp(xtemp, x);

/* free data */
   HYPRE_ParVectorDestroy_flt(btemp);
   HYPRE_ParVectorDestroy_flt(xtemp);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetPrecisionArray
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetPrecisionArray_mp(HYPRE_Solver solver,
                                HYPRE_Precision *precision_array)
{
   return (hypre_MPAMGSetPrecisionArray_mp ( (void *) solver, precision_array ) );
}

#endif
