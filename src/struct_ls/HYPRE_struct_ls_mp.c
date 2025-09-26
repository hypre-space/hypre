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

 #include "_hypre_struct_ls.h"

#ifdef HYPRE_MIXED_PRECISION

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_SMGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetup_mp( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   hypre_StructVector *btemp = NULL;
   hypre_StructVector *xtemp = NULL;
   /* get matrix precision to call appropriate setup */
   HYPRE_Precision precision = hypre_StructMatrixPrecision (A);

   /* call standard setup if precisions match */
   if(precision == hypre_StructVectorPrecision (b))
   {
      return HYPRE_StructSMGSetup( solver, A, b, x );
   }

   HYPRE_StructVectorCreate_pre(precision, 
                                hypre_StructMatrixComm(A),
                                hypre_StructMatrixGrid(A),
                                &btemp);
   HYPRE_StructVectorInitialize_pre( precision, btemp );
   HYPRE_StructVectorCreate_pre(precision, 
                                hypre_StructMatrixComm(A),
                                hypre_StructMatrixGrid(A),
                                &xtemp);
   HYPRE_StructVectorInitialize_pre( precision, xtemp );

   /* copy from double-precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_StructVectorCopy_mp(b, btemp);
   HYPRE_StructVectorCopy_mp(x, xtemp);

   /* call setup */
   HYPRE_StructSMGSetup_pre( precision, solver, A, btemp, xtemp );

   /* copy from single precision {btemp,xtemp} to double-precision {b,x} */
   HYPRE_StructVectorCopy_mp(btemp, b);
   HYPRE_StructVectorCopy_mp(xtemp, x);

   /* free data */
   HYPRE_StructVectorDestroy_pre(precision, btemp);
   HYPRE_StructVectorDestroy_pre(precision, xtemp);

   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_SMGSetup
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_StructSMGSolve_mp( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   HYPRE_StructVector btemp = NULL;
   HYPRE_StructVector xtemp = NULL;

   HYPRE_StructVectorCreate_flt(hypre_StructMatrixComm(A),
                                hypre_StructMatrixGrid(A),
                                &btemp);
   HYPRE_StructVectorInitialize_flt( btemp );
   HYPRE_StructVectorCreate_flt(hypre_StructMatrixComm(A),
                                hypre_StructMatrixGrid(A),
                                &xtemp);
   HYPRE_StructVectorInitialize_flt( xtemp );

   /* copy from double-precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_StructVectorCopy_mp(b, btemp);
   HYPRE_StructVectorCopy_mp(x, xtemp);

   /* call solve */
   HYPRE_StructSMGSolve_flt( solver, A, btemp, xtemp );

   /* copy from single precision {btemp,xtemp} to double-precision {b,x} */
   HYPRE_StructVectorCopy_mp(btemp, b);
   HYPRE_StructVectorCopy_mp(xtemp, x);

   /* free data */
   HYPRE_StructVectorDestroy_flt(btemp);
   HYPRE_StructVectorDestroy_flt(xtemp);

   return 0;
}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_PFMGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetup_mp( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x      )
{
   HYPRE_StructVector btemp = NULL;
   HYPRE_StructVector xtemp = NULL;

   HYPRE_StructVectorCreate_flt(hypre_StructMatrixComm(A),
                                hypre_StructMatrixGrid(A),
                                &btemp);
   HYPRE_StructVectorInitialize_flt( btemp );
   HYPRE_StructVectorCreate_flt(hypre_StructMatrixComm(A),
                                hypre_StructMatrixGrid(A),
                                &xtemp);
   HYPRE_StructVectorInitialize_flt( xtemp );

   /* copy from double-precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_StructVectorCopy_mp(b, btemp);
   HYPRE_StructVectorCopy_mp(x, xtemp);

   /* call setup */
   HYPRE_StructPFMGSetup_flt( solver, A, btemp, xtemp );

   /* copy from single precision {btemp,xtemp} to double-precision {b,x} */
   HYPRE_StructVectorCopy_mp(btemp, b);
   HYPRE_StructVectorCopy_mp(xtemp, x);

   /* free data */
   HYPRE_StructVectorDestroy_flt(btemp);
   HYPRE_StructVectorDestroy_flt(xtemp);

   return 0;

}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_PFMGSolve
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_StructPFMGSolve_mp( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x      )
{
   HYPRE_StructVector btemp = NULL;
   HYPRE_StructVector xtemp = NULL;

   HYPRE_StructVectorCreate_flt(hypre_StructMatrixComm(A),
                                hypre_StructMatrixGrid(A),
                                &btemp);
   HYPRE_StructVectorInitialize_flt( btemp );
   HYPRE_StructVectorCreate_flt(hypre_StructMatrixComm(A),
                                hypre_StructMatrixGrid(A),
                                &xtemp);
   HYPRE_StructVectorInitialize_flt( xtemp );

   /* copy from double-precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_StructVectorCopy_mp(b, btemp);
   HYPRE_StructVectorCopy_mp(x, xtemp);

   /* call setup */
   HYPRE_StructPFMGSolve_flt( solver, A, btemp, xtemp );

   /* copy from single precision {btemp,xtemp} to double-precision {b,x} */
   HYPRE_StructVectorCopy_mp(btemp, b);
   HYPRE_StructVectorCopy_mp(xtemp, x);

   /* free data */
   HYPRE_StructVectorDestroy_flt(btemp);
   HYPRE_StructVectorDestroy_flt(xtemp);

   return 0;
}

#endif
