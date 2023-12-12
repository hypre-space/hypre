/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *
 *****************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * hypre_SparseMSGCreateRAPOp
 *
 *   Wrapper for 2 and 3d CreateRAPOp routines which set up new coarse
 *   grid structures.
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_SparseMSGCreateRAPOp( hypre_StructMatrix *R,
                            hypre_StructMatrix *A,
                            hypre_StructMatrix *P,
                            hypre_StructGrid   *coarse_grid,
                            HYPRE_Int           cdir        )
{
   hypre_StructMatrix    *RAP = NULL;
   hypre_StructStencil   *stencil;

   stencil = hypre_StructMatrixStencil(A);

   switch (hypre_StructStencilNDim(stencil))
   {
      case 2:
         RAP = hypre_SparseMSG2CreateRAPOp(R, A, P, coarse_grid, cdir);
         break;

      case 3:
         RAP = hypre_SparseMSG3CreateRAPOp(R, A, P, coarse_grid, cdir);
         break;
   }

   return RAP;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetupRAPOp
 *
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetupRAPOp( hypre_StructMatrix *R,
                           hypre_StructMatrix *A,
                           hypre_StructMatrix *P,
                           HYPRE_Int           cdir,
                           hypre_Index         cindex,
                           hypre_Index         cstride,
                           hypre_Index         stridePR,
                           hypre_StructMatrix *Ac       )
{
   HYPRE_Int ierr = 0;

   hypre_StructStencil   *stencil;

   stencil = hypre_StructMatrixStencil(A);

   switch (hypre_StructStencilNDim(stencil))
   {

      case 2:

         /*--------------------------------------------------------------------
          *    Set lower triangular (+ diagonal) coefficients
          *--------------------------------------------------------------------*/
         ierr = hypre_SparseMSG2BuildRAPSym(A, P, R, cdir,
                                            cindex, cstride, stridePR, Ac);

         /*--------------------------------------------------------------------
          *    For non-symmetric A, set upper triangular coefficients as well
          *--------------------------------------------------------------------*/
         if (!hypre_StructMatrixSymmetric(A))
            ierr += hypre_SparseMSG2BuildRAPNoSym(A, P, R, cdir,
                                                  cindex, cstride, stridePR, Ac);

         break;

      case 3:

         /*--------------------------------------------------------------------
          *    Set lower triangular (+ diagonal) coefficients
          *--------------------------------------------------------------------*/
         ierr = hypre_SparseMSG3BuildRAPSym(A, P, R, cdir,
                                            cindex, cstride, stridePR, Ac);

         /*--------------------------------------------------------------------
          *    For non-symmetric A, set upper triangular coefficients as well
          *--------------------------------------------------------------------*/
         if (!hypre_StructMatrixSymmetric(A))
            ierr += hypre_SparseMSG3BuildRAPNoSym(A, P, R, cdir,
                                                  cindex, cstride, stridePR, Ac);

         break;

   }

   hypre_StructMatrixAssemble(Ac);

   return ierr;
}
