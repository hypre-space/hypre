/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * hypre_PFMGCreateRAPOp
 *
 *   Wrapper for 2 and 3d CreateRAPOp routines which set up new coarse
 *   grid structures.
 *
 *   The parameter rap_type controls which lower level routines are
 *   used.
 *      rap_type = 0   Use optimized code for computing Galerkin operators
 *                     for special, common stencil patterns: 5 & 9 pt in
 *                     2d and 7, 19 & 27 in 3d.
 *      rap_type = 1   Use PARFLOW formula for coarse grid operator. Used
 *                     only with 5pt in 2d and 7pt in 3d.
 *      rap_type = 2   General purpose Galerkin code.
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_PFMGCreateRAPOp( hypre_StructMatrix *R,
                       hypre_StructMatrix *A,
                       hypre_StructMatrix *P,
                       hypre_StructGrid   *coarse_grid,
                       HYPRE_Int           cdir,
                       HYPRE_Int           rap_type    )
{
   hypre_StructMatrix    *RAP = NULL;
   hypre_StructStencil   *stencil;
   HYPRE_Int              P_stored_as_transpose = 0;
   HYPRE_Int              constant_coefficient;

   stencil = hypre_StructMatrixStencil(A);

   if (rap_type == 0)
   {
      switch (hypre_StructStencilNDim(stencil))
      {
         case 2:
            RAP = hypre_PFMG2CreateRAPOp(R, A, P, coarse_grid, cdir);
            break;

         case 3:
            RAP = hypre_PFMG3CreateRAPOp(R, A, P, coarse_grid, cdir);
            break;
      }
   }

   else if (rap_type == 1)
   {
      switch (hypre_StructStencilNDim(stencil))
      {
         case 2:
            RAP =  hypre_PFMGCreateCoarseOp5(R, A, P, coarse_grid, cdir);
            break;

         case 3:
            RAP =  hypre_PFMGCreateCoarseOp7(R, A, P, coarse_grid, cdir);
            break;
      }
   }
   else if (rap_type == 2)
   {
      RAP = hypre_SemiCreateRAPOp(R, A, P, coarse_grid, cdir,
                                  P_stored_as_transpose);
   }


   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient == 2 && rap_type == 0 )
   {
      /* A has variable diagonal, so, in the Galerkin case, P (and R) is
         entirely variable coefficient.  Thus RAP will be variable coefficient */
      hypre_StructMatrixSetConstantCoefficient( RAP, 0 );
   }
   else
   {
      hypre_StructMatrixSetConstantCoefficient( RAP, constant_coefficient );
   }

   return RAP;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetupRAPOp
 *
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment.
 *
 *   The parameter rap_type controls which lower level routines are
 *   used.
 *      rap_type = 0   Use optimized code for computing Galerkin operators
 *                     for special, common stencil patterns: 5 & 9 pt in
 *                     2d and 7, 19 & 27 in 3d.
 *      rap_type = 1   Use PARFLOW formula for coarse grid operator. Used
 *                     only with 5pt in 2d and 7pt in 3d.
 *      rap_type = 2   General purpose Galerkin code.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetupRAPOp( hypre_StructMatrix *R,
                      hypre_StructMatrix *A,
                      hypre_StructMatrix *P,
                      HYPRE_Int           cdir,
                      hypre_Index         cindex,
                      hypre_Index         cstride,
                      HYPRE_Int           rap_type,
                      hypre_StructMatrix *Ac      )
{
   HYPRE_Int              P_stored_as_transpose = 0;
   hypre_StructStencil   *stencil;

   hypre_StructMatrix    *Ac_tmp;

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_MemoryLocation data_location_A = hypre_StructGridDataLocation(hypre_StructMatrixGrid(A));
   HYPRE_MemoryLocation data_location_Ac = hypre_StructGridDataLocation(hypre_StructMatrixGrid(Ac));
   HYPRE_Int constant_coefficient = hypre_StructMatrixConstantCoefficient(Ac);
   if ( data_location_A != data_location_Ac )
   {
      Ac_tmp = hypre_PFMGCreateRAPOp(R, A, P, hypre_StructMatrixGrid(Ac), cdir, rap_type);
      hypre_StructMatrixSymmetric(Ac_tmp) = hypre_StructMatrixSymmetric(Ac);
      hypre_StructMatrixConstantCoefficient(Ac_tmp) = hypre_StructMatrixConstantCoefficient(Ac);
      hypre_StructGridDataLocation(hypre_StructMatrixGrid(Ac)) = data_location_A;
      HYPRE_StructMatrixInitialize(Ac_tmp);
   }
   else
   {
      Ac_tmp = Ac;
   }
#else
   Ac_tmp = Ac;
#endif
   stencil = hypre_StructMatrixStencil(A);

   if (rap_type == 0)
   {
      switch (hypre_StructStencilNDim(stencil))
      {
         case 2:
            /*--------------------------------------------------------------------
             *    Set lower triangular (+ diagonal) coefficients
             *--------------------------------------------------------------------*/
            hypre_PFMG2BuildRAPSym(A, P, R, cdir, cindex, cstride, Ac_tmp);

            /*--------------------------------------------------------------------
             *    For non-symmetric A, set upper triangular coefficients as well
             *--------------------------------------------------------------------*/
            if (!hypre_StructMatrixSymmetric(A))
            {
               hypre_PFMG2BuildRAPNoSym(A, P, R, cdir, cindex, cstride, Ac_tmp);
            }

            break;

         case 3:

            /*--------------------------------------------------------------------
             *    Set lower triangular (+ diagonal) coefficients
             *--------------------------------------------------------------------*/
            hypre_PFMG3BuildRAPSym(A, P, R, cdir, cindex, cstride, Ac_tmp);

            /*--------------------------------------------------------------------
             *    For non-symmetric A, set upper triangular coefficients as well
             *--------------------------------------------------------------------*/
            if (!hypre_StructMatrixSymmetric(A))
            {
               hypre_PFMG3BuildRAPNoSym(A, P, R, cdir, cindex, cstride, Ac_tmp);
            }

            break;
      }
   }

   else if (rap_type == 1)
   {
      switch (hypre_StructStencilNDim(stencil))
      {
         case 2:
            hypre_PFMGBuildCoarseOp5(A, P, R, cdir, cindex, cstride, Ac_tmp);
            break;

         case 3:
            hypre_PFMGBuildCoarseOp7(A, P, R, cdir, cindex, cstride, Ac_tmp);
            break;
      }
   }

   else if (rap_type == 2)
   {
      hypre_SemiBuildRAP(A, P, R, cdir, cindex, cstride,
                         P_stored_as_transpose, Ac_tmp);
   }

   hypre_StructMatrixAssemble(Ac_tmp);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if ( data_location_A != data_location_Ac )
   {
      if (constant_coefficient == 0)
      {
         hypre_TMemcpy(hypre_StructMatrixDataConst(Ac), hypre_StructMatrixData(Ac_tmp), HYPRE_Complex,
                       hypre_StructMatrixDataSize(Ac_tmp), HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      }
      else if (constant_coefficient == 1)
      {
         hypre_TMemcpy(hypre_StructMatrixDataConst(Ac), hypre_StructMatrixDataConst(Ac_tmp), HYPRE_Complex,
                       hypre_StructMatrixDataConstSize(Ac_tmp), HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      }
      else if (constant_coefficient == 2)
      {
         hypre_TMemcpy(hypre_StructMatrixDataConst(Ac), hypre_StructMatrixDataConst(Ac_tmp), HYPRE_Complex,
                       hypre_StructMatrixDataConstSize(Ac_tmp), HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_StructStencil *stencil_c       = hypre_StructMatrixStencil(Ac);
         HYPRE_Int stencil_size  = hypre_StructStencilSize(stencil_c);
         HYPRE_Complex       *Acdiag = hypre_StructMatrixDataConst(Ac) + stencil_size;
         hypre_TMemcpy(Acdiag, hypre_StructMatrixData(Ac_tmp), HYPRE_Complex,
                       hypre_StructMatrixDataSize(Ac_tmp), HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      }

      hypre_HandleStructExecPolicy(hypre_handle()) = data_location_Ac == HYPRE_MEMORY_DEVICE ?
                                                     HYPRE_EXEC_DEVICE : HYPRE_EXEC_HOST;
      hypre_StructGridDataLocation(hypre_StructMatrixGrid(Ac)) = data_location_Ac;
      hypre_StructMatrixAssemble(Ac);
      hypre_HandleStructExecPolicy(hypre_handle()) = data_location_A == HYPRE_MEMORY_DEVICE ?
                                                     HYPRE_EXEC_DEVICE : HYPRE_EXEC_HOST;
      hypre_StructMatrixDestroy(Ac_tmp);
   }
#endif

   return hypre_error_flag;
}

