/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "smg.h"

#define OLDRAP 1
#define NEWRAP 0

/*--------------------------------------------------------------------------
 * Wrapper for 2 and 3d CreateRAPOp routines which set up new coarse
 * grid structures.
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_SMGCreateRAPOp( hypre_StructMatrix *R,
                      hypre_StructMatrix *A,
                      hypre_StructMatrix *PT,
                      hypre_StructGrid   *coarse_grid )
{
   hypre_StructMatrix    *RAP = NULL;
   hypre_StructStencil   *stencil;

#if NEWRAP
   HYPRE_Int              cdir;
   HYPRE_Int              P_stored_as_transpose = 1;
#endif

   stencil = hypre_StructMatrixStencil(A);

#if OLDRAP
   switch (hypre_StructStencilNDim(stencil))
   {
      case 2:
         RAP = hypre_SMG2CreateRAPOp(R, A, PT, coarse_grid);
         break;

      case 3:
         RAP = hypre_SMG3CreateRAPOp(R, A, PT, coarse_grid);
         break;
   }
#endif

#if NEWRAP
   switch (hypre_StructStencilNDim(stencil))
   {
      case 2:
         cdir = 1;
         RAP = hypre_SemiCreateRAPOp(R, A, PT, coarse_grid, cdir,
                                     P_stored_as_transpose);
         break;

      case 3:
         cdir = 2;
         RAP = hypre_SemiCreateRAPOp(R, A, PT, coarse_grid, cdir,
                                     P_stored_as_transpose);
         break;
   }
#endif

   return RAP;
}

/*--------------------------------------------------------------------------
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetupRAPOp( hypre_StructMatrix *R,
                     hypre_StructMatrix *A,
                     hypre_StructMatrix *PT,
                     hypre_StructMatrix *Ac,
                     hypre_Index         cindex,
                     hypre_Index         cstride )
{
#if NEWRAP
   HYPRE_Int              cdir;
   HYPRE_Int              P_stored_as_transpose = 1;
#endif

   hypre_StructStencil   *stencil;
   hypre_StructMatrix    *Ac_tmp;
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_MemoryLocation data_location_A = hypre_StructGridDataLocation(hypre_StructMatrixGrid(A));
   HYPRE_MemoryLocation data_location_Ac = hypre_StructGridDataLocation(hypre_StructMatrixGrid(Ac));
   if (data_location_A != data_location_Ac)
   {
      Ac_tmp = hypre_SMGCreateRAPOp(R, A, PT, hypre_StructMatrixGrid(Ac));
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
#if OLDRAP
   switch (hypre_StructStencilNDim(stencil))
   {

      case 2:

         /*--------------------------------------------------------------------
          *    Set lower triangular (+ diagonal) coefficients
          *--------------------------------------------------------------------*/
         hypre_SMG2BuildRAPSym(A, PT, R, Ac_tmp, cindex, cstride);

         /*--------------------------------------------------------------------
          *    For non-symmetric A, set upper triangular coefficients as well
          *--------------------------------------------------------------------*/
         if (!hypre_StructMatrixSymmetric(A))
         {
            hypre_SMG2BuildRAPNoSym(A, PT, R, Ac_tmp, cindex, cstride);
            /*-----------------------------------------------------------------
             *    Collapse stencil for periodic probems on coarsest grid.
             *-----------------------------------------------------------------*/
            hypre_SMG2RAPPeriodicNoSym(Ac_tmp, cindex, cstride);
         }
         else
         {
            /*-----------------------------------------------------------------
             *    Collapse stencil for periodic problems on coarsest grid.
             *-----------------------------------------------------------------*/
            hypre_SMG2RAPPeriodicSym(Ac_tmp, cindex, cstride);
         }

         break;

      case 3:

         /*--------------------------------------------------------------------
          *    Set lower triangular (+ diagonal) coefficients
          *--------------------------------------------------------------------*/
         hypre_SMG3BuildRAPSym(A, PT, R, Ac_tmp, cindex, cstride);

         /*--------------------------------------------------------------------
          *    For non-symmetric A, set upper triangular coefficients as well
          *--------------------------------------------------------------------*/
         if (!hypre_StructMatrixSymmetric(A))
         {
            hypre_SMG3BuildRAPNoSym(A, PT, R, Ac_tmp, cindex, cstride);
            /*-----------------------------------------------------------------
             *    Collapse stencil for periodic probems on coarsest grid.
             *-----------------------------------------------------------------*/
            hypre_SMG3RAPPeriodicNoSym(Ac_tmp, cindex, cstride);
         }
         else
         {
            /*-----------------------------------------------------------------
             *    Collapse stencil for periodic problems on coarsest grid.
             *-----------------------------------------------------------------*/
            hypre_SMG3RAPPeriodicSym(Ac_tmp, cindex, cstride);
         }

         break;

   }
#endif

#if NEWRAP
   switch (hypre_StructStencilNDim(stencil))
   {

      case 2:
         cdir = 1;
         hypre_SemiBuildRAP(A, PT, R, cdir, cindex, cstride,
                            P_stored_as_transpose, Ac_tmp);
         break;

      case 3:
         cdir = 2;
         hypre_SemiBuildRAP(A, PT, R, cdir, cindex, cstride,
                            P_stored_as_transpose, Ac_tmp);
         break;

   }
#endif

   hypre_StructMatrixAssemble(Ac_tmp);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (data_location_A != data_location_Ac)
   {

      hypre_TMemcpy(hypre_StructMatrixDataConst(Ac), hypre_StructMatrixData(Ac_tmp), HYPRE_Complex,
                    hypre_StructMatrixDataSize(Ac_tmp), HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_SetDeviceOff();
      hypre_StructGridDataLocation(hypre_StructMatrixGrid(Ac)) = data_location_Ac;
      hypre_StructMatrixAssemble(Ac);
      hypre_SetDeviceOn();
      hypre_StructMatrixDestroy(Ac_tmp);
   }
#endif
   return hypre_error_flag;
}
