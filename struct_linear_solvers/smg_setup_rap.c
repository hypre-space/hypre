/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "smg.h"

/*--------------------------------------------------------------------------
 * hypre_SMGCreateRAPOp
 *
 *   Wrapper for 2 and 3d CreateRAPOp routines which set up new coarse
 *   grid structures.
 *--------------------------------------------------------------------------*/
 
hypre_StructMatrix *
hypre_SMGCreateRAPOp( hypre_StructMatrix *R,
                      hypre_StructMatrix *A,
                      hypre_StructMatrix *PT,
                      hypre_StructGrid   *coarse_grid )
{
   hypre_StructMatrix    *RAP;
   hypre_StructStencil   *stencil;

   stencil = hypre_StructMatrixStencil(A);

   switch (hypre_StructStencilDim(stencil)) 
   {
      case 2:
      RAP = hypre_SMG2CreateRAPOp(R ,A, PT, coarse_grid);
      break;
    
      case 3:
      RAP = hypre_SMG3CreateRAPOp(R ,A, PT, coarse_grid);
      break;
   } 

   return RAP;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetupRAPOp
 *
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment. 
 *--------------------------------------------------------------------------*/
 
int
hypre_SMGSetupRAPOp( hypre_StructMatrix *R,
                     hypre_StructMatrix *A,
                     hypre_StructMatrix *PT,
                     hypre_StructMatrix *Ac,
                     hypre_Index         cindex,
                     hypre_Index         cstride )
{
   int ierr = 0;
 
   hypre_StructStencil   *stencil;

   stencil = hypre_StructMatrixStencil(A);

   switch (hypre_StructStencilDim(stencil)) 
   {

      case 2:

      /*--------------------------------------------------------------------
       *    Set lower triangular (+ diagonal) coefficients
       *--------------------------------------------------------------------*/
      ierr = hypre_SMG2BuildRAPSym(A, PT, R, Ac, cindex, cstride);

      /*--------------------------------------------------------------------
       *    For non-symmetric A, set upper triangular coefficients as well
       *--------------------------------------------------------------------*/
      if(!hypre_StructMatrixSymmetric(A))
      {
         ierr += hypre_SMG2BuildRAPNoSym(A, PT, R, Ac, cindex, cstride);
         /*-----------------------------------------------------------------
          *    Collapse stencil for periodic probems on coarsest grid.
          *-----------------------------------------------------------------*/
         ierr = hypre_SMG2RAPPeriodicNoSym(Ac, cindex, cstride);
      }
      else
      {
         /*-----------------------------------------------------------------
          *    Collapse stencil for periodic problems on coarsest grid.
          *-----------------------------------------------------------------*/
         ierr = hypre_SMG2RAPPeriodicSym(Ac, cindex, cstride);
      }

      break;

      case 3:

      /*--------------------------------------------------------------------
       *    Set lower triangular (+ diagonal) coefficients
       *--------------------------------------------------------------------*/
      ierr = hypre_SMG3BuildRAPSym(A, PT, R, Ac, cindex, cstride);

      /*--------------------------------------------------------------------
       *    For non-symmetric A, set upper triangular coefficients as well
       *--------------------------------------------------------------------*/
      if(!hypre_StructMatrixSymmetric(A))
      {
         ierr += hypre_SMG3BuildRAPNoSym(A, PT, R, Ac, cindex, cstride);
         /*-----------------------------------------------------------------
          *    Collapse stencil for periodic probems on coarsest grid.
          *-----------------------------------------------------------------*/
         ierr = hypre_SMG3RAPPeriodicNoSym(Ac, cindex, cstride);
      }
      else
      {
         /*-----------------------------------------------------------------
          *    Collapse stencil for periodic problems on coarsest grid.
          *-----------------------------------------------------------------*/
         ierr = hypre_SMG3RAPPeriodicSym(Ac, cindex, cstride);
      }

      break;

   }

   hypre_StructMatrixAssemble(Ac);

   return ierr;
}

