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
#include "pfmg.h"

#define OLDRAP 1
#define NEWRAP 0

/*--------------------------------------------------------------------------
 * hypre_PFMGCreateRAPOp
 *
 *   Wrapper for 2 and 3d CreateRAPOp routines which set up new coarse
 *   grid structures.
 *--------------------------------------------------------------------------*/
 
hypre_StructMatrix *
hypre_PFMGCreateRAPOp( hypre_StructMatrix *R,
                       hypre_StructMatrix *A,
                       hypre_StructMatrix *P,
                       hypre_StructGrid   *coarse_grid,
                       int                 cdir        )
{
   hypre_StructMatrix    *RAP;
   hypre_StructStencil   *stencil;

   int                    P_stored_as_transpose = 0;

#if OLDRAP
   stencil = hypre_StructMatrixStencil(A);

   switch (hypre_StructStencilDim(stencil)) 
   {
      case 2:
      RAP = hypre_PFMG2CreateRAPOp(R ,A, P, coarse_grid, cdir);
      break;
    
      case 3:
      RAP = hypre_PFMG3CreateRAPOp(R ,A, P, coarse_grid, cdir);
      break;
   } 
#endif

#if NEWRAP
   RAP = hypre_SemiCreateRAPOp(R ,A, P, coarse_grid, cdir,
                                     P_stored_as_transpose);

#endif


   return RAP;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetupRAPOp
 *
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment. 
 *--------------------------------------------------------------------------*/
 
int
hypre_PFMGSetupRAPOp( hypre_StructMatrix *R,
                      hypre_StructMatrix *A,
                      hypre_StructMatrix *P,
                      int                 cdir,
                      hypre_Index         cindex,
                      hypre_Index         cstride,
                      hypre_StructMatrix *Ac      )
{
   int ierr = 0;
 
   hypre_StructStencil   *stencil;

   int                    P_stored_as_transpose = 0;


#if OLDRAP
   stencil = hypre_StructMatrixStencil(A);

   switch (hypre_StructStencilDim(stencil)) 
   {

      case 2:

      /*--------------------------------------------------------------------
       *    Set lower triangular (+ diagonal) coefficients
       *--------------------------------------------------------------------*/
      ierr = hypre_PFMG2BuildRAPSym(A, P, R, cdir, cindex, cstride, Ac);

      /*--------------------------------------------------------------------
       *    For non-symmetric A, set upper triangular coefficients as well
       *--------------------------------------------------------------------*/
      if(!hypre_StructMatrixSymmetric(A))
         ierr += hypre_PFMG2BuildRAPNoSym(A, P, R, cdir, cindex, cstride, Ac);

      break;

      case 3:

      /*--------------------------------------------------------------------
       *    Set lower triangular (+ diagonal) coefficients
       *--------------------------------------------------------------------*/
      ierr = hypre_PFMG3BuildRAPSym(A, P, R, cdir, cindex, cstride, Ac);

      /*--------------------------------------------------------------------
       *    For non-symmetric A, set upper triangular coefficients as well
       *--------------------------------------------------------------------*/
      if(!hypre_StructMatrixSymmetric(A))
         ierr += hypre_PFMG3BuildRAPNoSym(A, P, R, cdir, cindex, cstride, Ac);

      break;

   }
#endif

#if NEWRAP
   ierr = hypre_SemiBuildRAP(A, P, R, cdir, cindex, cstride,
                                    P_stored_as_transpose, Ac);
#endif


   hypre_StructMatrixAssemble(Ac);

   return ierr;
}

