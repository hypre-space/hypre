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
 * zzz_SMGNewRAPOp
 *
 *   Wrapper for 2 and 3d NewRAPOp routines which set up new coarse
 *   grid structures.
 *--------------------------------------------------------------------------*/
 
zzz_StructMatrix *
zzz_SMGNewRAPOp( zzz_StructMatrix *R,
                 zzz_StructMatrix *A,
                 zzz_StructMatrix *PT )
{
   zzz_StructMatrix    *RAP;
   zzz_StructStencil   *stencil;

   stencil = zzz_StructMatrixStencil(A);

   switch (zzz_StructStencilDim(stencil)) 
   {
      case 2:
      RAP = zzz_SMG2NewRAPOp(R ,A, PT);
      break;
    
      case 3:
      RAP = zzz_SMG2NewRAPOp(R ,A, PT);
      break;
   } 

   return RAP;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetupRAPOp
 *
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment. 
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGSetupRAPOp( zzz_StructMatrix *R,
                   zzz_StructMatrix *A,
                   zzz_StructMatrix *PT,
                   zzz_StructMatrix *Ac,
                   zzz_Index        *cindex,
                   zzz_Index        *cstride )
{
   int ierr;
 
   zzz_StructStencil   *stencil;

   stencil = zzz_StructMatrixStencil(A);

   switch (zzz_StructStencilDim(stencil)) 
   {

      case 2:

/*--------------------------------------------------------------------------
 *    Set lower triangular (+ diagonal) coefficients
 *--------------------------------------------------------------------------*/
      ierr = zzz_SMG2BuildRAPSym(A, PT, R, Ac, cindex, cstride);

/*--------------------------------------------------------------------------
 *    For non-symmetric A, set upper triangular coefficients as well
 *--------------------------------------------------------------------------*/
      if(!zzz_StructMatrixSymmetric(A))
         ierr += zzz_SMG2BuildRAPNoSym(A, PT, R, Ac, cindex, cstride);

      break;

      case 3:

/*--------------------------------------------------------------------------
 *    Set lower triangular (+ diagonal) coefficients
 *--------------------------------------------------------------------------*/
      ierr = zzz_SMG3BuildRAPSym(A, PT, R, Ac, cindex, cstride);

/*--------------------------------------------------------------------------
 *    For non-symmetric A, set upper triangular coefficients as well
 *--------------------------------------------------------------------------*/
      if(!zzz_StructMatrixSymmetric(A))
         ierr += zzz_SMG3BuildRAPNoSym(A, PT, R, Ac, cindex, cstride);

      break;

   }

   return ierr;
}

