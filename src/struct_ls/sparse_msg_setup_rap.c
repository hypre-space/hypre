/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

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
   hypre_StructMatrix    *RAP;
   hypre_StructStencil   *stencil;

   stencil = hypre_StructMatrixStencil(A);

   switch (hypre_StructStencilDim(stencil)) 
   {
      case 2:
      RAP = hypre_SparseMSG2CreateRAPOp(R ,A, P, coarse_grid, cdir);
      break;
    
      case 3:
      RAP = hypre_SparseMSG3CreateRAPOp(R ,A, P, coarse_grid, cdir);
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

   switch (hypre_StructStencilDim(stencil)) 
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
      if(!hypre_StructMatrixSymmetric(A))
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
      if(!hypre_StructMatrixSymmetric(A))
         ierr += hypre_SparseMSG3BuildRAPNoSym(A, P, R, cdir,
                                               cindex, cstride, stridePR, Ac);

      break;

   }

   hypre_StructMatrixAssemble(Ac);

   return ierr;
}

