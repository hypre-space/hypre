/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_struct_ls.h"
#include "bamg.h"

/*--------------------------------------------------------------------------
 * hypre_BAMGCreateRAPOp
 *
 *   Wrapper for CreateRAPOp routines which set up new coarse grid structures.
 *--------------------------------------------------------------------------*/

  hypre_StructMatrix *
hypre_BAMGCreateRAPOp( hypre_StructMatrix *R,
    hypre_StructMatrix *A,
    hypre_StructMatrix *P,
    hypre_StructGrid   *coarse_grid,
    HYPRE_Int           cdir )
{
  hypre_StructMatrix    *RAP;
  hypre_StructStencil   *stencil;
  HYPRE_Int              P_stored_as_transpose = 0;
  HYPRE_Int              constant_coefficient;

  stencil = hypre_StructMatrixStencil(A);

  RAP = hypre_SemiCreateRAPOp(R ,A, P, coarse_grid, cdir,
      P_stored_as_transpose);

  constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

  hypre_StructMatrixSetConstantCoefficient( RAP, constant_coefficient );

  return RAP;
}

/*--------------------------------------------------------------------------
 * hypre_BAMGSetupRAPOp
 *
 * Wrapper for routines to calculate entries in RAP. Incomplete error handling at the moment. 
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_BAMGSetupRAPOp( hypre_StructMatrix *R,
    hypre_StructMatrix *A,
    hypre_StructMatrix *P,
    HYPRE_Int           cdir,
    hypre_Index         cindex,
    hypre_Index         cstride,
    hypre_StructMatrix *Ac      )
{
  HYPRE_Int              P_stored_as_transpose = 0;
  hypre_StructStencil   *stencil;

  stencil = hypre_StructMatrixStencil(A);

  hypre_SemiBuildRAP(A, P, R, cdir, cindex, cstride,
      P_stored_as_transpose, Ac);

  hypre_StructMatrixAssemble(Ac);

  return hypre_error_flag;
}

