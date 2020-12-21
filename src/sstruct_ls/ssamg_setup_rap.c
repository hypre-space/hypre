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

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeRAP
 *
 *   Wrapper for 2D and 3D SSAMGComputeRAP routines which sets up new coarse
 *   grid structures.
 *
 *   if the non_galerkin option is turned on, then use the PARFLOW formula
 *   for computing the coarse grid operator (works only with 5pt stencils in
 *   2D and 7pt stencils in 3D). If non_galerkin is turned off, then it uses
 *   the general purpose matrix-matrix multiplication function (SStructMatmult)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGComputeRAP( hypre_SStructMatrix   *A,
                       hypre_SStructMatrix   *P,
                       hypre_SStructGrid     *cgrid,
                       HYPRE_Int             *cdir_p,
                       HYPRE_Int              non_galerkin,
                       hypre_SStructMatrix  **Ac_ptr )
{
   hypre_SStructMatrix *Ac;

   if (non_galerkin)
   {
   }
   else
   {
      hypre_SStructMatPtAP(A, P, &Ac);
   }

   *Ac_ptr = Ac;

   return hypre_error_flag;
}
