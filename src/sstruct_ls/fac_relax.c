/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *  FAC relaxation. Refinement patches are solved using system pfmg
 *  relaxation.
 ******************************************************************************/

#include "headers.h"
#include "fac.h"

#define DEBUG 0

HYPRE_Int
hypre_FacLocalRelax(void                 *relax_vdata,
                    hypre_SStructPMatrix *A,
                    hypre_SStructPVector *x,
                    hypre_SStructPVector *b,
                    HYPRE_Int             num_relax,
                    HYPRE_Int            *zero_guess)
{
   hypre_SysPFMGRelaxSetPreRelax(relax_vdata);
   hypre_SysPFMGRelaxSetMaxIter(relax_vdata, num_relax);
   hypre_SysPFMGRelaxSetZeroGuess(relax_vdata, *zero_guess);
   hypre_SysPFMGRelax(relax_vdata, A, b, x);
   zero_guess = 0;

   return 0;
}

