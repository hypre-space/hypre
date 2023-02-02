/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *  FAC relaxation. Refinement patches are solved using system pfmg
 *  relaxation.
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
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

