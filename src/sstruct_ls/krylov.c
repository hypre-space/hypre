/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_sstruct_ls.h"


HYPRE_Int hypre_SStructKrylovCopyVector( void *x, void *y );

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovIdentitySetup( void *vdata,
                                  void *A,
                                  void *b,
                                  void *x )

{
   HYPRE_UNUSED_VAR(vdata);
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructKrylovIdentity( void *vdata,
                             void *A,
                             void *b,
                             void *x )

{
   HYPRE_UNUSED_VAR(vdata);
   HYPRE_UNUSED_VAR(A);

   return ( hypre_SStructKrylovCopyVector(b, x) );
}
