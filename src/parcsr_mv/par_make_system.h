/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_PAR_MAKE_SYSTEM
#define hypre_PAR_MAKE_SYSTEM

typedef struct
{
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *x;
   hypre_ParVector    *b;
} HYPRE_ParCSR_System_Problem;

#endif /* hypre_PAR_MAKE_SYSTEM */

