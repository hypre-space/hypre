/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_SHAREDDOFCOMM_HEADER
#define hypre_SHAREDDOFCOMM_HEADER

typedef struct
{
   HYPRE_BigInt  row;
   HYPRE_Int     ncols;
   HYPRE_BigInt *cols;
   HYPRE_Real   *data;

} hypre_MaxwellOffProcRow;

#endif
