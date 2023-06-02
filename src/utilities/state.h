/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_STATE_HEADER
#define hypre_STATE_HEADER

/*--------------------------------------------------------------------------
 * hypre library state
 *--------------------------------------------------------------------------*/

typedef enum hypre_State_enum
{
   HYPRE_STATE_NONE        = 0,
   HYPRE_STATE_INITIALIZED = 1,
   HYPRE_STATE_FINALIZED   = 2
} hypre_State;

extern hypre_State hypre__global_state;

#endif /* hypre_STATE_HEADER */
