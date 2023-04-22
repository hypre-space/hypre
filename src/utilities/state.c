/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/* Global variable: library state (initialized, finalized, or none) */
hypre_State hypre__global_state = HYPRE_STATE_NONE;

/*--------------------------------------------------------------------------
 * HYPRE_Initialized
 *
 * Public function for hypre_Initialized
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_Initialized( void )
{
   return hypre_Initialized();
}

/*--------------------------------------------------------------------------
 * HYPRE_Finalized
 *
 * Public function for hypre_Finalized
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_Finalized( void )
{
   return hypre_Finalized();
}

/*--------------------------------------------------------------------------
 * hypre_Initialized
 *
 * This function returns True when the library has been initialized, but not
 * finalized yet.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_Initialized( void )
{
   return (hypre__global_state == HYPRE_STATE_INITIALIZED);
}

/*--------------------------------------------------------------------------
 * hypre_Finalized
 *
 * This function returns True when the library is in finalized state;
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_Finalized( void )
{
   return (hypre__global_state == HYPRE_STATE_FINALIZED);
}

/*--------------------------------------------------------------------------
 * hypre_SetInitialized
 *
 * This function sets the library state to initialized
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetInitialized( void )
{
   hypre__global_state = HYPRE_STATE_INITIALIZED;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SetFinalized
 *
 * This function sets the library state to finalized
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetFinalized( void )
{
   hypre__global_state = HYPRE_STATE_FINALIZED;

   return hypre_error_flag;
}
