/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_Struct interface mixed precision functions
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

#ifdef HYPRE_MIXED_PRECISION

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCopy
 * copies data from x to y
 * y has its own data array, so this is a deep copy in that sense.
 * The grid and other size information are not copied - they are
 * assumed to be consistent already.
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_StructVectorCopy_mp( HYPRE_StructVector x, HYPRE_StructVector y )
{
   return ( hypre_StructVectorCopy_mp( (hypre_StructVector_mp *)x,
                                       (hypre_StructVector_mp *)y ) );
}

#endif
