/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_parcsr interface mixed precision functions
 *
 *****************************************************************************/

#include "HYPRE_parcsr_mv_mp.h"
#include "hypre_parcsr_mv_mp.h"


#ifdef HYPRE_MIXED_PRECISION
/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParVectorCopy_mp( HYPRE_ParVector x,
                     HYPRE_ParVector y )
{
   return ( hypre_ParVectorCopy_mp( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

#endif
