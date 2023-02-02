/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/


#include "_hypre_utilities.h"

/* This file will eventually contain functions needed to support
   a runtime decision of whether to use the assumed partition */


/* returns 1 if the assumed partition is in use */
HYPRE_Int HYPRE_AssumedPartitionCheck(void)
{
   return 1;
}

