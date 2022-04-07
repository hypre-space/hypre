/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Common.h header file - common definitions and parameters; also hides
 * HYPRE-specific definitions
 *
 *****************************************************************************/

#include <stdio.h>

#if 1 /* HYPRE */
#include "HYPRE_config.h"
#include "_hypre_utilities.h"
#include "fortran.h"
#ifdef HYPRE_USING_ESSL
#define ESSL
#endif
#else /* not HYPRE */
#include "mpi.h"
#endif

#ifndef _COMMON_H
#define _COMMON_H

#define PARASAILS_MAXLEN  300000 /* maximum nz in a pattern - can grow */
#define PARASAILS_NROWS   300000 /* maximum rows stored per proc - can grow */

#ifndef ABS
#define ABS(x) (((x)<0)?(-(x)):(x))
#endif
#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

#define PARASAILS_EXIT              \
{                                   \
   hypre_fprintf(stderr, "Exiting...\n"); \
   fflush(NULL);                    \
   hypre_MPI_Abort(hypre_MPI_COMM_WORLD, -1);   \
}

#endif /* _COMMON_H */
