/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Common.h header file - partially used to hide HYPRE-specific definitions
 * from the rest of the ParaSails source.  Also used for common definitions
 * and parameters.
 *
 *****************************************************************************/

#ifndef _COMMON_H
#define _COMMON_H

#include <stdio.h>

#if 1 /* HYPRE */
#include "HYPRE_config.h"
#include "utilities.h"
#else
#include "mpi.h"
#endif

#define MAX_NPES          1024
#define ROW_REQ_TAG        222
#define ROW_REPI_TAG       223
#define ROW_REPV_TAG       224
#define DIAG_VALS_TAG      225
#define DIAG_INDS_TAG      226
#define ROWPATT_MAXLEN   50021 /* a prime number */
#define DIAGSCALE_MAXLEN 50021

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
   fprintf(stderr, "Exiting...\n"); \
   fflush(NULL);                    \
   MPI_Abort(MPI_COMM_WORLD, -1);   \
}

#endif /* _COMMON_H */
