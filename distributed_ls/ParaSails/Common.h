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
 * Common.h header file - common definitions and parameters; also hides
 * HYPRE-specific definitions
 *
 *****************************************************************************/

#include <stdio.h>

#if 0 /* HYPRE */
#include "HYPRE_config.h"
#include "utilities.h"
#include "fortran.h"
#define ESSL HYPRE_USING_ESSL
#else
#include "mpi.h"
#define hypre_F90_NAME(name) name##_
#endif

#ifndef _COMMON_H
#define _COMMON_H

#define MAX_NPES          1024
#define ROW_REQ_TAG        222
#define ROW_REPI_TAG       223
#define ROW_REPV_TAG       224
#define DIAG_VALS_TAG      225
#define DIAG_INDS_TAG      226
#define ROWPATT_MAXLEN   50021
/*
#define DIAGSCALE_MAXLEN 50021
*/

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
