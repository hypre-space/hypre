/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.3 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Common.h header file - common definitions and parameters; also hides
 * HYPRE-specific definitions
 *
 *****************************************************************************/

#include <stdio.h>

#if 1 /* HYPRE */
#include "HYPRE_config.h"
#include "utilities.h"
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
   fprintf(stderr, "Exiting...\n"); \
   fflush(NULL);                    \
   MPI_Abort(MPI_COMM_WORLD, -1);   \
}

#endif /* _COMMON_H */
