/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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
 * $Revision$
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Header file for HYPRE_utilities library
 *
 *****************************************************************************/

#ifndef HYPRE_UTILITIES_HEADER
#define HYPRE_UTILITIES_HEADER

#include <HYPRE_config.h>

#ifndef HYPRE_SEQUENTIAL
#include "mpi.h"
#endif

#ifdef HYPRE_USING_OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Before a version of HYPRE goes out the door, increment the version
 * number and check in this file (for CVS to substitute the Date).
 */
#define HYPRE_Version() "PACKAGE_STRING  $Date$ Compiled: " __DATE__ " " __TIME__

#ifdef HYPRE_USE_PTHREADS
#ifndef hypre_MAX_THREADS
#define hypre_MAX_THREADS 128
#endif
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_SEQUENTIAL
typedef int MPI_Comm;
#endif


/*--------------------------------------------------------------------------
 * HYPRE error codes
 *--------------------------------------------------------------------------*/

#define HYPRE_ERROR_GENERIC         1   /* generic error */
#define HYPRE_ERROR_MEMORY          2   /* unable to allocate memory */
#define HYPRE_ERROR_ARG             4   /* argument error */
/* bits 4-8 are reserved for the index of the argument error */
#define HYPRE_ERROR_CONV          256   /* method did not converge as expected */


/*--------------------------------------------------------------------------
 * HYPRE error user functions
 *--------------------------------------------------------------------------*/

/* Return the current hypre error flag */
int HYPRE_GetError();

/* Check if the given error flag contains the given error code */
int HYPRE_CheckError(int hypre_ierr, int hypre_error_code);

/* Return the index of the argument (counting from 1) where
   argument error (HYPRE_ERROR_ARG) has occured */
int HYPRE_GetErrorArg();

/* Describe the given error flag in the given string */
void HYPRE_DescribeError(int hypre_ierr, char *descr);

/* Clears the hypre error flag */
int HYPRE_ClearAllErrors();

/* Clears the given error code from the hypre error flag */
int HYPRE_ClearError(int hypre_error_code);

#ifdef __cplusplus
}
#endif

#endif
