/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.33 $
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

/*#ifdef HYPRE_USING_OPENMP
#include <omp.h>
#endif*/

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Before a version of HYPRE goes out the door, increment the version
 * number and check in this file (for CVS to substitute the Date).
 */
#define HYPRE_Version() "HYPRE_RELEASE_NAME  $Date: 2010/12/20 19:27:44 $ Compiled: " __DATE__ " " __TIME__

/*--------------------------------------------------------------------------
 * Big int stuff
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_BIGINT
typedef long long int HYPRE_Int;
#define HYPRE_MPI_INT MPI_LONG_LONG_INT
#else 
typedef int HYPRE_Int;
#define HYPRE_MPI_INT MPI_INT
#endif

/*--------------------------------------------------------------------------
 * Sequential MPI stuff
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_SEQUENTIAL
typedef HYPRE_Int MPI_Comm;
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
HYPRE_Int HYPRE_GetError();

/* Check if the given error flag contains the given error code */
HYPRE_Int HYPRE_CheckError(HYPRE_Int hypre_ierr, HYPRE_Int hypre_error_code);

/* Return the index of the argument (counting from 1) where
   argument error (HYPRE_ERROR_ARG) has occured */
HYPRE_Int HYPRE_GetErrorArg();

/* Describe the given error flag in the given string */
void HYPRE_DescribeError(HYPRE_Int hypre_ierr, char *descr);

/* Clears the hypre error flag */
HYPRE_Int HYPRE_ClearAllErrors();

/* Clears the given error code from the hypre error flag */
HYPRE_Int HYPRE_ClearError(HYPRE_Int hypre_error_code);

/*--------------------------------------------------------------------------
 * HYPRE AP user functions
 *--------------------------------------------------------------------------*/

/*Checks whether the AP is on */
HYPRE_Int HYPRE_AssumedPartitionCheck();


#ifdef __cplusplus
}
#endif

#endif
