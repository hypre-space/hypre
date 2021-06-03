/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

/*--------------------------------------------------------------------------
 * Big int stuff
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_BIGINT)
typedef long long int HYPRE_BigInt;
typedef long long int HYPRE_Int;
#define HYPRE_MPI_BIG_INT MPI_LONG_LONG_INT
#define HYPRE_MPI_INT MPI_LONG_LONG_INT

#elif defined(HYPRE_MIXEDINT)
typedef long long int HYPRE_BigInt;
typedef int HYPRE_Int;
#define HYPRE_MPI_BIG_INT MPI_LONG_LONG_INT
#define HYPRE_MPI_INT MPI_INT

#else /* default */
typedef int HYPRE_BigInt;
typedef int HYPRE_Int;
#define HYPRE_MPI_BIG_INT MPI_INT
#define HYPRE_MPI_INT MPI_INT
#endif

/*--------------------------------------------------------------------------
 * Real and Complex types
 *--------------------------------------------------------------------------*/

#include <float.h>

#if defined(HYPRE_SINGLE)
typedef float HYPRE_Real;
#define HYPRE_REAL_MAX FLT_MAX
#define HYPRE_REAL_MIN FLT_MIN
#define HYPRE_REAL_EPSILON FLT_EPSILON
#define HYPRE_REAL_MIN_EXP FLT_MIN_EXP
#define HYPRE_MPI_REAL MPI_FLOAT

#elif defined(HYPRE_LONG_DOUBLE)
typedef long double HYPRE_Real;
#define HYPRE_REAL_MAX LDBL_MAX
#define HYPRE_REAL_MIN LDBL_MIN
#define HYPRE_REAL_EPSILON LDBL_EPSILON
#define HYPRE_REAL_MIN_EXP DBL_MIN_EXP
#define HYPRE_MPI_REAL MPI_LONG_DOUBLE

#else /* default */
typedef double HYPRE_Real;
#define HYPRE_REAL_MAX DBL_MAX
#define HYPRE_REAL_MIN DBL_MIN
#define HYPRE_REAL_EPSILON DBL_EPSILON
#define HYPRE_REAL_MIN_EXP DBL_MIN_EXP
#define HYPRE_MPI_REAL MPI_DOUBLE
#endif

#if defined(HYPRE_COMPLEX)
typedef double _Complex HYPRE_Complex;
#define HYPRE_MPI_COMPLEX MPI_C_DOUBLE_COMPLEX  /* or MPI_LONG_DOUBLE ? */

#else  /* default */
typedef HYPRE_Real HYPRE_Complex;
#define HYPRE_MPI_COMPLEX HYPRE_MPI_REAL
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
 * HYPRE init/finalize
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_Init();
HYPRE_Int HYPRE_Finalize();

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

/* Print GPU information */
HYPRE_Int HYPRE_PrintDeviceInfo();

/*--------------------------------------------------------------------------
 * HYPRE Version routines
 *--------------------------------------------------------------------------*/

/* RDF: This macro is used by the FEI code.  Want to eventually remove. */
#define HYPRE_VERSION "HYPRE_RELEASE_NAME Date Compiled: " __DATE__ " " __TIME__

/**
 * Allocates and returns a string with version number information in it.
 **/
HYPRE_Int
HYPRE_Version( char **version_ptr );

/**
 * Returns version number information in integer form.  Use 'NULL' for values
 * not needed.  The argument {\tt single} is a single sortable integer
 * representation of the release number.
 **/
HYPRE_Int
HYPRE_VersionNumber( HYPRE_Int  *major_ptr,
                     HYPRE_Int  *minor_ptr,
                     HYPRE_Int  *patch_ptr,
                     HYPRE_Int  *single_ptr );

/*--------------------------------------------------------------------------
 * HYPRE AP user functions
 *--------------------------------------------------------------------------*/

/*Checks whether the AP is on */
HYPRE_Int HYPRE_AssumedPartitionCheck();

/*--------------------------------------------------------------------------
 * HYPRE memory location
 *--------------------------------------------------------------------------*/

typedef enum _HYPRE_MemoryLocation
{
   HYPRE_MEMORY_UNDEFINED = -1,
   HYPRE_MEMORY_HOST          ,
   HYPRE_MEMORY_DEVICE
} HYPRE_MemoryLocation;

HYPRE_Int HYPRE_SetMemoryLocation(HYPRE_MemoryLocation memory_location);
HYPRE_Int HYPRE_GetMemoryLocation(HYPRE_MemoryLocation *memory_location);

#include <stdlib.h>

/*--------------------------------------------------------------------------
 * HYPRE execution policy
 *--------------------------------------------------------------------------*/

typedef enum _HYPRE_ExecutionPolicy
{
   HYPRE_EXEC_UNDEFINED = -1,
   HYPRE_EXEC_HOST          ,
   HYPRE_EXEC_DEVICE
} HYPRE_ExecutionPolicy;

HYPRE_Int HYPRE_SetExecutionPolicy(HYPRE_ExecutionPolicy exec_policy);
HYPRE_Int HYPRE_GetExecutionPolicy(HYPRE_ExecutionPolicy *exec_policy);
HYPRE_Int HYPRE_SetStructExecutionPolicy(HYPRE_ExecutionPolicy exec_policy);
HYPRE_Int HYPRE_GetStructExecutionPolicy(HYPRE_ExecutionPolicy *exec_policy);

/*--------------------------------------------------------------------------
 * HYPRE UMPIRE
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_SetUmpireDevicePoolSize(size_t nbytes);
HYPRE_Int HYPRE_SetUmpireUMPoolSize(size_t nbytes);
HYPRE_Int HYPRE_SetUmpireHostPoolSize(size_t nbytes);
HYPRE_Int HYPRE_SetUmpirePinnedPoolSize(size_t nbytes);
HYPRE_Int HYPRE_SetUmpireDevicePoolName(const char *pool_name);
HYPRE_Int HYPRE_SetUmpireUMPoolName(const char *pool_name);
HYPRE_Int HYPRE_SetUmpireHostPoolName(const char *pool_name);
HYPRE_Int HYPRE_SetUmpirePinnedPoolName(const char *pool_name);

/*--------------------------------------------------------------------------
 * HYPRE GPU memory pool
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_SetGPUMemoryPoolSize(HYPRE_Int bin_growth, HYPRE_Int min_bin, HYPRE_Int max_bin, size_t max_cached_bytes);

/*--------------------------------------------------------------------------
 * HYPRE handle
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_SetSpGemmUseCusparse( HYPRE_Int use_cusparse );
HYPRE_Int HYPRE_SetUseGpuRand( HYPRE_Int use_curand );

#ifdef __cplusplus
}
#endif

#endif
