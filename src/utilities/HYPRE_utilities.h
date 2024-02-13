/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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

#define HYPRE_BIG_INT_MAX LLONG_MAX
#define HYPRE_BIG_INT_MIN LLONG_MIN
#define HYPRE_INT_MAX LLONG_MAX
#define HYPRE_INT_MIN LLONG_MIN

#define HYPRE_MPI_BIG_INT MPI_LONG_LONG_INT
#define HYPRE_MPI_INT MPI_LONG_LONG_INT

#elif defined(HYPRE_MIXEDINT)
typedef long long int HYPRE_BigInt;
typedef int HYPRE_Int;

#define HYPRE_BIG_INT_MAX LLONG_MAX
#define HYPRE_BIG_INT_MIN LLONG_MIN
#define HYPRE_INT_MAX INT_MAX
#define HYPRE_INT_MIN INT_MIN

#define HYPRE_MPI_BIG_INT MPI_LONG_LONG_INT
#define HYPRE_MPI_INT MPI_INT

#else /* default */
typedef int HYPRE_BigInt;
typedef int HYPRE_Int;

#define HYPRE_BIG_INT_MAX INT_MAX
#define HYPRE_BIG_INT_MIN INT_MIN
#define HYPRE_INT_MAX INT_MAX
#define HYPRE_INT_MIN INT_MIN

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
#if defined(FLT_TRUE_MIN)
#define HYPRE_REAL_TRUE_MIN FLT_TRUE_MIN
#else
#define HYPRE_REAL_TRUE_MIN FLT_MIN
#endif
#define HYPRE_REAL_EPSILON FLT_EPSILON
#define HYPRE_REAL_MIN_EXP FLT_MIN_EXP
#define HYPRE_MPI_REAL MPI_FLOAT

#elif defined(HYPRE_LONG_DOUBLE)
typedef long double HYPRE_Real;
#define HYPRE_REAL_MAX LDBL_MAX
#define HYPRE_REAL_MIN LDBL_MIN
#if defined(LDBL_TRUE_MIN)
#define HYPRE_REAL_TRUE_MIN LDBL_TRUE_MIN
#else
#define HYPRE_REAL_TRUE_MIN LDBL_MIN
#endif
#define HYPRE_REAL_EPSILON LDBL_EPSILON
#define HYPRE_REAL_MIN_EXP DBL_MIN_EXP
#define HYPRE_MPI_REAL MPI_LONG_DOUBLE

#else /* default */
typedef double HYPRE_Real;
#define HYPRE_REAL_MAX DBL_MAX
#define HYPRE_REAL_MIN DBL_MIN
#if defined(DBL_TRUE_MIN)
#define HYPRE_REAL_TRUE_MIN DBL_TRUE_MIN
#else
#define HYPRE_REAL_TRUE_MIN DBL_MIN
#endif
#define HYPRE_REAL_EPSILON DBL_EPSILON
#define HYPRE_REAL_MIN_EXP DBL_MIN_EXP
#define HYPRE_MPI_REAL MPI_DOUBLE
#endif

#if defined(HYPRE_COMPLEX)
/* support for float double and long double complex types */
#if defined(HYPRE_SINGLE)
typedef float _Complex HYPRE_Complex;
#define HYPRE_MPI_COMPLEX MPI_C_FLOAT_COMPLEX
#elif defined(HYPRE_LONG_DOUBLE)
typedef long double _Complex HYPRE_Complex;
#define HYPRE_MPI_COMPLEX MPI_C_LONG_DOUBLE_COMPLEX
#else /* default */
typedef double _Complex HYPRE_Complex;
#define HYPRE_MPI_COMPLEX MPI_C_DOUBLE_COMPLEX  /* or MPI_LONG_DOUBLE ? */
#endif
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
#define HYPRE_MAX_FILE_NAME_LEN  1024   /* longest filename length used in hypre */
#define HYPRE_MAX_MSG_LEN        2048   /* longest message length */

/*--------------------------------------------------------------------------
 * HYPRE init/finalize
 *--------------------------------------------------------------------------*/

/**
 * (Required) Initializes the hypre library.
 **/

HYPRE_Int HYPRE_Initialize(void);

/**
 * (Required) Initializes the hypre library. This function is provided for backward compatibility.
 * Please, use HYPRE_Initialize instead.
 **/

#define HYPRE_Init() HYPRE_Initialize()

/**
 * (Optional) Initializes GPU features in the hypre library.
 **/

HYPRE_Int HYPRE_DeviceInitialize(void);

/**
 * (Required) Finalizes the hypre library.
 **/

HYPRE_Int HYPRE_Finalize(void);

/**
 * (Optional) Returns true if the hypre library has been initialized but not finalized yet.
 **/

HYPRE_Int HYPRE_Initialized(void);

/**
 * (Optional) Returns true if the hypre library has been finalized but not re-initialized yet.
 **/

HYPRE_Int HYPRE_Finalized(void);

/*--------------------------------------------------------------------------
 * HYPRE error user functions
 *--------------------------------------------------------------------------*/


/* Return an aggregate error code representing the collective status of all ranks */
HYPRE_Int HYPRE_GetGlobalError(MPI_Comm comm);

/* Return the current hypre error flag */
HYPRE_Int HYPRE_GetError(void);

/* Check if the given error flag contains the given error code */
HYPRE_Int HYPRE_CheckError(HYPRE_Int hypre_ierr, HYPRE_Int hypre_error_code);

/* Return the index of the argument (counting from 1) where
   argument error (HYPRE_ERROR_ARG) has occured */
HYPRE_Int HYPRE_GetErrorArg(void);

/* Describe the given error flag in the given string */
void HYPRE_DescribeError(HYPRE_Int hypre_ierr, char *descr);

/* Clears the hypre error flag */
HYPRE_Int HYPRE_ClearAllErrors(void);

/* Clears the given error code from the hypre error flag */
HYPRE_Int HYPRE_ClearError(HYPRE_Int hypre_error_code);

/* Set behavior for printing errors: mode 0 = stderr, mode 1 = memory buffer */
HYPRE_Int HYPRE_SetPrintErrorMode(HYPRE_Int mode);

/* Return a buffer of error messages and clear them in hypre */
HYPRE_Int HYPRE_GetErrorMessages(char **buffer, HYPRE_Int *bufsz);

/* Print the error messages and clear them in hypre */
HYPRE_Int HYPRE_PrintErrorMessages(MPI_Comm comm);

/* Print GPU information */
HYPRE_Int HYPRE_PrintDeviceInfo(void);

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
HYPRE_Int HYPRE_AssumedPartitionCheck(void);

/*--------------------------------------------------------------------------
 * HYPRE memory location
 *--------------------------------------------------------------------------*/

typedef enum _HYPRE_MemoryLocation
{
   HYPRE_MEMORY_UNDEFINED = -1,
   HYPRE_MEMORY_HOST,
   HYPRE_MEMORY_DEVICE
} HYPRE_MemoryLocation;

/**
 * (Optional) Sets the default (abstract) memory location.
 **/

HYPRE_Int HYPRE_SetMemoryLocation(HYPRE_MemoryLocation memory_location);

/**
 * (Optional) Gets a pointer to the default (abstract) memory location.
 **/

HYPRE_Int HYPRE_GetMemoryLocation(HYPRE_MemoryLocation *memory_location);

#include <stdlib.h>

/*--------------------------------------------------------------------------
 * HYPRE execution policy
 *--------------------------------------------------------------------------*/

typedef enum _HYPRE_ExecutionPolicy
{
   HYPRE_EXEC_UNDEFINED = -1,
   HYPRE_EXEC_HOST,
   HYPRE_EXEC_DEVICE
} HYPRE_ExecutionPolicy;

/**
 * (Optional) Sets the default execution policy.
 **/

HYPRE_Int HYPRE_SetExecutionPolicy(HYPRE_ExecutionPolicy exec_policy);

/**
 * (Optional) Gets a pointer to the default execution policy.
 **/

HYPRE_Int HYPRE_GetExecutionPolicy(HYPRE_ExecutionPolicy *exec_policy);

/**
 * (Optional) Returns a string denoting the execution policy passed as input.
 **/

const char* HYPRE_GetExecutionPolicyName(HYPRE_ExecutionPolicy exec_policy);

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

HYPRE_Int HYPRE_SetGPUMemoryPoolSize(HYPRE_Int bin_growth, HYPRE_Int min_bin, HYPRE_Int max_bin,
                                     size_t max_cached_bytes);

/*--------------------------------------------------------------------------
 * HYPRE handle
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_SetSpTransUseVendor( HYPRE_Int use_vendor );
HYPRE_Int HYPRE_SetSpMVUseVendor( HYPRE_Int use_vendor );
/* Backwards compatibility with HYPRE_SetSpGemmUseCusparse() */
#define HYPRE_SetSpGemmUseCusparse(use_vendor) HYPRE_SetSpGemmUseVendor(use_vendor)
HYPRE_Int HYPRE_SetSpGemmUseVendor( HYPRE_Int use_vendor );
HYPRE_Int HYPRE_SetUseGpuRand( HYPRE_Int use_curand );
HYPRE_Int HYPRE_SetGpuAwareMPI( HYPRE_Int use_gpu_aware_mpi );

/*--------------------------------------------------------------------------
 * Base objects
 *--------------------------------------------------------------------------*/

/* RDF: How should we provide reference documentation for this (and above)? */

/* Base public solver struct */
struct hypre_Solver_struct;
typedef struct hypre_Solver_struct *HYPRE_Solver;

/* Base public matrix struct */
struct hypre_Matrix_struct;
typedef struct hypre_Matrix_struct *HYPRE_Matrix;

/* Base public vector struct */
struct hypre_Vector_struct;
typedef struct hypre_Vector_struct *HYPRE_Vector;

/* Base function pointers */

/* RDF: Note that only PtrToSolverFcn is needed at the user level right now to
 * keep backward compatibility with SetPrecond().  In general, do we want these
 * at the user level?  I guess it doesn't hurt. */

/* RDF: Also note that PtrToSolverFcn is defined again in 'HYPRE_krylov.h' and
 * probably needs to be for the reference manual. */

typedef HYPRE_Int (*HYPRE_PtrToSolverFcn)(HYPRE_Solver,
                                          HYPRE_Matrix,
                                          HYPRE_Vector,
                                          HYPRE_Vector);
typedef HYPRE_Int (*HYPRE_PtrToDestroyFcn)(HYPRE_Solver);

#ifdef __cplusplus
}
#endif

#endif
