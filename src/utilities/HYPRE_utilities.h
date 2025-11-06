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

#ifdef HYPRE_MIXED_PRECISION
#include "_hypre_utilities_mup_def.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup Utilities Utilities
 *
 * Various utilities available in hypre.
 *
 * @{
 **/

/*===== BEGIN 1 - IGNORE CODE IN DOCS =====*/  /*! \cond */

/*--------------------------------------------------------------------------
 * BigInt and MixedInt
 *--------------------------------------------------------------------------*/

#include <limits.h>

#if defined(HYPRE_BIGINT)
typedef long long int HYPRE_BigInt;
typedef long long int HYPRE_Int;

#define HYPRE_BIG_INT_MAX LLONG_MAX
#define HYPRE_BIG_INT_MIN LLONG_MIN
#define HYPRE_INT_MAX LLONG_MAX
#define HYPRE_INT_MIN LLONG_MIN

#define HYPRE_MPI_BIG_INT MPI_LONG_LONG_INT
#define HYPRE_MPI_INT MPI_LONG_LONG_INT
#define HYPRE_INT_MAX LLONG_MAX
#define HYPRE_INT_MIN LLONG_MIN

#elif defined(HYPRE_MIXEDINT)
typedef long long int HYPRE_BigInt;
typedef int HYPRE_Int;

#define HYPRE_BIG_INT_MAX LLONG_MAX
#define HYPRE_BIG_INT_MIN LLONG_MIN
#define HYPRE_INT_MAX INT_MAX
#define HYPRE_INT_MIN INT_MIN

#define HYPRE_MPI_BIG_INT MPI_LONG_LONG_INT
#define HYPRE_MPI_INT MPI_INT
#define HYPRE_INT_MAX INT_MAX
#define HYPRE_INT_MIN INT_MIN

#else /* default */
typedef int HYPRE_BigInt;
typedef int HYPRE_Int;

#define HYPRE_BIG_INT_MAX INT_MAX
#define HYPRE_BIG_INT_MIN INT_MIN
#define HYPRE_INT_MAX INT_MAX
#define HYPRE_INT_MIN INT_MIN

#define HYPRE_MPI_BIG_INT MPI_INT
#define HYPRE_MPI_INT MPI_INT
#define HYPRE_INT_MAX INT_MAX
#define HYPRE_INT_MIN INT_MIN
#endif

static inline HYPRE_Int
hypre_IntSafeMult(HYPRE_Int a, HYPRE_Int b)
{
   if (a == 0 || b == 0)
   {
      return 0;
   }

   if (a > HYPRE_INT_MAX / b)
   {
      return HYPRE_INT_MAX;
   }

   return a * b;
}

static inline HYPRE_BigInt
hypre_BigIntSafeMult(HYPRE_BigInt a, HYPRE_BigInt b)
{
   if (a == 0 || b == 0)
   {
      return 0;
   }

   if (a > HYPRE_BIG_INT_MAX / b)
   {
      return HYPRE_BIG_INT_MAX;
   }

   return a * b;
}

/*--------------------------------------------------------------------------
 * Real and Complex Types
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

/* This allows us to avoid using 'float', 'double', and 'longdouble' in hypre.
 * NOTE: This must be defined here in the external header file for use in the
 * mixed precision code. */
typedef double       hypre_double;
typedef float        hypre_float;
typedef long double  hypre_long_double;

/*--------------------------------------------------------------------------
 * Sequential MPI stuff
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_SEQUENTIAL
typedef HYPRE_Int MPI_Comm;
#endif

/*--------------------------------------------------------------------------
 * HYPRE AP user functions
 *--------------------------------------------------------------------------*/

/* Checks whether the AP is on */
/* TODO (VPM): this function is provided for backwards compatibility
   and will be removed in a future release */
HYPRE_Int HYPRE_AssumedPartitionCheck(void);

/*===== END 1 - IGNORE CODE IN DOCS =====*/  /*! \endcond */

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Multiprecision
 *
 * @{
 **/

/* object precision options and API are available to users at all times */

/**
 * Available precisions.
 **/
typedef enum
{
   HYPRE_REAL_SINGLE,
   HYPRE_REAL_DOUBLE,
   HYPRE_REAL_LONGDOUBLE

} HYPRE_Precision;

/**
 * Set the global default runtime precision.
 **/
HYPRE_Int
HYPRE_SetGlobalPrecision(HYPRE_Precision precision);

/**
 * Get the global default runtime precision.
 **/
HYPRE_Int
HYPRE_GetGlobalPrecision(HYPRE_Precision *precision);

/**@}*/

/*===== BEGIN 2 - IGNORE CODE IN DOCS =====*/  /*! \cond */

/* RDF: This probably needs to be renamed to something like HYPRE_COMPILE_PRECISION */
#ifndef HYPRE_OBJECT_PRECISION
#if defined(HYPRE_SINGLE)
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_SINGLE
#elif defined(HYPRE_LONG_DOUBLE)
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_LONGDOUBLE
#else
#define HYPRE_OBJECT_PRECISION HYPRE_REAL_DOUBLE
#endif
#endif

#ifdef HYPRE_MIXED_PRECISION

#ifdef hypre_DEFINE_GLOBAL_MP
#define hypre_DEFINE_GLOBAL 1
#endif

#ifndef HYPRE_CURRENTPRECISION_FUNC
#if defined(HYPRE_SINGLE)
#define HYPRE_CURRENTPRECISION_FUNC(a) a##_flt
#elif defined(HYPRE_LONG_DOUBLE)
#define HYPRE_CURRENTPRECISION_FUNC(a) a##long_dbl
#else
#define HYPRE_CURRENTPRECISION_FUNC(a) a##_dbl
#endif
#endif

#endif

/* need to define this when not mixed precision */
#ifndef HYPRE_MIXED_PRECISION
#define hypre_DEFINE_GLOBAL 1
#endif

/*===== END 2 - IGNORE CODE IN DOCS =====*/  /*! \endcond */

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Error Codes
 *
 * @{
 **/

#define HYPRE_ERROR_GENERIC         1   /* generic error */
#define HYPRE_ERROR_MEMORY          2   /* unable to allocate memory */
#define HYPRE_ERROR_ARG             4   /* argument error */
/* bits 4-8 are reserved for the index of the argument error */
#define HYPRE_ERROR_CONV          256   /* method did not converge as expected */
#define HYPRE_MAX_FILE_NAME_LEN  1024   /* longest filename length used in hypre */
#define HYPRE_MAX_MSG_LEN        2048   /* longest message length */

/*--------------------------------------------------------------------------
 * HYPRE error user functions
 *--------------------------------------------------------------------------*/

/** Return an aggregate error code representing the collective status of all ranks **/
HYPRE_Int HYPRE_GetGlobalError(MPI_Comm comm);

/** Return the current hypre error flag **/
HYPRE_Int HYPRE_GetError(void);

/** Check if the given error flag contains the given error code **/
HYPRE_Int HYPRE_CheckError(HYPRE_Int hypre_ierr, HYPRE_Int hypre_error_code);

/**
 * Return the index of the argument (counting from 1) where argument error
 * (HYPRE_ERROR_ARG) has occured
 **/
HYPRE_Int HYPRE_GetErrorArg(void);

/** Describe the given error flag in the given string **/
void HYPRE_DescribeError(HYPRE_Int hypre_ierr, char *descr);

/** Clear the hypre error flag **/
HYPRE_Int HYPRE_ClearAllErrors(void);

/** Clear the given error code from the hypre error flag **/
HYPRE_Int HYPRE_ClearError(HYPRE_Int hypre_error_code);

/** Set behavior for printing errors: mode 0 = stderr, mode 1 = memory buffer **/
HYPRE_Int HYPRE_SetPrintErrorMode(HYPRE_Int mode);

/**
 * Set which error code messages to record for printing: code is an error code
 * such as HYPRE_ERROR_CONV, code -1 = all codes, verbosity 0 = do not record
 **/
HYPRE_Int HYPRE_SetPrintErrorVerbosity(HYPRE_Int code, HYPRE_Int verbosity);

/** Return a buffer of error messages and clear them in hypre **/
HYPRE_Int HYPRE_GetErrorMessages(char **buffer, HYPRE_Int *bufsz);

/** Print the error messages and clear them in hypre **/
HYPRE_Int HYPRE_PrintErrorMessages(MPI_Comm comm);

/** Clear the error messages in hypre and free any related memory allocated **/
HYPRE_Int HYPRE_ClearErrorMessages(void);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Initialize and Finalize
 *
 * @{
 **/

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

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Miscellaneous Information
 *
 * @{
 **/

/** Print GPU information **/
HYPRE_Int HYPRE_PrintDeviceInfo(void);

/**
 * @brief Prints the memory usage of the current process.
 *
 * This function prints the memory usage details of the process to standard output.
 * It provides information such as the virtual memory size, resident set size,
 * and other related statistics including GPU memory usage for device builds.
 *
 * @param[in] comm      The MPI communicator. This parameter allows the function
 *                      to print memory usage information for the process within
 *                      the context of an MPI program.
 *
 * @param[in] level     The level of detail in the memory statistics output.
 *                        - 1 : Display memory usage statistics for each MPI rank.
 *                        - 2 : Display aggregate memory usage statistics over MPI ranks.
 *
 * @param[in] function  The name of the function from which `HYPRE_MemoryPrintUsage`
 *                      is called. This is typically set to `__func__`, which
 *                      automatically captures the name of the calling function.
 *                      This variable can also be used to denote a region name.
 *
 * @param[in] line      The line number in the source file where `HYPRE_MemoryPrintUsage`
 *                      is called. This is typically set to `__LINE__`, which
 *                      automatically captures the line number. The line number can be
 *                      omitted by passing a negative value to this variable.
 *
 * @return              Returns an integer status code. `0` indicates success, while
 *                      a non-zero value indicates an error occurred.
 *
 * @note                The function is designed to be platform-independent but
 *                      may provide different levels of detail depending on the
 *                      underlying operating system (e.g., Linux, macOS). However,
 *                      this function does not lead to correct memory usage statistics
 *                      on Windows platforms.
 **/
HYPRE_Int HYPRE_MemoryPrintUsage(MPI_Comm comm, HYPRE_Int level, const char *function,
                                 HYPRE_Int line);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Library Version Information
 *
 * @{
 **/

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

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Memory Management
 *
 * @{
 **/

/** Memory location **/
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

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Execution Policy
 *
 * @{
 **/

/** Execution Policy **/
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

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Umpire and GPU Memory Pooling
 *
 * @{
 **/

/**
 * @brief Sets the size of the Umpire device memory pool.
 *
 * @param[in] nbytes The size of the device memory pool in bytes.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUmpireDevicePoolSize(size_t nbytes);

/**
 * @brief Sets the size of the Umpire unified memory pool.
 *
 * @param[in] nbytes The size of the unified memory pool in bytes.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUmpireUMPoolSize(size_t nbytes);

/**
 * @brief Sets the size of the Umpire host memory pool.
 *
 * @param[in] nbytes The size of the host memory pool in bytes.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUmpireHostPoolSize(size_t nbytes);

/**
 * @brief Sets the size of the Umpire pinned memory pool.
 *
 * @param[in] nbytes The size of the pinned memory pool in bytes.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUmpirePinnedPoolSize(size_t nbytes);

/**
 * @brief Sets the name of the Umpire device memory pool.
 *
 * @param[in] pool_name The name to assign to the device memory pool.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUmpireDevicePoolName(const char *pool_name);

/**
 * @brief Sets the name of the Umpire unified memory pool.
 *
 * @param[in] pool_name The name to assign to the unified memory pool.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUmpireUMPoolName(const char *pool_name);

/**
 * @brief Sets the name of the Umpire host memory pool.
 *
 * @param[in] pool_name The name to assign to the host memory pool.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUmpireHostPoolName(const char *pool_name);

/**
 * @brief Sets the name of the Umpire pinned memory pool.
 *
 * @param[in] pool_name The name to assign to the pinned memory pool.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUmpirePinnedPoolName(const char *pool_name);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Miscellaneous
 *
 * @{
 **/

/**
 * Sets the logging level for the HYPRE library.
 *
 * The following options are available for \e log_level:
 *
 *    - 0 : (default) No messaging.
 *    - 1 : Display memory usage statistics for each MPI rank.
 *    - 2 : Display aggregate memory usage statistics over MPI ranks.
 *
 * @note Log level codes can be combined using bitwise OR to enable multiple
 *       logging behaviors simultaneously.
 *
 * @param log_level The logging level to set.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetLogLevel(HYPRE_Int log_level);

/**
 * Specifies the algorithm used for sparse matrix transposition in device builds.
 *
 * The following options are available for \e use_vendor:
 *
 *    - 0 : Use hypre's internal implementation.
 *    - 1 : (default) Use the vendor library's implementation. This includes:
 *          - cuSPARSE for CUDA (HYPRE_USING_CUSPARSE)
 *          - rocSPARSE for HIP (HYPRE_USING_ROCSPARSE)
 *          - oneMKL for SYCL   (HYPRE_USING_ONEMKLSPARSE)
 *
 * @param use_vendor Indicates whether to use the internal or vendor-provided implementation.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetSpTransUseVendor(HYPRE_Int use_vendor);

/**
 * Specifies the algorithm used for sparse matrix/vector multiplication in device builds.
 *
 * The following options are available for \e use_vendor:
 *
 *    - 0 : Use hypre's internal implementation.
 *    - 1 : (default) Use the vendor library's implementation. This includes:
 *          - cuSPARSE for CUDA (HYPRE_USING_CUSPARSE)
 *          - rocSPARSE for HIP (HYPRE_USING_ROCSPARSE)
 *          - oneMKL for SYCL   (HYPRE_USING_ONEMKLSPARSE)
 *
 * @param use_vendor Indicates whether to use the internal or vendor-provided implementation.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetSpMVUseVendor(HYPRE_Int use_vendor);

/**
 * Specifies the algorithm used for sparse matrix/matrix multiplication in device builds.
 *
 * The following options are available for \e use_vendor:
 *
 *    - 0 : Use hypre's internal implementation.
 *    - 1 : Use the vendor library's implementation. This includes:
 *          - cuSPARSE for CUDA (HYPRE_USING_CUSPARSE)
 *          - rocSPARSE for HIP (HYPRE_USING_ROCSPARSE)
 *          - oneMKL for SYCL   (HYPRE_USING_ONEMKLSPARSE)
 *
 * @param use_vendor Indicates whether to use the internal or vendor-provided implementation.
 *
 * @note The default value is 1, except for CUDA builds, which is zero.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetSpGemmUseVendor( HYPRE_Int use_vendor );

/* Backwards compatibility with HYPRE_SetSpGemmUseCusparse() */
#define HYPRE_SetSpGemmUseCusparse(use_vendor) HYPRE_SetSpGemmUseVendor(use_vendor)

/**
 * Specifies the algorithm used for generating random numbers in device builds.
 *
 * The following options are available for \e use_curand:
 *
 *    - 0 : random numbers are generated on the host and copied to device memory.
 *    - 1 : (default) Use the vendor library's implementation. This includes:
 *          - cuSPARSE for CUDA (HYPRE_USING_CUSPARSE)
 *          - rocSPARSE for HIP (HYPRE_USING_ROCSPARSE)
 *          - oneMKL for SYCL   (HYPRE_USING_ONEMKLSPARSE)
 *
 * @param use_curand Indicates whether to use the vendor-provided implementation or not.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetUseGpuRand( HYPRE_Int use_curand );

/**
 * Configures the usage of GPU-aware MPI for communication in device builds.
 *
 * The following options are available for \e use_gpu_aware_mpi:
 *
 *    - 0 : MPI buffers are transferred between device and host memory. Communication occurs on the host.
 *    - 1 : MPI communication is performed directly from the device using device-resident buffers.
 *
 * @param use_gpu_aware_mpi Specifies whether to enable GPU-aware MPI communication or not.
 *
 * @note This option requires hypre to be configured with GPU-aware MPI support for it to take effect.
 *
 * @return Returns hypre's global error code, where 0 indicates success.
 **/
HYPRE_Int HYPRE_SetGpuAwareMPI( HYPRE_Int use_gpu_aware_mpi );

/**@}*/
/**@}*/

/*===== BEGIN 3 - IGNORE CODE IN DOCS =====*/  /*! \cond */

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

/*===== END 3 - IGNORE CODE IN DOCS =====*/  /*! \endcond */

#ifdef __cplusplus
}
#endif

#ifdef HYPRE_MIXED_PRECISION
/* The following is for user compiles and the order is important.  The first
 * header ensures that we do not change prototype names in user files or in the
 * second header file.  The second header contains all the prototypes needed by
 * users for mixed precision. */
#ifndef hypre_MP_BUILD
#include "_hypre_utilities_mup_undef.h"
#include "HYPRE_utilities_mup.h"
#endif
#endif

#endif
