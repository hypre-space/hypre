/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ERROR_HEADER
#define hypre_ERROR_HEADER

#include <assert.h>

/*--------------------------------------------------------------------------
 * Global variable used in hypre error checking
 *--------------------------------------------------------------------------*/

extern HYPRE_Int hypre__global_error;
#define hypre_error_flag  hypre__global_error

/*--------------------------------------------------------------------------
 * HYPRE error macros
 *--------------------------------------------------------------------------*/

void hypre_error_handler(const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg);

#define hypre_error(IERR)  hypre_error_handler(__FILE__, __LINE__, IERR, NULL)
#define hypre_error_w_msg(IERR, msg)  hypre_error_handler(__FILE__, __LINE__, IERR, msg)
#define hypre_error_in_arg(IARG)  hypre_error(HYPRE_ERROR_ARG | IARG<<3)

#if defined(HYPRE_DEBUG)
/* host assert */
#define hypre_assert(EX) do { if (!(EX)) { fprintf(stderr, "[%s, %d] hypre_assert failed: %s\n", __FILE__, __LINE__, #EX); hypre_error(1); assert(0); } } while (0)
/* device assert */
#if defined(HYPRE_USING_CUDA)
#define hypre_device_assert(EX) assert(EX)
#elif defined(HYPRE_USING_HIP)
/* FIXME: Currently, asserts in device kernels in HIP do not behave well */
#define hypre_device_assert(EX)
#endif
#else /* #ifdef HYPRE_DEBUG */
/* this is to silence compiler's unused variable warnings */
#ifdef __cplusplus
#define hypre_assert(EX) do { if (0) { static_cast<void> (EX); } } while (0)
#else
#define hypre_assert(EX) do { if (0) { (void) (EX); } } while (0)
#endif
#define hypre_device_assert(EX)
#endif

#endif /* hypre_ERROR_HEADER */

