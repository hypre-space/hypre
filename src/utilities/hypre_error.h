/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ERROR_HEADER
#define hypre_ERROR_HEADER

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
#ifdef NDEBUG
#define hypre_assert(EX)
#else
#define hypre_assert(EX) if (!(EX)) {hypre_fprintf(stderr,"hypre_assert failed: %s\n", #EX); hypre_error(1);}
#endif

#endif

