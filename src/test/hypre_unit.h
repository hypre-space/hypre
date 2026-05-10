/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_TEST_UNIT_HEADER
#define hypre_TEST_UNIT_HEADER

#include "_hypre_utilities.h"

#define HYPRE_UNIT_PRINT_TEST_RESULT(my_id, error, comm)                  \
   do                                                                     \
   {                                                                      \
      HYPRE_Int hypre_unit_global_error = 0;                              \
      hypre_MPI_Allreduce(&(error), &hypre_unit_global_error, 1,          \
                          HYPRE_MPI_INT, hypre_MPI_MAX, comm);           \
      if ((my_id) == 0)                                                   \
      {                                                                   \
         hypre_printf("%s\n", hypre_unit_global_error ? "FAILED" : "PASSED"); \
      }                                                                   \
      (error) = hypre_unit_global_error;                                  \
   } while (0)

#define HYPRE_UNIT_CHECK_INT(error, name, got, expected)                  \
   do                                                                     \
   {                                                                      \
      if ((got) != (expected))                                            \
      {                                                                   \
         hypre_printf("  %s mismatch: got %d expected %d\n",              \
                      (name), (HYPRE_Int) (got), (HYPRE_Int) (expected)); \
         (error) = 1;                                                     \
      }                                                                   \
   } while (0)

#define HYPRE_UNIT_CHECK_BIGINT(error, name, got, expected)               \
   do                                                                     \
   {                                                                      \
      if ((got) != (expected))                                            \
      {                                                                   \
         hypre_printf("  %s mismatch: got %lld expected %lld\n",          \
                      (name), (long long) (got), (long long) (expected)); \
         (error) = 1;                                                     \
      }                                                                   \
   } while (0)

#define HYPRE_UNIT_CHECK_REAL(error, name, got, expected, tol)            \
   do                                                                     \
   {                                                                      \
      HYPRE_Real hypre_unit_diff = hypre_abs((got) - (expected));         \
      if (hypre_unit_diff > (tol))                                        \
      {                                                                   \
         hypre_printf("  %s mismatch: got %.17e expected %.17e\n",        \
                      (name), (double) (got), (double) (expected));       \
         (error) = 1;                                                     \
      }                                                                   \
   } while (0)

#endif /* hypre_TEST_UNIT_HEADER */
