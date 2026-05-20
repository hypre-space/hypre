/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_TEST_UNIT_HEADER
#define hypre_TEST_UNIT_HEADER

#include "_hypre_utilities.h"

#define HYPRE_UNIT_TEST_SETUP(comm, min_procs, test_comm, test_my_id)     \
   do                                                                     \
   {                                                                      \
      HYPRE_Int hypre_unit_my_id;                                         \
      HYPRE_Int hypre_unit_num_procs;                                     \
      hypre_MPI_Comm_rank((comm), &hypre_unit_my_id);                     \
      hypre_MPI_Comm_size((comm), &hypre_unit_num_procs);                 \
      if (hypre_unit_num_procs >= (min_procs))                            \
      {                                                                   \
         HYPRE_Int hypre_unit_part =                                     \
            (hypre_unit_my_id < (min_procs)) ? 1 : hypre_MPI_UNDEFINED;   \
         hypre_MPI_Comm_split((comm), hypre_unit_part, hypre_unit_my_id,  \
                              &(test_comm));                              \
      }                                                                   \
      else                                                                \
      {                                                                   \
         if (hypre_unit_my_id == 0)                                       \
         {                                                                \
            hypre_printf("%s: Skipping (requires at least %d processors)\n", \
                         __func__, (HYPRE_Int) (min_procs));              \
         }                                                                \
         hypre_MPI_Barrier((comm));                                       \
         return 0;                                                        \
      }                                                                   \
      if ((test_comm) == hypre_MPI_COMM_NULL)                             \
      {                                                                   \
         hypre_MPI_Barrier((comm));                                       \
         return 0;                                                        \
      }                                                                   \
      hypre_MPI_Comm_rank((test_comm), &(test_my_id));                    \
      if ((test_my_id) == 0)                                              \
      {                                                                   \
         hypre_printf("%s (%d procs): ", __func__,                       \
                      (HYPRE_Int) (min_procs));                           \
      }                                                                   \
   } while (0)

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
         hypre_printf("  %s mismatch: got %b expected %b\n",              \
                      (name), (HYPRE_BigInt) (got),                       \
                      (HYPRE_BigInt) (expected));                         \
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
                      (name), (HYPRE_Real) (got),                         \
                      (HYPRE_Real) (expected));                           \
         (error) = 1;                                                     \
      }                                                                   \
   } while (0)

#endif /* hypre_TEST_UNIT_HEADER */
