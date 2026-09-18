/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Verify that HYPRE_Finalize safely returns before HYPRE_Initialize.
 *--------------------------------------------------------------------------*/

#include "HYPRE.h"
#include "_hypre_utilities.h"

hypre_int
main(hypre_int argc, char *argv[])
{
   HYPRE_Int my_id = 0;
   HYPRE_Int num_procs = 0;
   HYPRE_Int error = 0;
   HYPRE_Int global_error = 0;
   HYPRE_Int return_code = 0;

   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   /* Deliberately do not call HYPRE_Initialize(). */
   return_code = HYPRE_Finalize();

   if (return_code != 0)
   {
      ++error;
   }

   hypre_MPI_Allreduce(&error, &global_error, 1,
                       HYPRE_MPI_INT, hypre_MPI_MAX, hypre_MPI_COMM_WORLD);

   if (my_id == 0)
   {
      hypre_printf("Finalize before initialize (%d procs): %s\n",
                   num_procs, global_error ? "FAILED" : "PASSED");
   }

   hypre_MPI_Finalize();

   return global_error;
}
