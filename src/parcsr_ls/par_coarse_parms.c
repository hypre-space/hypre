/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/**
  Generates global coarse_size and dof_func for next coarser level

  Notes:
  \begin{itemize}
  \item The routine returns the following:
  \begin{itemize}
  \item an integer array containing the
  function values for the local coarse points
  \item the global number of coarse points
  \end{itemize}
  \end{itemize}

  {\bf Input files:}
  _hypre_parcsr_ls.h

  @return Error code.

  @param comm [IN]
  MPI Communicator
  @param local_num_variables [IN]
  number of points on local processor
  @param dof_func [IN]
  array that contains the function numbers for all local points
  @param CF_marker [IN]
  marker array for coarse points
  @param coarse_dof_func_ptr [OUT]
  pointer to array which contains the function numbers for local coarse points
  @param coarse_pnts_global [OUT]
  pointer to array which contains the number of the first coarse point on each  processor and the total number of coarse points in its last element

  @see */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCoarseParmsHost(MPI_Comm         comm,
                               HYPRE_Int        local_num_variables,
                               HYPRE_Int        num_functions,
                               hypre_IntArray  *dof_func,
                               hypre_IntArray  *CF_marker,
                               hypre_IntArray **coarse_dof_func_ptr,
                               HYPRE_BigInt    *coarse_pnts_global)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_COARSE_PARAMS] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int     i;
   HYPRE_BigInt  local_coarse_size = 0;
   HYPRE_Int    *coarse_dof_func;

   /*--------------------------------------------------------------
    *----------------------------------------------------------------*/

   for (i = 0; i < local_num_variables; i++)
   {
      if (hypre_IntArrayData(CF_marker)[i] == 1)
      {
         local_coarse_size++;
      }
   }

   if (num_functions > 1)
   {
      *coarse_dof_func_ptr = hypre_IntArrayCreate(local_coarse_size);
      hypre_IntArrayInitialize(*coarse_dof_func_ptr);
      coarse_dof_func = hypre_IntArrayData(*coarse_dof_func_ptr);

      local_coarse_size = 0;
      for (i = 0; i < local_num_variables; i++)
      {
         if (hypre_IntArrayData(CF_marker)[i] == 1)
         {
            coarse_dof_func[local_coarse_size++] = hypre_IntArrayData(dof_func)[i];
         }
      }
   }

   {
      HYPRE_BigInt scan_recv;
      hypre_MPI_Scan(&local_coarse_size, &scan_recv, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      /* first point in my range */
      coarse_pnts_global[0] = scan_recv - local_coarse_size;

      /* first point in next proc's range */
      coarse_pnts_global[1] = scan_recv;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_COARSE_PARAMS] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGCoarseParms(MPI_Comm         comm,
                           HYPRE_Int        local_num_variables,
                           HYPRE_Int        num_functions,
                           hypre_IntArray  *dof_func,
                           hypre_IntArray  *CF_marker,
                           hypre_IntArray **coarse_dof_func_ptr,
                           HYPRE_BigInt    *coarse_pnts_global)
{
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec;

   if (num_functions > 1)
   {
      exec = hypre_GetExecPolicy2(hypre_IntArrayMemoryLocation(CF_marker),
                                  hypre_IntArrayMemoryLocation(dof_func));
   }
   else
   {
      exec = hypre_GetExecPolicy1(hypre_IntArrayMemoryLocation(CF_marker));
   }

   if (exec == HYPRE_EXEC_DEVICE)
   {
      return hypre_BoomerAMGCoarseParmsDevice(comm, local_num_variables, num_functions, dof_func,
                                              CF_marker, coarse_dof_func_ptr, coarse_pnts_global);
   }
   else
#endif
   {
      return hypre_BoomerAMGCoarseParmsHost(comm, local_num_variables, num_functions, dof_func,
                                            CF_marker, coarse_dof_func_ptr, coarse_pnts_global);
   }
}
