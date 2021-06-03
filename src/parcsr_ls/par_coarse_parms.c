/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "_hypre_parcsr_ls.h"



/*==========================================================================*/
/*==========================================================================*/
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
  @param coarse_pnts_global_ptr [OUT]
  pointer to array which contains the number of the first coarse point on each  processor and the total number of coarse points in its last element

  @see */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCoarseParms(MPI_Comm       comm,
                           HYPRE_Int      local_num_variables,
                           HYPRE_Int      num_functions,
                           HYPRE_Int     *dof_func,
                           HYPRE_Int     *CF_marker,
                           HYPRE_Int    **coarse_dof_func_ptr,
                      	   HYPRE_BigInt **coarse_pnts_global_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_COARSE_PARAMS] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int      i;
   HYPRE_Int	  num_procs;
   HYPRE_BigInt   local_coarse_size = 0;

   HYPRE_Int	 *coarse_dof_func;
   HYPRE_BigInt	 *coarse_pnts_global;

   /*--------------------------------------------------------------
    *----------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm,&num_procs);

   for (i = 0; i < local_num_variables; i++)
   {
      if (CF_marker[i] == 1)
      {
         local_coarse_size++;
      }
   }
   if (num_functions > 1)
   {
      coarse_dof_func = hypre_CTAlloc(HYPRE_Int, local_coarse_size, HYPRE_MEMORY_HOST);

      local_coarse_size = 0;
      for (i = 0; i < local_num_variables; i++)
      {
         if (CF_marker[i] == 1)
         {
            coarse_dof_func[local_coarse_size++] = dof_func[i];
         }
      }

      *coarse_dof_func_ptr = coarse_dof_func;
   }

   {
      HYPRE_BigInt scan_recv;

      coarse_pnts_global = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
      hypre_MPI_Scan(&local_coarse_size, &scan_recv, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      /* first point in my range */
      coarse_pnts_global[0] = scan_recv - local_coarse_size;

      /* first point in next proc's range */
      coarse_pnts_global[1] = scan_recv;
   }

   if (*coarse_pnts_global_ptr)
   {
      hypre_TFree(*coarse_pnts_global_ptr, HYPRE_MEMORY_HOST);
   }
   *coarse_pnts_global_ptr = coarse_pnts_global;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_COARSE_PARAMS] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}
