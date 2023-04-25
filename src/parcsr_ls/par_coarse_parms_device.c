/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

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
hypre_BoomerAMGCoarseParmsDevice(MPI_Comm          comm,
                                 HYPRE_Int         local_num_variables,
                                 HYPRE_Int         num_functions,
                                 hypre_IntArray   *dof_func,
                                 hypre_IntArray   *CF_marker,
                                 hypre_IntArray  **coarse_dof_func_ptr,
                                 HYPRE_BigInt     *coarse_pnts_global)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_COARSE_PARAMS] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int      ierr = 0;
   HYPRE_BigInt   local_coarse_size = 0;

   /*--------------------------------------------------------------
    *----------------------------------------------------------------*/

#if defined(HYPRE_USING_SYCL)
   local_coarse_size = HYPRE_ONEDPL_CALL( std::count_if,
                                          hypre_IntArrayData(CF_marker),
                                          hypre_IntArrayData(CF_marker) + local_num_variables,
                                          equal<HYPRE_Int>(1) );
#else
   local_coarse_size = HYPRE_THRUST_CALL( count_if,
                                          hypre_IntArrayData(CF_marker),
                                          hypre_IntArrayData(CF_marker) + local_num_variables,
                                          equal<HYPRE_Int>(1) );
#endif

   if (num_functions > 1)
   {
      *coarse_dof_func_ptr = hypre_IntArrayCreate(local_coarse_size);
      hypre_IntArrayInitialize_v2(*coarse_dof_func_ptr, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      hypreSycl_copy_if( hypre_IntArrayData(dof_func),
                         hypre_IntArrayData(dof_func) + local_num_variables,
                         hypre_IntArrayData(CF_marker),
                         hypre_IntArrayData(*coarse_dof_func_ptr),
                         equal<HYPRE_Int>(1) );
#else
      HYPRE_THRUST_CALL( copy_if,
                         hypre_IntArrayData(dof_func),
                         hypre_IntArrayData(dof_func) + local_num_variables,
                         hypre_IntArrayData(CF_marker),
                         hypre_IntArrayData(*coarse_dof_func_ptr),
                         equal<HYPRE_Int>(1) );
#endif
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

   return (ierr);
}

HYPRE_Int
hypre_BoomerAMGInitDofFuncDevice( HYPRE_Int *dof_func,
                                  HYPRE_Int  local_size,
                                  HYPRE_Int  offset,
                                  HYPRE_Int  num_functions )
{
#if defined(HYPRE_USING_SYCL)
   hypreSycl_sequence(dof_func,
                      dof_func + local_size,
                      offset);

   HYPRE_ONEDPL_CALL( std::transform,
                      dof_func,
                      dof_func + local_size,
                      dof_func,
                      modulo<HYPRE_Int>(num_functions) );
#else
   HYPRE_THRUST_CALL( sequence,
                      dof_func,
                      dof_func + local_size,
                      offset,
                      1 );

   HYPRE_THRUST_CALL( transform,
                      dof_func,
                      dof_func + local_size,
                      dof_func,
                      modulo<HYPRE_Int>(num_functions) );
#endif

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)

