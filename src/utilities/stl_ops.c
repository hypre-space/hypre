/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#include <math.h>

/*--------------------------------------------------------------------
 * hypre_IntSequence
 *
 * Generate a linear sequence of integers from 0 to size-1 and store
 * them in the provided data array.
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_IntSequence(HYPRE_MemoryLocation  memory_location,
                  HYPRE_Int             size,
                  HYPRE_Int            *data)
{
#if !defined (HYPRE_USING_GPU)
   HYPRE_UNUSED_VAR(memory_location);
#endif

   HYPRE_Int   i;

#if defined (HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_sequence(data, data + size, 0);
#else
      HYPRE_THRUST_CALL(sequence, data, data + size);
#endif
   }
   else
#endif
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         data[i] = i;
      }
   }

   return hypre_error_flag;
}
