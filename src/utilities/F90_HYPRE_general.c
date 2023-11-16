/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

void
hypre_F90_IFACE(hypre_init, HYPRE_INIT)
(hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_Initialize();
}

void
hypre_F90_IFACE(hypre_initialize, HYPRE_INITIALIZE)
(hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_Initialize();
}

void
hypre_F90_IFACE(hypre_finalize, HYPRE_FINALIZE)
(hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_Finalize();
}

void
hypre_F90_IFACE(hypre_setmemorylocation, HYPRE_SETMEMORYLOCATION)
(hypre_F90_Int *memory_location, hypre_F90_Int *ierr)
{
   HYPRE_MemoryLocation loc = (HYPRE_MemoryLocation) * memory_location;
   *ierr = (hypre_F90_Int) HYPRE_SetMemoryLocation(loc);
}

void
hypre_F90_IFACE(hypre_setexecutionpolicy, HYPRE_SETEXECUTIONPOLICY)
(hypre_F90_Int *exec_policy, hypre_F90_Int *ierr)
{
   HYPRE_ExecutionPolicy exec = (HYPRE_ExecutionPolicy) * exec_policy;

   *ierr = (hypre_F90_Int) HYPRE_SetExecutionPolicy(exec);
}

void
hypre_F90_IFACE(hypre_setspgemmusevendor, HYPRE_SETSPGEMMUSEVENDOR)
(hypre_F90_Int *use_vendor, hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SetSpGemmUseVendor(*use_vendor);
}

#ifdef __cplusplus
}
#endif
