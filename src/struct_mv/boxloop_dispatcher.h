/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_BOXLOOP_DISPATCHER_HEADER
#define HYPRE_BOXLOOP_DISPATCHER_HEADER

#define CONCAT2(x, y) x ## y
#define XCONCAT2(x, y) CONCAT2(x, y)

#define HYPRE_BOXLOOP(BEGIN, BEGINARGS, BODY, END, ENDARGS)                                  \
{                                                                                            \
   const HYPRE_MemoryLocation memory_location = hypre_HandleMemoryLocation(hypre_handle());  \
   const HYPRE_ExecutionPolicy exec_policy = hypre_GetExecPolicy1(memory_location);          \
   if (exec_policy == HYPRE_EXEC_HOST)                                                       \
   {                                                                                         \
      XCONCAT2(BEGIN, Host) BEGINARGS BODY XCONCAT2(END, Host) ENDARGS                       \
   }                                                                                         \
   else if (exec_policy == HYPRE_EXEC_DEVICE)                                                \
   {                                                                                         \
      XCONCAT2(BEGIN, Device) BEGINARGS BODY XCONCAT2(END, Device) ENDARGS                   \
   }                                                                                         \
}

#endif

