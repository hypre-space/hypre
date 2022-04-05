/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef MEM_DH_DH
#define MEM_DH_DH

/* #include "euclid_common.h" */

extern void  Mem_dhCreate(Mem_dh *m);
extern void  Mem_dhDestroy(Mem_dh m);

extern void *Mem_dhMalloc(Mem_dh m, size_t size);
extern void  Mem_dhFree(Mem_dh m, void *ptr);
  /* preceeding two are called via the macros
     MALLOC_DH and FREE_DH; see "euclid_config.h"
   */

extern void  Mem_dhPrint(Mem_dh m, FILE* fp, bool allPrint);
  /* prints memory usage statistics;  "allPrint" is only
     meaningful when running in MPI mode.
   */

#endif
