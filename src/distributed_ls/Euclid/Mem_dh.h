/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




#ifndef MEM_DH_DH
#define MEM_DH_DH

#include "euclid_common.h"

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
