/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#if defined(HYPRE_USE_GPU) && defined(HYPRE_USE_MANAGED)
//#define CUDAMEMATTACHTYPE cudaMemAttachGlobal
//#define CUDAMEMATTACHTYPE cudaMemAttachHost
#define HYPRE_GPU_USE_PINNED 1
#define HYPRE_USE_MANAGED_SCALABLE 1
#endif

