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

#if HYPRE_USING_PGCC_SMP
#define HYPRE_SMP_PRIVATE \
HYPRE_BOX_SMP_PRIVATE,hypre__nx,hypre__ny,hypre__nz,hypre__block
#include "../utilities/hypre_smp_forloop.h"
#else
#define HYPRE_SMP_PRIVATE \
HYPRE_BOX_SMP_PRIVATE,hypre__nx,hypre__ny,hypre__nz
#include "../utilities/hypre_smp_forloop.h"
#endif
#undef HYPRE_BOX_SMP_PRIVATE

