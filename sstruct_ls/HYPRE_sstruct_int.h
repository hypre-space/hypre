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




#ifndef HYPRE_PARCSR_INTERFACE_INTERPRETER
#define HYPRE_PARCSR_INTERFACE_INTERPRETER

#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "_hypre_sstruct_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

int
hypre_SStructPVectorSetRandomValues( hypre_SStructPVector *pvector, int seed);

int
hypre_SStructVectorSetRandomValues( hypre_SStructVector *vector, int seed);

int
hypre_SStructSetRandomValues( void *v, int seed);

int
HYPRE_SStructSetupInterpreter( mv_InterfaceInterpreter *i );

int
HYPRE_SStructSetupMatvec(HYPRE_MatvecFunctions * mv);

#ifdef __cplusplus
}
#endif

#endif
