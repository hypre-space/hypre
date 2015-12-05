/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




#ifndef HYPRE_PARCSR_INTERFACE_INTERPRETER
#define HYPRE_PARCSR_INTERFACE_INTERPRETER

#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "_hypre_struct_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

int
hypre_StructVectorSetRandomValues( hypre_StructVector *vector, int seed);

int
hypre_StructSetRandomValues( void *v, int seed);

int
HYPRE_StructSetupInterpreter( mv_InterfaceInterpreter *i );

int
HYPRE_StructSetupMatvec(HYPRE_MatvecFunctions * mv);


#ifdef __cplusplus
}
#endif

#endif
