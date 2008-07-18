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


#ifdef __cplusplus
extern "C" {
#endif

int
hypre_ParCSRMultiVectorPrint( void* x_, const char* fileName );

void*
hypre_ParCSRMultiVectorRead( MPI_Comm comm, void* ii_, const char* fileName );

int
HYPRE_ParCSRSetupInterpreter( mv_InterfaceInterpreter *i );

int
HYPRE_ParCSRSetupMatvec(HYPRE_MatvecFunctions * mv);

#ifdef __cplusplus
}
#endif

#endif
