/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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
