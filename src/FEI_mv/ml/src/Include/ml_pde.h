/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/




/* ******************************************************************** */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */        
/* ******************************************************************** */

/* ******************************************************************** */
/* Declaration of the ML_PDE functions                                  */
/* ******************************************************************** */
/* Author        : Charles Tong (LLNL)                                  */
/* Date          : February, 2000                                       */
/* ******************************************************************** */

#ifndef _MLPDE__
#define _MLPDE__

#include "ml_common.h"
#include "mli_solver.h"

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

extern int ML_PDE_GenMat(MLI_Solver*,int);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif

