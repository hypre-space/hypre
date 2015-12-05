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




#ifndef ML_AGG_MIN_ENERGY
#define ML_AGG_MIN_ENERGY

#include "ml_common.h"
#include "ml_include.h"

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" 
{
#endif
#endif

int ML_AGG_Gen_Prolongator_MinEnergy(ML *ml,int level, int clevel, void *data);
int ML_AGG_Gen_Restriction_MinEnergy(ML *ml,int level, int clevel, void *data);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif
#endif
