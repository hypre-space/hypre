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

#ifndef __MLCHECK__
#define __MLCHECK__

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

#include "ml_common.h"

extern void ML_check_it(double sol[], double rhs[], ML *ml);

extern void ML_interp_check(ML *ml, int, int);

extern int  ML_Check(ML *ml);

extern int ML_Reitzinger_Check_Hierarchy(ML *ml, ML_Operator **Tmat_array, int incr_or_decr);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif
#endif

