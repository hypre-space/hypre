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




/******************************************************************************
 *
 * OrderStat.h header file.
 *
 *****************************************************************************/

#ifndef _ORDERSTAT_H
#define _ORDERSTAT_H

#include "_hypre_utilities.h"

double randomized_select(double *a, HYPRE_Int p, HYPRE_Int r, HYPRE_Int i);
void shell_sort(const HYPRE_Int n, HYPRE_Int x[]);

#endif /* _ORDERSTAT_H */
