/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * OrderStat.h header file.
 *
 *****************************************************************************/

#ifndef _ORDERSTAT_H
#define _ORDERSTAT_H

double randomized_select(double *a, int p, int r, int i);
void shell_sort(const int n, int x[]);

#endif /* _ORDERSTAT_H */
