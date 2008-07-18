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




#ifndef GET_ROW_DH
#define GET_ROW_DH

#include "euclid_common.h"

/* "row" refers to global row number */

extern void EuclidGetDimensions(void *A, int *beg_row, int *rowsLocal, int *rowsGlobal);
extern void EuclidGetRow(void *A, int row, int *len, int **ind, double **val);
extern void EuclidRestoreRow(void *A, int row, int *len, int **ind, double **val);

extern int EuclidReadLocalNz(void *A);

extern void PrintMatUsingGetRow(void* A, int beg_row, int m,
                          int *n2o_row, int *n2o_col, char *filename);


#endif

