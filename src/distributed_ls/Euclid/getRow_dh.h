/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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
 * $Revision: 2.3 $
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

