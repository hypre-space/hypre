/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Matrix.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Common.h"
#include "Mem.h"

#ifndef _MATRIX_H
#define _MATRIX_H

typedef struct
{
    MPI_Comm comm;

    int      beg_row;
    int      end_row;
    int     *beg_rows;
    int     *end_rows;

    Mem     *mem;

    int     *lens;
    int    **inds;
    double **vals;

    int     num_recv;
    int     num_send;

    int     sendlen;
    int     recvlen;

    int    *sendind;
    double *sendbuf;
    double *recvbuf;

    MPI_Request *recv_req;
    MPI_Request *send_req;
    MPI_Request *recv_req2;
    MPI_Request *send_req2;
    MPI_Status  *statuses;

    struct numbering *numb;
}
Matrix;

Matrix *MatrixCreate(MPI_Comm comm, int beg_row, int end_row);
Matrix *MatrixCreateLocal(int beg_row, int end_row);
void MatrixDestroy(Matrix *mat);
void MatrixSetRow(Matrix *mat, int row, int len, int *ind, double *val);
void MatrixGetRow(Matrix *mat, int row, int *lenp, int **indp, double **valp);
int  MatrixRowPe(Matrix *mat, int row);
void MatrixPrint(Matrix *mat, char *filename);
void MatrixRead(Matrix *mat, char *filename);
void RhsRead(double *rhs, Matrix *mat, char *filename);
int  MatrixNnz(Matrix *mat);

void MatrixComplete(Matrix *mat);
void MatrixMatvec(Matrix *mat, double *x, double *y);
void MatrixMatvecSerial(Matrix *mat, double *x, double *y);
void MatrixMatvecTrans(Matrix *mat, double *x, double *y);

#endif /* _MATRIX_H */
