/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
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

    HYPRE_Int      beg_row;
    HYPRE_Int      end_row;
    HYPRE_Int     *beg_rows;
    HYPRE_Int     *end_rows;

    Mem     *mem;

    HYPRE_Int     *lens;
    HYPRE_Int    **inds;
    double **vals;

    HYPRE_Int     num_recv;
    HYPRE_Int     num_send;

    HYPRE_Int     sendlen;
    HYPRE_Int     recvlen;

    HYPRE_Int    *sendind;
    double *sendbuf;
    double *recvbuf;

    hypre_MPI_Request *recv_req;
    hypre_MPI_Request *send_req;
    hypre_MPI_Request *recv_req2;
    hypre_MPI_Request *send_req2;
    hypre_MPI_Status  *statuses;

    struct numbering *numb;
}
Matrix;

Matrix *MatrixCreate(MPI_Comm comm, HYPRE_Int beg_row, HYPRE_Int end_row);
Matrix *MatrixCreateLocal(HYPRE_Int beg_row, HYPRE_Int end_row);
void MatrixDestroy(Matrix *mat);
void MatrixSetRow(Matrix *mat, HYPRE_Int row, HYPRE_Int len, HYPRE_Int *ind, double *val);
void MatrixGetRow(Matrix *mat, HYPRE_Int row, HYPRE_Int *lenp, HYPRE_Int **indp, double **valp);
HYPRE_Int  MatrixRowPe(Matrix *mat, HYPRE_Int row);
void MatrixPrint(Matrix *mat, char *filename);
void MatrixRead(Matrix *mat, char *filename);
void RhsRead(double *rhs, Matrix *mat, char *filename);
HYPRE_Int  MatrixNnz(Matrix *mat);

void MatrixComplete(Matrix *mat);
void MatrixMatvec(Matrix *mat, double *x, double *y);
void MatrixMatvecSerial(Matrix *mat, double *x, double *y);
void MatrixMatvecTrans(Matrix *mat, double *x, double *y);

#endif /* _MATRIX_H */
