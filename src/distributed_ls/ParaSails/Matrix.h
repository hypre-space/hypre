/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

#ifdef __cplusplus
extern "C" {
#endif
	
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
    HYPRE_Real **vals;

    HYPRE_Int     num_recv;
    HYPRE_Int     num_send;

    HYPRE_Int     sendlen;
    HYPRE_Int     recvlen;

    HYPRE_Int    *sendind;
    HYPRE_Real *sendbuf;
    HYPRE_Real *recvbuf;

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
void MatrixSetRow(Matrix *mat, HYPRE_Int row, HYPRE_Int len, HYPRE_Int *ind, HYPRE_Real *val);
void MatrixGetRow(Matrix *mat, HYPRE_Int row, HYPRE_Int *lenp, HYPRE_Int **indp, HYPRE_Real **valp);
HYPRE_Int  MatrixRowPe(Matrix *mat, HYPRE_Int row);
void MatrixPrint(Matrix *mat, char *filename);
void MatrixRead(Matrix *mat, char *filename);
void RhsRead(HYPRE_Real *rhs, Matrix *mat, char *filename);
HYPRE_Int  MatrixNnz(Matrix *mat);

void MatrixComplete(Matrix *mat);
void MatrixMatvec(Matrix *mat, HYPRE_Real *x, HYPRE_Real *y);
void MatrixMatvecSerial(Matrix *mat, HYPRE_Real *x, HYPRE_Real *y);
void MatrixMatvecTrans(Matrix *mat, HYPRE_Real *x, HYPRE_Real *y);

#ifdef __cplusplus
}
#endif

#endif /* _MATRIX_H */
