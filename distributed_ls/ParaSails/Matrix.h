/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matrix.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "mpi.h"
#include "Mem.h"
#include "Hash.h"

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

    int     matvec_setup;

    double *recvbuf;
    double *sendbuf;

    int     recvlen;

    int     sendlen;
    int    *sendind;

    int     num_recv;
    int     num_send;

    int    *global_to_local;
    int    *local_to_global;

    Hash   *hash_numbering;

    MPI_Request *recv_req;
    MPI_Request *send_req;
    MPI_Request *recv_req2;
    MPI_Request *send_req2;
    MPI_Status  *statuses;
}
Matrix;

#if 0 /* old */
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
}
Matrix;
#endif

Matrix *MatrixCreate(MPI_Comm comm, int beg_row, int end_row);
void MatrixDestroy(Matrix *mat);
void MatrixSetRow(Matrix *mat, int row, int len, int *ind, double *val);
void MatrixGetRow(Matrix *mat, int row, int *lenp, int **indp, double **valp);
int  MatrixRowPe(Matrix *mat, int row);
void MatrixPrint(Matrix *mat, char *filename);
void MatrixRead(Matrix *mat, char *filename);
void RhsRead(double *rhs, Matrix *mat, char *filename);
int  MatrixNnz(Matrix *mat);

void MatrixMatvec(Matrix *mat, double *x, double *y);
void MatrixMatvecTrans(Matrix *mat, double *x, double *y);
void MatrixMatvecComplete(Matrix *mat);

#endif /* _MATRIX_H */
