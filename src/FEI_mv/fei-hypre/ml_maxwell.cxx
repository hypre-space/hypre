/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

// *************************************************************************
// test program for HYPRE_LinSysCore
// *************************************************************************

//***************************************************************************
// system includes
//---------------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//**************************************************************************
// HYPRE includes
//---------------------------------------------------------------------------

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_LinSysCore.h"

#define HABS(x) ((x) > 0 ? (x) : -(x))

//**************************************************************************
// local defines and local and external functions
//---------------------------------------------------------------------------

void fei_hypre_test(int, char **);
void hypre_read_matrix(double **val, int **ia, int **ja, int *N, int *M,
                       char *matfile);
void hypre_read_rhs(double **val, int *N, char *rhsfile);

//***************************************************************************
// main program
//***************************************************************************

int main(int argc, char *argv[])
{
    fei_hypre_test(argc, argv);
}

//***************************************************************************
// a test program
//***************************************************************************

void fei_hypre_test(int argc, char *argv[])
{
    int    i, j, k, my_rank, num_procs, nrows, status;
    int    *ia, *ja, ncnt, index, ncols, iterations;
    int    *rowLengths, **colIndices;
    double *val, *rhs, *sol;
    char   tname[20], *paramString = new char[100];
    HYPRE_ParCSRMatrix G_csr;
    Data               data;

    //------------------------------------------------------------------
    // initialize parallel platform
    //------------------------------------------------------------------

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    HYPRE_LinSysCore H(MPI_COMM_WORLD);

    //------------------------------------------------------------------
    // read edge matrix
    //------------------------------------------------------------------

    hypre_read_matrix(&val,&ia,&ja,&nrows,&ncols,"Edge.ij");
    H.createMatricesAndVectors(nrows, 1, nrows);
    rowLengths = new int[nrows];
    colIndices = new int*[nrows];
    for (i = 0; i < nrows; i++)
    {
       ncnt = ia[i+1] - ia[i];
       rowLengths[i] = ncnt;
       colIndices[i] = new int[ncnt];
       k = 0;
       for (j = ia[i]; j < ia[i+1]; j++) colIndices[i][k++] = ja[j] + 1;
    }
    H.allocateMatrix(colIndices, rowLengths);
    delete [] rowLengths;
    for (i = 0; i < nrows; i++)
    {
       ncnt = ia[i+1] - ia[i];
       index = i + 1;
       H.sumIntoSystemMatrix(index, ncnt, &val[ia[i]], colIndices[i]);
    }
    H.matrixLoadComplete();
    for (i = 0; i < nrows; i++ ) delete [] colIndices[i];
    delete [] colIndices;
    delete [] ia;
    delete [] ja;
    delete [] val;

    //------------------------------------------------------------------
    // read gradient matrix
    //------------------------------------------------------------------

    HYPRE_LinSysCore G(MPI_COMM_WORLD);
    hypre_read_matrix(&val, &ia, &ja, &nrows, &ncols, "Grad.ij");
    if (nrows != ncols) G.HYPRE_LSC_SetColMap(0, ncols-1);
    G.createMatricesAndVectors(nrows, 1, nrows);
    rowLengths = new int[nrows];
    colIndices = new int*[nrows];
    for (i = 0; i < nrows; i++)
    {
       ncnt = ia[i+1] - ia[i];
       rowLengths[i] = ncnt;
       colIndices[i] = new int[ncnt];
       k = 0;
       for (j = ia[i]; j < ia[i+1]; j++) colIndices[i][k++] = ja[j] + 1;
    }
    G.allocateMatrix(colIndices, rowLengths);
    delete [] rowLengths;
    for (i = 0; i < nrows; i++)
    {
       ncnt = ia[i+1] - ia[i];
       index = i + 1;
       G.sumIntoSystemMatrix(index, ncnt, &val[ia[i]], colIndices[i]);
    }
    G.matrixLoadComplete();
    for (i = 0; i < nrows; i++ ) delete [] colIndices[i];
    delete [] colIndices;
    delete [] ia;
    delete [] ja;
    delete [] val;
    HYPRE_IJMatrixGetObject(G.HYA_, (void**) &G_csr);
    data.setDataPtr((void *) G_csr);
    strcpy(tname, "GEN");
    data.setTypeName(tname);
    H.copyInMatrix(1.0, data);
    G.HYA_ = NULL;

    //------------------------------------------------------------------
    // load the right hand side
    //------------------------------------------------------------------

    hypre_read_rhs(&rhs, &i, "rhs.ij");
    if (i < 0)
    {
       rhs = new double[nrows];
       for (i = 0; i < nrows; i++) rhs[i] = 1.0;
    }
    for (i = 0; i < nrows; i++)
    {
       H.sumIntoRHSVector(1, &rhs[i], &i);
    }
    delete [] rhs;

    //------------------------------------------------------------------
    // set other parameters
    //------------------------------------------------------------------

    strcpy(paramString, "solver gmres");
    H.parameters(1, &paramString);
    strcpy(paramString, "gmresDim 300");
    H.parameters(1, &paramString);
    strcpy(paramString, "relativeNorm");
    H.parameters(1, &paramString);
    strcpy(paramString, "tolerance 1.0e-12");
    H.parameters(1, &paramString);
    strcpy(paramString, "preconditioner boomeramg");
    strcpy(paramString, "preconditioner mlmaxwell");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlStrongThreshold 0.08");
    H.parameters(1, &paramString);

    //------------------------------------------------------------------
    // solve the system
    //------------------------------------------------------------------

    strcpy(paramString, "outputLevel 3");
    H.parameters(1, &paramString);
    H.launchSolver(status, iterations);
    sol = new double[nrows];
    H.getSolution(sol, nrows);
    for (i = 0; i < 10; i++)
       printf("Solution %6d = %16.8e\n", i, sol[i]);
    delete [] sol;

    //------------------------------------------------------------------
    // clean up
    //------------------------------------------------------------------

    MPI_Finalize();
}

#if 0
    HYPRE_IJVectorGetObject(H.HYx_, (void**) &x_csr);
    HYPRE_IJVectorGetObject(H.HYb_, (void**) &b_csr);
    HYPRE_IJMatrixGetObject(H.HYA_, (void**) &A_csr);
#endif

//***************************************************************************
// read a matrix
//***************************************************************************

void hypre_read_matrix(double **val, int **ia, int **ja, int *N, int *M,
                       char *matfile)
{
    int    i, nrows, ncols, nnz, icount, rowindex, colindex, curr_row;
    int    *mat_ia, *mat_ja;
    double *mat_a, value;
    FILE   *fp;

    /*------------------------------------------------------------------*/
    /* read matrix file                                                 */
    /*------------------------------------------------------------------*/

    printf("Reading matrix file = %s \n", matfile);
    fp = fopen(matfile, "r");
    if (fp == NULL)
    {
       printf("File not found = %s \n", matfile);
       exit(1);
    }
    fscanf(fp, "%d %d %d", &nnz, &nrows, &ncols);
    mat_ia = new int[nrows+1];
    mat_ja = new int[nnz];
    mat_a  = new double[nnz];
    mat_ia[0] = 0;

    curr_row = 0;
    icount   = 0;
    for (i = 0; i < nnz; i++)
    {
       fscanf(fp, "%d %d %lg", &rowindex, &colindex, &value);
       if (rowindex != curr_row) mat_ia[++curr_row] = icount;
       if (rowindex < 0 || rowindex >= nrows)
          printf("Error reading row %d (curr_row = %d)\n",rowindex,curr_row);
       if (colindex < 0 || colindex >= ncols)
          printf("Error reading col %d (rowindex = %d)\n",colindex,rowindex);
       if (HABS(value) > 1.0e-12)
       {
          mat_ja[icount] = colindex;
          mat_a[icount++] = value;
       }
    }
    fclose(fp);
    for (i = curr_row+1; i <= nrows; i++) mat_ia[i] = icount;
    (*val) = mat_a;
    (*ia)  = mat_ia;
    (*ja)  = mat_ja;
    (*N)   = nrows;
    (*M)   = ncols;
    printf("matrix has %6d rows and %7d nonzeros\n",nrows,mat_ia[nrows]);
    return;
}

//***************************************************************************
// read a right hand side
//***************************************************************************

void hypre_read_rhs(double **val, int *N, char *rhsfile)
{
    int    i, nrows, rowindex;
    double *rhs, value;
    FILE   *fp;

    /*------------------------------------------------------------------*/
    /* read matrix file                                                 */
    /*------------------------------------------------------------------*/

    printf("Reading rhs file = %s \n", rhsfile);
    fp = fopen(rhsfile, "r");
    if (fp == NULL)
    {
       printf("File not found = %s \n", rhsfile);
       (*N) = -1;
       return;
    }
    fscanf(fp, "%d", &nrows);
    rhs = new double[nrows];

    for (i = 0; i < nrows; i++)
    {
       fscanf(fp, "%d %lg", &rowindex, &value);
       if (rowindex < 0 || rowindex >= nrows)
          printf("Error reading row %d (curr_row = %d)\n",rowindex,i);
       rhs[i] = value;
    }
    fclose(fp);
    (*N)   = nrows;
    (*val) = rhs;
    return;
}


