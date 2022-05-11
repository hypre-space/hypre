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
#include "utilities/utilities.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../parcsr_mv/HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_FEI_includes.h"
#include "HYPRE_LinSysCore.h"

//**************************************************************************
// local defines and local and external functions
//---------------------------------------------------------------------------

void fei_hypre_domaindecomposition(int, char **);
void fei_hypre_test(int, char **);

extern "C" {
int  HYPRE_LSI_DDAMGSolve(HYPRE_ParCSRMatrix A_csr, HYPRE_ParVector x_csr,
                         HYPRE_ParVector b_csr );
void HYPRE_LSI_Get_IJAMatrixFromFile(double **val, int **ia,
     int **ja, int *N, double **rhs, char *matfile, char *rhsfile);
}

//***************************************************************************
// main program
//***************************************************************************

main(int argc, char *argv[])
{
    fei_hypre_test(argc, argv);
}

//***************************************************************************
// a test program
//***************************************************************************

void fei_hypre_test(int argc, char *argv[])
{
    int    i, j, k, my_rank, num_procs, nrows, nnz, mybegin, myend, status;
    int    *ia, *ja, ncnt, index, chunksize, iterations, local_nrows;
    int    *rowLengths, **colIndices, blksize=1, *list, prec;
    double *val, *rhs, ddata, ddata_max;

    //------------------------------------------------------------------
    // initialize parallel platform
    //------------------------------------------------------------------

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    HYPRE_LinSysCore H(MPI_COMM_WORLD);

    //------------------------------------------------------------------
    // read the matrix and rhs and broadcast
    //------------------------------------------------------------------

    if ( my_rank == 0 ) {
       HYPRE_LSI_Get_IJAMatrixFromFile(&val, &ia, &ja, &nrows,
                                &rhs, "matrix.data", "rhs.data");
       nnz = ia[nrows];
       MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);

    } else {
       MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);
       ia  = new int[nrows+1];
       ja  = new int[nnz];
       val = new double[nnz];
       rhs = new double[nrows];

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    chunksize = nrows / blksize;
    if ( chunksize * blksize != nrows )
    {
       printf("Cannot put into matrix blocks with block size 3\n");
       exit(1);
    }
    chunksize = chunksize / num_procs;
    mybegin = chunksize * my_rank * blksize;
    myend   = chunksize * (my_rank + 1) * blksize - 1;
    if ( my_rank == num_procs-1 ) myend = nrows - 1;
    printf("Processor %d : begin/end = %d %d\n", my_rank, mybegin, myend);
    fflush(stdout);

    //------------------------------------------------------------------
    // create matrix in the HYPRE context
    //------------------------------------------------------------------

    local_nrows = myend - mybegin + 1;
    H.createMatricesAndVectors(nrows, mybegin+1, local_nrows);

    rowLengths = new int[local_nrows];
    colIndices = new int*[local_nrows];
    for ( i = mybegin; i < myend+1; i++ )
    {
       ncnt = ia[i+1] - ia[i];
       rowLengths[i-mybegin] = ncnt;
       colIndices[i-mybegin] = new int[ncnt];
       k = 0;
       for (j = ia[i]; j < ia[i+1]; j++) colIndices[i-mybegin][k++] = ja[j];
    }

    H.allocateMatrix(colIndices, rowLengths);

    for ( i = mybegin; i < myend+1; i++ ) delete [] colIndices[i-mybegin];
    delete [] colIndices;
    delete [] rowLengths;

    //------------------------------------------------------------------
    // load the matrix
    //------------------------------------------------------------------

    for ( i = mybegin; i <= myend; i++ ) {
       ncnt = ia[i+1] - ia[i];
       index = i + 1;
       H.sumIntoSystemMatrix(index, ncnt, &val[ia[i]], &ja[ia[i]]);
    }
    H.matrixLoadComplete();
    free( ia );
    free( ja );
    free( val );

    //------------------------------------------------------------------
    // load the right hand side
    //------------------------------------------------------------------

    for ( i = mybegin; i <= myend; i++ )
    {
       index = i;
       H.sumIntoRHSVector(1, &rhs[i], &index);
    }
    free( rhs );

    //------------------------------------------------------------------
    // set other parameters
    //------------------------------------------------------------------

    char *paramString = new char[100];

    strcpy(paramString, "version");
    H.parameters(1, &paramString);
    strcpy(paramString, "solver gmres");
    H.parameters(1, &paramString);
    strcpy(paramString, "relativeNorm");
    H.parameters(1, &paramString);
    strcpy(paramString, "tolerance 1.0e-6");
    H.parameters(1, &paramString);
    if ( my_rank == 0 )
    {
       printf("preconditioner (diagonal,parasails,boomeramg,ml,pilut,ddilut) : ");
       scanf("%d", &prec);
    }
    MPI_Bcast(&prec,  1, MPI_INT, 0, MPI_COMM_WORLD);
    switch (prec)
    {
       case 0 : strcpy(paramString, "preconditioner diagonal");
                break;
       case 1 : strcpy(paramString, "preconditioner parasails");
                break;
       case 2 : strcpy(paramString, "preconditioner boomeramg");
                break;
       case 3 : strcpy(paramString, "preconditioner ml");
                break;
       case 4 : strcpy(paramString, "preconditioner pilut");
                break;
       case 5 : strcpy(paramString, "preconditioner ddilut");
                break;
       default : strcpy(paramString, "preconditioner parasails");
                break;
    }

    H.parameters(1, &paramString);
    strcpy(paramString, "gmresDim 300");
    H.parameters(1, &paramString);

    strcpy(paramString, "ddilutFillin 0.0");
    H.parameters(1, &paramString);

    strcpy(paramString, "amgRelaxType hybrid");
    H.parameters(1, &paramString);
    strcpy(paramString, "amgRelaxWeight 0.5");
    H.parameters(1, &paramString);
    strcpy(paramString, "amgStrongThreshold 0.08");
    H.parameters(1, &paramString);
    strcpy(paramString, "amgNumSweeps 2");
    H.parameters(1, &paramString);

    strcpy(paramString, "mlNumPresweeps 2");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlNumPostsweeps 2");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlPresmootherType bgs");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlPostsmootherType bgs");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlRelaxWeight 0.5");
    H.parameters(1, &paramString);
    strcpy(paramString, "mlStrongThreshold 0.08");
    H.parameters(1, &paramString);

    strcpy(paramString, "parasailsNlevels 1");
    H.parameters(1, &paramString);
    strcpy(paramString, "parasailsThreshold 0.1");
    H.parameters(1, &paramString);

    //------------------------------------------------------------------
    // solve the system
    //------------------------------------------------------------------

    strcpy(paramString, "outputLevel 1");
    H.parameters(1, &paramString);
/*
    strcpy(paramString, "schurReduction");
    H.parameters(1, &paramString);
*/
    H.launchSolver(status, iterations);
/*
    strcpy(paramString, "preconditioner reuse");
    H.parameters(1, &paramString);
    ddata = 0.0;
    for ( i = H.localStartRow_; i <= H.localEndRow_; i++ )
    {
       H.putInitialGuess(&i, &ddata, 1);
    }
    H.launchSolver(status, iterations);
    ddata = 0.0;
    for ( i = H.localStartRow_; i <= H.localEndRow_; i++ )
    {
       H.putInitialGuess(&i, &ddata, 1);
    }
    H.launchSolver(status, iterations);
*/

    if ( status != 1 )
    {
       printf("%4d : HYPRE_LinSysCore : solve unsuccessful.\n", my_rank);
    }
    else if ( my_rank == 0 )
    {
       printf("HYPRE_LinSysCore : solve successful.\n", my_rank);
       printf("              iteration count = %4d\n", iterations);
    }

    if ( my_rank == 0 )
    {
       //for ( i = H.localStartRow_-1; i < H.localEndRow_; i++ )
       //{
       //   HYPRE_IJVectorGetLocalComponents(H.currX_,1,&i, NULL, &ddata);
       //   //H.getSolnEntry(i, ddata);
       //   printf("sol(%d): %e\n", i, ddata);
       //}
    }

    //------------------------------------------------------------------
    // clean up
    //------------------------------------------------------------------

    MPI_Finalize();
}

//***************************************************************************
// driver program for domain decomposition
//***************************************************************************

void fei_hypre_domaindecomposition(int argc, char *argv[])
{
    int                i, j, k, nrows, nnz, global_nrows;
    int                num_procs, status, rowCnt, relaxType[4];
    int                *ia, *ja, ncnt, index, chunksize, myRank;
    int                local_nrows, eqnNum, *rowLengths, **colIndices;
    int                blksize=1, *list, *colInd, *newColInd;
    int                rowSize, newRowSize, maxRowSize=0, num_iterations;
    int                myBegin, myEnd;
    double             *val, *rhs, *colVal, *newColVal, ddata;
    MPI_Comm           newComm, dummyComm;

    HYPRE_Solver       SeqPrecon;
    HYPRE_Solver       PSolver;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    r_csr;

    //******************************************************************
    // initialize parallel platform
    //------------------------------------------------------------------

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    HYPRE_LinSysCore H(MPI_COMM_WORLD);

    //******************************************************************
    // read the matrix and rhs and broadcast
    //------------------------------------------------------------------

    if ( myRank == 0 ) {
       HYPRE_LSI_Get_IJAMatrixFromFile(&val, &ia, &ja, &nrows,
                                &rhs, "matrix.data", "rhs.data");
       nnz = ia[nrows];
       MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);

    } else {
       MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);
       ia  = new int[nrows+1];
       ja  = new int[nnz];
       val = new double[nnz];
       rhs = new double[nrows];

       MPI_Bcast(ia,  nrows+1, MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(ja,  nnz,     MPI_INT,    0, MPI_COMM_WORLD);
       MPI_Bcast(val, nnz,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(rhs, nrows,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    chunksize = nrows / blksize;
    if ( chunksize * blksize != nrows )
    {
       printf("Cannot put into matrix blocks with block size 3\n");
       exit(1);
    }
    chunksize = chunksize / num_procs;
    myBegin = chunksize * myRank * blksize;
    myEnd   = chunksize * (myRank + 1) * blksize - 1;
    if ( myRank == num_procs-1 ) myEnd = nrows - 1;
    printf("Processor %d : begin/end = %d %d\n", myRank, myBegin, myEnd);
    fflush(stdout);

    //******************************************************************
    // create and load the global matrix in the HYPRE context
    //------------------------------------------------------------------

    local_nrows = myEnd - myBegin + 1;
    MPI_Allreduce(&local_nrows, &global_nrows,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    H.createMatricesAndVectors(nrows, myBegin+1, local_nrows);

    rowLengths = new int[local_nrows];
    colIndices = new int*[local_nrows];
    for ( i = myBegin; i < myEnd+1; i++ )
    {
       ncnt = ia[i+1] - ia[i];
       rowLengths[i-myBegin] = ncnt;
       colIndices[i-myBegin] = new int[ncnt];
       k = 0;
       for (j = ia[i]; j < ia[i+1]; j++) colIndices[i-myBegin][k++] = ja[j];
    }

    H.allocateMatrix(colIndices, rowLengths);

    for ( i = myBegin; i < myEnd+1; i++ ) delete [] colIndices[i-myBegin];
    delete [] colIndices;
    delete [] rowLengths;

    for ( i = myBegin; i <= myEnd; i++ )
    {
       ncnt = ia[i+1] - ia[i];
       index = i + 1;
       H.sumIntoSystemMatrix(index, ncnt, &val[ia[i]], &ja[ia[i]]);
    }
    H.matrixLoadComplete();
    free( ia );
    free( ja );
    free( val );

    //******************************************************************
    // load the right hand side
    //------------------------------------------------------------------

    for ( i = myBegin; i <= myEnd; i++ )
    {
       index = i + 1;
       H.sumIntoRHSVector(1, &rhs[i], &index);
    }
    free( rhs );

    //******************************************************************
    // call solver
    //------------------------------------------------------------------

    //---old_IJ---------------------------------------------------------
    //x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYx_);
    //b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYb_);
    //A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(H.HYA_);
    //---new_IJ---------------------------------------------------------
    HYPRE_IJVectorGetObject(H.HYx_, (void**) &x_csr);
    HYPRE_IJVectorGetObject(H.HYb_, (void**) &b_csr);
    HYPRE_IJMatrixGetObject(H.HYA_, (void**) &A_csr);
    //------------------------------------------------------------------

    HYPRE_LSI_DDAMGSolve(A_csr,x_csr,b_csr);

    MPI_Finalize();
}

