
#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>
#include <assert.h>

#define dabs(x) ((x > 0 ) ? x : -(x))

//---------------------------------------------------------------------------
// parcsr_matrix_vector.h is put here instead of in HYPRE_LinSysCore.h 
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "utilities/utilities.h"
#include "base/Data.h"
#include "base/basicTypes.h"
#include "base/Utils.h"
#include "base/LinearSystemCore.h"
#include "HYPRE_LinSysCore.h"

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
    fei_hypre_domaindecomposition(argc, argv);
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

    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYx_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYb_);
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(H.HYA_);
    HYPRE_LSI_DDAMGSolve(A_csr,x_csr,b_csr);
 
    MPI_Finalize();
}

