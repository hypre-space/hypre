/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>
#include <assert.h>

//---------------------------------------------------------------------------
// parcsr_matrix_vector.h is put here instead of in HYPRE_LinSysCore.h 
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "parcsr_matrix_vector/parcsr_matrix_vector.h"

#include "utilities/utilities.h"
#include "base/Data.h"
#include "base/basicTypes.h"
#include "base/Utils.h"
#include "base/LinearSystemCore.h"
#include "HYPRE_LinSysCore.h"

//***************************************************************************
//***************************************************************************
// This section investigates the use of domain decomposition preconditioner
// using AMG.  
//***************************************************************************
//***************************************************************************

HYPRE_IJMatrix localA;
HYPRE_IJVector localb;
HYPRE_IJVector localx;
int            myBegin, myEnd, myRank;
int            interior_nrows, *offRowLengths;
int            **offColInd;
int            *remap_array;
double         **offColVal;
MPI_Comm       parComm;      

//***************************************************************************
// Compute y = E^T A E x where A is the global matrix and x and y are
// global vectors
//***************************************************************************

int HYPRE_Precondition( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A_csr,
                        HYPRE_ParVector x_csr,
                        HYPRE_ParVector y_csr )
{
   int                i, j, index, local_nrows, *temp_list, global_nrows;
   hypre_ParVector    *x_par;
   hypre_ParVector    *y_par;
   hypre_Vector       *x_par_local;
   hypre_Vector       *y_par_local;
   double             *x_par_data;
   double             *y_par_data ;
   double             *temp_vect;
   double             *y2_data;
   hypre_ParVector    *Lx_par;
   hypre_Vector       *Lx_local;
   double             *Lx_data;
   HYPRE_ParCSRMatrix LA_csr;
   HYPRE_ParVector    Lx_csr;
   HYPRE_ParVector    Lb_csr, tv_csr;
   HYPRE_IJVector     tv;
   hypre_ParVector    *tv_par;
   hypre_Vector       *tv_par_local;
   double             *tv_par_data;
   double             localSum;

   // -----------------------------------------------------------------
   // initialization and fetch double arrays for b and x (global)
   // -----------------------------------------------------------------

   local_nrows = myEnd - myBegin + 1;
   MPI_Allreduce(&local_nrows, &global_nrows,1,MPI_INT,MPI_SUM,parComm);
   x_par       = (hypre_ParVector *) x_csr;
   x_par_local = hypre_ParVectorLocalVector(x_par);
   x_par_data  = hypre_VectorData(x_par_local);
   y_par       = (hypre_ParVector *) y_csr;
   y_par_local = hypre_ParVectorLocalVector(y_par);
   y_par_data  = hypre_VectorData(y_par_local);
   HYPRE_IJVectorCreate(parComm, &tv, global_nrows);
   HYPRE_IJVectorSetLocalStorageType(tv, HYPRE_PARCSR);
   HYPRE_IJVectorSetLocalPartitioning(tv,myBegin,myEnd+1);
   HYPRE_IJVectorAssemble(tv);
   HYPRE_IJVectorInitialize(tv);
   HYPRE_IJVectorZeroLocalComponents(tv);
   tv_csr       = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(tv);
   tv_par       = (hypre_ParVector *) tv_csr;
   tv_par_local = hypre_ParVectorLocalVector(tv_par);
   tv_par_data  = hypre_VectorData(tv_par_local);

   // -----------------------------------------------------------------
   // apply E^T to x => tv
   // - tv(int) = x(int) 
   // - solve interior using AMG with tv(int) as rhs
   // - adjust tv(border) based on interior solution
   // -----------------------------------------------------------------

   for (i=0; i<local_nrows; i++) tv_par_data[i] = x_par_data[i];

   temp_list = new int[interior_nrows];
   temp_vect = new double[interior_nrows];
   for (i=0; i<interior_nrows; i++) temp_list[i] = i;
   for (i=0; i<local_nrows; i++) 
   {
      if (remap_array[i] >= 0 && remap_array[i] < interior_nrows) 
         temp_vect[remap_array[i]] = x_par_data[i];
   }
   HYPRE_IJVectorSetLocalComponents(localb,interior_nrows,temp_list,
                                    NULL,temp_vect);
   HYPRE_IJVectorZeroLocalComponents(localx);
   delete [] temp_list;
   delete [] temp_vect;

   LA_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(localA);
   Lx_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localx);
   Lb_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localb);
   HYPRE_ParAMGSolve( solver, LA_csr, Lb_csr, Lx_csr );

   Lx_par   = (hypre_ParVector *) Lx_csr;
   Lx_local = hypre_ParVectorLocalVector(Lx_par);
   Lx_data  = hypre_VectorData(Lx_local);
   for (i=0; i<local_nrows; i++) 
   {
      if ( remap_array[i] >= 0 ) 
      {
         for (j=0; j<offRowLengths[i]; j++) 
         {
            index = offColInd[i][j];
            tv_par_data[index] -= (Lx_data[remap_array[i]] * offColVal[i][j]);
         }
         //tv_par_data[i] = Lx_data[remap_array[i]];
      }
   }

   // -----------------------------------------------------------------
   // apply E to tv => y
   // - y(border) = tv(border)
   // - solve inteior using the border data 
   // - y(int) = x(int) + E * x(border)
   //   + x(border) => right hand side of interior system
   //   + E * x(border) : apply AMG
   //   + y(int) = x(int) + E * x(border)
   // -----------------------------------------------------------------
  
   for (i=0; i<local_nrows; i++) y_par_data[i] = tv_par_data[i];

   temp_list = new int[interior_nrows];
   temp_vect = new double[interior_nrows];
   for (i=0; i<interior_nrows; i++) temp_list[i] = i;
   for (i=0; i<local_nrows; i++) 
   {
      if ( remap_array[i] >= 0 && remap_array[i] < interior_nrows) 
      {
         temp_vect[remap_array[i]] = 0.0;
         for (j=0; j<offRowLengths[i]; j++) 
            temp_vect[remap_array[i]] += 
               (offColVal[i][j]*tv_par_data[offColInd[i][j]]);
      } else if ( remap_array[i] >= interior_nrows) 
        printf("WARNING : index out of range.\n");
   }

   HYPRE_IJVectorSetLocalComponents(localb,interior_nrows,temp_list,
                                    NULL,temp_vect);
   HYPRE_IJVectorZeroLocalComponents(localx);
   delete [] temp_list;
   delete [] temp_vect;

   LA_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(localA);
   Lx_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localx);
   Lb_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localb);
   HYPRE_ParAMGSolve( solver, LA_csr, Lb_csr, Lx_csr );

   Lx_par   = (hypre_ParVector *) Lx_csr;
   Lx_local = hypre_ParVectorLocalVector(Lx_par);
   Lx_data  = hypre_VectorData(Lx_local);
   for (i=0; i<local_nrows; i++) 
   {
      if (remap_array[i] >= 0) y_par_data[i] -= Lx_data[remap_array[i]];
   }

   return 0;
}

//***************************************************************************
// perform 1 V-cycle of AMG locally
//***************************************************************************
int HYPRE_SeqAMGSolve( HYPRE_Solver solver,
                       HYPRE_ParCSRMatrix A_csr,
                       HYPRE_ParVector b_csr,
                       HYPRE_ParVector x_csr )
{
   int                i, j, local_nrows, *indlist, global_nrows;
   hypre_ParVector    *b_par;
   hypre_ParVector    *x_par;
   hypre_Vector       *x_par_local;
   double             *x_par_data;
   hypre_Vector       *b_par_local;
   double             *b_par_data ;
   double             *b2_data;
   double             *x2_data;
   double             rnorm, initnorm;
   HYPRE_ParCSRMatrix LA_csr;
   HYPRE_ParVector    Lx_csr;
   HYPRE_ParVector    Lb_csr, tv_csr;
   HYPRE_IJVector     tv;

   // -----------------------------------------------------------------
   // initialization and fetch double arrays for b and x (global)
   // -----------------------------------------------------------------

   local_nrows = myEnd - myBegin + 1;
   MPI_Allreduce(&local_nrows, &global_nrows,1,MPI_INT,MPI_SUM,parComm);
   x_par       = (hypre_ParVector *) x_csr;
   x_par_local = hypre_ParVectorLocalVector(x_par);
   x_par_data  = hypre_VectorData(x_par_local);
   b_par       = (hypre_ParVector *) b_csr;
   b_par_local = hypre_ParVectorLocalVector(b_par);
   b_par_data  = hypre_VectorData(b_par_local);

   // -----------------------------------------------------------------
   // copy b and x to local array and use local AMG solve 
   // -----------------------------------------------------------------
  
   indlist = new int[interior_nrows];
   b2_data  = new double[interior_nrows];
   x2_data  = new double[interior_nrows];
   for (i=0; i<interior_nrows; i++) indlist[i] = i;
   for (i=0; i<local_nrows; i++) 
   {
      if ( remap_array[i] < 0 ) x_par_data[i] = b_par_data[i];
   }
   for (i=0; i<local_nrows; i++) 
   {
      if ( remap_array[i] >= 0 ) 
      {
         b2_data[remap_array[i]] = b_par_data[i];
         for (j=0; j<offRowLengths[i]; j++) 
            b2_data[remap_array[i]] -= 
               (offColVal[i][j]*x_par_data[offColInd[i][j]]);
      }
   }
   HYPRE_IJVectorSetLocalComponents(localb,interior_nrows,indlist,NULL,b2_data);
   for (i=0; i<local_nrows; i++) 
      if ( remap_array[i] >= 0 ) x2_data[remap_array[i]] = x_par_data[i];
   HYPRE_IJVectorSetLocalComponents(localx,interior_nrows,indlist,NULL,x2_data);
   LA_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(localA);
   Lx_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localx);
   Lb_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localb);
   HYPRE_ParAMGSolve( solver, LA_csr, Lb_csr, Lx_csr );

   // -----------------------------------------------------------------
   // put the local result back to global array
   // -----------------------------------------------------------------

   hypre_ParVector    *u = (hypre_ParVector *) Lx_csr;
   hypre_Vector       *u_local = hypre_ParVectorLocalVector(u);
   double             *u_data  = hypre_VectorData(u_local);
   for (i=0; i<local_nrows; i++) 
      if ( remap_array[i] >= 0 ) x_par_data[i] = u_data[remap_array[i]];
   delete [] indlist;
   delete [] b2_data;
   delete [] x2_data;

   // -----------------------------------------------------------------
   // put the local result back to global array
   // -----------------------------------------------------------------

   //HYPRE_IJVectorCreate(parComm, &tv, global_nrows);
   //HYPRE_IJVectorSetLocalStorageType(tv, HYPRE_PARCSR);
   //HYPRE_IJVectorSetLocalPartitioning(tv,myBegin,myEnd+1);
   //HYPRE_IJVectorAssemble(tv);
   //HYPRE_IJVectorInitialize(tv);
   //tv_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(tv);
   //HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, x_csr, 0.0, tv_csr );
   //HYPRE_ParAMGSolve( solver, LA_csr, tv_csr, Lx_csr );
   //for (i=0; i<local_nrows; i++) x_data[i] = u_data[i];

   int    rowSize, *colInd;
   double *colVal;
   if ( myRank == 10 )
   {
      for ( i = 0; i < local_nrows; i++ )
      {
         HYPRE_ParCSRMatrixGetRow(LA_csr,i,&rowSize,&colInd,&colVal);
         for ( j = 0; j < rowSize; j++ )
            printf("A(%d,%d) = %e;\n", i+1, colInd[j]+1, colVal[j]);
         HYPRE_ParCSRMatrixRestoreRow(LA_csr,i,&rowSize,&colInd,&colVal);
      }
      for ( i = 0; i < local_nrows; i++ )
      {
         printf("x(%d) = %e;\n", i+1, u_data[i]);
      }
      for ( i = 0; i < local_nrows; i++ )
      {
         printf("b(%d) = %e;\n", i+1, b_par_data[i]);
      }
   }
   return 0;
}

//***************************************************************************
//***************************************************************************

void fei_hypre_dd(int argc, char *argv[])
{
    int                i, j, k, k1, k2, nrows, nnz, global_nrows;
    int                num_procs, status;
    int                *ia, *ja, ncnt, index, chunksize, iterations;
    int                local_nrows, eqnNum, *rowLengths, **colIndices;
    int                blksize=1, *list, *colInd, *newColInd;
    int                rowSize, newRowSize, maxRowSize=0;
    double             *val, *rhs, ddata, ddata_max, *colVal, *newColVal;
    MPI_Comm           newComm, dummyComm;

    int                its, rowCnt;
    double             initnorm, rnorm, rho, rhom1, beta, sigma, alpha;
    HYPRE_IJVector     pvec, apvec, tv, zvec;
    HYPRE_Solver       SeqPrecon;
    HYPRE_Solver       PSolver;
    HYPRE_ParCSRMatrix A_csr;
    HYPRE_ParVector    x_csr;
    HYPRE_ParVector    b_csr;
    HYPRE_ParVector    r_csr;
    HYPRE_ParVector    p_csr;
    HYPRE_ParVector    z_csr;
    HYPRE_ParVector    ap_csr;
    HYPRE_ParVector    tv_csr;
    hypre_ParVector    *x_par;
    hypre_ParVector    *r_par;
    hypre_ParVector    *p_par;
    hypre_ParVector    *ap_par;
    hypre_ParVector    *tv_par;
    hypre_ParVector    *z_par;

    int                maxiter = 500;
    double             offdiag_norm, *alpha_array;
    double             *rnorm_array, **Tmat, init_offdiag_norm;
    double             app, aqq, arr, ass, apq, sign, tau, t, c, s;

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
       H.HYFEI_Get_IJAMatrixFromFile(&val, &ia, &ja, &nrows,
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
    delete [] ia;
    delete [] ja;
    delete [] val;
    
    //******************************************************************
    // load the right hand side 
    //------------------------------------------------------------------

    for ( i = myBegin; i <= myEnd; i++ ) 
    {
       index = i + 1;
       H.sumIntoRHSVector(1, &rhs[i], &index);
    }
    delete [] rhs;

    //******************************************************************
    // create and load a local matrix 
    //------------------------------------------------------------------

    local_nrows = myEnd - myBegin + 1;
    for ( i = 0; i < num_procs; i++ )
    {
       if ( myRank == i )
          MPI_Comm_split(MPI_COMM_WORLD, i+1, 0, &newComm);
       else
          MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 1, &dummyComm);
    }
    MPI_Comm_rank(newComm, &i);
    MPI_Comm_size(newComm, &j);
    parComm = MPI_COMM_WORLD;

    //------------------------------------------------------------------
    // find out how many rows are interior rows
    //------------------------------------------------------------------


    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(H.HYA_);
    remap_array = new int[local_nrows];
    for ( i = 0; i < local_nrows; i++ ) remap_array[i] = 0;
    for ( i = myBegin; i <= myEnd; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       for ( j = 0; j < rowSize; j++ )
          if ( colInd[j] < myBegin || colInd[j] > myEnd ) 
             {remap_array[i-myBegin] = -1; break;}
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }
    interior_nrows = 0;
    for ( i = 0; i < local_nrows; i++ ) 
       if ( remap_array[i] == 0 ) remap_array[i] = interior_nrows++;

    //------------------------------------------------------------------
    // construct the local matrix (only the border nodes)
    //------------------------------------------------------------------

    HYPRE_IJMatrixCreate(newComm,&localA,interior_nrows,interior_nrows);
    HYPRE_IJMatrixSetLocalStorageType(localA, HYPRE_PARCSR);
    HYPRE_IJMatrixSetLocalSize(localA, interior_nrows, interior_nrows);
    rowLengths = new int[interior_nrows];
    offRowLengths = new int[local_nrows];
    rowCnt = 0;
    for ( i = myBegin; i <= myEnd; i++ )
    {
       offRowLengths[i-myBegin] = 0;
       if ( remap_array[i-myBegin] >= 0 )
       {
          rowLengths[rowCnt] = 0;
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          for ( j = 0; j < rowSize; j++ )
          {
             if ( colInd[j] >= myBegin && colInd[j] <= myEnd ) 
             {
                if (remap_array[colInd[j]-myBegin] >= 0) rowLengths[rowCnt]++;
                else offRowLengths[i-myBegin]++;
             }
          }
          nnz += rowLengths[rowCnt];
          maxRowSize = (rowLengths[rowCnt] > maxRowSize) ? 
                        rowLengths[rowCnt] : maxRowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          rowCnt++;
       }
    }
    HYPRE_IJMatrixSetRowSizes(localA, rowLengths);
    HYPRE_IJMatrixInitialize(localA);
    newColInd = new int[maxRowSize];
    newColVal = new double[maxRowSize];
    rowCnt = 0;
    offColInd = new int*[local_nrows];
    offColVal = new double*[local_nrows];
    for ( i = 0; i < local_nrows; i++ )
    {
       if ( offRowLengths[i] > 0 )
       {
          offColInd[i] = new int[offRowLengths[i]];
          offColVal[i] = new double[offRowLengths[i]];
       }
       else
       {
          offColInd[i] = NULL;
          offColVal[i] = NULL;
       }
    }
    for ( i = 0; i < local_nrows; i++ )
    {
       eqnNum = myBegin + i;
       if  ( remap_array[i] >= 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,eqnNum,&rowSize,&colInd,&colVal);
          nnz = 0;
          k = 0;
          for ( j = 0; j < rowSize; j++ )
          {
             if ( colInd[j] >= myBegin && colInd[j] <= myEnd ) 
             {
                if ( remap_array[colInd[j]-myBegin] >= 0 )
                {
                   newColInd[nnz] = remap_array[colInd[j]-myBegin];
                   newColVal[nnz++] = colVal[j];
                }
                else
                {
                   offColInd[i][k] = colInd[j]-myBegin;
                   offColVal[i][k++] = colVal[j];
                }
             }
          }
          if ( k != offRowLengths[i] )
             printf("WARNING : k != offRowLengths[i]\n");
          HYPRE_ParCSRMatrixRestoreRow(A_csr,eqnNum,&rowSize,&colInd,&colVal);
          HYPRE_IJMatrixInsertRow(localA,nnz,rowCnt,newColInd,newColVal);
          rowCnt++;
       }
    }
    delete [] newColInd;
    delete [] newColVal;
    HYPRE_IJMatrixAssemble(localA);

    //******************************************************************
    // create and load local vectors 
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(newComm, &localx, interior_nrows);
    HYPRE_IJVectorSetLocalStorageType(localx, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(localx, 0, interior_nrows);
    HYPRE_IJVectorAssemble(localx);
    HYPRE_IJVectorInitialize(localx);
    HYPRE_IJVectorZeroLocalComponents(localx);
    HYPRE_IJVectorCreate(newComm, &localb, interior_nrows);
    HYPRE_IJVectorSetLocalStorageType(localb, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(localb, 0, interior_nrows);
    HYPRE_IJVectorAssemble(localb);
    HYPRE_IJVectorInitialize(localb);
    HYPRE_IJVectorZeroLocalComponents(localb);

    //******************************************************************
    // create an AMG context
    //------------------------------------------------------------------

    HYPRE_ParAMGCreate(&SeqPrecon);
    HYPRE_ParAMGSetMaxIter(SeqPrecon, 1);
    HYPRE_ParAMGSetCycleType(SeqPrecon, 1);
    HYPRE_ParAMGSetMaxLevels(SeqPrecon, 25);
    HYPRE_ParAMGSetTol(SeqPrecon, 1.0E-16);
    HYPRE_ParAMGSetMeasureType(SeqPrecon, 0);
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(localA);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localb);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(localb);
    //HYPRE_ParAMGSetIOutDat(SeqPrecon, 2);
    //HYPRE_ParAMGSetDebugFlag(SeqPrecon, 1);
    HYPRE_ParAMGSetup( SeqPrecon, A_csr, b_csr, x_csr);
    MPI_Barrier(MPI_COMM_WORLD);

    //******************************************************************
    // solve using CG 
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(parComm, &pvec, global_nrows);
    HYPRE_IJVectorSetLocalStorageType(pvec, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(pvec,myBegin,myEnd+1);
    HYPRE_IJVectorAssemble(pvec);
    HYPRE_IJVectorInitialize(pvec);
    HYPRE_IJVectorZeroLocalComponents(pvec);
    HYPRE_IJVectorCreate(parComm, &apvec, global_nrows);
    HYPRE_IJVectorSetLocalStorageType(apvec, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(apvec,myBegin,myEnd+1);
    HYPRE_IJVectorAssemble(apvec);
    HYPRE_IJVectorInitialize(apvec);
    HYPRE_IJVectorZeroLocalComponents(apvec);
    HYPRE_IJVectorCreate(parComm, &zvec, global_nrows);
    HYPRE_IJVectorSetLocalStorageType(zvec, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(zvec,myBegin,myEnd+1);
    HYPRE_IJVectorAssemble(zvec);
    HYPRE_IJVectorInitialize(zvec);
    HYPRE_IJVectorZeroLocalComponents(zvec);
    HYPRE_IJVectorCreate(parComm, &tv, global_nrows);
    HYPRE_IJVectorSetLocalStorageType(tv, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(tv,myBegin,myEnd+1);
    HYPRE_IJVectorAssemble(tv);
    HYPRE_IJVectorInitialize(tv);
    HYPRE_IJVectorZeroLocalComponents(tv);
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(H.HYA_);
    x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYx_);
    b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYb_);
    r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(H.HYr_);
    p_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(pvec);
    ap_csr = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(apvec);
    z_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(zvec);
    tv_csr = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(tv);
    x_par  = (hypre_ParVector *)  x_csr;
    p_par  = (hypre_ParVector *)  p_csr;
    tv_par = (hypre_ParVector *)  tv_csr;
    ap_par = (hypre_ParVector *)  ap_csr;
    r_par  = (hypre_ParVector *)  r_csr;
    z_par  = (hypre_ParVector *)  z_csr;
    HYPRE_ParVectorCopy( b_csr, r_csr );
    HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
    HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
    initnorm = rnorm = sqrt( rnorm );
    if ( myRank == 0 )
       printf("CG with DDAMG preconditioner initial norm = %e\n", initnorm);

    alpha_array = new double[maxiter+1];
    rnorm_array = new double[maxiter+1];
    Tmat = new double*[maxiter+1];
    for ( i = 0; i <= maxiter; i++ )
    {
       Tmat[i] = new double[maxiter+1];
       for ( j = 0; j <= maxiter; j++ ) Tmat[i][j] = 0.0;
       Tmat[i][i] = 1.0;
    }

    its = 0;
    rnorm_array[0] = initnorm;
    while ( rnorm / initnorm > 1.0E-8 && its < maxiter )
    {
       //HYPRE_ParVectorCopy( r_csr, z_csr );
       HYPRE_Precondition( SeqPrecon, A_csr, r_csr, z_csr);
       HYPRE_ParVectorInnerProd( r_csr, z_csr, &rho);
       if ( its == 0 ) beta = 0.0;
       else
       {
          beta = rho / rhom1;
          Tmat[its-1][its] = -beta;
       }
       HYPRE_ParVectorCopy( p_csr, tv_csr );
       HYPRE_ParVectorCopy( z_csr, p_csr );
       hypre_ParVectorAxpy(beta, tv_par, p_par);
       HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, p_csr, 0.0, ap_csr );
       //HYPRE_DD_Matvec( SeqPrecon, A_csr, p_csr, ap_csr );
       HYPRE_ParVectorInnerProd( p_csr, ap_csr, &sigma);
       alpha = rho / sigma;
       alpha_array[its] = sigma;
       hypre_ParVectorAxpy(alpha, p_par, x_par);
       hypre_ParVectorAxpy(-alpha, ap_par, r_par);
       HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       rhom1 = rho;
       rnorm = sqrt( rnorm );
       its++;
       rnorm_array[its] = rnorm;
       if ( myRank == 0 )
          printf("Iteration %4d : rnorm = %e\n", its, rnorm);
    }

    // -----------------------------------------------------------------
    // construct T
    // -----------------------------------------------------------------

    /*
    maxiter = its;
    Tmat[0][0] = alpha_array[0];
    for ( i = 1; i < maxiter; i++ )
    {
       Tmat[i][i]=alpha_array[i]+alpha_array[i-1]*Tmat[i-1][i]*Tmat[i-1][i];
    }
    for ( i = 0; i < maxiter; i++ )
    {
       Tmat[i][i+1] *= alpha_array[i];
       Tmat[i+1][i] = Tmat[i][i+1];
       rnorm_array[i] = 1.0 / rnorm_array[i];
    }
    for ( i = 0; i < maxiter; i++ )
    {
       k1 = i - 1;
       if ( k1 < 0 ) k1 = 0;
       k2 = i + 1;
       if ( k2 >= maxiter ) k2 = maxiter;
       for ( j = 0; j < maxiter; j++ )
          Tmat[i][j] = Tmat[i][j] * rnorm_array[i] * rnorm_array[j];
    }
    */

    // -----------------------------------------------------------------
    // diagonalize T using Jacobi iteration
    // -----------------------------------------------------------------

    /*
    offdiag_norm = 0.0;
    for ( i = 0; i < maxiter; i++ )
       for ( j = 0; j < i; j++ ) offdiag_norm += (Tmat[i][j] * Tmat[i][j]);
    offdiag_norm *= 2.0;
    init_offdiag_norm = offdiag_norm;

    while ( offdiag_norm > init_offdiag_norm * 1.0E-8 )
    {
       for ( i = 1; i < maxiter; i++ )
       {
          for ( j = 0; j < i; j++ )
          {
             apq = Tmat[i][j];
             if ( apq != 0.0 )
             {
                app = Tmat[j][j];
                aqq = Tmat[i][i];
                tau = ( aqq - app ) / (2.0 * apq);
                sign = (tau >= 0.0) ? 1.0 : -1.0;
                t  = sign / (tau * sign + sqrt(1.0 + tau * tau));
                c  = 1.0 / sqrt( 1.0 + t * t );
                s  = t * c;
                for ( k = 0; k < maxiter; k++ )
                {
                   arr = Tmat[j][k];
                   ass = Tmat[i][k];
                   Tmat[j][k] = c * arr - s * ass;
                   Tmat[i][k] = s * arr + c * ass;
                }
                for ( k = 0; k < maxiter; k++ )
                {
                   arr = Tmat[k][j];
                   ass = Tmat[k][i];
                   Tmat[k][j] = c * arr - s * ass;
                   Tmat[k][i] = s * arr + c * ass;
                }
             }
          }
       }
       offdiag_norm = 0.0;
       for ( i = 0; i < maxiter; i++ )
          for ( j = 0; j < i; j++ ) offdiag_norm += (Tmat[i][j] * Tmat[i][j]);
       offdiag_norm *= 2.0;
       //for ( i = 0; i < maxiter; i++ )
       //   printf("%13.6e %13.6e %13.6e %13.6e %13.6e\n", Tmat[i][0],
       //          Tmat[i][1], Tmat[i][2], Tmat[i][3], Tmat[i][4]);
       printf("offdiag_norm = %e (%e)\n",offdiag_norm,init_offdiag_norm);
    }
    */

    // -----------------------------------------------------------------
    // search for max and min eigenvalue
    // -----------------------------------------------------------------

    /*
    t = Tmat[0][0];
    for ( i = 1; i < maxiter; i++ )
       t = (Tmat[i][i] > t) ? Tmat[i][i] : t;
    printf("max eigenvalue = %e\n", t);
    t = Tmat[0][0];
    for ( i = 1; i < maxiter; i++ )
       t = (Tmat[i][i] < t) ? Tmat[i][i] : t;
    printf("min eigenvalue = %e\n", t);
    */

    //------------------------------------------------------------------
    // clean up 
    //------------------------------------------------------------------

    MPI_Finalize();
}

