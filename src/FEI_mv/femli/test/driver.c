/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.13 $
 ***********************************************************************EHEADER*/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "util/mli_utils.h"
#include "base/mli_defs.h"
#include "cintface/cmli.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#define habs(x) ((x)>=0 ? x : -x)

HYPRE_ParCSRMatrix GenerateLaplacian9pt(MPI_Comm, int, int, int, int, int, 
                                        int, double *);
int GenerateConvectionDiffusion3D(MPI_Comm, int, int, int, int, int, int,
                      int, int, int, double, double, double *, 
                      HYPRE_ParCSRMatrix *,HYPRE_ParVector*);
int GenerateConvectionDiffusion3D2(MPI_Comm, int, int, int, int, int, int,
                      int, int, int, double, double, double *, 
                      HYPRE_ParCSRMatrix *,HYPRE_ParVector*);
int GenerateRugeStuben1(MPI_Comm,double,double,int,HYPRE_ParCSRMatrix*,
                        HYPRE_ParVector*);
int GenerateRugeStuben2(MPI_Comm,double,int,HYPRE_ParCSRMatrix*,HYPRE_ParVector*);
int GenerateRugeStuben3(MPI_Comm,double,int,HYPRE_ParCSRMatrix*, HYPRE_ParVector*);
int GenerateStuben(MPI_Comm,double,int,HYPRE_ParCSRMatrix*,HYPRE_ParVector*);

int main(int argc, char **argv)
{
   int                j, nx=401, ny=401, nz=1, P, Q, R, p, q, r, nprocs;
   int                mypid, startRow, fdScheme;
   int                *partition, globalSize, localSize, nsweeps, rowSize;
   int                *colInd, k, *procCnts, *offsets, *rowCnts, ftype;
   int                ndofs=3, nullDim=6, testProb=2, solver=2, scaleFlag=0;
   int                fleng, rleng, status, amgMethod=0, scaleFlag2=0;
   char               *targv[10], fname[100], rhsFname[100], methodName[10];
   char               argStr[40];
   double             *values, *nullVecs, *scaleVec, *colVal, *gscaleVec;
   double             *rhsVector=NULL, alpha, beta, *weights, Lvalue=4.0;
   double             epsilon=0.001, Pweight, dtemp;
   HYPRE_IJMatrix     newIJA;
   HYPRE_IJVector     IJrhs;
   HYPRE_ParCSRMatrix HYPREA;
   hypre_ParCSRMatrix *hypreA;
   hypre_ParVector    *sol, *rhs;
   CMLI               *cmli;
   CMLI_Matrix        *cmliMat;
   CMLI_Method        *cmliMethod;
   CMLI_Vector        *csol, *crhs;
   MLI_Function       *funcPtr;

   /* ------------------------------------------------------------- *
    * machine setup
    * ------------------------------------------------------------- */

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);

   /* ------------------------------------------------------------- *
    * problem setup
    * ------------------------------------------------------------- */

/*
   if ( mypid == 0 )
   {
      printf("Which test problem (0-6) : ");
      scanf("%d", &testProb);
   }
   MPI_Bcast(&testProb, 1, MPI_INT, 0, MPI_COMM_WORLD);
*/
testProb = 0;
   if ( testProb < 0 ) testProb = 0;
   if ( testProb > 6 ) testProb = 6;
   if ( testProb != 1 && testProb != 6)
   {
/*
      if ( mypid == 0 )
      {
         printf("nx = ny = ? ");
         scanf("%d", &nx);
      }
      MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
*/
      nx = 96;
      ny = nx;
      nz = nx;
   }

   /* --- Test problem 1 --- */
   if ( testProb == 0 )
   {
      dtemp = (double) nprocs;
      dtemp = pow(dtemp,1.0000001/3.0);
      R = (int) (dtemp);
      dtemp = (double) nprocs / (double) R;
      dtemp = pow(dtemp,1.0000001/2.0);
      Q = (int) (dtemp);
      P = nprocs/R/Q;
      p = mypid % P;
      q = (( mypid - p)/P) % Q;
      r = ( mypid - p - P*q)/( P*Q );
      values = (double *) calloc(9, sizeof(double));
      values[3] = -1.0;
      values[2] = -1.0;
      values[1] = -1.0;
      values[0] = 6.00;
/*
      if ( mypid == 0 )
      {
         printf("enter alpha : ");
         scanf("%lg", &alpha);
         printf("enter beta : ");
         scanf("%lg", &beta);
         printf("enter difference scheme (0 - upwind, center otherwise) : ");
         scanf("%d", &fdScheme);
      }
      MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&fdScheme, 1, MPI_INT, 0, MPI_COMM_WORLD);
*/
alpha = 0.0;
beta = 0.0;
fdScheme = 1;
printf("generate problem\n");
      if (fdScheme == 0)
         GenerateConvectionDiffusion3D2(MPI_COMM_WORLD,nx,ny,nz,P,Q,R,p, 
                        q, r, alpha, beta, values, &HYPREA, 
                        (HYPRE_ParVector *) &rhs); 
      else if (fdScheme == 1)
         GenerateConvectionDiffusion3D(MPI_COMM_WORLD,nx,ny,nz,P,Q,R,p, 
                        q, r, alpha, beta, values, &HYPREA, 
                        (HYPRE_ParVector *) &rhs); 
      else
      {
         values[8] = -1.0;
         values[7] = -1.0;
         values[6] = -1.0;
         values[5] = -1.0;
         values[4] = 8.0;
         values[3] = -1.0;
         values[2] = -1.0;
         values[1] = -1.0;
         values[0] = 8.0;
         dtemp = (double) nprocs;
         dtemp = pow(dtemp,1.0000001/2.0);
         Q = (int) (dtemp);
         P = nprocs/Q;
         p = mypid % P;
         q = ((mypid - p)/P) % Q;
         HYPREA = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(MPI_COMM_WORLD,
                                  nx, ny, P, Q, p, q, values);
         HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
         HYPRE_IJVectorCreate(MPI_COMM_WORLD, partition[mypid],
                              partition[mypid+1]-1, &IJrhs);
         free( partition );
         HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
         HYPRE_IJVectorInitialize(IJrhs);
         HYPRE_IJVectorAssemble(IJrhs);
         HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
         HYPRE_IJVectorSetObjectType(IJrhs, -1);
         HYPRE_IJVectorDestroy(IJrhs);
         hypre_ParVectorSetConstantValues( rhs, 1.0 );
      }
printf("generate problem done\n");
      free( values );
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
      if ( scaleFlag == 1 ) 
      {
         scaleVec  = (double *) malloc(localSize*sizeof(double));
         for ( j = startRow; j < startRow+localSize; j++ ) 
            scaleVec[j-startRow] = 1.0e6 * (0.5 * random() / RAND_MAX + 1.0e-1);
         HYPRE_IJMatrixCreate(MPI_COMM_WORLD, startRow, startRow+localSize-1,
                              startRow, startRow+localSize-1, &newIJA);
         HYPRE_IJMatrixSetObjectType(newIJA, HYPRE_PARCSR);
         rowCnts = (int *) malloc( localSize * sizeof(int) );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,NULL);
            rowCnts[j-startRow] = rowSize;
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,NULL);
         }
         HYPRE_IJMatrixSetRowSizes(newIJA, rowCnts);
         HYPRE_IJMatrixInitialize(newIJA);
         free( rowCnts );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,&colVal);
            for ( k = 0; k < rowSize; k++ ) 
               colVal[k] = colVal[k] * scaleVec[j-startRow];
            HYPRE_IJMatrixSetValues(newIJA, 1, &rowSize, (const int *) &j,
                   (const int *) colInd, (const double *) colVal);
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,&colVal);
         }
         HYPRE_IJMatrixAssemble(newIJA);
         HYPRE_ParCSRMatrixDestroy(HYPREA);
         HYPRE_IJMatrixGetObject(newIJA, (void **) &HYPREA);
         HYPRE_IJMatrixSetObjectType(newIJA, -1);
         HYPRE_IJMatrixDestroy(newIJA);
         free( scaleVec );
      } 
      else if ( scaleFlag == 2 ) 
      {
         scaleVec  = (double *) malloc(localSize*sizeof(double));
         gscaleVec = (double *) malloc(globalSize*sizeof(double));
         for ( j = startRow; j < startRow+localSize; j++ ) 
            scaleVec[j-startRow] = 1.0e6 * (0.5 * random() / RAND_MAX + 1.0e-1);
         alpha = 0.0;
         beta = 1.0e20;
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            if (scaleVec[j-startRow] > alpha) alpha = scaleVec[j-startRow]; 
            if (scaleVec[j-startRow] < beta) beta = scaleVec[j-startRow]; 
         }
         printf("scaling min/max = %e %e %d\n", beta, alpha, RAND_MAX);
         procCnts = (int *) malloc( nprocs * sizeof(int) );
         offsets   = (int *) malloc( nprocs * sizeof(int) );
         MPI_Allgather(&localSize,1,MPI_INT,procCnts,1,MPI_INT,MPI_COMM_WORLD);
         offsets[0] = 0;
         for ( j = 1; j < nprocs; j++ )
            offsets[j] = offsets[j-1] + procCnts[j-1];
         MPI_Allgatherv(scaleVec, localSize, MPI_DOUBLE, gscaleVec,
                        procCnts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
         free( procCnts );
         free( offsets );
         HYPRE_IJMatrixCreate(MPI_COMM_WORLD, startRow, startRow+localSize-1,
                              startRow, startRow+localSize-1, &newIJA);
         HYPRE_IJMatrixSetObjectType(newIJA, HYPRE_PARCSR);
         rowCnts = (int *) malloc( localSize * sizeof(int) );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,NULL);
            rowCnts[j-startRow] = rowSize;
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,NULL);
         }
         HYPRE_IJMatrixSetRowSizes(newIJA, rowCnts);
         HYPRE_IJMatrixInitialize(newIJA);
         free( rowCnts );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,&colVal);
            for ( k = 0; k < rowSize; k++ ) 
               colVal[k] = colVal[k] * gscaleVec[colInd[k]];
            HYPRE_IJMatrixSetValues(newIJA, 1, &rowSize, (const int *) &j,
                   (const int *) colInd, (const double *) colVal);
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,&colVal);
         }
         HYPRE_IJMatrixAssemble(newIJA);
         HYPRE_ParCSRMatrixDestroy(HYPREA);
         HYPRE_IJMatrixGetObject(newIJA, (void **) &HYPREA);
         HYPRE_IJMatrixSetObjectType(newIJA, -1);
         HYPRE_IJMatrixDestroy(newIJA);
         free( gscaleVec );
         nullVecs = ( double *) malloc(localSize * sizeof(double));
         for ( j = 0; j < localSize; j++ ) nullVecs[j] = 1.0 / scaleVec[j]; 
         free( scaleVec );
      }
      if ( scaleFlag2 == 1 ) 
      {
         scaleVec  = (double *) malloc(localSize*sizeof(double));
         gscaleVec = (double *) malloc(globalSize*sizeof(double));
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            scaleVec[j-startRow] = 0.0;
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,&colVal);
            for (k = 0; k < rowSize; k++)
               if ( colInd[k] == j ) scaleVec[j-startRow] = colVal[k]; 
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,&colVal);
         }
         for (j = 0; j < localSize; j++) 
         {
            if ( scaleVec[j] <= 0.0 ) 
            {
               printf("Proc %d : diag %d = %e <= 0 \n",mypid,j,scaleVec[j]); 
               exit(1);
            }
            scaleVec[j] = 1.0/sqrt(scaleVec[j]); 
         }
         procCnts = (int *) malloc( nprocs * sizeof(int) );
         offsets   = (int *) malloc( nprocs * sizeof(int) );
         MPI_Allgather(&localSize,1,MPI_INT,procCnts,1,MPI_INT,MPI_COMM_WORLD);
         offsets[0] = 0;
         for ( j = 1; j < nprocs; j++ )
            offsets[j] = offsets[j-1] + procCnts[j-1];
         MPI_Allgatherv(scaleVec, localSize, MPI_DOUBLE, gscaleVec,
                        procCnts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
         free( procCnts );
         free( offsets );
         HYPRE_IJMatrixCreate(MPI_COMM_WORLD, startRow, startRow+localSize-1,
                              startRow, startRow+localSize-1, &newIJA);
         HYPRE_IJMatrixSetObjectType(newIJA, HYPRE_PARCSR);
         rowCnts = (int *) malloc( localSize * sizeof(int) );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,NULL);
            rowCnts[j-startRow] = rowSize;
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,NULL);
         }
         HYPRE_IJMatrixSetRowSizes(newIJA, rowCnts);
         HYPRE_IJMatrixInitialize(newIJA);
         free( rowCnts );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,&colVal);
            for ( k = 0; k < rowSize; k++ ) 
            {
               colVal[k] = colVal[k] * gscaleVec[colInd[k]] * gscaleVec[j];
               if ( colInd[k] == j && habs(colVal[k]-1.0) > 1.0e-8 )
                  printf("Proc %d : diag %d(%d) = %e != 1.0\n",mypid,j,k,
                         colVal[k]);
            }
            HYPRE_IJMatrixSetValues(newIJA, 1, &rowSize, (const int *) &j,
                   (const int *) colInd, (const double *) colVal);
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,&colVal);
         }
         HYPRE_IJMatrixAssemble(newIJA);
         HYPRE_ParCSRMatrixDestroy(HYPREA);
         HYPRE_IJMatrixGetObject(newIJA, (void **) &HYPREA);
         HYPRE_IJMatrixSetObjectType(newIJA, -1);
         HYPRE_IJMatrixDestroy(newIJA);
         free( gscaleVec );
         nullVecs = ( double *) malloc(localSize * sizeof(double));
         for ( j = 0; j < localSize; j++ ) nullVecs[j] = 1.0 / scaleVec[j]; 
         free( scaleVec );
      } else nullVecs = NULL;
   }

   /* --- Test problem 2 --- */
   else if ( testProb == 1 )
   {
      if ( mypid == 0 )
      {
         printf("Matrix file name : ");
         scanf("%s", fname);
         fleng = strlen(fname);
         fleng++;
         printf("\nMatrix file type (0 - Tumin, 1 - IJA) : ");
         scanf("%d", &ftype);
         printf("rhs file name : ");
         scanf("%s", rhsFname);
         rleng = strlen(rhsFname);
         rleng++;
      }
      MPI_Bcast(&fleng, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ftype, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&rleng, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(fname, fleng, MPI_CHAR, 0, MPI_COMM_WORLD);
      MPI_Bcast(rhsFname, rleng, MPI_CHAR, 0, MPI_COMM_WORLD);
      if ( ftype == 0 )
      {
         MLI_Utils_HypreMatrixReadTuminFormat(fname,MPI_COMM_WORLD,ndofs,
                     (void **) &HYPREA, scaleFlag, &scaleVec);
      }
      else
      {
/*
         MLI_Utils_HypreMatrixReadIJAFormat(fname,MPI_COMM_WORLD,ndofs,
                     (void **) &HYPREA, scaleFlag, &scaleVec);
*/
         MLI_Utils_HypreParMatrixReadIJAFormat(fname,MPI_COMM_WORLD,
                     (void **) &HYPREA, scaleFlag, &scaleVec);
      }
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
      rhsVector = ( double *) malloc(localSize * sizeof(double));
      status = MLI_Utils_DoubleParVectorRead(rhsFname, MPI_COMM_WORLD, 
                               localSize, startRow, rhsVector);
      if ( status < 0 )
      {
         free( rhsVector );
         rhsVector = NULL;
      }
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, partition[mypid],
                               partition[mypid+1]-1, &IJrhs);
      free( partition );
      HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJrhs);
      HYPRE_IJVectorAssemble(IJrhs);
      if ( rhsVector != NULL )
      {
         colInd = (int *) malloc(localSize * sizeof(int));
         for (j = 0; j < localSize; j++) colInd[j] = startRow + j;
         HYPRE_IJVectorSetValues(IJrhs, localSize, (const int *) colInd,
                                 (const double *) rhsVector);
         free(colInd);
      }
      HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
      HYPRE_IJVectorSetObjectType(IJrhs, -1);
      HYPRE_IJVectorDestroy(IJrhs);
      if ( rhsVector == NULL )
         hypre_ParVectorSetConstantValues( rhs, 1.0 );
      else free(rhsVector);

      if ( scaleFlag2 == 1 ) 
      {
         scaleVec  = (double *) malloc(localSize*sizeof(double));
         gscaleVec = (double *) malloc(globalSize*sizeof(double));
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            scaleVec[j-startRow] = 0.0;
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,&colVal);
            for (k = 0; k < rowSize; k++)
               if ( colInd[k] == j ) scaleVec[j-startRow] = colVal[k]; 
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,&colVal);
         }
         for (j = 0; j < localSize; j++) 
         {
            if ( scaleVec[j] <= 0.0 ) 
            {
               printf("Proc %d : diag %d = %e <= 0 \n",mypid,j,scaleVec[j]); 
               exit(1);
            }
            scaleVec[j] = 1.0/sqrt(scaleVec[j]); 
         }
         procCnts = (int *) malloc( nprocs * sizeof(int) );
         offsets   = (int *) malloc( nprocs * sizeof(int) );
         MPI_Allgather(&localSize,1,MPI_INT,procCnts,1,MPI_INT,MPI_COMM_WORLD);
         offsets[0] = 0;
         for ( j = 1; j < nprocs; j++ )
            offsets[j] = offsets[j-1] + procCnts[j-1];
         MPI_Allgatherv(scaleVec, localSize, MPI_DOUBLE, gscaleVec,
                        procCnts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
         free( procCnts );
         free( offsets );
         HYPRE_IJMatrixCreate(MPI_COMM_WORLD, startRow, startRow+localSize-1,
                              startRow, startRow+localSize-1, &newIJA);
         HYPRE_IJMatrixSetObjectType(newIJA, HYPRE_PARCSR);
         rowCnts = (int *) malloc( localSize * sizeof(int) );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,NULL);
            rowCnts[j-startRow] = rowSize;
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,NULL);
         }
         HYPRE_IJMatrixSetRowSizes(newIJA, rowCnts);
         HYPRE_IJMatrixInitialize(newIJA);
         free( rowCnts );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,&colVal);
            for ( k = 0; k < rowSize; k++ ) 
            {
               colVal[k] = colVal[k] * gscaleVec[colInd[k]] * gscaleVec[j];
               if ( colInd[k] == j && habs(colVal[k]-1.0) > 1.0e-8 )
                  printf("Proc %d : diag %d(%d) = %e != 1.0\n",mypid,j,k,
                         colVal[k]);
            }
            HYPRE_IJMatrixSetValues(newIJA, 1, &rowSize, (const int *) &j,
                   (const int *) colInd, (const double *) colVal);
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,&colVal);
         }
         HYPRE_IJMatrixAssemble(newIJA);
         HYPRE_ParCSRMatrixDestroy(HYPREA);
         HYPRE_IJMatrixGetObject(newIJA, (void **) &HYPREA);
         HYPRE_IJMatrixSetObjectType(newIJA, -1);
         HYPRE_IJMatrixDestroy(newIJA);
         free( gscaleVec );
         nullVecs = ( double *) malloc(localSize * sizeof(double));
         for ( j = 0; j < localSize; j++ ) nullVecs[j] = 1.0 / scaleVec[j]; 
         free( scaleVec );
      } else nullVecs = NULL;
#if 0
      nullVecs = ( double *) malloc(localSize * nullDim * sizeof(double));
/*
      MLI_Utils_DoubleParVectorRead("rigid_body_mode01",MPI_COMM_WORLD,
                              localSize, startRow, nullVecs);
      MLI_Utils_DoubleParVectorRead("rigid_body_mode02",MPI_COMM_WORLD,
                              localSize, startRow, &nullVecs[localSize]);
      MLI_Utils_DoubleParVectorRead("rigid_body_mode03",MPI_COMM_WORLD,
                              localSize, startRow, &nullVecs[localSize*2]);
*/
      MLI_Utils_DoubleParVectorRead("rigid_body_mode01",MPI_COMM_WORLD,
                              localSize, 0, nullVecs);
      MLI_Utils_DoubleParVectorRead("rigid_body_mode02",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize]);
      MLI_Utils_DoubleParVectorRead("rigid_body_mode03",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize*2]);
      if ( scaleFlag )
      {
         for ( j = 0; j < localSize; j++ ) 
            scaleVec[j] = sqrt(scaleVec[j]); 
         for ( j = 0; j < localSize; j++ ) nullVecs[j] *= scaleVec[j]; 
         for ( j = 0; j < localSize; j++ ) 
            nullVecs[localSize+j] *= scaleVec[j]; 
         for ( j = 0; j < localSize; j++ ) 
            nullVecs[2*localSize+j] *= scaleVec[j]; 
      }
      if ( nullDim > 3 )
      {
         MLI_Utils_DoubleParVectorRead("rigid_body_mode04",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize*3]);
         MLI_Utils_DoubleParVectorRead("rigid_body_mode05",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize*4]);
         MLI_Utils_DoubleParVectorRead("rigid_body_mode06",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize*5]);
         if ( scaleFlag )
         {
            for ( j = 0; j < localSize; j++ ) 
               nullVecs[3*localSize+j] *= scaleVec[j]; 
            for ( j = 0; j < localSize; j++ ) 
               nullVecs[4*localSize+j] *= scaleVec[j]; 
            for ( j = 0; j < localSize; j++ ) 
               nullVecs[5*localSize+j] *= scaleVec[j]; 
         }
      }
#endif
   }

   /* --- Test problem 3 --- */
   else if ( testProb == 2 )
   {
      if ( mypid == 0 )
      {
         printf("Convection diffusion equation (1) : L = (0-4) ? ");
         scanf("%lg", &Lvalue);
         printf("Convection diffusion equation (1) : epsilon = ");
         scanf("%lg", &epsilon);
      }
      MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&Lvalue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      GenerateRugeStuben1(MPI_COMM_WORLD,Lvalue,epsilon,nx,&HYPREA,
                          (HYPRE_ParVector *) &rhs);
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
   }   

   /* --- Test problem 4 --- */
   else if ( testProb == 3 )
   {
      if ( mypid == 0 )
      {
         printf("Convection diffusion equation (2) : epsilon = ");
         scanf("%lg", &epsilon);
      }
      MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      GenerateRugeStuben2(MPI_COMM_WORLD,epsilon,nx,&HYPREA,
                          (HYPRE_ParVector *) &rhs);
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
   }   

   /* --- Test problem 5 --- */
   else if ( testProb == 4 )
   {
      if ( mypid == 0 )
      {
         printf("Convection diffusion equation (3) : epsilon = ");
         scanf("%lg", &epsilon);
      }
      MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      GenerateRugeStuben3(MPI_COMM_WORLD,epsilon,nx,&HYPREA,
                          (HYPRE_ParVector *) &rhs);
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
   }   

   /* --- Test problem 6 --- */
   else if ( testProb == 5 )
   {
      if ( mypid == 0 )
      {
         printf("Convection diffusion equation (4) : epsilon = ");
         scanf("%lg", &epsilon);
      }
      MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      GenerateStuben(MPI_COMM_WORLD,epsilon,nx,&HYPREA,
                     (HYPRE_ParVector *) &rhs);
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
   }   

   /* --- Test problem 7 --- */
   else if ( testProb == 6 )
   {
      if ( mypid == 0 )
      {
         printf("HB Matrix file name : ");
         scanf("%s", fname);
         fleng = strlen(fname);
         fleng++;
      }
      MLI_Utils_HypreMatrixReadHBFormat(fname,MPI_COMM_WORLD,
                                        (void **) &HYPREA);
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
      rhsVector = ( double *) malloc(localSize * sizeof(double));
      for (j = 0; j < localSize; j++) rhsVector[j] = 1.0;
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, partition[mypid],
                           partition[mypid+1]-1, &IJrhs);
      free( partition );
      HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJrhs);
      HYPRE_IJVectorAssemble(IJrhs);
      colInd = (int *) malloc(localSize * sizeof(int));
      for (j = 0; j < localSize; j++) colInd[j] = startRow + j;
      HYPRE_IJVectorSetValues(IJrhs, localSize, (const int *) colInd,
                              (const double *) rhsVector);
      free(colInd);
      HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
      HYPRE_IJVectorSetObjectType(IJrhs, -1);
      HYPRE_IJVectorDestroy(IJrhs);
      free(rhsVector);
      nullVecs = NULL;
   }

   hypreA = (hypre_ParCSRMatrix *) HYPREA;
   HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
   sol = hypre_ParVectorCreate(MPI_COMM_WORLD, globalSize, partition);
   hypre_ParVectorInitialize( sol );
   hypre_ParVectorSetConstantValues( sol, 0.0 );

   funcPtr = (MLI_Function *) malloc( sizeof( MLI_Function ) );
   MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
   csol = MLI_VectorCreate(sol, "HYPRE_ParVector", funcPtr);
   crhs = MLI_VectorCreate(rhs, "HYPRE_ParVector", funcPtr);

   /* ------------------------------------------------------------- *
    * problem setup
    * ------------------------------------------------------------- */

   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   cmliMat = MLI_MatrixCreate((void*) hypreA,"HYPRE_ParCSR",funcPtr);
   free( funcPtr );
   cmli = MLI_Create( MPI_COMM_WORLD );
/*
   if ( mypid == 0 )
   {
      printf("Which AMG to use (0 - RSAMG, 1 - SAAMG, 2 - CRAMG) : ");
      scanf("%d", &amgMethod);
   }
   MPI_Bcast(&amgMethod, 1, MPI_INT, 0, MPI_COMM_WORLD);
*/
amgMethod = 1;
   if       (amgMethod == 0) strcpy(methodName, "AMGRS");
   else  if (amgMethod == 1) strcpy(methodName, "AMGSA");
   else                      strcpy(methodName, "AMGCR");

   cmliMethod = MLI_MethodCreate(methodName, MPI_COMM_WORLD);
   MLI_MethodSetParams(cmliMethod, "setNumLevels 10", 0, NULL);
   MLI_MethodSetParams(cmliMethod, "setMaxIterations 1", 0, NULL);
   MLI_MethodSetParams(cmliMethod, "setMinCoarseSize 25", 0, NULL);
   MLI_MethodSetParams(cmliMethod, "setCoarseSolver SuperLU", 2, targv);
   if (! strcmp(methodName, "AMGRS"))
   {
      if (mypid == 0)
      {
         printf("RSAMG number of sweeps = ");
         scanf("%d", &nsweeps);
      }
      MPI_Bcast(&nsweeps, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if ( nsweeps <= 0 ) nsweeps = 1;
      weights = (double *) malloc( sizeof(double)*nsweeps );
      //if ( mypid == 0 )
      //{
      //   printf("RSAMG relaxation weights = ");
      //   scanf("%lg", &weights[0]);
      //}
      weights[0] = 2.0 / 3.0;
      //MPI_Bcast(&weights[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      //if ( weights[0] <= 0.0 ) weights[0] = 0.67; 
      for ( j = 1; j < nsweeps; j++ ) weights[j] = weights[0];
      targv[0] = (char *) &nsweeps;
      targv[1] = (char *) weights;
      MLI_MethodSetParams(cmliMethod, "setPreSmoother Jacobi", 2, targv);
      MLI_MethodSetParams(cmliMethod, "setPostSmoother Jacobi", 2, targv);
      free(weights);
      //MLI_MethodSetParams(cmliMethod, "setCoarsenScheme cljp", 0, NULL);
      MLI_MethodSetParams(cmliMethod, "setStrengthThreshold 0.5", 0, NULL);
      if ( mypid == 0 )
      {
         printf("RSAMG smootherPrintRNorm ? (0 for no, 1 for yes) = ");
         scanf("%d", &j);
      }
      MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if ( j != 0 )
         MLI_MethodSetParams(cmliMethod, "setSmootherPrintRNorm", 0, NULL);
      if ( mypid == 0 )
      {
         printf("RSAMG smootherFindOmega ? (0 for no, 1 for yes) = ");
         scanf("%d", &j);
      }
      MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if ( j != 0 )
         MLI_MethodSetParams(cmliMethod, "setSmootherFindOmega", 0, NULL);
      if ( mypid == 0 )
      {
         printf("RSAMG nonsymmetric ? (0 for no, 1 for yes) = ");
         scanf("%d", &j);
      }
      MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if ( j != 0 )
         MLI_MethodSetParams(cmliMethod, "useNonsymmetric", 0, NULL);
/*
MLI_MethodSetParams(cmliMethod, "useInjectionForR", 0, NULL);
*/
   }
   else if (! strcmp(methodName, "AMGSA"))
   {
/*
      if ( mypid == 0 )
      {
         printf("SAAMG number of sweeps = ");
         scanf("%d", &nsweeps);
      }
      MPI_Bcast(&nsweeps, 1, MPI_INT, 0, MPI_COMM_WORLD);
*/
nsweeps = 1;
      if ( nsweeps <= 0 ) nsweeps = 1;
      weights = (double *) malloc( sizeof(double)*nsweeps );
/*
      if ( mypid == 0 )
      {
         printf("SAAMG relaxation weights = ");
         scanf("%lg", &weights[0]);
      }
      MPI_Bcast(&weights[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
*/
weights[0] = 1.0;
      if ( weights[0] <= 0.0 ) weights[0] = 1.0; 
      for ( j = 1; j < nsweeps; j++ ) weights[j] = weights[0];
      targv[0] = (char *) &nsweeps;
      targv[1] = (char *) weights;
      MLI_MethodSetParams(cmliMethod, "setPreSmoother Jacobi", 2, targv);
      MLI_MethodSetParams(cmliMethod, "setPostSmoother Jacobi", 2, targv);
      free(weights);
      MLI_MethodSetParams(cmliMethod, "setStrengthThreshold 0.08", 0, NULL);
      MLI_MethodSetParams(cmliMethod, "setCalibrationSize 0", 0, NULL);
      MLI_MethodSetParams(cmliMethod, "useSAMGDDExt", 0, NULL);
/*
      if ( mypid == 0 )
      {
         printf("SAAMG Pweight weights = ");
         scanf("%lg", &Pweight);
      }
      MPI_Bcast(&Pweight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
*/
Pweight = 0.0;
      if ( Pweight < 0.0 ) Pweight = 0.0;
      sprintf( argStr, "setPweight %e", Pweight);
      MLI_MethodSetParams(cmliMethod, argStr, 0, NULL);
/*
      if ( mypid == 0 )
      {
         printf("SAAMG nonsymmetric ? (0 for no, 1 for yes) = ");
         scanf("%d", &j);
      }
      MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);
*/
j = 0;
      if ( j != 0 )
         MLI_MethodSetParams(cmliMethod, "useNonsymmetric", 0, NULL);
   }
   else if (! strcmp(methodName, "AMGCR"))
   {
      if ( mypid == 0 )
      {
         printf("CRAMG number of sweeps = ");
         scanf("%d", &nsweeps);
      }
      MPI_Bcast(&nsweeps, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if ( nsweeps <= 0 ) nsweeps = 1;
      weights = (double *) malloc( sizeof(double)*nsweeps );
      //weights[0] = 2.0 / 3.0;
      weights[0] = 1.0 / 1.0;
      MPI_Bcast(&weights[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if ( weights[0] <= 0.0 ) weights[0] = 1.0; 
      for ( j = 1; j < nsweeps; j++ ) weights[j] = weights[0];
      targv[0] = (char *) &nsweeps;
      targv[1] = (char *) weights;
      MLI_MethodSetParams(cmliMethod, "setSmoother Jacobi", 2, targv);
      free(weights);
      if ( mypid == 0 )
      {
         printf("Use MIS ? (0 for no, otherwise yes) ");
         scanf("%d", &j);
      }
      MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if ( j != 0 )
      {
         MLI_MethodSetParams(cmliMethod, "useMIS", 0, NULL);
      }
      if ( mypid == 0 )
      {
         printf("Use CR ? (0 for no, otherwise yes) ");
         scanf("%d", &j);
      }
      MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if ( j != 0 )
      {
         MLI_MethodSetParams(cmliMethod, "setNumTrials 100", 0, NULL);
      }
      else
      {
         MLI_MethodSetParams(cmliMethod, "setNumTrials 1", 0, NULL);
      }
      MLI_MethodSetParams(cmliMethod, "setTargetMu 0.8", 0, NULL);
      MLI_MethodSetParams(cmliMethod, "setPDegree 2", 0, NULL);
   }
   MLI_MethodSetParams(cmliMethod, "setOutputLevel 2", 0, NULL);

   if ( testProb == 0 )
   {
      ndofs    = 1;
      nullDim  = 1;
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &nullDim;
      targv[2] = (char *) nullVecs;
      targv[3] = (char *) &localSize;
      if ( ! strcmp(methodName, "AMGSA") )
         MLI_MethodSetParams(cmliMethod, "setNullSpace", 4, targv);
      free( nullVecs );
   }
   if ( testProb == 1 )
   {
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &nullDim;
      targv[2] = (char *) nullVecs;
      targv[3] = (char *) &localSize;
      if ( ! strcmp(methodName, "AMGSA") )
         MLI_MethodSetParams(cmliMethod, "setNullSpace", 4, targv);
      free( nullVecs );
   }
   MLI_MethodSetParams(cmliMethod, "print", 0, NULL);
   MLI_SetMethod(cmli, cmliMethod);
   MLI_SetSystemMatrix(cmli, 0, cmliMat);
   MLI_SetOutputLevel(cmli, 2);

/*
   if ( mypid == 0 )
   {
      printf("outer Krylov solver (0 - none, 1 - CG, 2 - GMRES) : ");
      scanf("%d", &solver);
   }
   MPI_Bcast(&solver, 1, MPI_INT, 0, MPI_COMM_WORLD);
*/
solver = 2;
   if ( solver < 0 ) solver = 0;
   if ( solver > 2 ) solver = 2;

   if ( solver == 0 )
   {
      MLI_Setup(cmli);
      MLI_Solve(cmli, csol, crhs);
   } 
   else if ( solver == 1 )
   {
      MLI_Utils_HyprePCGSolve(cmli, (HYPRE_Matrix) HYPREA, 
                             (HYPRE_Vector) rhs, (HYPRE_Vector) sol);
   }
   else
   {
      MLI_Utils_HypreGMRESSolve(cmli, (HYPRE_Matrix) HYPREA, 
                              (HYPRE_Vector) rhs, (HYPRE_Vector) sol, "mli");
   }
   MLI_Print( cmli );
   MLI_Destroy( cmli );
   MLI_MatrixDestroy( cmliMat );
   MLI_VectorDestroy( csol );
   MLI_VectorDestroy( crhs );
   MLI_MethodDestroy( cmliMethod );
   MPI_Barrier(MPI_COMM_WORLD);
   exit(0);
   MPI_Finalize();
   return 0;
}

/* **************************************************************** *
 * problem generation
 * ---------------------------------------------------------------- */

/* ---------------------------------------------------------------- *
   convection diffusion equation where (value, alpha, beta) determine 
   the diffusion and convection terms
 * ---------------------------------------------------------------- */

int hypre_GeneratePartitioning(int, int, int**);

int hypre_mapCD( int  ix, int  iy, int  iz, int  p, int  q, int  r,
     int  P, int  Q, int  R, int *nx_part, int *ny_part, int *nz_part,
     int *global_part )
{
   int nx_local, ny_local, ix_local, iy_local, iz_local;
   int global_index, proc_num;
 
   proc_num = r*P*Q + q*P + p;
   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];
   ix_local = ix - nx_part[p];
   iy_local = iy - ny_part[q];
   iz_local = iz - nz_part[r];
   global_index = global_part[proc_num] 
      + (iz_local*ny_local+iy_local)*nx_local + ix_local;

   return global_index;
}

/* ---------------------------------------------------------------- *
   convection diffusion equation where (centered difference)
 * ---------------------------------------------------------------- */

int GenerateConvectionDiffusion3D( MPI_Comm comm, int nx, int ny, int nz, 
            int P, int Q, int R, int p, int q, int r, double alpha, 
            double beta, double  *value, HYPRE_ParCSRMatrix *rA, 
            HYPRE_ParVector *rrhs )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag, *offd;
   HYPRE_IJVector     IJrhs;
   hypre_ParVector    *rhs;

   int                *diag_i, *diag_j, *offd_i, *offd_j;
   double             *diag_data, *offd_data;
   int                *global_part, ix, iy, iz, cnt, o_cnt, local_num_rows; 
   int                *col_map_offd, row_index, i,j, *partition;
   int                nx_local, ny_local, nz_local;
   int                nx_size, ny_size, nz_size, num_cols_offd, grid_size;
   int                *nx_part, *ny_part, *nz_part;
   int                num_procs, my_id, P_busy, Q_busy, R_busy;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   grid_size = nx*ny*nz;

   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);
   hypre_GeneratePartitioning(nz,R,&nz_part);

   global_part = hypre_CTAlloc(int,P*Q*R+1);

   global_part[0] = 0;
   cnt = 1;
   for (iz = 0; iz < R; iz++)
   {
      nz_size = nz_part[iz+1]-nz_part[iz];
      for (iy = 0; iy < Q; iy++)
      {
         ny_size = ny_part[iy+1]-ny_part[iy];
         for (ix = 0; ix < P; ix++)
         {
            nx_size = nx_part[ix+1] - nx_part[ix];
            global_part[cnt] = global_part[cnt-1];
            global_part[cnt++] += nx_size*ny_size*nz_size;
         }
      }
   }

   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];
   nz_local = nz_part[r+1] - nz_part[r];

   my_id = r*(P*Q) + q*P + p;
   num_procs = P*Q*R;

   local_num_rows = nx_local*ny_local*nz_local;
   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);

   P_busy = hypre_min(nx,P);
   Q_busy = hypre_min(ny,Q);
   R_busy = hypre_min(nz,R);

   num_cols_offd = 0;
   if (p) num_cols_offd += ny_local*nz_local;
   if (p < P_busy-1) num_cols_offd += ny_local*nz_local;
   if (q) num_cols_offd += nx_local*nz_local;
   if (q < Q_busy-1) num_cols_offd += nx_local*nz_local;
   if (r) num_cols_offd += nx_local*ny_local;
   if (r < R_busy-1) num_cols_offd += nx_local*ny_local;

   if (!local_num_rows) num_cols_offd = 0;

   col_map_offd = hypre_CTAlloc(int, num_cols_offd);

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt-1];
            offd_i[o_cnt] = offd_i[o_cnt-1];
            diag_i[cnt]++;
            if (iz > nz_part[r]) 
               diag_i[cnt]++;
            else
            {
               if (iz) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iy > ny_part[q]) 
               diag_i[cnt]++;
            else
            {
               if (iy) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix > nx_part[p]) 
               diag_i[cnt]++;
            else
            {
               if (ix) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (ix+1 < nx_part[p+1]) 
               diag_i[cnt]++;
            else
            {
               if (ix+1 < nx) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (iy+1 < ny_part[q+1]) 
               diag_i[cnt]++;
            else
            {
               if (iy+1 < ny) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz+1 < nz_part[r+1]) 
               diag_i[cnt]++;
            else
            {
               if (iz+1 < nz) 
               {
                  offd_i[o_cnt]++;
               }
            }
            cnt++;
            o_cnt++;
         }
      }
   }

   diag_j = hypre_CTAlloc(int, diag_i[local_num_rows]);
   diag_data = hypre_CTAlloc(double, diag_i[local_num_rows]);

   if (num_procs > 1)
   {
      offd_j = hypre_CTAlloc(int, offd_i[local_num_rows]);
      offd_data = hypre_CTAlloc(double, offd_i[local_num_rows]);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0];
            if (iz > nz_part[r]) 
            {
               if (value[3] != 0.0)
               {
                  diag_j[cnt] = row_index-nx_local*ny_local;
                  diag_data[cnt++] = value[3];
               }
            }
            else
            {
               if (iz) 
               {
                  if (value[3] != 0.0)
                  {
                     offd_j[o_cnt] = hypre_mapCD(ix,iy,iz-1,p,q,r-1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                     offd_data[o_cnt++] = value[3];
                  }
               }
            }
            if (iy > ny_part[q]) 
            {
               diag_j[cnt] = row_index-nx_local;
               diag_data[cnt++] = value[2] - 0.5 * beta / (double)(ny - 1);
            }
            else
            {
               if (iy) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix,iy-1,iz,p,q-1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[2] - 0.5 * beta / (double)(ny - 1);
               }
            }
            if (ix > nx_part[p]) 
            {
               diag_j[cnt] = row_index-1;
               diag_data[cnt++] = value[1] - 0.5 * alpha / (double) (nx-1);
            }
            else
            {
               if (ix) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix-1,iy,iz,p-1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[1] - 0.5 * alpha / (double) (nx-1);
               }
            }
            if (ix+1 < nx_part[p+1]) 
            {
               diag_j[cnt] = row_index+1;
               diag_data[cnt++] = value[1] + 0.5 * alpha / (double) (nx-1);
            }
            else
            {
               if (ix+1 < nx) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix+1,iy,iz,p+1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[1] + 0.5 * alpha / (double) (nx-1);
               }
            }
            if (iy+1 < ny_part[q+1]) 
            {
               diag_j[cnt] = row_index+nx_local;
               diag_data[cnt++] = value[2] + 0.5 * beta / (double) (ny-1);
            }
            else
            {
               if (iy+1 < ny) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix,iy+1,iz,p,q+1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[2] + 0.5 * beta / (double) (ny-1);
               }
            }
            if (iz+1 < nz_part[r+1]) 
            {
               if (value[3] != 0.0)
               {
                  diag_j[cnt] = row_index+nx_local*ny_local;
                  diag_data[cnt++] = value[3];
               }
            }
            else
            {
               if (iz+1 < nz) 
               {
                  if (value[3] != 0.0)
                  {
                     offd_j[o_cnt] = hypre_mapCD(ix,iy,iz+1,p,q,r+1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                     offd_data[o_cnt++] = value[3];
                  }
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
      for (i=0; i < num_cols_offd; i++)
         col_map_offd[i] = offd_j[i];
   	
      qsort0(col_map_offd, 0, num_cols_offd-1);

      for (i=0; i < num_cols_offd; i++)
         for (j=0; j < num_cols_offd; j++)
            if (offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
   }

   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, num_cols_offd,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);

   hypre_ParCSRMatrixColMapOffd(A) = col_map_offd;

   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;

   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;
   if (num_cols_offd)
   {
      hypre_CSRMatrixJ(offd) = offd_j;
      hypre_CSRMatrixData(offd) = offd_data;
   }

   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   HYPRE_IJVectorCreate(comm, partition[my_id], partition[my_id+1]-1, &IJrhs);
   free( partition );
   HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(IJrhs);
   HYPRE_IJVectorAssemble(IJrhs);
   HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
   HYPRE_IJVectorSetObjectType(IJrhs, -1);
   HYPRE_IJVectorDestroy(IJrhs);
   hypre_ParVectorSetConstantValues( rhs, 1.0 );
   (*rA) = (HYPRE_ParCSRMatrix) A;
   (*rrhs) = (HYPRE_ParVector) rhs;
   return (0);
}

/* ---------------------------------------------------------------- *
   convection diffusion equation where (upwind difference)
 * ---------------------------------------------------------------- */

int GenerateConvectionDiffusion3D2( MPI_Comm comm, int nx, int ny, int nz, 
            int P, int Q, int R, int p, int q, int r, double alpha, 
            double beta, double  *value, HYPRE_ParCSRMatrix *rA, 
            HYPRE_ParVector *rrhs )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag, *offd;
   HYPRE_IJVector     IJrhs;
   hypre_ParVector    *rhs;

   int                *diag_i, *diag_j, *offd_i, *offd_j;
   double             *diag_data, *offd_data;
   int                *global_part, ix, iy, iz, cnt, o_cnt, local_num_rows; 
   int                *col_map_offd, row_index, i,j, *partition;
   int                nx_local, ny_local, nz_local;
   int                nx_size, ny_size, nz_size, num_cols_offd, grid_size;
   int                *nx_part, *ny_part, *nz_part;
   int                num_procs, my_id, P_busy, Q_busy, R_busy;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

printf("nx,ny,nz = %d %d %d\n", nx, ny, nz);
   grid_size = nx*ny*nz;

   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);
   hypre_GeneratePartitioning(nz,R,&nz_part);

   global_part = hypre_CTAlloc(int,P*Q*R+1);

   global_part[0] = 0;
   cnt = 1;
   for (iz = 0; iz < R; iz++)
   {
      nz_size = nz_part[iz+1]-nz_part[iz];
      for (iy = 0; iy < Q; iy++)
      {
         ny_size = ny_part[iy+1]-ny_part[iy];
         for (ix = 0; ix < P; ix++)
         {
            nx_size = nx_part[ix+1] - nx_part[ix];
            global_part[cnt] = global_part[cnt-1];
            global_part[cnt++] += nx_size*ny_size*nz_size;
         }
      }
   }

   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];
   nz_local = nz_part[r+1] - nz_part[r];

   my_id = r*(P*Q) + q*P + p;
   num_procs = P*Q*R;

   local_num_rows = nx_local*ny_local*nz_local;
   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);

   P_busy = hypre_min(nx,P);
   Q_busy = hypre_min(ny,Q);
   R_busy = hypre_min(nz,R);

   num_cols_offd = 0;
   if (p) num_cols_offd += ny_local*nz_local;
   if (p < P_busy-1) num_cols_offd += ny_local*nz_local;
   if (q) num_cols_offd += nx_local*nz_local;
   if (q < Q_busy-1) num_cols_offd += nx_local*nz_local;
   if (r) num_cols_offd += nx_local*ny_local;
   if (r < R_busy-1) num_cols_offd += nx_local*ny_local;

   if (!local_num_rows) num_cols_offd = 0;

   col_map_offd = hypre_CTAlloc(int, num_cols_offd);

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt-1];
            offd_i[o_cnt] = offd_i[o_cnt-1];
            diag_i[cnt]++;
            if (iz > nz_part[r]) 
               diag_i[cnt]++;
            else
            {
               if (iz) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iy > ny_part[q]) 
               diag_i[cnt]++;
            else
            {
               if (iy) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix > nx_part[p]) 
               diag_i[cnt]++;
            else
            {
               if (ix) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (ix+1 < nx_part[p+1]) 
               diag_i[cnt]++;
            else
            {
               if (ix+1 < nx) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (iy+1 < ny_part[q+1]) 
               diag_i[cnt]++;
            else
            {
               if (iy+1 < ny) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz+1 < nz_part[r+1]) 
               diag_i[cnt]++;
            else
            {
               if (iz+1 < nz) 
               {
                  offd_i[o_cnt]++;
               }
            }
            cnt++;
            o_cnt++;
         }
      }
   }

   diag_j = hypre_CTAlloc(int, diag_i[local_num_rows]);
   diag_data = hypre_CTAlloc(double, diag_i[local_num_rows]);

   if (num_procs > 1)
   {
      offd_j = hypre_CTAlloc(int, offd_i[local_num_rows]);
      offd_data = hypre_CTAlloc(double, offd_i[local_num_rows]);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0] + 0.5 * alpha / (double)(nx - 1) +
                               0.5 * beta / (double) (ny - 1);
            if (iz > nz_part[r]) 
            {
               if (value[3] != 0.0)
               {
                  diag_j[cnt] = row_index-nx_local*ny_local;
                  diag_data[cnt++] = value[3];
               }
            }
            else
            {
               if (iz) 
               {
                  if (value[3] != 0.0)
                  {
                     offd_j[o_cnt] = hypre_mapCD(ix,iy,iz-1,p,q,r-1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                     offd_data[o_cnt++] = value[3];
                  }
               }
            }
            if (iy > ny_part[q]) 
            {
               diag_j[cnt] = row_index-nx_local;
               diag_data[cnt++] = value[2] - 0.5 * beta / (double)(ny - 1);
            }
            else
            {
               if (iy) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix,iy-1,iz,p,q-1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[2] - 0.5 * beta / (double)(ny - 1);
               }
            }
            if (ix > nx_part[p]) 
            {
               diag_j[cnt] = row_index-1;
               diag_data[cnt++] = value[1] - 0.5 * alpha / (double) (nx-1);
            }
            else
            {
               if (ix) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix-1,iy,iz,p-1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[1] - 0.5 * alpha / (double) (nx-1);
               }
            }
            if (ix+1 < nx_part[p+1]) 
            {
               diag_j[cnt] = row_index+1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix+1 < nx) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix+1,iy,iz,p+1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (iy+1 < ny_part[q+1]) 
            {
               diag_j[cnt] = row_index+nx_local;
               diag_data[cnt++] = value[2];
            }
            else
            {
               if (iy+1 < ny) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix,iy+1,iz,p,q+1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[2];
               }
            }
            if (iz+1 < nz_part[r+1]) 
            {
               if (value[3] != 0.0)
               {
                  diag_j[cnt] = row_index+nx_local*ny_local;
                  diag_data[cnt++] = value[3];
               }
            }
            else
            {
               if (iz+1 < nz) 
               {
                  if (value[3] != 0.0)
                  {
                     offd_j[o_cnt] = hypre_mapCD(ix,iy,iz+1,p,q,r+1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                     offd_data[o_cnt++] = value[3];
                  }
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
      for (i=0; i < num_cols_offd; i++)
         col_map_offd[i] = offd_j[i];
   	
      qsort0(col_map_offd, 0, num_cols_offd-1);

      for (i=0; i < num_cols_offd; i++)
         for (j=0; j < num_cols_offd; j++)
            if (offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
   }

   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, num_cols_offd,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);

   hypre_ParCSRMatrixColMapOffd(A) = col_map_offd;

   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;

   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;
   if (num_cols_offd)
   {
      hypre_CSRMatrixJ(offd) = offd_j;
      hypre_CSRMatrixData(offd) = offd_data;
   }

   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   HYPRE_IJVectorCreate(comm, partition[my_id], partition[my_id+1]-1, &IJrhs);
   free( partition );
   HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(IJrhs);
   HYPRE_IJVectorAssemble(IJrhs);
   HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
   HYPRE_IJVectorSetObjectType(IJrhs, -1);
   HYPRE_IJVectorDestroy(IJrhs);
   hypre_ParVectorSetConstantValues( rhs, 1.0 );
   (*rA) = (HYPRE_ParCSRMatrix) A;
   (*rrhs) = (HYPRE_ParVector) rhs;
   return (0);
}
/* ---------------------------------------------------------------- *
   convection diffusion equation 
   - epsilon del^2 u + a(x,y) u_x + b(x,y) u_y
     a(x,y) = sin (L * pi/8), b(x,y) = cos(L*pi/8), L - input
 * ---------------------------------------------------------------- */

int GenerateRugeStuben1(MPI_Comm comm, double Lval, double epsilon, int n, 
                        HYPRE_ParCSRMatrix *rA, HYPRE_ParVector *rrhs) 
{
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *rhs;
   hypre_CSRMatrix    *diag, *offd;
   HYPRE_IJVector     IJrhs;

   int    *diag_i, *diag_j, *offd_i, mypid;
   double *diag_data, sum1, L=2.0;

   int *global_part, cnt, local_num_rows, *partition; 
   int row_index, i,j, grid_size;
   double h, ac, bc, pi=3.1415928, mu_x, mu_y;

   h = 1.0 / (n + 1.0);
   ac = sin(Lval * pi / 8.0);
   bc = cos(Lval * pi / 8.0);
   if      ( ac * h >= epsilon )  mu_x = epsilon / ( 2 * ac * h );
   else if ( ac * h < - epsilon ) mu_x = 1 + epsilon / ( 2 * ac * h );
   else                           mu_x = 0.5;
   if       ( bc* h >= epsilon )  mu_y = epsilon / ( 2 * bc* h );
   else if ( bc* h < - epsilon )  mu_y = 1 + epsilon / ( 2 * bc* h );
   else                           mu_y = 0.5;

   grid_size = n * n;
   global_part = hypre_CTAlloc(int,2);
   global_part[0] = 0;
   global_part[1] = grid_size;
   local_num_rows = grid_size;
   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);
   for ( i = 0; i <= local_num_rows; i++ ) offd_i[i] = 0; 
   diag_j = hypre_CTAlloc(int, 5*local_num_rows);
   diag_data = hypre_CTAlloc(double, 5*local_num_rows);
   cnt = 0;
   diag_i[0] = 0;
   for ( j = 0; j < n; j++ ) 
   {
      for ( i = 0; i < n; i++ ) 
      {
         row_index = j * n + i;
         cnt++;
         sum1 = 0.0;
         if ( j > 0 )
         {
            diag_j[cnt] = row_index - n;
            diag_data[cnt++] = - epsilon + bc * h * (mu_y - 1);
         }
         sum1 = sum1 + epsilon - bc * h * (mu_y - 1);
         if ( i > 0 )
         {
            diag_j[cnt] = row_index - 1;
            diag_data[cnt++] = - epsilon + ac * h * (mu_x - 1);
         }
         sum1 = sum1 + epsilon - ac * h * (mu_x - 1);
         if ( i < n-1 )
         {
            diag_j[cnt] = row_index + 1;
            diag_data[cnt++] = - epsilon + ac * h * mu_x;
         }
         sum1 = sum1 + epsilon - ac * h * mu_x;
         if ( j < n-1 )
         {
            diag_j[cnt] = row_index + n;
            diag_data[cnt++] = - epsilon + bc * h * mu_y;
         }
         sum1 = sum1 + epsilon - bc * h * mu_y;
         diag_j[diag_i[row_index]] = row_index;
         diag_data[diag_i[row_index]] = sum1;
         diag_i[row_index+1] = cnt;
      }
   }
      
   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, 0,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);
/*
   hypre_ParCSRMatrixColMapOffd(A) = NULL;
*/
   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;
   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   MPI_Comm_rank(comm, &mypid);
   HYPRE_IJVectorCreate(comm, partition[mypid], partition[mypid+1]-1, &IJrhs);
   free( partition );
   HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(IJrhs);
   HYPRE_IJVectorAssemble(IJrhs);
   HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
   HYPRE_IJVectorSetObjectType(IJrhs, -1);
   HYPRE_IJVectorDestroy(IJrhs);
   hypre_ParVectorSetConstantValues( rhs, 1.0 );
   (*rA) = (HYPRE_ParCSRMatrix) A;
   (*rrhs) = (HYPRE_ParVector) rhs;
   return (0);
}

/* ---------------------------------------------------------------- *
   convection diffusion equation 
   - epsilon del^2 u + a(x,y) u_x + b(x,y) u_y
     a(x,y) = (2y-1)(1-x^2), b(x,y) = 2xy(y-1)
 * ---------------------------------------------------------------- */

int GenerateRugeStuben2(MPI_Comm comm, double epsilon, int n, 
                        HYPRE_ParCSRMatrix *rA, HYPRE_ParVector *rrhs) 
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag, *offd;
   hypre_ParVector    *rhs;
   HYPRE_IJVector     IJrhs;

   int                *diag_i, *diag_j, *offd_i;
   double             *diag_data, sum1;

   int                *global_part, cnt, local_num_rows, *partition; 
   int                mypid, row_index, i,j, grid_size;
   double             h, ac, bc, mu_x, mu_y;

   h = 1.0 / (n + 1.0);

   grid_size = n * n;
   global_part = hypre_CTAlloc(int,2);
   global_part[0] = 0;
   global_part[1] = grid_size;
   local_num_rows = grid_size;
   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);
   for ( i = 0; i <= local_num_rows; i++ ) offd_i[i] = 0; 
   diag_j = hypre_CTAlloc(int, 5*local_num_rows);
   diag_data = hypre_CTAlloc(double, 5*local_num_rows);
   cnt = 0;
   diag_i[0] = 0;
   for ( j = 0; j < n; j++ ) 
   {
      for ( i = 0; i < n; i++ ) 
      {
         row_index = j * n + i;
         cnt++;
         sum1 = 0.0;
         ac = ( 2.0 * j * h - 1.0 ) * ( 1.0 - i * i * h * h );
         bc = 2.0 * i * h * j * h * ( j * h - 1.0);
         if      ( ac * h >= epsilon )  mu_x = epsilon / ( 2 * ac * h );
         else if ( ac * h < - epsilon ) mu_x = 1 + epsilon / ( 2 * ac * h );
         else                           mu_x = 0.5;
         if ( bc* h > epsilon )         mu_y = epsilon / ( 2 * bc* h );
         else if ( bc* h < - epsilon )  mu_y = 1 + epsilon / ( 2 * bc* h );
         else                           mu_y = 0.5;
         if ( j > 0 )
         {
            diag_j[cnt] = row_index - n;
            diag_data[cnt++] = - epsilon + bc * h * (mu_y - 1);
         }
         sum1 = sum1 + epsilon - bc * h * (mu_y - 1);
         if ( i > 0 )
         {
            diag_j[cnt] = row_index - 1;
            diag_data[cnt++] = - epsilon + ac * h * (mu_x - 1);
         }
         sum1 = sum1 + epsilon - ac * h * (mu_x - 1);
         if ( i < n-1 )
         {
            diag_j[cnt] = row_index + 1;
            diag_data[cnt++] = - epsilon + ac * h * mu_x;
         }
         sum1 = sum1 + epsilon - ac * h * mu_x;
         if ( j < n-1 )
         {
            diag_j[cnt] = row_index + n;
            diag_data[cnt++] = - epsilon + bc * h * mu_y;
         }
         sum1 = sum1 + epsilon - bc * h * mu_y;
         diag_j[diag_i[row_index]] = row_index;
         diag_data[diag_i[row_index]] = sum1;
         diag_i[row_index+1] = cnt;
      }
   }
      
   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, 0,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);
/*
   hypre_ParCSRMatrixColMapOffd(A) = NULL;
*/
   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;
   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   MPI_Comm_rank(comm, &mypid);
   HYPRE_IJVectorCreate(comm, partition[mypid], partition[mypid+1]-1, &IJrhs);
   free( partition );
   HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(IJrhs);
   HYPRE_IJVectorAssemble(IJrhs);
   HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
   HYPRE_IJVectorSetObjectType(IJrhs, -1);
   HYPRE_IJVectorDestroy(IJrhs);
   hypre_ParVectorSetConstantValues( rhs, 1.0 );
   (*rA) = (HYPRE_ParCSRMatrix) A;
   (*rrhs) = (HYPRE_ParVector) rhs;
   return (0);
}

/* ---------------------------------------------------------------- *
   convection diffusion equation 
   - epsilon del^2 u + a(x,y) u_x + b(x,y) u_y
     a(x,y) = 4x(x-1)(1-2y), b(x,y) = -4y(y-1)(1-2x)
 * ---------------------------------------------------------------- */

int GenerateRugeStuben3(MPI_Comm comm, double epsilon, int n,
                        HYPRE_ParCSRMatrix *rA,HYPRE_ParVector *rrhs) 
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag, *offd;
   hypre_ParVector    *rhs;
   HYPRE_IJVector     IJrhs;
   int                *diag_i, *diag_j, *offd_i;
   double             *diag_data, sum1;
   int                *global_part, cnt, local_num_rows, *partition; 
   int                row_index, i,j, grid_size, mypid, nprocs;
   double             h, ac, bc, mu_x, mu_y;

   MPI_Comm_size(comm, &nprocs);
   if ( nprocs != 1 )
   {  
      printf("GenerateStuben ERROR : nprocs > 1\n");
      exit(1);
   }

   h = 1.0 / (n + 1.0);

   grid_size = n * n;
   global_part = hypre_CTAlloc(int,2);
   global_part[0] = 0;
   global_part[1] = grid_size;
   local_num_rows = grid_size;
   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);
   for ( i = 0; i <= local_num_rows; i++ ) offd_i[i] = 0; 
   diag_j = hypre_CTAlloc(int, 5*local_num_rows);
   diag_data = hypre_CTAlloc(double, 5*local_num_rows);
   cnt = 0;
   diag_i[0] = 0;
   for ( j = 0; j < n; j++ ) 
   {
      for ( i = 0; i < n; i++ ) 
      {
         row_index = j * n + i;
         cnt++;
         sum1 = 0.0;
         ac = 4.0 * i * h * ( i * h - 1.0 ) * ( 1.0 - 2 * j * h );
         bc = -4.0 * j * h * ( j * h - 1.0) * ( 1.0 - 2 * i * h );
         if      ( ac * h >= epsilon )  mu_x = epsilon / ( 2 * ac * h );
         else if ( ac * h < - epsilon ) mu_x = 1 + epsilon / ( 2 * ac * h );
         else                           mu_x = 0.5;
         if ( bc* h > epsilon )         mu_y = epsilon / ( 2 * bc* h );
         else if ( bc* h < - epsilon )  mu_y = 1 + epsilon / ( 2 * bc* h );
         else                           mu_y = 0.5;
         if ( j > 0 )
         {
            diag_j[cnt] = row_index - n;
            diag_data[cnt++] = - epsilon + bc * h * (mu_y - 1);
         }
         sum1 = sum1 + epsilon - bc * h * (mu_y - 1);
         if ( i > 0 )
         {
            diag_j[cnt] = row_index - 1;
            diag_data[cnt++] = - epsilon + ac * h * (mu_x - 1);
         }
         sum1 = sum1 + epsilon - ac * h * (mu_x - 1);
         if ( i < n-1 )
         {
            diag_j[cnt] = row_index + 1;
            diag_data[cnt++] = - epsilon + ac * h * mu_x;
         }
         sum1 = sum1 + epsilon - ac * h * mu_x;
         if ( j < n-1 )
         {
            diag_j[cnt] = row_index + n;
            diag_data[cnt++] = - epsilon + bc * h * mu_y;
         }
         sum1 = sum1 + epsilon - bc * h * mu_y;
         diag_j[diag_i[row_index]] = row_index;
         diag_data[diag_i[row_index]] = sum1;
         diag_i[row_index+1] = cnt;
      }
   }
      
   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, 0,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);
/*
   hypre_ParCSRMatrixColMapOffd(A) = NULL;
*/
   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;
   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   MPI_Comm_rank(comm, &mypid);
   HYPRE_IJVectorCreate(comm, partition[mypid], partition[mypid+1]-1, &IJrhs);
   free( partition );
   HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(IJrhs);
   HYPRE_IJVectorAssemble(IJrhs);
   HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
   HYPRE_IJVectorSetObjectType(IJrhs, -1);
   HYPRE_IJVectorDestroy(IJrhs);
   hypre_ParVectorSetConstantValues( rhs, 1.0 );
   (*rA) = (HYPRE_ParCSRMatrix) A;
   (*rrhs) = (HYPRE_ParVector) rhs;
   return (0);
}

/* ---------------------------------------------------------------- *
   convection diffusion equation 
   - epsilon del^2 u + a(x,y) u_x + b(x,y) u_y
     a(x,y) = -sin(pi*x)cos(pi*y), b(x,y)=sin(pi*y)cos(pi*x)
 * ---------------------------------------------------------------- */

int GenerateStuben(MPI_Comm comm, double epsilon, int n,
                   HYPRE_ParCSRMatrix *rA, HYPRE_ParVector *rrhs) 
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *diag, *offd;
   int                *diag_i, *diag_j, *offd_i;
   double             *diag_data, sum1, *rhsVec;
   int                *global_part, cnt, local_num_rows, *colInd; 
   int                row_index, i,j, grid_size, mypid, nprocs, *partition;
   double             h, ac, bc, pi=3.1415928;
   hypre_ParVector    *rhs;
   HYPRE_IJVector     IJrhs;

   MPI_Comm_size(comm, &nprocs);
   if ( nprocs != 1 )
   {  
      printf("GenerateStuben ERROR : nprocs > 1\n");
      exit(1);
   }
   MPI_Comm_rank(comm, &mypid);

   h = 1.0 / (n + 1.0);

   grid_size = n * n;
   global_part = hypre_CTAlloc(int,2);
   global_part[0] = 0;
   global_part[1] = grid_size;
   local_num_rows = grid_size;
   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);
   for ( i = 0; i <= local_num_rows; i++ ) offd_i[i] = 0; 
   diag_j = hypre_CTAlloc(int, 5*local_num_rows);
   diag_data = hypre_CTAlloc(double, 5*local_num_rows);
   cnt = 0;
   diag_i[0] = 0;
   rhsVec = (double *) malloc(grid_size * sizeof(double));
   for ( j = 0; j < n; j++ ) 
   {
      for ( i = 0; i < n; i++ ) 
      {
         row_index = j * n + i;
         cnt++;
         rhsVec[row_index] = 1.0;
         sum1 = 0.0;
         ac = - sin(pi*(i+0.5)*h) * cos(pi*(j+0.5)*h); 
         bc = sin(pi*(j+0.5)*h) * cos(pi*(i+0.5)*h); 
         if ( j > 0 )
         {
            diag_j[cnt] = row_index - n;
            diag_data[cnt++] = - epsilon - bc * h;
         }
         sum1 = sum1 + epsilon + bc * h;
         if ( j == 0 ) 
            rhsVec[row_index] -= (epsilon+bc*h)*
                                 (sin(pi*(i+1)*h)+sin(13.0*pi*(i+1)*h));
         if ( i > 0 )
         {
            diag_j[cnt] = row_index - 1;
            diag_data[cnt++] = - epsilon - ac * h;
         }
         sum1 = sum1 + epsilon + ac * h;
         if ( i == 0 ) 
            rhsVec[row_index] -= (epsilon+ac*h)*
                                 (sin(pi*(j+1)*h)+sin(13.0*pi*(j+1)*h));
         if ( i < n-1 )
         {
            diag_j[cnt] = row_index + 1;
            diag_data[cnt++] = - epsilon;
         }
         sum1 = sum1 + epsilon;
         if ( i == (n-1) ) 
            rhsVec[row_index] -= epsilon *
                                 (sin(pi*(j+1)*h)+sin(13.0*pi*(j+1)*h));
         if ( j < n-1 )
         {
            diag_j[cnt] = row_index + n;
            diag_data[cnt++] = - epsilon;
         }
         sum1 = sum1 + epsilon;
         if ( j == (n-1) ) 
            rhsVec[row_index] -= epsilon *
                                 (sin(pi*(i+1)*h)+sin(13.0*pi*(i+1)*h));
         diag_j[diag_i[row_index]] = row_index;
         diag_data[diag_i[row_index]] = sum1;
         diag_i[row_index+1] = cnt;
      }
   }
      
   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, 0,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);
   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;
   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   HYPRE_IJVectorCreate(comm, partition[mypid], partition[mypid+1]-1, &IJrhs);
   free( partition );
   HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(IJrhs);
   colInd = (int *) malloc(grid_size * sizeof(int));
   for (j = 0; j < grid_size; j++) colInd[j] = j;
   HYPRE_IJVectorSetValues(IJrhs, grid_size, (const int *) colInd,
                                 (const double *) rhsVec);
   free( colInd );
   free( rhsVec );
   HYPRE_IJVectorAssemble(IJrhs);
   HYPRE_IJVectorGetObject(IJrhs, (void**) &rhs);
   HYPRE_IJVectorSetObjectType(IJrhs, -1);
   HYPRE_IJVectorDestroy(IJrhs);
   hypre_ParVectorSetConstantValues( rhs, 1.0 );
   (*rA) = (HYPRE_ParCSRMatrix) A;
   (*rrhs) = (HYPRE_ParVector) rhs;
   return (0);
}

