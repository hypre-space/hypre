/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_LinSysCore.h"

#define habs(x) ((x > 0) ? x : -(x))

//---------------------------------------------------------------------------
// _hypre_parcsr_mv.h is put here instead of in HYPRE_LinSysCore.h
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "parcsr_mv/_hypre_parcsr_mv.h"

//---------------------------------------------------------------------------
// These are external functions needed internally here
//---------------------------------------------------------------------------

extern "C" {
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix**);
   int HYPRE_LSI_Search(int*, int, int);
   void hypre_qsort0(int *, int, int);
   void hypre_qsort1(int *, double *, int, int);
}

//*****************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSchurReducedSystem()
{
    int    i, j, ierr, ncnt, ncnt2;
    int    nRows, globalNRows, StartRow, EndRow, colIndex;
    int    nSchur, *schurList, globalNSchur, *globalSchurList;
    int    CStartRow, CNRows, CNCols, CGlobalNRows, CGlobalNCols;
    int    CTStartRow, CTNRows, CTNCols, CTGlobalNRows, CTGlobalNCols;
    int    MStartRow, MNRows, MNCols, MGlobalNRows, MGlobalNCols;
    int    rowSize, rowCount, rowIndex, maxRowSize, newRowSize;
    int    *CMatSize, *CTMatSize, *MMatSize, *colInd, *newColInd, one=1;
    int    *tempList, *recvCntArray, *displArray, *colInd2, rowSize2;
    int    procIndex, *ProcNRows, *ProcNSchur, searchIndex, CStartCol;
    double *colVal, *colVal2, *newColVal, *diagonal, ddata, maxdiag, mindiag;

    HYPRE_IJMatrix     Cmat, CTmat, Mmat;
    HYPRE_ParCSRMatrix A_csr, C_csr, CT_csr, M_csr, S_csr;
    HYPRE_IJVector     f1, f2, f2hat;
    HYPRE_ParVector    f1_csr, f2hat_csr;

    //******************************************************************
    // initial clean up and set up
    //------------------------------------------------------------------

    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
       printf("       buildSchurSystem begins....\n");
    if ( HYA21_    != NULL ) HYPRE_IJMatrixDestroy(HYA21_);
    if ( HYA12_    != NULL ) HYPRE_IJMatrixDestroy(HYA12_);
    if ( HYinvA22_ != NULL ) HYPRE_IJMatrixDestroy(HYinvA22_);
    if ( reducedB_ != NULL ) HYPRE_IJVectorDestroy(reducedB_);
    if ( reducedX_ != NULL ) HYPRE_IJVectorDestroy(reducedX_);
    if ( reducedR_ != NULL ) HYPRE_IJVectorDestroy(reducedR_);
    if ( reducedA_ != NULL ) HYPRE_IJMatrixDestroy(reducedA_);
    HYA21_    = NULL;
    HYA12_    = NULL;
    HYinvA22_ = NULL;
    reducedB_ = NULL;
    reducedX_ = NULL;
    reducedR_ = NULL;
    reducedA_ = NULL;

    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    nRows    = localEndRow_ - localStartRow_ + 1;
    HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       printf("%4d : buildSchurSystem - StartRow/EndRow = %d %d\n",mypid_,
                                        StartRow,EndRow);

    //******************************************************************
    // construct local and global information about where the constraints
    // are (this is given by user or searched within this code)
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // get information about processor offsets and globalNRows
    // (ProcNRows, globalNRows)
    //------------------------------------------------------------------

    ProcNRows   = new int[numProcs_];
    tempList    = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = StartRow;
    MPI_Allreduce(tempList, ProcNRows, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    MPI_Allreduce(&nRows, &globalNRows,1,MPI_INT,MPI_SUM,comm_);

    //******************************************************************
    // compose the local and global Schur node lists
    //------------------------------------------------------------------

    nSchur = 0;
    for ( i = StartRow; i <= EndRow; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       searchIndex = globalNRows + 1;
       for (j = 0; j < rowSize; j++)
       {
          colIndex = colInd[j];
          if ( colIndex < searchIndex && colVal[j] != 0.0 )
             searchIndex = colIndex;
       }
       if ( searchIndex < i ) nSchur++;
       //searchIndex = -1;
       //for (j = 0; j < rowSize; j++)
       //{
       //   colIndex = colInd[j];
       //   if ( colIndex < i && colVal[j] != 0.0 )
       //      if ( colIndex > searchIndex ) searchIndex = colIndex;
       //}
       //if ( searchIndex >= StartRow ) nSchur++;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }
    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       printf("%4d : buildSchurSystem - nSchur = %d\n",mypid_,nSchur);

    //------------------------------------------------------------------
    // allocate array for storing indices of selected nodes
    //------------------------------------------------------------------

    if ( nSchur > 0 ) schurList = new int[nSchur];
    else              schurList = NULL;

    //------------------------------------------------------------------
    // compose the list of rows having zero diagonal
    // (nSchur, schurList)
    //------------------------------------------------------------------

    nSchur = 0;
    for ( i = StartRow; i <= EndRow; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       searchIndex = globalNRows + 1;
       for (j = 0; j < rowSize; j++)
       {
          colIndex = colInd[j];
          if ( colIndex < searchIndex && colVal[j] != 0.0 )
             searchIndex = colIndex;
       }
       if ( searchIndex < i ) schurList[nSchur++] = i;
       //searchIndex = -1;
       //for (j = 0; j < rowSize; j++)
       //{
       //   colIndex = colInd[j];
       //   if ( colIndex < i && colVal[j] != 0.0 )
       //      if ( colIndex > searchIndex ) searchIndex = colIndex;
       //}
       //if ( searchIndex >= StartRow ) schurList[nSchur++] = i;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }

    //------------------------------------------------------------------
    // compose the global list of rows having zero diagonal
    // (globalNSchur, globalSchurList)
    //------------------------------------------------------------------

    MPI_Allreduce(&nSchur, &globalNSchur, 1, MPI_INT, MPI_SUM,comm_);

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       if ( globalNSchur == 0 && mypid_ == 0 )
          printf("buildSchurSystem WARNING : nSchur = 0 on all processors.\n");
    }
    if ( globalNSchur == 0 )
    {
       schurReduction_ = 0;
       delete [] ProcNRows;
       return;
    }

    if ( globalNSchur > 0 ) globalSchurList = new int[globalNSchur];
    else                    globalSchurList = NULL;

    recvCntArray = new int[numProcs_];
    displArray   = new int[numProcs_];
    MPI_Allgather(&nSchur, 1, MPI_INT, recvCntArray, 1, MPI_INT, comm_);
    displArray[0] = 0;
    for ( i = 1; i < numProcs_; i++ )
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    MPI_Allgatherv(schurList, nSchur, MPI_INT, globalSchurList,
                   recvCntArray, displArray, MPI_INT, comm_);
    delete [] recvCntArray;
    delete [] displArray;

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE2 )
    {
       for ( i = 0; i < nSchur; i++ )
          printf("%4d : buildSchurSystem - schurList %d = %d\n",mypid_,
                 i,schurList[i]);
    }

    //------------------------------------------------------------------
    // get information about processor offsets for nSchur
    // (ProcNSchur)
    //------------------------------------------------------------------

    ProcNSchur = new int[numProcs_];
    tempList   = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nSchur;
    MPI_Allreduce(tempList, ProcNSchur, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    globalNSchur = 0;
    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ )
    {
       globalNSchur  += ProcNSchur[i];
       ncnt2         = ProcNSchur[i];
       ProcNSchur[i] = ncnt;
       ncnt          += ncnt2;
    }

    //******************************************************************
    // construct Cmat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of Cmat
    //------------------------------------------------------------------

    CNRows = nSchur;
    CNCols = nRows - nSchur;
    CGlobalNRows = globalNSchur;
    CGlobalNCols = globalNRows - globalNSchur;
    CStartRow    = ProcNSchur[mypid_];

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d : buildSchurSystem : CStartRow  = %d\n",mypid_,CStartRow);
       printf("%4d : buildSchurSystem : CGlobalDim = %d %d\n", mypid_,
                                        CGlobalNRows, CGlobalNCols);
       printf("%4d : buildSchurSystem : CLocalDim  = %d %d\n",mypid_,
                                        CNRows, CNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for Cmat
    //------------------------------------------------------------------

    CStartCol = ProcNRows[mypid_] - ProcNSchur[mypid_];
    ierr  = HYPRE_IJMatrixCreate(comm_, CStartRow, CStartRow+CNRows-1,
				 CStartCol, CStartCol+CNCols-1, &Cmat);
    ierr += HYPRE_IJMatrixSetObjectType(Cmat, HYPRE_PARCSR);
    hypre_assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros per row in Cmat and call set up
    //------------------------------------------------------------------

    maxRowSize = 0;
    CMatSize = new int[CNRows];

    for ( i = 0; i < nSchur; i++ )
    {
       rowIndex = schurList[i];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++)
       {
          colIndex = colInd[j];
	  searchIndex = hypre_BinarySearch(globalSchurList,colIndex,
                                           globalNSchur);
          if (searchIndex < 0) newRowSize++;
          else if ( colVal[j] != 0.0 )
          {
             if ( HYOutputLevel_ & HYFEI_SCHURREDUCE3 )
             {
                printf("%4d : buildSchurSystem WARNING : A22 block != 0\n",
                       mypid_);
                printf(" Cmat[%4d,%4d] = %e\n",rowIndex,colIndex,colVal[j]);
             }
          }
       }
       CMatSize[i] = newRowSize;
       maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(Cmat, CMatSize);
    ierr += HYPRE_IJMatrixInitialize(Cmat);
    hypre_assert(!ierr);
    delete [] CMatSize;

    //------------------------------------------------------------------
    // load Cmat extracted from A
    //------------------------------------------------------------------

    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];
    rowCount  = CStartRow;
    for ( i = 0; i < nSchur; i++ )
    {
       rowIndex = schurList[i];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++)
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
	     searchIndex = HYPRE_LSI_Search(globalSchurList,colIndex,
                                            globalNSchur);
             if ( searchIndex < 0 )
             {
                searchIndex = - searchIndex - 1;
                for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                procIndex--;
                colIndex = colInd[j] - searchIndex;
                //colIndex = colInd[j]-ProcNSchur[procIndex]-searchIndex;
                newColInd[newRowSize]   = colIndex;
                newColVal[newRowSize++] = colVal[j];
                if ( colIndex < 0 || colIndex >= CGlobalNCols )
                {
                   if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
                   {
                      printf("%4d : buildSchurSystem WARNING - Cmat ", mypid_);
                      printf("out of range %d - %d (%d)\n", rowCount,colIndex,
                              CGlobalNCols);
                   }
                }
                if ( newRowSize > maxRowSize+1 )
                {
                   if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
                   {
                      printf("%4d buildSchurSystem : WARNING - ",mypid_);
                      printf("passing array boundary(1).\n");
                   }
                }
             }
          }
       }
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       HYPRE_IJMatrixSetValues(Cmat, 1, &newRowSize, (const int *) &rowCount,
		(const int *) newColInd, (const double *) newColVal);
       rowCount++;
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(Cmat);
    HYPRE_IJMatrixGetObject(Cmat, (void **) &C_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) C_csr);

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             printf("%4d buildSchurSystem : matrix Cmat assembled %d.\n",
                                           mypid_,CStartRow);
             fflush(stdout);
             for ( i = CStartRow; i < CStartRow+nSchur; i++ )
             {
                HYPRE_ParCSRMatrixGetRow(C_csr,i,&rowSize,&colInd,&colVal);
                printf("Cmat ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(C_csr,i,&rowSize,&colInd,&colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(MPI_COMM_WORLD);
       }
    }

    //******************************************************************
    // construct the diagonal Mmat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of Mmat
    //------------------------------------------------------------------

    MNRows = nRows - nSchur;
    MNCols = nRows - nSchur;
    MGlobalNRows = globalNRows - globalNSchur;
    MGlobalNCols = globalNRows - globalNSchur;
    MStartRow    = ProcNRows[mypid_] - ProcNSchur[mypid_];
    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d : buildSchurSystem - MStartRow  = %d\n",mypid_,MStartRow);
       printf("%4d : buildSchurSystem - MGlobalDim = %d %d\n", mypid_,
                                        MGlobalNRows, MGlobalNCols);
       printf("%4d : buildSchurSystem - MLocalDim  = %d %d\n",mypid_,
                                        MNRows, MNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for Mmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_, MStartRow, MStartRow+MNRows-1,
				 MStartRow, MStartRow+MNCols-1, &Mmat);
    ierr += HYPRE_IJMatrixSetObjectType(Mmat, HYPRE_PARCSR);

    MMatSize = new int[MNRows];
    for ( i = 0; i < MNRows; i++ ) MMatSize[i] = 1;
    ierr  = HYPRE_IJMatrixSetRowSizes(Mmat, MMatSize);
    ierr += HYPRE_IJMatrixInitialize(Mmat);
    hypre_assert(!ierr);
    delete [] MMatSize;

    //------------------------------------------------------------------
    // load Mmat
    //------------------------------------------------------------------

    maxdiag = -1.0E10;
    mindiag =  1.0E10;
    diagonal = new double[MNRows];
    rowIndex = MStartRow;
    ierr     = 0;
    for ( i = StartRow; i <= EndRow; i++ )
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          ncnt = 0;
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          for (j = 0; j < rowSize; j++)
          {
             colIndex = colInd[j];
             if ( colIndex == i && colVal[j] != 0.0 )
             {
                ddata = 1.0 / colVal[j];
                maxdiag = ( colVal[j] > maxdiag ) ? colVal[j] : maxdiag;
                mindiag = ( colVal[j] < mindiag ) ? colVal[j] : mindiag;
                break;
             }
             if ( colVal[j] != 0.0 ) ncnt++;
          }
          if ( j == rowSize )
          {
             if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
                printf("%4d : buildSchurSystem WARNING - diag[%d] not found\n",
                     mypid_, i);
             ierr = 1;
          }
          else if ( ncnt > 1 ) ierr = 1;
          diagonal[rowIndex-MStartRow] = ddata;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_IJMatrixSetValues(Mmat, 1, &one, (const int *) &rowIndex,
		(const int *) &rowIndex, (const double *) &ddata);
          rowIndex++;
       }
    }
    ddata = maxdiag;
    MPI_Allreduce(&ddata, &maxdiag, 1, MPI_DOUBLE, MPI_MAX, comm_);
    ddata = -mindiag;
    MPI_Allreduce(&ddata, &mindiag, 1, MPI_DOUBLE, MPI_MAX, comm_);
    mindiag = - mindiag;
    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1))
    {
       printf("%4d : buildSchurSystem : max diagonal = %e\n",mypid_,maxdiag);
       printf("%4d : buildSchurSystem : min diagonal = %e\n",mypid_,mindiag);
    }

    //------------------------------------------------------------------
    // finally assemble Mmat
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(Mmat);
    HYPRE_IJMatrixGetObject(Mmat, (void **) &M_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) M_csr);

    //------------------------------------------------------------------
    // Error checking
    //------------------------------------------------------------------

    MPI_Allreduce(&ierr, &ncnt, 1, MPI_INT, MPI_SUM, comm_);
    if ( ncnt > 0 )
    {
       if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       {
          printf("buildSchurSystem WARNING : A11 not diagonal\n");
          printf("buildSchurSystem WARNING : reduction not performed.\n");
       }
       schurReduction_ = 0;
       delete [] ProcNRows;
       delete [] ProcNSchur;
       if ( nSchur > 0 )       delete [] schurList;
       if ( globalNSchur > 0 ) delete [] globalSchurList;
       HYPRE_IJMatrixDestroy(Cmat);
       return;
    }

    //******************************************************************
    // construct CTmat (transpose of Cmat)
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of CTmat
    //------------------------------------------------------------------

    CTNRows = CNCols;
    CTNCols = CNRows;
    CTGlobalNRows = CGlobalNCols;
    CTGlobalNCols = CGlobalNRows;
    CTStartRow    = ProcNRows[mypid_] - ProcNSchur[mypid_];

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d : buildSchurSystem - CTStartRow  = %d\n",mypid_,CTStartRow);
       printf("%4d : buildSchurSystem - CTGlobalDim = %d %d\n", mypid_,
                                        CTGlobalNRows, CTGlobalNCols);
       printf("%4d : buildSchurSystem - CTLocalDim  = %d %d\n",mypid_,
                                        CTNRows, CTNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for CTmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_, CTStartRow, CTStartRow+CTNRows-1,
                                 CStartRow, CStartRow+CTNCols-1, &CTmat);
    ierr += HYPRE_IJMatrixSetObjectType(CTmat, HYPRE_PARCSR);
    hypre_assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros per row in CTmat and call set up
    //------------------------------------------------------------------

    maxRowSize = 0;
    CTMatSize = new int[CTNRows];

    rowCount = 0;
    for ( i = StartRow; i <= EndRow; i++ )
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          newRowSize = 0;
          for (j = 0; j < rowSize; j++)
          {
             colIndex = colInd[j];
             searchIndex = hypre_BinarySearch(globalSchurList,colIndex,
                                              globalNSchur);
             if (searchIndex >= 0) newRowSize++;
          }
          //if ( newRowSize <= 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
          //   printf("%d : WARNING at row %d - empty row.\n", mypid_, i);
          if ( newRowSize <= 0 ) newRowSize = 1;
          CTMatSize[rowCount++] = newRowSize;
          maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(CTmat, CTMatSize);
    ierr += HYPRE_IJMatrixInitialize(CTmat);
    hypre_assert(!ierr);
    delete [] CTMatSize;

    //------------------------------------------------------------------
    // load CTmat extracted from A
    //------------------------------------------------------------------

    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];
    rowCount  = CTStartRow;

    for ( i = StartRow; i <= EndRow; i++ )
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          newRowSize = 0;
          for (j = 0; j < rowSize; j++)
          {
             colIndex = colInd[j];
             searchIndex = hypre_BinarySearch(globalSchurList,colIndex,
                                              globalNSchur);
             if (searchIndex >= 0)
             {
                newColInd[newRowSize] = searchIndex;
                if ( searchIndex >= globalNSchur )
                {
                   if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
                   {
                      printf("%4d : buildSchurSystem WARNING - CTmat ",mypid_);
                      printf("out of range %d - %d (%d)\n", rowCount,
                             searchIndex, globalNSchur);
                   }
                }
                newColVal[newRowSize++] = colVal[j];
             }
          }
          if ( newRowSize == 0 )
          {
             newColInd[0] = ProcNSchur[mypid_];
             newColVal[0] = 0.0;
             newRowSize = 1;
          }
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_IJMatrixSetValues(CTmat, 1, &newRowSize,
		(const int *) &rowCount, (const int *) newColInd,
		(const double *) newColVal);
          rowCount++;
       }
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(CTmat);
    HYPRE_IJMatrixGetObject(CTmat, (void **) &CT_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) CT_csr);

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             printf("%4d buildSchurSystem : matrix CTmat assembled %d.\n",
                                            mypid_,CTStartRow);
             fflush(stdout);
             for ( i = CTStartRow; i < CTStartRow+CTNRows; i++ )
             {
                HYPRE_ParCSRMatrixGetRow(CT_csr,i,&rowSize,&colInd,&colVal);
                printf("CTmat ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(CT_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(MPI_COMM_WORLD);
       }
    }

    //******************************************************************
    // perform the triple matrix product
    //------------------------------------------------------------------

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       printf("%4d : buildSchurSystem - Triple matrix product begins..\n",
              mypid_);
    hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix *) M_csr,
                                     (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix **) &S_csr);
    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       printf("%4d : buildSchurSystem - Triple matrix product ends\n",mypid_);

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             for ( i = CStartRow; i < CStartRow+CNRows; i++ )
             {
                HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd, &colVal);
                printf("Schur ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,&colVal);
             }
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    // *****************************************************************
    // form modified right hand side  (f2 = f2 - C*M*f1)
    // *****************************************************************

    //------------------------------------------------------------------
    // form f2hat = C*M*f1
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, CTStartRow, CTStartRow+CTNRows-1, &f1);
    HYPRE_IJVectorSetObjectType(f1, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f1);
    ierr += HYPRE_IJVectorAssemble(f1);
    hypre_assert(!ierr);
    HYPRE_IJVectorCreate(comm_, CStartRow, CStartRow+CNRows-1, &f2hat);
    HYPRE_IJVectorSetObjectType(f2hat, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    hypre_assert(!ierr);

    rowCount = CTStartRow;
    for ( i = StartRow; i <= EndRow; i++ )
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
          ddata *= diagonal[rowCount-CTStartRow];
          ierr = HYPRE_IJVectorSetValues(f1, 1, (const int *) &rowCount,
			(const double *) &ddata);
          hypre_assert( !ierr );
          rowCount++;
       }
    }

    HYPRE_IJVectorGetObject(f1, (void **) &f1_csr);
    HYPRE_IJVectorGetObject(f2hat, (void **) &f2hat_csr);
    HYPRE_ParCSRMatrixMatvec( 1.0, C_csr, f1_csr, 0.0, f2hat_csr );
    delete [] diagonal;
    HYPRE_IJVectorDestroy(f1);

    //------------------------------------------------------------------
    // form f2 = f2 - f2hat
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, CStartRow, CStartRow+CNRows-1, &f2);
    HYPRE_IJVectorSetObjectType(f2, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorAssemble(f2);
    hypre_assert(!ierr);

    rowCount = CStartRow;
    for ( i = 0; i < nSchur; i++ )
    {
       rowIndex = schurList[i];
       HYPRE_IJVectorGetValues(HYb_, 1, &rowIndex, &ddata);
       ddata = - ddata;
       ierr = HYPRE_IJVectorSetValues(f2, 1, (const int *) &rowCount,
			(const double *) &ddata);
       HYPRE_IJVectorGetValues(f2hat, 1, &rowCount, &ddata);
       HYPRE_IJVectorAddToValues(f2, 1, (const int *) &rowCount,
			(const double *) &ddata);
       HYPRE_IJVectorGetValues(f2, 1, &rowCount, &ddata);
       hypre_assert( !ierr );
       rowCount++;
    }
    HYPRE_IJVectorDestroy(f2hat);

    // *****************************************************************
    // set up the system with the new matrix
    // *****************************************************************

    ierr  = HYPRE_IJMatrixCreate(comm_, CStartRow, CStartRow+CNRows-1,
				 CStartRow, CStartRow+CNRows-1, &reducedA_);
    ierr += HYPRE_IJMatrixSetObjectType(reducedA_, HYPRE_PARCSR);
    hypre_assert(!ierr);

    //------------------------------------------------------------------
    // compute row sizes for the Schur complement
    //------------------------------------------------------------------

    CMatSize = new int[CNRows];
    maxRowSize = 0;
    for ( i = CStartRow; i < CStartRow+CNRows; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd,NULL);
       rowIndex = schurList[i-CStartRow];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize2,&colInd2,NULL);
       newRowSize = rowSize + rowSize2;
       newColInd = new int[newRowSize];
       for (j = 0; j < rowSize; j++)  newColInd[j] = colInd[j];
       ncnt = 0;
       for (j = 0; j < rowSize2; j++)
       {
          colIndex = colInd2[j];
          searchIndex = hypre_BinarySearch(globalSchurList,colIndex,
                                           globalNSchur);
          if ( searchIndex >= 0 )
          {
             newColInd[rowSize+ncnt] = colInd2[j];
             ncnt++;
          }
       }
       newRowSize = rowSize + ncnt;
       hypre_qsort0(newColInd, 0, newRowSize-1);
       ncnt = 0;
       for ( j = 1; j < newRowSize; j++ )
       {
          if ( newColInd[j] != newColInd[ncnt] )
          {
             ncnt++;
             newColInd[ncnt] = newColInd[j];
          }
       }
       if ( newRowSize > 0 ) ncnt++;
       CMatSize[i-CStartRow] = ncnt;
       maxRowSize = ( ncnt > maxRowSize ) ? ncnt : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,NULL);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize2,&colInd2,NULL);
       delete [] newColInd;
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(reducedA_, CMatSize);
    ierr += HYPRE_IJMatrixInitialize(reducedA_);
    hypre_assert(!ierr);
    delete [] CMatSize;

    //------------------------------------------------------------------
    // load and assemble the Schur complement matrix
    //------------------------------------------------------------------

    for ( i = CStartRow; i < CStartRow+CNRows; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd,&colVal);
       rowIndex = schurList[i-CStartRow];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize2,&colInd2,&colVal2);
       newRowSize = rowSize + rowSize2;
       newColInd = new int[newRowSize];
       newColVal = new double[newRowSize];
       for (j = 0; j < rowSize; j++)
       {
          newColInd[j] = colInd[j];
          newColVal[j] = colVal[j];
       }
       ncnt = 0;
       for (j = 0; j < rowSize2; j++)
       {
          colIndex = colInd2[j];
          searchIndex = hypre_BinarySearch(globalSchurList,colIndex,
                                           globalNSchur);
          if ( searchIndex >= 0 )
          {
             newColInd[rowSize+ncnt] = searchIndex;
             newColVal[rowSize+ncnt] = - colVal2[j];
             ncnt++;
          }
       }
       newRowSize = rowSize + ncnt;
       hypre_qsort1(newColInd, newColVal, 0, newRowSize-1);
       ncnt = 0;
       for ( j = 1; j < newRowSize; j++ )
       {
          if ( newColInd[j] == newColInd[ncnt] )
          {
             newColVal[ncnt] += newColVal[j];
          }
          else
          {
             ncnt++;
             newColInd[ncnt] = newColInd[j];
             newColVal[ncnt] = newColVal[j];
          }
       }
       if ( newRowSize > 0 ) newRowSize = ++ncnt;
       ncnt = 0;
       ddata = 0.0;
       for ( j = 0; j < newRowSize; j++ )
          if ( habs(newColVal[j]) > ddata ) ddata = habs(newColVal[j]);
       for ( j = 0; j < newRowSize; j++ )
       {
          if ( habs(newColVal[j]) > ddata*1.0e-14 )
          {
             newColInd[ncnt] = newColInd[j];
             newColVal[ncnt++] = newColVal[j];
          }
       }
       newRowSize = ncnt;
       HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,&colVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize2,&colInd2,
                                    &colVal2);
       HYPRE_IJMatrixSetValues(reducedA_, 1, &newRowSize, (const int *) &i,
		(const int *) newColInd, (const double *) newColVal);

       delete [] newColInd;
       delete [] newColVal;
    }
    HYPRE_IJMatrixAssemble(reducedA_);
    HYPRE_ParCSRMatrixDestroy(S_csr);

    //------------------------------------------------------------------
    // create and initialize the reduced x, and create the reduced r
    //------------------------------------------------------------------

    ierr = HYPRE_IJVectorCreate(comm_,CStartRow,CStartRow+CNRows-1,&reducedX_);
    ierr = HYPRE_IJVectorSetObjectType(reducedX_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(reducedX_);
    ierr = HYPRE_IJVectorAssemble(reducedX_);
    hypre_assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm_,CStartRow,CStartRow+CNRows-1,&reducedR_);
    ierr = HYPRE_IJVectorSetObjectType(reducedR_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(reducedR_);
    ierr = HYPRE_IJVectorAssemble(reducedR_);
    hypre_assert(!ierr);

    //------------------------------------------------------------------
    // save A21 and invA22 for solution recovery
    //------------------------------------------------------------------

    reducedB_ = f2;
    currA_ = reducedA_;
    currB_ = reducedB_;
    currR_ = reducedR_;
    currX_ = reducedX_;

    HYA21_    = CTmat;
    HYA12_    = Cmat;
    HYinvA22_ = Mmat;
    A21NRows_ = CTNRows;
    A21NCols_ = CTNCols;
    buildSchurInitialGuess();

    //------------------------------------------------------------------
    // final clean up
    //------------------------------------------------------------------

    delete [] globalSchurList;
    selectedList_ = schurList;
    delete [] ProcNRows;
    delete [] ProcNSchur;

    if ( colValues_ != NULL )
    {
       for ( j = 0; j < localEndRow_-localStartRow_+1; j++ )
          if ( colValues_[j] != NULL ) delete [] colValues_[j];
       delete [] colValues_;
       colValues_ = NULL;
    }
    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
       printf("       buildSchurSystem ends....\n");
}

//*****************************************************************************
// build the solution vector for Schur-reduced systems
//-----------------------------------------------------------------------------

double HYPRE_LinSysCore::buildSchurReducedSoln()
{
    int                i, *int_array, *gint_array, x2NRows, x2GlobalNRows;
    int                ierr, rowNum, startRow, startRow2, index, localNRows;
    double             ddata, rnorm;
    HYPRE_ParCSRMatrix A_csr, A21_csr, A22_csr;
    HYPRE_ParVector    x_csr, x2_csr, r_csr, b_csr;
    HYPRE_IJVector     R1, x2;

    if ( HYA21_ == NULL || HYinvA22_ == NULL )
    {
       printf("buildSchurReducedSoln WARNING : A21 or A22 absent.\n");
       return (0.0);
    }
    else
    {
       //---------------------------------------------------------------
       // compute A21 * sol
       //---------------------------------------------------------------

       int_array  = new int[numProcs_];
       gint_array = new int[numProcs_];
       x2NRows    = A21NRows_;
       for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
       int_array[mypid_] = x2NRows;
       MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
       x2GlobalNRows = 0;
       for ( i = 0; i < numProcs_; i++ ) x2GlobalNRows += gint_array[i];
       rowNum = 0;
       for ( i = 0; i < mypid_; i++ ) rowNum += gint_array[i];
       startRow = rowNum;
       startRow2 = localStartRow_ - 1 - rowNum;
       delete [] int_array;
       delete [] gint_array;
       localNRows = localEndRow_ - localStartRow_ + 1 - A21NRows_;
       ierr = HYPRE_IJVectorCreate(comm_, startRow, startRow+x2NRows-1, &R1);
       ierr = HYPRE_IJVectorSetObjectType(R1, HYPRE_PARCSR);
       ierr = HYPRE_IJVectorInitialize(R1);
       ierr = HYPRE_IJVectorAssemble(R1);
       hypre_assert(!ierr);
       HYPRE_IJMatrixGetObject(HYA21_, (void **) &A21_csr);
       HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
       HYPRE_IJVectorGetObject(R1, (void **) &r_csr);
       HYPRE_ParCSRMatrixMatvec( -1.0, A21_csr, x_csr, 0.0, r_csr );

       //-------------------------------------------------------------
       // f2 - A21 * sol
       //-------------------------------------------------------------

       rowNum = startRow;
       if ( selectedList_ != NULL )
       {
          for ( i = localStartRow_-1; i < localEndRow_; i++ )
          {
             if (HYPRE_LSI_Search(selectedList_,i,localNRows)<0)
             {
                HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
                HYPRE_IJVectorAddToValues(R1, 1, (const int *) &rowNum,
				(const double *) &ddata);
                rowNum++;
             }
          }
       }
       else
       {
          for ( i = localStartRow_-1; i < localEndRow_-A21NCols_; i++ )
          {
             HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
             HYPRE_IJVectorAddToValues(R1, 1, (const int *) &rowNum,
			(const double *) &ddata);
             HYPRE_IJVectorGetValues(R1, 1, &rowNum, &ddata);
             rowNum++;
          }
       }

       //-------------------------------------------------------------
       // inv(A22) * (f2 - A21 * sol)
       //-------------------------------------------------------------

       ierr = HYPRE_IJVectorCreate(comm_, startRow, startRow+x2NRows-1, &x2);
       ierr = HYPRE_IJVectorSetObjectType(x2, HYPRE_PARCSR);
       ierr = HYPRE_IJVectorInitialize(x2);
       ierr = HYPRE_IJVectorAssemble(x2);
       hypre_assert(!ierr);
       HYPRE_IJMatrixGetObject(HYinvA22_, (void **) &A22_csr);
       HYPRE_IJVectorGetObject(R1, (void **) &r_csr);
       HYPRE_IJVectorGetObject(x2, (void **) &x2_csr);
       HYPRE_ParCSRMatrixMatvec( 1.0, A22_csr, r_csr, 0.0, x2_csr );

       //-------------------------------------------------------------
       // inject final solution to the solution vector
       //-------------------------------------------------------------

       if ( selectedList_ != NULL )
       {
          for ( i = startRow2; i < startRow2+localNRows; i++ )
          {
             HYPRE_IJVectorGetValues(reducedX_, 1, &i, &ddata);
             index = selectedList_[i-startRow2];
             HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &index,
			(const double *) &ddata);
          }
          rowNum = localStartRow_ - 1;
          for ( i = startRow; i < startRow+A21NRows_; i++ )
          {
             HYPRE_IJVectorGetValues(x2, 1, &i, &ddata);
             while (HYPRE_LSI_Search(selectedList_,rowNum,localNRows)>=0)
                rowNum++;
             HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &rowNum,
			(const double *) &ddata);
             rowNum++;
          }
       }
       else
       {
          for ( i = startRow2; i < startRow2+localNRows; i++ )
          {
             HYPRE_IJVectorGetValues(reducedX_, 1, &i, &ddata);
             index = localEndRow_ - A21NCols_ + i - startRow2;
             HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &index,
			(const double *) &ddata);
          }
          rowNum = localStartRow_ - 1;
          for ( i = startRow; i < startRow+A21NRows_; i++ )
          {
             HYPRE_IJVectorGetValues(x2, 1, &i, &ddata);
             HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &rowNum,
			(const double *) &ddata);
             rowNum++;
          }
       }

       //-------------------------------------------------------------
       // residual norm check
       //-------------------------------------------------------------

       HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
       HYPRE_IJVectorGetObject(HYx_, (void **) &x_csr);
       HYPRE_IJVectorGetObject(HYb_, (void **) &b_csr);
       HYPRE_IJVectorGetObject(HYr_, (void **) &r_csr);
       HYPRE_ParVectorCopy( b_csr, r_csr );
       HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       rnorm = sqrt( rnorm );
       if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 ) )
          printf("       buildReducedSystemSoln::final residual norm = %e\n",
                 rnorm);
    }
    currX_ = HYx_;

    //****************************************************************
    // clean up
    //----------------------------------------------------------------

    HYPRE_IJVectorDestroy(R1);
    HYPRE_IJVectorDestroy(x2);
    return rnorm;
}

//*****************************************************************************
// form initial solution for the reduced system
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSchurInitialGuess()
{
    int    i, ierr, EndRow, nSchur, *partition, CStartRow;
    int    *getIndices, *putIndices;
    double *dArray;
    HYPRE_ParVector hypre_x;

    //------------------------------------------------------------------
    // initial set up
    //------------------------------------------------------------------

    if (reducedX_ == HYx_ || reducedX_ == NULL || reducedA_ == NULL) return;
    EndRow    = localEndRow_ - 1;
    nSchur    = A21NCols_;
    if ( nSchur == 0 ) return;
    HYPRE_IJVectorGetObject(reducedX_, (void **) &hypre_x);
    partition = hypre_ParVectorPartitioning((hypre_ParVector *) hypre_x);
    CStartRow = partition[mypid_];

    //------------------------------------------------------------------
    // injecting initial guesses
    //------------------------------------------------------------------

    if ( selectedList_ != NULL ) getIndices = selectedList_;
    else
    {
       getIndices = new int[nSchur];
       for ( i = 0; i < nSchur; i++ ) getIndices[i] = EndRow+1-nSchur+i;
    }
    dArray     = new double[nSchur];
    putIndices = new int[nSchur];
    for ( i = 0; i < nSchur; i++ ) putIndices[i] = CStartRow + i;
    HYPRE_IJVectorGetValues(HYx_, nSchur, getIndices, dArray);
    ierr = HYPRE_IJVectorSetValues(reducedX_, nSchur,
                    (const int *) putIndices, (const double *) dArray);
    hypre_assert( !ierr );
    delete [] dArray;
    delete [] putIndices;
    if ( selectedList_ == NULL ) delete [] getIndices;
}

//*****************************************************************************
// form modified right hand side  (f2 = f2 - C*M*f1)
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSchurReducedRHS()
{

    int    i, ncnt, ncnt2, ierr, *colInd, CTStartRow, CStartRow, rowIndex;
    int    StartRow, EndRow;
    int    nSchur, *schurList, *ProcNRows, *ProcNSchur;
    int    globalNSchur, CTNRows, CTNCols, CTGlobalNRows, CTGlobalNCols;
    int    CNRows, *tempList, searchIndex, rowCount, rowSize;
    double ddata, ddata2, *colVal;
    HYPRE_IJMatrix     Cmat, Mmat;
    HYPRE_IJVector     f1, f2, f2hat;
    HYPRE_ParVector    f1_csr, f2hat_csr;
    HYPRE_ParCSRMatrix M_csr, C_csr;

    //******************************************************************
    // initial set up
    //------------------------------------------------------------------

    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
       printf("       buildSchurRHS begins....\n");
    if ( HYA21_ == NULL || HYinvA22_ == NULL )
    {
       printf("buildSchurReducedRHS WARNING : A21 or A22 absent.\n");
       return;
    }
    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;

    //------------------------------------------------------------------
    // get information about processor offsets and globalNRows
    // (ProcNRows, globalNRows)
    //------------------------------------------------------------------

    ProcNRows  = new int[numProcs_];
    tempList   = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = EndRow - StartRow + 1;
    MPI_Allreduce(tempList, ProcNRows, numProcs_, MPI_INT, MPI_SUM, comm_);
    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ )
    {
       ncnt2         = ProcNRows[i];
       ProcNRows[i]  = ncnt;
       ncnt          += ncnt2;
    }
    ProcNSchur = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = A21NCols_;
    MPI_Allreduce(tempList, ProcNSchur, numProcs_, MPI_INT, MPI_SUM, comm_);
    globalNSchur = 0;
    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ )
    {
       globalNSchur  += ProcNSchur[i];
       ncnt2         = ProcNSchur[i];
       ProcNSchur[i] = ncnt;
       ncnt          += ncnt2;
    }
    CStartRow  = ProcNSchur[mypid_];
    CTStartRow = ProcNRows[mypid_] - ProcNSchur[mypid_];
    delete [] ProcNRows;
    delete [] ProcNSchur;
    delete [] tempList;

    CTNRows = A21NRows_;
    CTNCols = A21NCols_;
    MPI_Allreduce(&CTNRows, &CTGlobalNRows, 1, MPI_INT, MPI_SUM, comm_);
    MPI_Allreduce(&CTNCols, &CTGlobalNCols, 1, MPI_INT, MPI_SUM, comm_);
    Cmat         = HYA12_;
    Mmat         = HYinvA22_;
    CNRows       = CTNCols;
    nSchur       = A21NCols_;
    schurList    = selectedList_;
    HYPRE_IJMatrixGetObject(Mmat, (void **) &M_csr);
    HYPRE_IJMatrixGetObject(Cmat, (void **) &C_csr);

    // *****************************************************************
    // form f2hat = C*M*f1
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, CTStartRow, CTStartRow+CTNRows-1, &f1);
    HYPRE_IJVectorSetObjectType(f1, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(f1);
    ierr = HYPRE_IJVectorAssemble(f1);
    hypre_assert(!ierr);
    HYPRE_IJVectorCreate(comm_, CStartRow, CStartRow+CNRows-1, &f2hat);
    HYPRE_IJVectorSetObjectType(f2hat, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr  = HYPRE_IJVectorAssemble(f2hat);
    hypre_assert(!ierr);

    rowCount = CTStartRow;
    if ( schurList != NULL )
    {
       for ( i = StartRow; i <= EndRow; i++ )
       {
          searchIndex = hypre_BinarySearch(schurList, i, nSchur);
          if ( searchIndex < 0 )
          {
             HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
             HYPRE_ParCSRMatrixGetRow(M_csr,rowCount,&rowSize,&colInd,&colVal);
             if ( rowSize != 1 ) printf("buildReducedRHS : WARNING.\n");
             if ( colVal[0] != 0.0 ) ddata *= colVal[0];
             ierr = HYPRE_IJVectorSetValues(f1, 1, (const int *) &rowCount,
			(const double *) &ddata);
             HYPRE_ParCSRMatrixRestoreRow(M_csr,rowCount,&rowSize,&colInd,
                                          &colVal);
             hypre_assert( !ierr );
             rowCount++;
          }
       }
    }
    else
    {
       for ( i = StartRow; i <= EndRow-nSchur; i++ )
       {
          HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
          HYPRE_ParCSRMatrixGetRow(M_csr,rowCount,&rowSize,&colInd,&colVal);
          if ( rowSize != 1 ) printf("buildReducedRHS : WARNING.\n");
          if ( colVal[0] != 0.0 ) ddata *= colVal[0];
          ierr = HYPRE_IJVectorSetValues(f1, 1, (const int *) &rowCount,
			(const double *) &ddata);
          HYPRE_ParCSRMatrixRestoreRow(M_csr,rowCount,&rowSize,&colInd,
                                       &colVal);
          hypre_assert( !ierr );
          rowCount++;
       }
    }
    HYPRE_IJVectorGetObject(f1, (void **) &f1_csr);
    HYPRE_IJVectorGetObject(f2hat, (void **) &f2hat_csr);
    HYPRE_ParCSRMatrixMatvec( 1.0, C_csr, f1_csr, 0.0, f2hat_csr );
    HYPRE_IJVectorDestroy(f1);

    //------------------------------------------------------------------
    // form f2 = f2 - f2hat
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, CStartRow, CStartRow+CNRows-1, &f2);
    HYPRE_IJVectorSetObjectType(f2, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorAssemble(f2);
    hypre_assert(!ierr);

    rowCount = CStartRow;
    for ( i = 0; i < nSchur; i++ )
    {
       if ( schurList != NULL ) rowIndex = schurList[i];
       else                     rowIndex = EndRow+1-nSchur+i;
       HYPRE_IJVectorGetValues(HYb_, 1, &rowIndex, &ddata);
       HYPRE_IJVectorGetValues(f2hat, 1, &rowCount,  &ddata2);
       ddata = ddata2 - ddata;
       ierr = HYPRE_IJVectorSetValues(f2, 1, (const int *) &rowCount,
			(const double *) &ddata);
       hypre_assert( !ierr );
       rowCount++;
    }
    HYPRE_IJVectorDestroy(f2hat);

    //******************************************************************
    // initialize current matrix system
    //------------------------------------------------------------------

    if ( reducedB_ != NULL ) HYPRE_IJVectorDestroy(reducedB_);
    reducedB_ = f2;
    currA_    = reducedA_;
    currB_    = reducedB_;
    currR_    = reducedR_;
    currX_    = reducedX_;

    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
       printf("       buildSchurRHS ends....\n");
}

//*****************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
// (This version is different from the previous one in that users are supposed
// to give hypre the number of rows in the reduced matrix starting from the
// bottom, and that the (2,2) block is not expected to be a zero block)
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSchurReducedSystem2()
{
    int    i, j, ierr, ncnt, one=1;
    int    nRows, globalNRows, StartRow, EndRow, colIndex;
    int    nSchur, globalNSchur;
    int    CStartRow, CNRows, CNCols, CGlobalNRows, CGlobalNCols;
    int    CTStartRow, CTNRows, CTNCols;
    int    MStartRow, MNRows, MNCols, MGlobalNRows, MGlobalNCols;
    int    rowSize, rowCount, rowIndex, maxRowSize, newRowSize;
    int    *CMatSize, *CTMatSize, *MMatSize, *colInd, *newColInd, *colInd2;
    int    *tempList, rowSize2;
    int    *ProcNRows, *ProcNSchur, searchIndex, CStartCol;
    double *colVal, *newColVal, *diagonal, ddata, maxdiag, mindiag, darray[2];
    double darray2[2], *colVal2, rowmax;

    HYPRE_IJMatrix     Cmat, CTmat, Mmat;
    HYPRE_ParCSRMatrix A_csr, C_csr, CT_csr, M_csr, S_csr;
    HYPRE_IJVector     f1, f2, f2hat;
    HYPRE_ParVector    f1_csr, f2hat_csr;

    //******************************************************************
    // output initial message and clean up
    //------------------------------------------------------------------

    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
       printf("buildSchurSystem (2) begins....\n");
    if ( HYA21_    != NULL ) HYPRE_IJMatrixDestroy(HYA21_);
    if ( HYA12_    != NULL ) HYPRE_IJMatrixDestroy(HYA12_);
    if ( HYinvA22_ != NULL ) HYPRE_IJMatrixDestroy(HYinvA22_);
    if ( reducedB_ != NULL ) HYPRE_IJVectorDestroy(reducedB_);
    if ( reducedX_ != NULL ) HYPRE_IJVectorDestroy(reducedX_);
    if ( reducedR_ != NULL ) HYPRE_IJVectorDestroy(reducedR_);
    if ( reducedA_ != NULL ) HYPRE_IJMatrixDestroy(reducedA_);
    HYA21_    = NULL;
    HYA12_    = NULL;
    HYinvA22_ = NULL;
    reducedB_ = NULL;
    reducedX_ = NULL;
    reducedR_ = NULL;
    reducedA_ = NULL;

    //******************************************************************
    // set up local information
    //------------------------------------------------------------------

    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    nRows    = localEndRow_ - localStartRow_ + 1;
    HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       printf("%4d : buildSchurSystem - StartRow/EndRow = %d %d\n",mypid_,
                                        StartRow,EndRow);

    //******************************************************************
    // construct global information about the matrix
    //------------------------------------------------------------------

    ProcNRows = new int[numProcs_];
    tempList  = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nRows;
    MPI_Allreduce(tempList, ProcNRows, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    globalNRows = 0;
    for ( i = 0; i < numProcs_; i++ )
    {
       ncnt = globalNRows;
       globalNRows += ProcNRows[i];
       ProcNRows[i] = ncnt;
    }

    //******************************************************************
    // perform an automatic search for nSchur
    //------------------------------------------------------------------

    nSchur = 0;
    for ( i = StartRow; i <= EndRow; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       searchIndex = globalNRows + 1;
       for (j = 0; j < rowSize; j++)
       {
          colIndex = colInd[j];
          if ( colIndex < searchIndex && colVal[j] != 0.0 )
             searchIndex = colIndex;
       }
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       if ( searchIndex >= i ) nSchur++;
       else                    break;
    }
    nSchur = EndRow - StartRow + 1 - nSchur;
    MPI_Allreduce(&nSchur, &globalNSchur, 1, MPI_INT, MPI_SUM,comm_);

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d buildSchurSystem : nSchur = %d\n",mypid_,nSchur);
       if ( globalNSchur == 0 && mypid_ == 0 )
          printf("buildSchurSystem WARNING : nSchur = 0 on all processors.\n");
    }
    if ( globalNSchur == 0 )
    {
       schurReduction_ = 0;
       delete [] ProcNRows;
       return;
    }

    //******************************************************************
    // construct global information about the reduced matrix
    //------------------------------------------------------------------

    ProcNSchur = new int[numProcs_];
    tempList   = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nSchur;
    MPI_Allreduce(tempList, ProcNSchur, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    globalNSchur = 0;
    for ( i = 0; i < numProcs_; i++ )
    {
       ncnt = globalNSchur;
       globalNSchur  += ProcNSchur[i];
       ProcNSchur[i] = ncnt;
    }

    //******************************************************************
    // construct Cmat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of Cmat
    //------------------------------------------------------------------

    CNRows = nSchur;
    CNCols = nRows - nSchur;
    CGlobalNRows = globalNSchur;
    CGlobalNCols = globalNRows - globalNSchur;
    CStartRow    = ProcNSchur[mypid_];

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d buildSchurSystem : CStartRow  = %d\n",mypid_,CStartRow);
       printf("%4d buildSchurSystem : CGlobalDim = %d %d\n", mypid_,
                                      CGlobalNRows, CGlobalNCols);
       printf("%4d buildSchurSystem : CLocalDim  = %d %d\n",mypid_,
                                         CNRows, CNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for Cmat
    //------------------------------------------------------------------

    CStartCol = ProcNRows[mypid_] - ProcNSchur[mypid_];
    ierr  = HYPRE_IJMatrixCreate(comm_, CStartRow, CStartRow+CNRows-1,
				 CStartCol, CStartCol+CNCols-1, &Cmat);
    ierr += HYPRE_IJMatrixSetObjectType(Cmat, HYPRE_PARCSR);
    hypre_assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros per row in Cmat and call set up
    //------------------------------------------------------------------

    maxRowSize = 0;
    CMatSize = new int[CNRows];

    for ( i = 0; i < nSchur; i++ )
    {
       rowIndex = EndRow - nSchur + i + 1;
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++)
       {
          colIndex = colInd[j];
          searchIndex = HYPRE_Schur_Search(colIndex, numProcs_, ProcNRows,
                                      ProcNSchur, globalNRows, globalNSchur);
          if (searchIndex < 0) newRowSize++;
       }
       CMatSize[i] = newRowSize;
       maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(Cmat, CMatSize);
    ierr += HYPRE_IJMatrixInitialize(Cmat);
    hypre_assert(!ierr);
    delete [] CMatSize;

    //------------------------------------------------------------------
    // load Cmat extracted from A
    //------------------------------------------------------------------

    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];
    rowCount  = CStartRow;
    for ( i = 0; i < nSchur; i++ )
    {
       rowIndex = EndRow - nSchur + i + 1;
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++)
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
             searchIndex = HYPRE_Schur_Search(colIndex, numProcs_, ProcNRows,
                                  ProcNSchur, globalNRows, globalNSchur);
             if ( searchIndex < 0 )
             {
                searchIndex = - searchIndex - 1;
                colIndex = searchIndex;
                newColInd[newRowSize]   = colIndex;
                newColVal[newRowSize++] = colVal[j];
                if ( colIndex < 0 || colIndex >= CGlobalNCols )
                {
                   if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
                   {
                      printf("%4d buildSchurSystem WARNING : Cmat ", mypid_);
                      printf("out of range %d - %d (%d)\n", rowCount, colIndex,
                              CGlobalNCols);
                   }
                }
                if ( newRowSize > maxRowSize+1 )
                {
                   if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
                   {
                      printf("%4d buildSchurSystem : WARNING - ",mypid_);
                      printf("passing array boundary(1).\n");
                   }
                }
             }
          }
       }
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       HYPRE_IJMatrixSetValues(Cmat, 1, &newRowSize, (const int *) &rowCount,
		(const int *) newColInd, (const double *) newColVal);
       rowCount++;
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(Cmat);
    HYPRE_IJMatrixGetObject(Cmat, (void **) &C_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) C_csr);

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             printf("%4d buildSchurSystem : matrix Cmat assembled %d.\n",
                                           mypid_,CStartRow);
             fflush(stdout);
             for ( i = CStartRow; i < CStartRow+nSchur; i++ )
             {
                HYPRE_ParCSRMatrixGetRow(C_csr,i,&rowSize,&colInd,&colVal);
                printf("Cmat ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(C_csr,i,&rowSize,&colInd,&colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(MPI_COMM_WORLD);
       }
    }

    //******************************************************************
    // construct the diagonal Mmat and CTmat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of Mmat and CTmat
    //------------------------------------------------------------------

    MNRows = nRows - nSchur;
    MNCols = nRows - nSchur;
    MGlobalNRows = globalNRows - globalNSchur;
    MGlobalNCols = globalNRows - globalNSchur;
    MStartRow    = ProcNRows[mypid_] - ProcNSchur[mypid_];

    CTNRows = nRows - nSchur;
    CTNCols = nSchur;
    CTStartRow    = ProcNRows[mypid_] - ProcNSchur[mypid_];

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d buildSchurSystem : MStartRow  = %d\n",mypid_,MStartRow);
       printf("%4d buildSchurSystem : MGlobalDim = %d %d\n", mypid_,
                                      MGlobalNRows, MGlobalNCols);
       printf("%4d buildSchurSystem : MLocalDim  = %d %d\n",mypid_,
                                      MNRows, MNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for Mmat and CTmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_, MStartRow, MStartRow+MNRows-1,
				 MStartRow, MStartRow+MNCols-1, &Mmat);
    ierr += HYPRE_IJMatrixSetObjectType(Mmat, HYPRE_PARCSR);
    hypre_assert(!ierr);
    ierr  = HYPRE_IJMatrixCreate(comm_, CTStartRow, CTStartRow+CTNRows-1,
				 CStartRow, CStartRow+CTNCols-1, &CTmat);
    ierr += HYPRE_IJMatrixSetObjectType(Mmat, HYPRE_PARCSR);
    hypre_assert(!ierr);

    //------------------------------------------------------------------
    // compute row sizes for Mmat
    //------------------------------------------------------------------

    MMatSize = new int[MNRows];
    for ( i = 0; i < MNRows; i++ ) MMatSize[i] = 1;
    ierr  = HYPRE_IJMatrixSetRowSizes(Mmat, MMatSize);
    ierr += HYPRE_IJMatrixInitialize(Mmat);
    hypre_assert(!ierr);
    delete [] MMatSize;

    //------------------------------------------------------------------
    // compute row sizes for CTmat
    //------------------------------------------------------------------

    maxRowSize = 0;
    CTMatSize = new int[CTNRows];
    rowCount = 0;
    for ( i = StartRow; i <= EndRow-nSchur; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++)
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 )
          {
             searchIndex = HYPRE_Schur_Search(colIndex, numProcs_, ProcNRows,
                                    ProcNSchur, globalNRows, globalNSchur);
             if (searchIndex >= 0) newRowSize++;
          }
       }
       if ( newRowSize <= 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE3) )
          printf("%d : WARNING at row %d - empty row.\n", mypid_, i);
       if ( newRowSize <= 0 ) newRowSize = 1;
       CTMatSize[rowCount++] = newRowSize;
       maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(CTmat, CTMatSize);
    ierr += HYPRE_IJMatrixInitialize(CTmat);
    hypre_assert(!ierr);
    delete [] CTMatSize;

    //------------------------------------------------------------------
    // load Mmat and CTmat
    //------------------------------------------------------------------

    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];
    maxdiag   = -1.0E10;
    mindiag   =  1.0E10;
    diagonal  = new double[MNRows];
    rowIndex  = MStartRow;
    ierr      = 0;
    for ( i = StartRow; i <= EndRow-nSchur; i++ )
    {
       ncnt = 0;
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++)
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 )
          {
             searchIndex = HYPRE_Schur_Search(colIndex, numProcs_, ProcNRows,
                                       ProcNSchur, globalNRows, globalNSchur);
             if (searchIndex >= 0)
             {
                newColInd[newRowSize] = searchIndex;
                if ( searchIndex >= globalNSchur )
                {
                   if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
                   {
                      printf("%4d : buildSchurSystem WARNING - CTmat ",mypid_);
                      printf("out of range %d - %d (%d)\n", rowIndex,
                             searchIndex, globalNSchur);
                   }
                }
                newColVal[newRowSize++] = colVal[j];
             }
             else if ( colIndex == i && colVal[j] != 0.0 )
             {
                ddata = 1.0 / colVal[j];
                ncnt++;
                maxdiag = ( colVal[j] > maxdiag ) ? colVal[j] : maxdiag;
                mindiag = ( colVal[j] < mindiag ) ? colVal[j] : mindiag;
             }
          }
       }
       if ( ncnt == 0 )
       {
          if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
             printf("%4d : buildSchurSystem WARNING - diag[%d] not found.\n",
                     mypid_, i);
          ierr = 1;
       }
       else if ( ncnt > 1 ) ierr = 1;
       if ( newRowSize == 0 )
       {
          newColInd[0] = ProcNSchur[mypid_];
          newColVal[0] = 0.0;
          newRowSize = 1;
       }
       diagonal[rowIndex-MStartRow] = ddata;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       HYPRE_IJMatrixSetValues(Mmat, 1, &one, (const int *) &rowIndex,
		(const int *) &rowIndex, (const double *) &ddata);
       HYPRE_IJMatrixSetValues(CTmat, 1, &newRowSize, (const int *) &rowIndex,
		(const int *) newColInd, (const double *) newColVal);
       rowIndex++;
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // output statistics about sizes of diagonal elements
    //------------------------------------------------------------------

    darray[0]  = maxdiag;
    darray[1]  = - mindiag;
    darray2[0] = maxdiag;
    darray2[1] = - mindiag;
    MPI_Allreduce(darray, darray2, 2, MPI_DOUBLE, MPI_MAX, comm_);
    maxdiag = darray2[0];
    mindiag = - darray2[1];
    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1))
    {
       printf("%4d : buildSchurSystem - max diagonal = %e\n",mypid_,maxdiag);
       printf("%4d : buildSchurSystem - min diagonal = %e\n",mypid_,mindiag);
    }

    //------------------------------------------------------------------
    // finally assemble Mmat
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(Mmat);
    HYPRE_IJMatrixGetObject(Mmat, (void **) &M_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) M_csr);
    HYPRE_IJMatrixAssemble(CTmat);
    HYPRE_IJMatrixGetObject(CTmat, (void **) &CT_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) CT_csr);

    //------------------------------------------------------------------
    // Error checking
    //------------------------------------------------------------------

    MPI_Allreduce(&ierr, &ncnt, 1, MPI_INT, MPI_SUM, comm_);
    if ( ncnt > 0 )
    {
       if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       {
          printf("%4d : buildSchurSystem WARNING - A11 not diagonal\n",mypid_);
          printf("%4d : buildSchurSystem WARNING - reduction not performed\n",
                 mypid_);
       }
       schurReduction_ = 0;
       delete [] ProcNRows;
       delete [] ProcNSchur;
       HYPRE_IJMatrixDestroy(Cmat);
       return;
    }

    //------------------------------------------------------------------
    // diagnostics (output CTmat)
    //------------------------------------------------------------------

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             printf("%4d buildSchurSystem : matrix CTmat assembled %d.\n",
                                            mypid_,CTStartRow);
             fflush(stdout);
             for ( i = CTStartRow; i < CTStartRow+CTNRows; i++ )
             {
                HYPRE_ParCSRMatrixGetRow(CT_csr,i,&rowSize,&colInd,&colVal);
                printf("CTmat ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(CT_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(MPI_COMM_WORLD);
       }
    }

    //******************************************************************
    // perform the triple matrix product
    //------------------------------------------------------------------

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       printf("%4d : buildSchurSystem - Triple matrix product begins..\n",
              mypid_);

    hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix *) M_csr,
                                     (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix **) &S_csr);

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
       printf("%4d : buildSchurSystem - Triple matrix product ends\n",
              mypid_);

    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             for ( i = CStartRow; i < CStartRow+CNRows; i++ ) {
                HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd, &colVal);
                printf("Schur ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,&colVal);
             }
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    // *****************************************************************
    // form modified right hand side  (f2 = f2 - C*M*f1)
    // *****************************************************************

    // *****************************************************************
    // form f2hat = C*M*f1
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, CTStartRow, CTStartRow+CTNRows-1, &f1);
    HYPRE_IJVectorSetObjectType(f1, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f1);
    ierr += HYPRE_IJVectorAssemble(f1);
    hypre_assert(!ierr);
    HYPRE_IJVectorCreate(comm_, CStartRow, CStartRow+CNRows-1, &f2hat);
    HYPRE_IJVectorSetObjectType(f2hat, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    hypre_assert(!ierr);

    rowCount = CTStartRow;
    for ( i = StartRow; i <= EndRow-nSchur; i++ )
    {
       HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
       ddata *= diagonal[rowCount-CTStartRow];
       ierr = HYPRE_IJVectorSetValues(f1, 1, (const int *) &rowCount,
			(const double *) &ddata);
       hypre_assert( !ierr );
       rowCount++;
    }
    HYPRE_IJVectorGetObject(f1, (void **) &f1_csr);
    HYPRE_IJVectorGetObject(f2hat, (void **) &f2hat_csr);
    HYPRE_ParCSRMatrixMatvec( 1.0, C_csr, f1_csr, 0.0, f2hat_csr );
    delete [] diagonal;
    HYPRE_IJVectorDestroy(f1);

    //------------------------------------------------------------------
    // form f2 = f2 - f2hat (and negate)
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, CStartRow, CStartRow+CNRows-1, &f2);
    HYPRE_IJVectorSetObjectType(f2, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorAssemble(f2);
    hypre_assert(!ierr);

    rowCount = CStartRow;
    for ( i = 0; i < nSchur; i++ )
    {
       rowIndex = EndRow - nSchur + i + 1;
       HYPRE_IJVectorGetValues(HYb_, 1, &rowIndex, &ddata);
       ddata = - ddata;
       ierr = HYPRE_IJVectorSetValues(f2, 1, (const int *) &rowCount,
		(const double *) &ddata);
       HYPRE_IJVectorGetValues(f2hat, 1, &rowCount, &ddata);
       HYPRE_IJVectorAddToValues(f2, 1, (const int *) &rowCount,
		(const double *) &ddata);
       HYPRE_IJVectorGetValues(f2, 1, &rowCount, &ddata);
       hypre_assert( !ierr );
       rowCount++;
    }
    HYPRE_IJVectorDestroy(f2hat);

    //******************************************************************
    // set up the system with the new matrix
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_, CStartRow, CStartRow+CNRows-1,
				 CStartRow, CStartRow+CNRows-1, &reducedA_);
    ierr += HYPRE_IJMatrixSetObjectType(reducedA_, HYPRE_PARCSR);
    hypre_assert(!ierr);

    //------------------------------------------------------------------
    // compute row sizes for the Schur complement
    //------------------------------------------------------------------

    CMatSize = new int[CNRows];
    maxRowSize = 0;
    for ( i = CStartRow; i < CStartRow+CNRows; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd,NULL);
       rowIndex = EndRow - nSchur + i + 1;
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize2,&colInd2,NULL);
       newRowSize = rowSize + rowSize2;
       newColInd = new int[newRowSize];
       for (j = 0; j < rowSize; j++)  newColInd[j] = colInd[j];
       ncnt = 0;
       for (j = 0; j < rowSize2; j++)
       {
          colIndex = colInd2[j];
          searchIndex = HYPRE_Schur_Search(colIndex, numProcs_, ProcNRows,
                                    ProcNSchur, globalNRows, globalNSchur);
          if ( searchIndex >= 0 )
          {
             newColInd[rowSize+ncnt] = colInd2[j];
             ncnt++;
          }
       }
       newRowSize = rowSize + ncnt;
       hypre_qsort0(newColInd, 0, newRowSize-1);
       ncnt = 0;
       for ( j = 1; j < newRowSize; j++ )
       {
          if ( newColInd[j] != newColInd[ncnt] )
          {
             ncnt++;
             newColInd[ncnt] = newColInd[j];
          }
       }
       if ( newRowSize > 0 ) ncnt++;
       CMatSize[i-CStartRow] = ncnt;
       maxRowSize = ( ncnt > maxRowSize ) ? ncnt : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,NULL);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize2,&colInd2,NULL);
       delete [] newColInd;
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(reducedA_, CMatSize);
    ierr += HYPRE_IJMatrixInitialize(reducedA_);
    hypre_assert(!ierr);
    delete [] CMatSize;

    //------------------------------------------------------------------
    // load and assemble the Schur complement matrix
    //------------------------------------------------------------------

    for ( i = CStartRow; i < CStartRow+CNRows; i++ )
    {
       HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd,&colVal);
       rowIndex = EndRow - nSchur + i + 1;
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize2,&colInd2,&colVal2);
       newRowSize = rowSize + rowSize2;
       newColInd = new int[newRowSize];
       newColVal = new double[newRowSize];
       for (j = 0; j < rowSize; j++)
       {
          newColInd[j] = colInd[j];
          newColVal[j] = colVal[j];
       }
       ncnt = 0;
       for (j = 0; j < rowSize2; j++)
       {
          colIndex = colInd2[j];
          searchIndex = HYPRE_Schur_Search(colIndex, numProcs_, ProcNRows,
                                      ProcNSchur, globalNRows, globalNSchur);
          if ( searchIndex >= 0 )
          {
             newColInd[rowSize+ncnt] = searchIndex;
             newColVal[rowSize+ncnt] = - colVal2[j];
             ncnt++;
          }
       }
       newRowSize = rowSize + ncnt;
       hypre_qsort1(newColInd, newColVal, 0, newRowSize-1);
       ncnt = 0;
       for ( j = 1; j < newRowSize; j++ )
       {
          if ( newColInd[j] == newColInd[ncnt] )
          {
             newColVal[ncnt] += newColVal[j];
          }
          else
          {
             ncnt++;
             newColInd[ncnt] = newColInd[j];
             newColVal[ncnt] = newColVal[j];
          }
       }
       if ( newRowSize > 0 ) newRowSize = ++ncnt;
       ncnt = 0;
       rowmax = 0.0;
       for ( j = 0; j < newRowSize; j++ )
          if ( habs(newColVal[j]) > rowmax ) rowmax = habs(newColVal[j]);
       for ( j = 0; j < newRowSize; j++ )
       {
          if ( habs(newColVal[j]) > rowmax*1.0e-14 )
          {
             newColInd[ncnt] = newColInd[j];
             newColVal[ncnt++] = newColVal[j];
          }
       }
       newRowSize = ncnt;
       // see how this can be done elegantly
       //if ( newRowSize == 1 && newColInd[0] == i && newColVal[0] < 0.0 )
       //   newColVal[0] = - newColVal[0];
       HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,&colVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize2,&colInd2,
                                    &colVal2);
       HYPRE_IJMatrixSetValues(reducedA_, 1, &newRowSize, (const int *) &i,
		(const int *) newColInd, (const double *) newColVal);
    }
    HYPRE_IJMatrixAssemble(reducedA_);

    //------------------------------------------------------------------
    // create the reduced x, right hand side and r
    //------------------------------------------------------------------

    ierr = HYPRE_IJVectorCreate(comm_,CStartRow,CStartRow+CNRows-1,&reducedX_);
    ierr = HYPRE_IJVectorSetObjectType(reducedX_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(reducedX_);
    ierr = HYPRE_IJVectorAssemble(reducedX_);
    hypre_assert(!ierr);
    ierr = HYPRE_IJVectorCreate(comm_,CStartRow,CStartRow+CNRows-1,&reducedR_);
    ierr = HYPRE_IJVectorSetObjectType(reducedR_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(reducedR_);
    ierr = HYPRE_IJVectorAssemble(reducedR_);
    hypre_assert(!ierr);

    reducedB_ = f2;
    currA_ = reducedA_;
    currB_ = reducedB_;
    currR_ = reducedR_;
    currX_ = reducedX_;

    //******************************************************************
    // save A21 and invA22 for solution recovery
    //------------------------------------------------------------------

    HYA21_  = CTmat;
    HYA12_  = Cmat;
    HYinvA22_ = Mmat;
    A21NRows_ = CTNRows;
    A21NCols_ = CTNCols;

    //------------------------------------------------------------------
    // final clean up
    //------------------------------------------------------------------

    delete [] ProcNRows;
    delete [] ProcNSchur;
    if ( colValues_ != NULL )
    {
       for ( j = 0; j < localEndRow_-localStartRow_+1; j++ )
          if ( colValues_[j] != NULL ) delete [] colValues_[j];
       delete [] colValues_;
       colValues_ = NULL;
    }
    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
       printf("       buildSchurSystem ends....\n");
}

//*****************************************************************************
// search to see if the key is in the range
//-----------------------------------------------------------------------------

int HYPRE_LinSysCore::HYPRE_Schur_Search(int key, int nprocs, int *Barray,
                             int *Sarray, int globalNrows, int globalNSchur)
{
   int  i, index1, index2, search_index, out_of_range, not_found;

   search_index = 0;
   out_of_range = 0;
   not_found    = 0;

   for ( i = 0; i < nprocs; i++ )
   {
      if ( i == (nprocs-1) )
      {
         index1 = globalNrows;
         index2 = index1 - globalNSchur;
      }
      else
      {
         index1 = Barray[i+1];
         index2 = index1 - Sarray[i+1];
      }
      if ( key >= index2 && key < index1 )
      {
         search_index += ( key - index2 );
         break;
      }
      else if ( key >= index1 )
      {
         search_index += ( index2 - index1 );
         out_of_range += ( index2 - Barray[i] );
      }
      else if ( key >= Barray[i] )
      {
         out_of_range += (key - Barray[i]);
         not_found = 1;
         break;
      }
      if ( i == (nprocs-1) ) out_of_range += (index1 - index2);
   }
   if ( not_found ) return (-out_of_range-1);
   else             return search_index;
}

