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

#include "utilities/utilities.h"
#include "Data.h"
#include "basicTypes.h"

#if defined(FEI_V13) 
#include "LinearSystemCore.1.3.h"
#elseif defined(FEI_V14)
#include "LinearSystemCore.1.4.h"
#else
#include "LinearSystemCore.h"
#include "LSC.h"
#endif

#include "HYPRE.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../parcsr_mv/HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_LinSysCore.h"

//---------------------------------------------------------------------------
// parcsr_mv.h is put here instead of in HYPRE_LinSysCore.h 
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "parcsr_mv/parcsr_mv.h"

//---------------------------------------------------------------------------
// These are external functions needed internally here
//---------------------------------------------------------------------------

extern "C" {
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix**);
   int HYPRE_LSI_Search(int*, int, int);
}

//******************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//------------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSchurReducedSystem()
{
    int    i, j, k, ierr, ncnt, ncnt2, diag_found;
    int    nRows, globalNRows, StartRow, EndRow, colIndex;
    int    nSchur, *schurList, globalNSchur, *globalSchurList;
    int    CStartRow, CNRows, CNCols, CGlobalNRows, CGlobalNCols;
    int    CTStartRow, CTNRows, CTNCols, CTGlobalNRows, CTGlobalNCols;
    int    MStartRow, MNRows, MNCols, MGlobalNRows, MGlobalNCols;
    int    rowSize, rowCount, rowIndex, maxRowSize, newRowSize;
    int    *CMatSize, *CTMatSize, *MMatSize, *colInd, *newColInd;
    int    *tempList, *recvCntArray, *displArray;
    int    procIndex, *ProcNRows, *ProcNSchur, searchIndex;
    double *colVal, *newColVal, *diagonal, ddata, maxdiag, mindiag;

    HYPRE_IJMatrix     Cmat, CTmat, Mmat, Smat;
    HYPRE_ParCSRMatrix A_csr, C_csr, CT_csr, M_csr, S_csr;
    HYPRE_IJVector     f1, f2, f2hat;
    HYPRE_ParVector    f1_csr, f2_csr, f2hat_csr;

    //******************************************************************
    // initial set up 
    //------------------------------------------------------------------

    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
    {
       printf("buildSchurSystem begins....\n");
    }
    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    nRows    = localEndRow_ - localStartRow_ + 1;
    A_csr    = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d buildSchurSystem : StartRow/EndRow = %d %d\n",mypid_,
                                         StartRow,EndRow);
    }

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

    //------------------------------------------------------------------
    // count the rows that have zero diagonals
    //------------------------------------------------------------------

    nSchur = 0;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       diag_found = 0;
       for (j = 0; j < rowSize; j++) 
          if (colInd[j] == i && colVal[j] != 0.0) {diag_found = 1; break;}
       if ( diag_found == 0 ) nSchur++;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }
    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d buildSchurSystem : nSchur = %d\n",mypid_,nSchur);
    }

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
       diag_found = 0;
       for (j = 0; j < rowSize; j++) 
          if (colInd[j] == i && colVal[j] != 0.0) {diag_found = 1; break;}
       if ( diag_found == 0 ) schurList[nSchur++] = i;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }

    //------------------------------------------------------------------
    // compose the global list of rows having zero diagonal
    // (globalNSchur, globalSchurList)
    //------------------------------------------------------------------

    MPI_Allreduce(&nSchur, &globalNSchur, 1, MPI_INT, MPI_SUM,comm_);

    if ( globalNSchur == 0 && mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1))
    {
       printf("buildSchurSystem WARNING : no row has 0 diagonal element.\n");
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
          printf("%4d buildSchurSystem : schurList %d = %d\n",mypid_,
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
       printf("%4d buildSchurSystem : CStartRow  = %d\n",mypid_,CStartRow);
       printf("%4d buildSchurSystem : CGlobalDim = %d %d\n", mypid_, 
                                      CGlobalNRows, CGlobalNCols);
       printf("%4d buildSchurSystem : CLocalDim  = %d %d\n",mypid_,
                                         CNRows, CNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for Cmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&Cmat,CGlobalNRows,CGlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(Cmat, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(Cmat, CNRows, CNCols);
    assert(!ierr);

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
             if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
             {
                printf("buildSchurSystem WARNING : lower diag block != 0.\n");
                printf("%4d : Cmat[%4d,%4d] = %e\n",rowIndex,colIndex,colVal[j]);
             }
          }
       }
       CMatSize[i] = newRowSize;
       maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(Cmat, CMatSize);
    ierr += HYPRE_IJMatrixInitialize(Cmat);
    assert(!ierr);
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
       HYPRE_IJMatrixInsertRow(Cmat,newRowSize,rowCount,newColInd,newColVal);
       rowCount++;
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix 
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(Cmat);
    C_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(Cmat);
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
       printf("%4d buildSchurSystem : MStartRow  = %d\n",mypid_,MStartRow);
       printf("%4d buildSchurSystem : MGlobalDim = %d %d\n", mypid_, 
                                      MGlobalNRows, MGlobalNCols);
       printf("%4d buildSchurSystem : MLocalDim  = %d %d\n",mypid_,
                                      MNRows, MNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for Mmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&Mmat,MGlobalNRows,MGlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(Mmat, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(Mmat, MNRows, MNCols);
    MMatSize = new int[MNRows];
    for ( i = 0; i < MNRows; i++ ) MMatSize[i] = 1;
    ierr  = HYPRE_IJMatrixSetRowSizes(Mmat, MMatSize);
    ierr += HYPRE_IJMatrixInitialize(Mmat);
    assert(!ierr);
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
                printf("%4d : buildSchurSystem WARNING : diag[%d] not found.\n",
                     mypid_, i);
             ierr = 1;
          } 
          else if ( ncnt > 1 ) ierr = 1;
          diagonal[rowIndex-MStartRow] = ddata;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_IJMatrixInsertRow(Mmat,1,rowIndex,&rowIndex,&ddata);
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
       printf("buildSchurSystem : max diagonal = %e\n", maxdiag);
       printf("buildSchurSystem : min diagonal = %e\n", mindiag);
    }

    //------------------------------------------------------------------
    // finally assemble Mmat
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(Mmat);
    M_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(Mmat);
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
       printf("%4d buildSchurSystem : CTStartRow  = %d\n",mypid_,CTStartRow);
       printf("%4d buildSchurSystem : CTGlobalDim = %d %d\n", mypid_, 
                                      CTGlobalNRows, CTGlobalNCols);
       printf("%4d buildSchurSystem : CTLocalDim  = %d %d\n",mypid_,
                                      CTNRows, CTNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for CTmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&CTmat,CTGlobalNRows,CTGlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(CTmat, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(CTmat, CTNRows, CTNCols);
    assert(!ierr);

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
          if ( newRowSize <= 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
             printf("%d : WARNING at row %d - empty row.\n", mypid_, i);
          if ( newRowSize <= 0 ) newRowSize = 1;
          CTMatSize[rowCount++] = newRowSize;
          maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(CTmat, CTMatSize);
    ierr += HYPRE_IJMatrixInitialize(CTmat);
    assert(!ierr);
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
                      printf("%4d buildSchurSystem WARNING : CTmat ", mypid_);
                      printf("out of range %d - %d (%d)\n", rowCount, searchIndex, 
                              globalNSchur);
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
          HYPRE_IJMatrixInsertRow(CTmat,newRowSize,rowCount,newColInd,newColVal);
          rowCount++;
       }
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix 
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(CTmat);
    CT_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(CTmat);
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
                HYPRE_ParCSRMatrixRestoreRow(CT_csr,i,&rowSize,&colInd,&colVal);
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
    {
       printf("%4d buildSchurSystem : Triple matrix product starts\n",mypid_);
    }
    hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix *) M_csr,
                                     (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix **) &S_csr);
    if ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 )
    {
       printf("%4d buildSchurSystem : Triple matrix product ends\n",mypid_);
    }

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

    HYPRE_IJVectorCreate(comm_, &f1, CTGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f1, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f1,CTStartRow,CTStartRow+CTNRows);
    ierr += HYPRE_IJVectorAssemble(f1);
    ierr += HYPRE_IJVectorInitialize(f1);
    assert(!ierr);

    HYPRE_IJVectorCreate(comm_, &f2hat, CGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2hat, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f2hat,CStartRow,CStartRow+CNRows);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2hat);
    assert(!ierr);

    rowCount = CTStartRow;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_IJVectorGetLocalComponents(HYb_, 1, &i, NULL, &ddata);
          ddata *= diagonal[rowCount-CTStartRow];
          ierr = HYPRE_IJVectorSetLocalComponents(f1,1,&rowCount,NULL,&ddata);
          assert( !ierr );
          rowCount++;
       }
    } 
        
    f1_csr     = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f1);
    f2hat_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f2hat);
    HYPRE_ParCSRMatrixMatvec( 1.0, C_csr, f1_csr, 0.0, f2hat_csr );
    delete [] diagonal;
    HYPRE_IJVectorDestroy(f1); 

    //------------------------------------------------------------------
    // form f2 = f2 - f2hat 
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, &f2, CGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f2,CStartRow,CStartRow+CNRows);
    ierr += HYPRE_IJVectorAssemble(f2);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2);
    assert(!ierr);

    rowCount = CStartRow;
    for ( i = 0; i < nSchur; i++ ) 
    {
       rowIndex = schurList[i];
       HYPRE_IJVectorGetLocalComponents(HYb_, 1, &rowIndex, NULL, &ddata);
       ddata = - ddata;
       ierr = HYPRE_IJVectorSetLocalComponents(f2,1,&rowCount,NULL,&ddata);
       HYPRE_IJVectorGetLocalComponents(f2hat, 1, &rowCount, NULL, &ddata);
       HYPRE_IJVectorAddToLocalComponents(f2,1,&rowCount,NULL,&ddata);
       HYPRE_IJVectorGetLocalComponents(f2, 1, &rowCount, NULL, &ddata);
       assert( !ierr );
       rowCount++;
    } 

    //******************************************************************
    // set up the system with the new matrix
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&reducedA_,CGlobalNRows,CGlobalNRows);
    ierr += HYPRE_IJMatrixSetLocalStorageType(reducedA_, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(reducedA_, CNRows, CNRows);
    assert(!ierr);
    CMatSize = new int[CNRows];
    maxRowSize = 0;
    for ( i = CStartRow; i < CStartRow+CNRows; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd,&colVal);
       CMatSize[i-CStartRow] = rowSize;
       maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,&colVal);
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(reducedA_, CMatSize);
    ierr += HYPRE_IJMatrixInitialize(reducedA_);
    assert(!ierr);
    delete [] CMatSize;
    for ( i = CStartRow; i < CStartRow+CNRows; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd,&colVal);
       HYPRE_IJMatrixInsertRow(reducedA_,rowSize,i,colInd,colVal);
       HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,&colVal);
    }
    HYPRE_IJMatrixAssemble(reducedA_);

    ierr = HYPRE_IJVectorCreate(comm_, &reducedX_, globalNSchur);
    ierr = HYPRE_IJVectorSetLocalStorageType(reducedX_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(reducedX_,CStartRow,
                                              CStartRow+CNRows);
    ierr = HYPRE_IJVectorAssemble(reducedX_);
    ierr = HYPRE_IJVectorInitialize(reducedX_);
    ierr = HYPRE_IJVectorZeroLocalComponents(reducedX_);
    assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm_, &reducedR_, globalNSchur);
    ierr = HYPRE_IJVectorSetLocalStorageType(reducedR_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(reducedR_,CStartRow,
                                              CStartRow+CNRows);
    ierr = HYPRE_IJVectorAssemble(reducedR_);
    ierr = HYPRE_IJVectorInitialize(reducedR_);
    ierr = HYPRE_IJVectorZeroLocalComponents(reducedR_);
    assert(!ierr);

    reducedB_ = f2;
    currA_ = reducedA_;
    currB_ = reducedB_;
    currR_ = reducedR_;
    currX_ = reducedX_;

    //******************************************************************
    // save A21 and invA22 for solution recovery
    //------------------------------------------------------------------

    if ( HYA21_ != NULL ) HYPRE_IJMatrixDestroy(HYA21_);
    HYA21_    = CTmat; 
    if ( HYA12_ != NULL ) HYPRE_IJMatrixDestroy(HYA12_);
    HYA12_    = Cmat; 
    if ( HYinvA22_ != NULL ) HYPRE_IJMatrixDestroy(HYinvA22_);
    HYinvA22_ = Mmat; 
    A21NRows_ = CTNRows;
    A21NCols_ = CTNCols;

    //------------------------------------------------------------------
    // final clean up
    //------------------------------------------------------------------

    delete [] globalSchurList;
    selectedList_ = schurList;
    delete [] ProcNRows;
    delete [] ProcNSchur;

/*
    if ( colIndices_ != NULL )
    {
       for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
          if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
       delete [] colIndices_;
       colIndices_ = NULL;
    }
    if ( rowLengths_ != NULL ) 
    {
       delete [] rowLengths_;
       rowLengths_ = NULL;
    }
*/
    if ( colValues_ != NULL )
    {
       for ( j = 0; j < localEndRow_-localStartRow_+1; j++ )
          if ( colValues_[j] != NULL ) delete [] colValues_[j];
       delete [] colValues_;
       colValues_ = NULL;
    }
    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
    {
       printf("buildSchurSystem ends....\n");
    }
}

//******************************************************************************
// build the solution vector for Schur-reduced systems 
//------------------------------------------------------------------------------

double HYPRE_LinSysCore::buildSchurReducedSoln()
{
    int                i, j, *int_array, *gint_array, x2NRows, x2GlobalNRows;
    int                ierr, rowNum, startRow, startRow2, index, localNRows;
    double             ddata, rnorm;
    HYPRE_ParCSRMatrix A_csr, A21_csr, A22_csr;
    HYPRE_ParVector    x_csr, x2_csr, r_csr, b_csr;
    HYPRE_IJVector     R1, x2; 

    if ( HYA21_ == NULL || HYinvA22_ == NULL )
    {
       printf("buildSchurReducedSoln ERROR : A21 or A22 absent.\n");
       exit(1);
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

       ierr = HYPRE_IJVectorCreate(comm_, &R1, x2GlobalNRows);
       ierr = HYPRE_IJVectorSetLocalStorageType(R1, HYPRE_PARCSR);
       HYPRE_IJVectorSetLocalPartitioning(R1,startRow,startRow+x2NRows);
       ierr = HYPRE_IJVectorAssemble(R1);
       ierr = HYPRE_IJVectorInitialize(R1);
       ierr = HYPRE_IJVectorZeroLocalComponents(R1);
       assert(!ierr);

       A21_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA21_);
       x_csr   = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(currX_);
       r_csr   = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(R1);

       HYPRE_ParCSRMatrixMatvec( -1.0, A21_csr, x_csr, 0.0, r_csr );

       //-------------------------------------------------------------
       // f2 - A21 * sol
       //-------------------------------------------------------------

       rowNum = startRow;
       for ( i = localStartRow_-1; i < localEndRow_; i++ )
       {
          if (HYPRE_LSI_Search(selectedList_,i,localNRows)<0)
          {
             HYPRE_IJVectorGetLocalComponents(HYb_, 1, &i, NULL, &ddata);
             HYPRE_IJVectorAddToLocalComponents(R1,1,&rowNum,NULL,&ddata);
             rowNum++;
          } 
       } 

       //-------------------------------------------------------------
       // inv(A22) * (f2 - A21 * sol)
       //-------------------------------------------------------------

       ierr = HYPRE_IJVectorCreate(comm_, &x2, x2GlobalNRows);
       ierr = HYPRE_IJVectorSetLocalStorageType(x2, HYPRE_PARCSR);
       HYPRE_IJVectorSetLocalPartitioning(x2,startRow,startRow+x2NRows);
       ierr = HYPRE_IJVectorAssemble(x2);
       ierr = HYPRE_IJVectorInitialize(x2);
       ierr = HYPRE_IJVectorZeroLocalComponents(x2);
       assert(!ierr);
       A22_csr = (HYPRE_ParCSRMatrix)HYPRE_IJMatrixGetLocalStorage(HYinvA22_);
       r_csr   = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(R1);
       x2_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(x2);
       HYPRE_ParCSRMatrixMatvec( 1.0, A22_csr, r_csr, 0.0, x2_csr );

       //-------------------------------------------------------------
       // inject final solution to the solution vector
       //-------------------------------------------------------------

       for ( i = startRow2; i < startRow2+localNRows; i++ )
       {
          HYPRE_IJVectorGetLocalComponents(reducedX_, 1, &i, NULL, &ddata);
          index = selectedList_[i-startRow2];
          HYPRE_IJVectorSetLocalComponents(HYx_,1,&index,NULL,&ddata);
       }
       rowNum = localStartRow_ - 1;
       for ( i = startRow; i < startRow+A21NRows_; i++ )
       {
          HYPRE_IJVectorGetLocalComponents(x2, 1, &i, NULL, &ddata);
          while (HYPRE_LSI_Search(selectedList_,rowNum,localNRows)>=0)
             rowNum++;
          HYPRE_IJVectorSetLocalComponents(HYx_,1,&rowNum,NULL,&ddata);
          rowNum++;
       } 

       //-------------------------------------------------------------
       // residual norm check 
       //-------------------------------------------------------------

       A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
       x_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYx_);
       b_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYb_);
       r_csr  = (HYPRE_ParVector)    HYPRE_IJVectorGetLocalStorage(HYr_);
       HYPRE_ParVectorCopy( b_csr, r_csr );
       HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
       HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
       rnorm = sqrt( rnorm );
       if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SCHURREDUCE1 ) )
          printf("buildReducedSystemSoln::final residual norm = %e\n", rnorm);
    } 
    currX_ = HYx_;

    //****************************************************************
    // clean up
    //----------------------------------------------------------------

    //HYPRE_IJMatrixDestroy(HYA21_); 
    //HYA21_ = NULL;
    //HYPRE_IJMatrixDestroy(HYinvA22_); 
    //HYinvA22_ = NULL;
    HYPRE_IJVectorDestroy(R1); 
    HYPRE_IJVectorDestroy(x2); 
    return rnorm;
}

//******************************************************************************
// form modified right hand side  (f2 = f2 - C*M*f1)
//------------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSchurReducedRHS()
{

    int    i, j, ncnt, ncnt2, ierr, *colInd, CTStartRow, CStartRow, rowIndex;
    int    StartRow, EndRow, nRows, nSchur, *schurList, *ProcNRows, *ProcNSchur;
    int    globalNSchur, CTNRows, CTNCols, CTGlobalNRows, CTGlobalNCols;
    int    CNRows, CGlobalNRows, *tempList, searchIndex, rowCount, rowSize;
    double ddata, *colVal;
    HYPRE_IJMatrix     Cmat, Mmat;
    HYPRE_IJVector     f1, f2, f2hat;
    HYPRE_ParVector    f1_csr, f2_csr, f2hat_csr;
    HYPRE_ParCSRMatrix M_csr, C_csr;

    //******************************************************************
    // initial set up 
    //------------------------------------------------------------------

    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
    {
       printf("buildSchurRHS begins....\n");
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
    tempList[mypid_] = StartRow;
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
    MPI_Allreduce(&CTGlobalNRows, &CTNRows, 1, MPI_INT, MPI_SUM, comm_);
    MPI_Allreduce(&CTGlobalNCols, &CTNCols, 1, MPI_INT, MPI_SUM, comm_);
    Cmat         = HYA12_; 
    Mmat         = HYinvA22_; 
    CNRows       = CTNCols;
    CGlobalNRows = CTGlobalNCols;
    nSchur       = A21NCols_;
    schurList    = selectedList_;
    CGlobalNRows = globalNSchur;
    M_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(Mmat);
    C_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(Cmat);

    // *****************************************************************
    // form f2hat = C*M*f1
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, &f1, CTGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f1, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f1,CTStartRow,CTStartRow+CTNRows);
    ierr  = HYPRE_IJVectorAssemble(f1);
    ierr += HYPRE_IJVectorInitialize(f1);
    assert(!ierr);

    HYPRE_IJVectorCreate(comm_, &f2hat, CGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2hat, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f2hat,CStartRow,CStartRow+CNRows);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2hat);
    assert(!ierr);

    rowCount = CTStartRow;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_IJVectorGetLocalComponents(HYb_, 1, &i, NULL, &ddata);
          HYPRE_ParCSRMatrixGetRow(M_csr,rowCount,&rowSize,&colInd,&colVal);
          if ( rowSize != 1 ) printf("buildReducedRHS : WARNING.\n");
          if ( colVal[0] != 0.0 ) ddata /= colVal[0];
          ierr = HYPRE_IJVectorSetLocalComponents(f1,1,&rowCount,NULL,&ddata);
          HYPRE_ParCSRMatrixRestoreRow(M_csr,rowCount,&rowSize,&colInd,&colVal);
          assert( !ierr );
          rowCount++;
       }
    } 
        
    f1_csr     = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f1);
    f2hat_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f2hat);
    HYPRE_ParCSRMatrixMatvec( 1.0, C_csr, f1_csr, 0.0, f2hat_csr );
    HYPRE_IJVectorDestroy(f1); 

    //------------------------------------------------------------------
    // form f2 = f2 - f2hat 
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, &f2, CGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f2,CStartRow,CStartRow+CNRows);
    ierr += HYPRE_IJVectorAssemble(f2);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2);
    assert(!ierr);

    rowCount = CStartRow;
    for ( i = 0; i < nSchur; i++ ) 
    {
       rowIndex = schurList[i];
       HYPRE_IJVectorGetLocalComponents(HYb_, 1, &rowIndex, NULL, &ddata);
       ddata = - ddata;
       ierr = HYPRE_IJVectorSetLocalComponents(f2,1,&rowCount,NULL,&ddata);
       HYPRE_IJVectorGetLocalComponents(f2hat, 1, &rowCount, NULL, &ddata);
       HYPRE_IJVectorAddToLocalComponents(f2,1,&rowCount,NULL,&ddata);
       HYPRE_IJVectorGetLocalComponents(f2, 1, &rowCount, NULL, &ddata);
       assert( !ierr );
       rowCount++;
    } 
    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SCHURREDUCE1) )
    {
       printf("buildSchurRHS ends....\n");
    }
}

