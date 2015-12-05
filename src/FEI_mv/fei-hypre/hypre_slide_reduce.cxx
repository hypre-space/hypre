/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/





//***************************************************************************
// system includes
//---------------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

//***************************************************************************
// HYPRE includes
//---------------------------------------------------------------------------

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_LinSysCore.h"
#include "HYPRE_LSI_mli.h"

#define HAVE_MLI

//***************************************************************************
// local defines and external functions
//---------------------------------------------------------------------------

#define habs(x) (((x) > 0.0) ? x : -(x))

extern "C" 
{
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix**);
   void qsort0(int *, int, int);
   void qsort1(int *, double *, int, int);
   int  HYPRE_LSI_Search(int*, int, int);
}

//*****************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//
// Additional assumptions are :
//
//    - a given slave equation and the corresponding constraint equation
//      reside in the same processor
//    - constraint equations are given at the end of the local matrix
//      (hence given by EndRow_-nConstr to EndRow_)
//    - each processor gets a contiguous block of equations, and processor
//      i+1 has equation numbers higher than those of processor i
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSlideReducedSystem()
{
    int    i, j, StartRow, EndRow, rowSize, *colInd, globalNConstr;
    int    nRows, *ProcNRows, *tempList, globalNRows, ncnt;
    int    globalNSelected, *globalSelectedList, *globalSelectedListAux;
    int    *ProcNConstr, isAConstr, ncnt2;
    double *colVal;
    HYPRE_ParCSRMatrix A_csr, RAP_csr;

    //******************************************************************
    // initial set up 
    //------------------------------------------------------------------

    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SLIDEREDUCE1) )
       printf("%4d : SlideReduction begins....\n",mypid_);
    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction - StartRow/EndRow = %d %d\n",mypid_,
                                      StartRow,EndRow);

    //******************************************************************
    // construct local and global information about where the constraints
    // are (this is given by user or searched within this code)
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // get the CSR matrix for A
    //------------------------------------------------------------------

    HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);

    //------------------------------------------------------------------
    // search the entire local matrix to find where the constraint
    // equations are, if not already given
    //------------------------------------------------------------------
    
    MPI_Allreduce(&nConstraints_,&globalNConstr,1,MPI_INT,MPI_SUM,comm_);
    if ( globalNConstr == 0 )
    {
       for ( i = EndRow; i >= StartRow; i-- ) 
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          isAConstr = 1;
          for (j = 0; j < rowSize; j++) 
             if ( colInd[j] == i && colVal[j] != 0.0 ) {isAConstr = 0; break;}
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          if ( isAConstr ) nConstraints_++;
          else             break;
       }
    }
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction - no. constr = %d\n",mypid_,nConstraints_);

    MPI_Allreduce(&nConstraints_, &globalNConstr, 1, MPI_INT, MPI_SUM, comm_);
    if ( globalNConstr == 0 ) return;

    //------------------------------------------------------------------
    // get information about nRows from all processors
    //------------------------------------------------------------------
 
    nRows       = localEndRow_ - localStartRow_ + 1;
    ProcNRows   = new int[numProcs_];
    tempList    = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nRows;
    MPI_Allreduce(tempList, ProcNRows, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction - localNRows = %d\n", mypid_, nRows);

    //------------------------------------------------------------------
    // compute the base NRows on each processor
    // (This is needed later on for column index conversion)
    //------------------------------------------------------------------

    globalNRows = 0;
    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ ) 
    {
       globalNRows   += ProcNRows[i];
       ncnt2          = ProcNRows[i];
       ProcNRows[i]   = ncnt;
       ncnt          += ncnt2;
    }

    //------------------------------------------------------------------
    // compose a global array marking where the constraint equations are
    //------------------------------------------------------------------
    
    globalNConstr = 0;
    tempList    = new int[numProcs_];
    ProcNConstr = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nConstraints_;
    MPI_Allreduce(tempList,ProcNConstr,numProcs_,MPI_INT,MPI_SUM,comm_);
    delete [] tempList;

    //------------------------------------------------------------------
    // compute the base nConstraints on each processor
    // (This is needed later on for column index conversion)
    //------------------------------------------------------------------

    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ ) 
    {
       globalNConstr += ProcNConstr[i];
       ncnt2          = ProcNConstr[i];
       ProcNConstr[i] = ncnt;
       ncnt          += ncnt2;
    }
   
    //******************************************************************
    // compose the local and global selected node lists
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // allocate array for storing indices of selected nodes
    //------------------------------------------------------------------

    globalNSelected = globalNConstr;
    if (globalNSelected > 0) 
    {
       globalSelectedList = new int[globalNSelected];
       globalSelectedListAux = new int[globalNSelected];
    }
    else globalSelectedList = globalSelectedListAux = NULL;
    if ( selectedList_    != NULL ) delete [] selectedList_;
    if ( selectedListAux_ != NULL ) delete [] selectedListAux_;
    if ( nConstraints_ > 0 ) 
    {
       selectedList_ = new int[nConstraints_];
       selectedListAux_ = new int[nConstraints_];
    }
    else selectedList_ = selectedListAux_ = NULL;
   
    //------------------------------------------------------------------
    // call the three parts
    //------------------------------------------------------------------

    buildSlideReducedSystemPartA(ProcNRows,ProcNConstr,globalNRows, 
                                 globalNConstr,globalSelectedList, 
                                 globalSelectedListAux);
    buildSlideReducedSystemPartB(ProcNRows,ProcNConstr,globalNRows,
                                 globalNConstr,globalSelectedList, 
                                 globalSelectedListAux, &RAP_csr);
    buildSlideReducedSystemPartC(ProcNRows,ProcNConstr,globalNRows,
                                 globalNConstr,globalSelectedList, 
                                 globalSelectedListAux, RAP_csr);

    //------------------------------------------------------------------
    // initialize global variables and clean up 
    //------------------------------------------------------------------

    currA_ = reducedA_;
    currB_ = reducedB_;
    currR_ = reducedR_;
    currX_ = reducedX_;
    delete [] globalSelectedList;
    delete [] globalSelectedListAux;
    delete [] ProcNRows;
    delete [] ProcNConstr;
    HYPRE_ParCSRMatrixDestroy(RAP_csr);
    //if ( HYA_ != NULL ) {HYPRE_IJMatrixDestroy(HYA_); HYA_ = NULL;}
    if ( colIndices_ != NULL )
    {
       for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
          if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
       delete [] colIndices_;
       colIndices_ = NULL;
    }
    if ( colValues_ != NULL )
    {
       for ( j = 0; j < localEndRow_-localStartRow_+1; j++ )
          if ( colValues_[j] != NULL ) delete [] colValues_[j];
       delete [] colValues_;
       colValues_ = NULL;
       if ( rowLengths_ != NULL ) 
       {
          delete [] rowLengths_;
          rowLengths_ = NULL;
       }
    }
}

//*****************************************************************************
// Part A of buildSlideReducedSystem : generate a selected equation list
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSlideReducedSystemPartA(int *ProcNRows, 
                            int *ProcNConstr, int globalNRows,
                            int globalNConstr, int *globalSelectedList, 
                            int *globalSelectedListAux)
{
    int    i, ncnt2, StartRow, EndRow, ncnt;;
    int    nSlaves, *constrListAux, colIndex, searchIndex, procIndex, ubound;
    int    j, k, ierr, rowSize, *colInd, *colInd2, rowIndex, nSelected;
    int    *recvCntArray, *displArray, *selectedList, *selectedListAux;
    int    rowSize2;
    double *colVal, searchValue, *dble_array, *colVal2;
    HYPRE_ParCSRMatrix A_csr;

    //------------------------------------------------------------------
    // get matrix information
    //------------------------------------------------------------------

    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
    nSelected       = nConstraints_;
    selectedList    = selectedList_;
    selectedListAux = selectedListAux_;

    //------------------------------------------------------------------
    // compose candidate slave list
    //------------------------------------------------------------------

    nSlaves       = 0;
    constrListAux = NULL;
    if ( nConstraints_ > 0 && constrList_ == NULL )
    {
       constrList_   = new int[EndRow-nConstraints_-StartRow+1];
       constrListAux = new int[EndRow-nConstraints_-StartRow+1];

       //------------------------------------------------------------------
       // candidates are those with 1 link to the constraint list
       //------------------------------------------------------------------

       for ( i = StartRow; i <= EndRow-nConstraints_; i++ ) 
       {
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          ncnt = 0;
          for (j = 0; j < rowSize; j++) 
          {
             colIndex = colInd[j];
             for (procIndex=0; procIndex < numProcs_; procIndex++ )
                if ( colIndex < ProcNRows[procIndex] ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows - 
                         (globalNConstr-ProcNConstr[procIndex-1]);
             else                          
                ubound = ProcNRows[procIndex] - (ProcNConstr[procIndex] - 
                                                 ProcNConstr[procIndex-1]); 

             //Note : include structural zeros by not checking for nonzero
             //if ( colIndex >= ubound && colVal[j] != 0.0 ) 
             if ( colIndex >= ubound ) 
             {
                ncnt++;
                searchIndex = colIndex;
             }
             if ( ncnt > 1 ) break;
          }
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          if ( j == rowSize && ncnt == 1 ) 
          {
             constrListAux[nSlaves] = searchIndex;
             constrList_[nSlaves++] = i;
          }
          if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE2 )
          {
             if ( j == rowSize && ncnt == 1 ) 
                printf("%4d : SlideReductionA - slave candidate %d = %d(%d)\n", 
                        mypid_, nSlaves-1, i, constrListAux[nSlaves-1]);
          }
       }
       if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       {
          printf("%4d : SlideReductionA - nSlave Candidate, nConstr = %d %d\n",
                 mypid_,nSlaves, nConstraints_);
       }
    }
    else if ( constrList_ != NULL )
    {
       if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 ) )
          printf("%4d : SlideReductionA WARNING - constraint list not empty\n",
                  mypid_);
    }   

    //---------------------------------------------------------------------
    // search the constraint equations for the selected nodes
    // (search for candidates column index with maximum magnitude)
    //---------------------------------------------------------------------
    
    nSelected   = 0;
    rowIndex    = -1;
    searchIndex = 0;

    for ( i = EndRow-nConstraints_+1; i <= EndRow; i++ ) 
    {
       ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       assert(!ierr);
       searchIndex = -1;
       searchValue = -1.0E10;
       for (j = 0; j < rowSize; j++) 
       {
          if (colVal[j] != 0.0 && colInd[j] >= StartRow 
                               && colInd[j] <= (EndRow-nConstraints_)) 
          {
             colIndex = hypre_BinarySearch(constrList_,colInd[j],nSlaves);
             if ( colIndex >= 0 && constrListAux[colIndex] != -1) 
             {
                 if ( habs(colVal[j]) > searchValue )
                 {
                    if (i != constrListAux[colIndex]) 
                    {
                       if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                       {
                          printf("%4d : SlideReductionA WARNING - slave %d",
                                  mypid_, colInd[j]);
                          printf(" candidate does not have constr %d\n", i);
                       }
                    }
                    searchValue = habs(colVal[j]);
                    searchIndex = colInd[j];
                 }
             }
          }
       } 
       if ( searchIndex >= 0 )
       {
          selectedList[nSelected++] = searchIndex;
          if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE2 )
          {
             printf("%4d : SlideReductionA - constraint %4d <=> slave %d \n",
                    mypid_,i,searchIndex);
          }
       } 
       else 
       {
          // get ready for error processing

          colInd2 = new int[rowSize];
          colVal2 = new double[rowSize];
          for ( j = 0; j < rowSize; j++ )
          {    
             colInd2[j] = colInd[j];
             colVal2[j] = colVal[j];
          }
          rowIndex = i;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          colInd = colInd2;
          colVal = colVal2;
          break;
       }
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }

    //---------------------------------------------------------------------
    // error processing
    //---------------------------------------------------------------------

    if ( searchIndex < 0 ) searchIndex = 1; else searchIndex = 0;
    MPI_Allreduce(&searchIndex, &ncnt,1,MPI_INT,MPI_MAX,comm_);

    if ( ncnt > 0 )
    {
       ncnt2 = 0;
       while ( ncnt2 < numProcs_ )
       { 
          if ( ncnt2 == mypid_ && rowIndex >= 0 )
          {
             printf("%4d : SlideReductionA ERROR - constraint number",mypid_);
             printf(" cannot be found for row %d\n", rowIndex);
             for (j = 0; j < rowSize; j++) 
             {
                printf("ROW %4d COL = %d VAL = %e\n",rowIndex,colInd[j],
                       colVal[j]);
                if (colVal[j] != 0.0 && colInd[j] >= StartRow 
                                     && colInd[j] <= (EndRow-nConstraints_)) 
                {
                   colIndex = colInd[j];
                   HYPRE_ParCSRMatrixGetRow(A_csr,colIndex,&rowSize2,&colInd2,
                                            &colVal2);
                   printf("      row %4d (%d) : \n",colIndex, rowSize2);
                   for (k = 0; k < rowSize2; k++) 
                      printf("    row %4d col = %d val = %e\n",colIndex,
                                            colInd2[k],colVal2[k]);
                   HYPRE_ParCSRMatrixRestoreRow(A_csr,colIndex,&rowSize2,
                                            &colInd2,&colVal2);
                }
             }
             printf("===================================================\n");
          }
          ncnt2++;
          MPI_Barrier(comm_);
       }
       MPI_Finalize();
       exit(1);
    }
    delete [] constrListAux;

    //------------------------------------------------------------------
    // sort the local selected node list and its auxiliary list, then
    // form a global list of selected nodes on each processor
    // form the corresponding auxiliary list for later pruning
    //------------------------------------------------------------------

    dble_array = new double[nSelected];
    for ( i = 0; i < nSelected; i++ ) dble_array[i] = (double) i; 
    if ( nSelected > 1 ) qsort1(selectedList, dble_array, 0, nSelected-1);
    for (i = 1; i < nSelected; i++) 
    {
       if ( selectedList[i] == selectedList[i-1] )
       {
          printf("%4d : SlideReductionA ERROR - repeated selected nodes %d \n", 
                 mypid_, selectedList[i]);
          exit(1);
       }
    }
    for (i = 0; i < nSelected; i++) selectedListAux[i] = (int) dble_array[i];
    delete [] dble_array;
    
    recvCntArray = new int[numProcs_];
    displArray   = new int[numProcs_];
    MPI_Allgather(&nSelected, 1, MPI_INT, recvCntArray, 1,MPI_INT, comm_);
    displArray[0] = 0;
    for ( i = 1; i < numProcs_; i++ ) 
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    for ( i = 0; i < nSelected; i++ ) 
       selectedListAux[i] += displArray[mypid_]; 
    MPI_Allgatherv(selectedList, nSelected, MPI_INT, globalSelectedList,
                   recvCntArray, displArray, MPI_INT, comm_);
    MPI_Allgatherv(selectedListAux, nSelected, MPI_INT, globalSelectedListAux,
                   recvCntArray, displArray, MPI_INT, comm_);
    for ( i = 0; i < nSelected; i++ ) 
       selectedListAux[i] -= displArray[mypid_]; 
    delete [] recvCntArray;
    delete [] displArray;

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE2 )
    {
       for ( i = 0; i < nSelected; i++ )
          printf("%4d : SlideReductionA - selectedList %d = %d(%d)\n",mypid_,
                 i,selectedList[i],selectedListAux[i]);
    }
}
 
//*****************************************************************************
// Part B of buildSlideReducedSystem : create submatrices
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSlideReducedSystemPartB(int *ProcNRows, 
                            int *ProcNConstr, int globalNRows, 
                            int globalNConstr, int *globalSelectedList, 
                            int *globalSelectedListAux, 
                            HYPRE_ParCSRMatrix *rap_csr)
{
    int    A21NRows, A21GlobalNRows, A21NCols, A21GlobalNCols, A21StartRow;
    int    i, j, nRows, ncnt, StartRow, A21StartCol;
    int    ierr, rowCount, maxRowSize, newEndRow, *A21MatSize, EndRow;
    int    rowIndex, rowSize, *colInd, rowSize2, colIndex, searchIndex;
    int    nnzA21, *newColInd, diagCount, newRowSize, procIndex;
    int    *recvCntArray, *displArray, *invA22MatSize, one=1;
    int    invA22NRows, invA22GlobalNRows, invA22NCols, invA22GlobalNCols;
    int    nSelected, *selectedList, *selectedListAux, globalNSelected;
    double *colVal, *newColVal, *diagonal, *extDiagonal;
    HYPRE_ParCSRMatrix A_csr, A21_csr, invA22_csr, RAP_csr;
    HYPRE_IJMatrix     A21, invA22;

    //******************************************************************
    // get matrix and constraint information
    //------------------------------------------------------------------

    nRows    = localEndRow_ - localStartRow_ + 1;
    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
    globalNSelected = globalNConstr;
    nSelected       = nConstraints_;
    selectedList    = selectedList_;
    selectedListAux = selectedListAux_;
   
    //------------------------------------------------------------------
    // calculate the dimension of A21
    //------------------------------------------------------------------

    A21NRows       = 2 * nConstraints_;
    A21NCols       = nRows - 2 * nConstraints_;
    A21GlobalNRows = 2 * globalNConstr;
    A21GlobalNCols = globalNRows - 2 * globalNConstr;
    A21StartRow    = 2 * ProcNConstr[mypid_];
    A21StartCol    = ProcNRows[mypid_] - 2 * ProcNConstr[mypid_];

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
    {
       printf("%4d : SlideReductionB - A21StartRow  = %d\n", mypid_,
                                       A21StartRow);
       printf("%4d : SlideReductionB - A21GlobalDim = %d %d\n", mypid_, 
                                       A21GlobalNRows, A21GlobalNCols);
       printf("%4d : SlideReductionB - A21LocalDim  = %d %d\n",mypid_,
                                       A21NRows, A21NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A21
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_, A21StartRow, A21StartRow+A21NRows-1,
				 A21StartCol, A21StartCol+A21NCols-1, &A21);
    ierr += HYPRE_IJMatrixSetObjectType(A21, HYPRE_PARCSR);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros in the first nConstraint row of A21
    // (which consists of the rows in selectedList), the nnz will
    // be reduced by excluding the constraint and selected slave columns
    //------------------------------------------------------------------

    rowCount   = 0;
    maxRowSize = 0;
    newEndRow  = EndRow - nConstraints_;
    A21MatSize = new int[A21NRows];

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             rowIndex = selectedList[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowSize2 = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 ) 
          {
	     searchIndex = hypre_BinarySearch(globalSelectedList,colIndex, 
                                              globalNSelected);
             if (searchIndex < 0 && 
                 (colIndex <= newEndRow || colIndex >= localEndRow_)) 
                rowSize2++;
          }
       }
       A21MatSize[rowCount] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowCount++;
    }

    //------------------------------------------------------------------
    // compute the number of nonzeros in the second nConstraint row of A21
    // (which consists of the rows in constraint equations), the nnz will
    // be reduced by excluding the selected slave columns only (since the
    // entries corresponding to the constraint columns are 0, and since
    // the selected matrix is a diagonal matrix, there is no need to 
    // search for slave equations in the off-processor list)
    //------------------------------------------------------------------

    rowCount = nSelected;
    for ( i = EndRow-nConstraints_+1; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowSize2 = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
	     searchIndex = hypre_BinarySearch(globalSelectedList,colIndex,
                                              globalNSelected); 
             if ( searchIndex < 0 ) rowSize2++;
          }
       }
       A21MatSize[rowCount] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowCount++;
    }
    nnzA21 = 0;
    for ( i = 0; i < 2*nConstraints_; i++ ) nnzA21 += A21MatSize[i];

    //------------------------------------------------------------------
    // after fetching the row sizes, set up A21 with such sizes
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(A21, A21MatSize);
    ierr += HYPRE_IJMatrixInitialize(A21);
    assert(!ierr);
    delete [] A21MatSize;

    //------------------------------------------------------------------
    // next load the first nConstraint row to A21 extracted from A
    // (at the same time, the D block is saved for future use)
    //------------------------------------------------------------------

    rowCount  = A21StartRow;
    if ( nConstraints_ > 0 ) diagonal = new double[nConstraints_];
    else                     diagonal = NULL;
    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];

    diagCount = 0;
    for ( i = 0; i < nSelected; i++ )
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             rowIndex = selectedList[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
             if (colIndex <= newEndRow || colIndex >= localEndRow_) 
             {
	        searchIndex = HYPRE_LSI_Search(globalSelectedList,colIndex, 
                                               globalNSelected); 
                if ( searchIndex < 0 ) 
                {
                   searchIndex = - searchIndex - 1;
                   for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                      if ( ProcNRows[procIndex] > colIndex ) break;
                   procIndex--;
                   colIndex = colInd[j]-ProcNConstr[procIndex]-searchIndex;
                   newColInd[newRowSize]   = colIndex;
                   newColVal[newRowSize++] = colVal[j];
                   if ( colIndex < 0 || colIndex >= A21GlobalNCols )
                   {
                      if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                      {
                         printf("%4d : SlideReductionB WARNING - A21 ",mypid_);
                         printf("out of range (%d,%d (%d))\n", rowCount, 
                                colIndex, A21GlobalNCols);
                      } 
                   } 
                   if ( newRowSize > maxRowSize+1 ) 
                   {
                      if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                      {
                         printf("%4d : SlideReductionB WARNING - ",mypid_);
                         printf("passing array boundary(1).\n");
                      }
                   }
                }
             }
             else if ( colIndex > newEndRow && colIndex <= EndRow ) 
             {
                if ( colVal[j] != 0.0 ) diagonal[diagCount++] = colVal[j];
                if ( habs(colVal[j]) < 1.0E-8 )
                {
                   if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                   {
                      printf("%4d : SlideReductionB WARNING - large entry ",
                             mypid_);
                      printf("in invA22\n");
                   }
                }
             }
          } 
       }

       HYPRE_IJMatrixSetValues(A21, 1, &newRowSize, (const int *) &rowCount,
		(const int *) newColInd, (const double *) newColVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       if ( diagCount != (i+1) )
       {
          printf("%4d : SlideReductionB ERROR (3) - %d %d.\n", mypid_,
                  diagCount,i+1);
          exit(1);
       }
       rowCount++;
    }

    //------------------------------------------------------------------
    // send the diagonal to each processor that needs them
    //------------------------------------------------------------------

    recvCntArray = new int[numProcs_];
    displArray   = new int[numProcs_];
    MPI_Allgather(&diagCount, 1, MPI_INT, recvCntArray, 1, MPI_INT, comm_);
    displArray[0] = 0;
    for ( i = 1; i < numProcs_; i++ ) 
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    ncnt = displArray[numProcs_-1] + recvCntArray[numProcs_-1];
    if ( ncnt > 0 ) extDiagonal = new double[ncnt];
    else            extDiagonal = NULL;
    MPI_Allgatherv(diagonal, diagCount, MPI_DOUBLE, extDiagonal,
                   recvCntArray, displArray, MPI_DOUBLE, comm_);
    diagCount = ncnt;
    delete [] recvCntArray;
    delete [] displArray;
    if ( diagonal != NULL ) delete [] diagonal;

    //------------------------------------------------------------------
    // next load the second nConstraint rows to A21 extracted from A
    //------------------------------------------------------------------

    for ( i = EndRow-nConstraints_+1; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex    = colInd[j];
	  searchIndex = HYPRE_LSI_Search(globalSelectedList,colIndex,
                                         globalNSelected); 
          if ( searchIndex < 0 && colVal[j] != 0.0 ) 
          {
             searchIndex = - searchIndex - 1;
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             procIndex--;
             colIndex = colInd[j] - ProcNConstr[procIndex] - searchIndex;
             newColInd[newRowSize]   = colIndex;
             newColVal[newRowSize++] = colVal[j];
             if ( colIndex < 0 || colIndex >= A21GlobalNCols )
             {
                if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                   printf("%4d : SlideReductionB WARNING - A21(%d,%d(%d))\n",
                          mypid_, rowCount, colIndex, A21GlobalNCols);
             } 
             if ( newRowSize > maxRowSize+1 ) 
             {
                if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                {
                   printf("%4d : SlideReductionB WARNING - ",mypid_);
                   printf("passing array boundary(2).\n");
                }
             }
          } 
       }
       HYPRE_IJMatrixSetValues(A21, 1, &newRowSize, (const int *) &rowCount,
		(const int *) newColInd, (const double *) newColVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowCount++;
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix and sanitize
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(A21);
    HYPRE_IJMatrixGetObject(A21, (void **) &A21_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A21_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       ncnt = 0;
       MPI_Barrier(comm_);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d : SlideReductionB - matrix A21 assembled %d.\n",
                                        mypid_,A21StartRow);
             fflush(stdout);
             for ( i = A21StartRow; i < A21StartRow+2*nConstraints_; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(A21_csr,i,&rowSize,&colInd,&colVal);
                printf("A21 ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(A21_csr, i, &rowSize,
                                             &colInd, &colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(comm_);
       }
    }

    //******************************************************************
    // construct invA22
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of invA22
    //------------------------------------------------------------------

    invA22NRows       = A21NRows;
    invA22NCols       = invA22NRows;
    invA22GlobalNRows = A21GlobalNRows;
    invA22GlobalNCols = invA22GlobalNRows;
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
    {
       printf("%4d : SlideReductionB - A22GlobalDim = %d %d\n", mypid_, 
                        invA22GlobalNRows, invA22GlobalNCols);
       printf("%4d : SlideReductionB - A22LocalDim  = %d %d\n", mypid_, 
                        invA22NRows, invA22NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A22
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixCreate(comm_, A21StartRow, A21StartRow+invA22NRows-1,
                           A21StartRow, A21StartRow+invA22NCols-1, &invA22);
    ierr += HYPRE_IJMatrixSetObjectType(invA22, HYPRE_PARCSR);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the no. of nonzeros in the first nConstraint row of invA22
    //------------------------------------------------------------------

    maxRowSize  = 0;
    invA22MatSize = new int[invA22NRows];
    for ( i = 0; i < nConstraints_; i++ ) invA22MatSize[i] = 1;

    //------------------------------------------------------------------
    // compute the number of nonzeros in the second nConstraints row of 
    // invA22 (consisting of [D and A22 block])
    //------------------------------------------------------------------

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             rowIndex = selectedList[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowSize2 = 1;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 ) 
          {
             if ( colIndex >= StartRow && colIndex <= newEndRow ) 
             {
	        searchIndex = hypre_BinarySearch(selectedList, colIndex, 
                                                 nSelected); 
                if ( searchIndex >= 0 ) rowSize2++;
             } 
             else if ( colIndex < StartRow || colIndex > EndRow ) 
             {
	        searchIndex = hypre_BinarySearch(globalSelectedList,colIndex, 
                                                 globalNSelected); 
                if ( searchIndex >= 0 ) rowSize2++;
             }
          }
       }
       invA22MatSize[nConstraints_+i] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }

    //------------------------------------------------------------------
    // after fetching the row sizes, set up invA22 with such sizes
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(invA22, invA22MatSize);
    ierr += HYPRE_IJMatrixInitialize(invA22);
    assert(!ierr);
    delete [] invA22MatSize;

    //------------------------------------------------------------------
    // next load the first nConstraints_ row to invA22 extracted from A
    // (that is, the D block)
    //------------------------------------------------------------------

    maxRowSize++;
    newColInd = new int[maxRowSize];
    newColVal = new double[maxRowSize];

    for ( i = 0; i < diagCount; i++ ) 
    {
       extDiagonal[i] = 1.0 / extDiagonal[i];
    }
    for ( i = 0; i < nConstraints_; i++ ) 
    {
       newColInd[0] = A21StartRow + nConstraints_ + i; 
       rowIndex     = A21StartRow + i;
       if ( newColInd[0] < 0 || newColInd[0] >= invA22GlobalNCols )
       {
          if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
             printf("%4d : SlideReductionB WARNING - A22 (%d, %d (%d))\n", 
                    mypid_, rowIndex, newColInd[0], invA22GlobalNCols);
       } 
       newColVal[0] = extDiagonal[A21StartRow/2+i];
       ierr = HYPRE_IJMatrixSetValues(invA22, 1, &one, (const int *) &rowIndex,
			(const int *) newColInd, (const double *) newColVal);
       assert(!ierr);
    }

    //------------------------------------------------------------------
    // next load the second nConstraints_ rows to A22 extracted from A
    //------------------------------------------------------------------

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             rowIndex = selectedList[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 1;
       newColInd[0] = A21StartRow + i;
       newColVal[0] = extDiagonal[A21StartRow/2+i]; 
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 )
          {
	     searchIndex = hypre_BinarySearch(globalSelectedList,colIndex, 
                                              globalNSelected); 
             if ( searchIndex >= 0 ) 
             {
                searchIndex = globalSelectedListAux[searchIndex];
                for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                if ( procIndex == numProcs_ )
                {
                   newColInd[newRowSize] = searchIndex + globalNConstr; 
                }
                else
                {
                   newColInd[newRowSize] = searchIndex + 
                                           ProcNConstr[procIndex]; 
                }
                if ( newColInd[newRowSize] < 0 || 
                     newColInd[newRowSize] >= invA22GlobalNCols )
                {
                   if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                      printf("%4d : SlideReductionB WARNING - A22(%d,%d,%d)\n",
                          mypid_,rowCount,newColInd[newRowSize],
                          invA22GlobalNCols);
                } 
                newColVal[newRowSize++] = - extDiagonal[A21StartRow/2+i] * 
                                        colVal[j] * extDiagonal[searchIndex];
                if ( newRowSize > maxRowSize )
                {
                   if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                   {
                      printf("%4d : SlideReductionB WARNING - ",mypid_);
                      printf("passing array boundary(3).\n");
                   }
                }
      	     } 
	  } 
       }
       rowCount = A21StartRow + nConstraints_ + i;
       ierr = HYPRE_IJMatrixSetValues(invA22, 1, &newRowSize, 
		(const int *) &rowCount, (const int *) newColInd, 
		(const double *) newColVal);
       assert(!ierr);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }
    delete [] newColInd;
    delete [] newColVal;
    delete [] extDiagonal;

    //------------------------------------------------------------------
    // finally assemble the matrix and sanitize
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(invA22);
    HYPRE_IJMatrixGetObject(invA22, (void **) &invA22_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) invA22_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       ncnt = 0;
       MPI_Barrier(comm_);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d : SlideReductionB - invA22 \n", mypid_);
             for ( i = A21StartRow; i < A21StartRow+2*nConstraints_; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(invA22_csr,i,&rowSize,&colInd,
                                         &colVal);
                printf("invA22 ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(invA22_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(comm_);
          ncnt++;
       }
    }

    //******************************************************************
    // perform the triple matrix product
    //------------------------------------------------------------------

    HYPRE_IJMatrixGetObject(A21, (void **) &A21_csr);
    HYPRE_IJMatrixGetObject(invA22, (void **) &invA22_csr);
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReductionB - Triple matrix product starts\n",mypid_);

    hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix *) invA22_csr,
                                     (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix **) &RAP_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReductionB - Triple matrix product ends\n", mypid_);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       MPI_Barrier(comm_);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             for ( i = A21StartRow; i < A21StartRow+A21NCols; i++ ) {
                HYPRE_ParCSRMatrixGetRow(RAP_csr,i,&rowSize,&colInd, &colVal);
                printf("RAP ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(RAP_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
          }
          MPI_Barrier(comm_);
          ncnt++;
       }
    }

    //******************************************************************
    // set global objects and checking
    //------------------------------------------------------------------

    HYA21_     = A21; 
    HYinvA22_  = invA22; 
    (*rap_csr) = RAP_csr;

    MPI_Allreduce(&nnzA21,&ncnt,1,MPI_INT,MPI_SUM,comm_);
    if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 ))
       printf("       SlideReductionB : NNZ of A21 = %d\n", ncnt);
}

//*****************************************************************************
// Part C of buildSlideReducedSystem : create subvectors and wrap up
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSlideReducedSystemPartC(int *ProcNRows,
                            int *ProcNConstr, int globalNRows,
                            int globalNConstr, int *globalSelectedList,
                            int *globalSelectedListAux, 
                            HYPRE_ParCSRMatrix RAP_csr)
{
    int    i, j, nRows, StartRow, EndRow; 
    int    newNRows, *reducedAMatSize, reducedAStartRow;
    int    rowCount, rowIndex, newRowSize, rowSize, rowSize2, *newColInd;
    int    *colInd, *colInd2, colIndex, searchIndex, ubound, ncnt, ierr;
    int    A21NRows, A21NCols, A21GlobalNRows, A21GlobalNCols, A21StartRow;
    int    A12NRows, A12NCols, A12GlobalNRows, A12GlobalNCols, A12StartRow;
    int    *A12MatSize, newEndRow, globalNSelected, procIndex, nnzA12;
    int    nSelected, *selectedList, *selectedListAux, A21StartCol;
    double *colVal, *colVal2, *newColVal, ddata;
    HYPRE_IJMatrix     A12, reducedA;
    HYPRE_ParCSRMatrix A_csr, A12_csr, reducedA_csr, invA22_csr;
    HYPRE_IJVector     f2, f2hat;
    HYPRE_ParVector    f2_csr, f2hat_csr, reducedB_csr;

    //------------------------------------------------------------------
    // get matrix and constraint information
    //------------------------------------------------------------------

    nRows     = localEndRow_ - localStartRow_ + 1;
    StartRow  = localStartRow_ - 1;
    EndRow    = localEndRow_ - 1;
    newEndRow = EndRow - nConstraints_;
    newEndRow = EndRow - nConstraints_;
    HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);
    globalNSelected = globalNConstr;
    nSelected       = nConstraints_;
    selectedList    = selectedList_;
    selectedListAux = selectedListAux_;
   
    //------------------------------------------------------------------
    // first calculate the dimension of the reduced matrix
    //------------------------------------------------------------------

    newNRows       = nRows - 2 * nConstraints_;
    A21StartCol = ProcNRows[mypid_] - 2 * ProcNConstr[mypid_];
    ierr  = HYPRE_IJMatrixCreate(comm_, A21StartCol, A21StartCol+newNRows-1,
                             A21StartCol, A21StartCol+newNRows-1, &reducedA);
    ierr += HYPRE_IJMatrixSetObjectType(reducedA, HYPRE_PARCSR);
    assert(!ierr);

    //------------------------------------------------------------------
    // set up reducedA with proper sizes
    //------------------------------------------------------------------

    reducedAMatSize  = new int[newNRows];
    reducedAStartRow = ProcNRows[mypid_] - 2 * ProcNConstr[mypid_];
    rowCount = reducedAStartRow;
    rowIndex = 0;

    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )  
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          ierr = HYPRE_ParCSRMatrixGetRow(RAP_csr,rowCount,&rowSize2,
                                          &colInd2, &colVal2);
          assert( !ierr );
          newRowSize = rowSize + rowSize2;
          newColInd = new int[newRowSize];
          for (j = 0; j < rowSize; j++)  newColInd[j] = colInd[j]; 
          for (j = 0; j < rowSize2; j++) newColInd[rowSize+j] = colInd2[j];
          qsort0(newColInd, 0, newRowSize-1);
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
          reducedAMatSize[rowIndex++] = ncnt;
         
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          ierr = HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowCount,&rowSize2,
                                              &colInd2,&colVal2);
          delete [] newColInd;
          assert( !ierr );
          rowCount++;
       }
    }

    //------------------------------------------------------------------
    // create a matrix context for reducedA
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(reducedA, reducedAMatSize);
    ierr += HYPRE_IJMatrixInitialize(reducedA);
    assert(!ierr);
    delete [] reducedAMatSize;

    //------------------------------------------------------------------
    // load the reducedA matrix 
    //------------------------------------------------------------------

    rowCount = reducedAStartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr, i, &rowSize, &colInd, &colVal);
          HYPRE_ParCSRMatrixGetRow(RAP_csr,rowCount,&rowSize2,&colInd2,
                                   &colVal2);
          newRowSize = rowSize + rowSize2;
          newColInd  = new int[newRowSize];
          newColVal  = new double[newRowSize];
          ncnt       = 0;
                  
          for ( j = 0; j < rowSize; j++ ) 
          {
             colIndex = colInd[j];
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows-(globalNConstr-ProcNConstr[numProcs_-1]);
             else
                ubound = ProcNRows[procIndex] - 
                         (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
             procIndex--;
             if ( colIndex < ubound ) 
             {
                searchIndex = HYPRE_LSI_Search(globalSelectedList,colIndex, 
                                               globalNSelected); 
                if ( searchIndex < 0 ) 
                {
                   searchIndex = - searchIndex - 1;
                   newColInd[ncnt] = colIndex - ProcNConstr[procIndex] - 
                                     searchIndex;
                   newColVal[ncnt++] = colVal[j]; 
                }
             }
          }
          for ( j = 0; j < rowSize2; j++ ) 
          {
             newColInd[ncnt+j] = colInd2[j]; 
             newColVal[ncnt+j] = - colVal2[j]; 
          }
          newRowSize = ncnt + rowSize2;
          qsort1(newColInd, newColVal, 0, newRowSize-1);
          ncnt = 0;
          for ( j = 0; j < newRowSize; j++ ) 
          {
             if ( j != ncnt && newColInd[j] == newColInd[ncnt] ) 
                newColVal[ncnt] += newColVal[j];
             else if ( newColInd[j] != newColInd[ncnt] ) 
             {
                ncnt++;
                newColVal[ncnt] = newColVal[j];
                newColInd[ncnt] = newColInd[j];
             }  
          } 
          newRowSize = ncnt + 1;
          ierr = HYPRE_IJMatrixSetValues(reducedA, 1, &newRowSize, 
			(const int *) &rowCount, (const int *) newColInd, 
			(const double *) newColVal);
          assert(!ierr);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowCount,&rowSize2,&colInd2,
                                       &colVal2);
          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
    }

    //------------------------------------------------------------------
    // assemble the reduced matrix
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(reducedA);
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReductionC - reducedAStartRow = %d\n", mypid_, 
               reducedAStartRow);

    HYPRE_IJMatrixGetObject(reducedA, (void **) &reducedA_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       MPI_Barrier(comm_);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             for ( i = reducedAStartRow; 
                   i < reducedAStartRow+nRows-2*nConstraints_; i++ )
             {
                printf("%d : reducedA ROW %d\n", mypid_, i);
                ierr = HYPRE_ParCSRMatrixGetRow(reducedA_csr,i,&rowSize,
                                                &colInd, &colVal);
                //qsort1(colInd, colVal, 0, rowSize-1);
                for ( j = 0; j < rowSize; j++ )
                   if ( colVal[j] != 0.0 )
                      printf("%4d %4d %20.13e\n", i, colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(reducedA_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(comm_);
          ncnt++;
       }
    }

    // *****************************************************************
    // form modified right hand side  (f1 = f1 - A12*invA22*f2)
    // *****************************************************************

    // *****************************************************************
    // form f2hat = invA22 * f2
    //------------------------------------------------------------------

    A21NRows       = 2 * nConstraints_;
    A21NCols       = nRows - 2 * nConstraints_;
    A21GlobalNRows = 2 * globalNConstr;
    A21GlobalNCols = globalNRows - 2 * globalNConstr;
    A21StartRow    = 2 * ProcNConstr[mypid_];

    HYPRE_IJVectorCreate(comm_, A21StartRow, A21StartRow+A21NRows-1, &f2);
    HYPRE_IJVectorSetObjectType(f2, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorAssemble(f2);
    assert(!ierr);

    HYPRE_IJVectorCreate(comm_, A21StartRow, A21StartRow+A21NRows-1, &f2hat);
    HYPRE_IJVectorSetObjectType(f2hat, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    assert(!ierr);

    colInd = new int[nSelected*2];
    colVal = new double[nSelected*2];

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux[j] == i ) 
          {
             colInd[i] = selectedList[j];
             break;
          }
       }
       if ( colInd[i] < 0 )
       {
          printf("%4d : SlideReductionC ERROR - out of range %d\n", mypid_,
                  colInd[i]);
          exit(1);
       }
    }
    for ( i = 0; i < nSelected; i++ ) 
    {
       colInd[nSelected+i] = EndRow - nConstraints_ + i + 1;
    }
    HYPRE_IJVectorGetValues(HYb_, 2*nSelected, colInd, colVal);
    for ( i = 0; i < nSelected*2; i++ ) colInd[i] = A21StartRow + i;
    ierr = HYPRE_IJVectorSetValues(f2,2*nSelected,(const int *) colInd,
			(const double *) colVal);
    assert( !ierr );
    HYPRE_IJVectorGetObject(f2, (void **) &f2_csr);
    HYPRE_IJVectorGetObject(f2hat, (void **) &f2hat_csr);
    HYPRE_IJMatrixGetObject(HYinvA22_, (void **) &invA22_csr);
    HYPRE_ParCSRMatrixMatvec( 1.0, invA22_csr, f2_csr, 0.0, f2hat_csr );
    delete [] colVal;
    delete [] colInd;
    HYPRE_IJVectorDestroy(f2); 

    // *****************************************************************
    // set up A12 with proper sizes before forming f2til = A12 * f2hat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of A12
    //------------------------------------------------------------------

    A12NRows       = A21NCols;
    A12NCols       = A21NRows;
    A12GlobalNRows = A21GlobalNCols;
    A12GlobalNCols = A21GlobalNRows;
    A12MatSize     = new int[A12NRows];
    A12StartRow    = ProcNRows[mypid_] - 2 * ProcNConstr[mypid_];
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
    {
       printf("%4d : SlideReductionC - A12GlobalDim = %d %d\n", mypid_, 
                        A12GlobalNRows, A12GlobalNCols);
       printf("%4d : SlideReductionC - A12LocalDim  = %d %d\n", mypid_, 
                        A12NRows, A12NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A12
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_, A21StartCol, A21StartCol+A12NRows-1,
                          A21StartRow, A21StartRow+A12NCols-1, &A12);
    ierr += HYPRE_IJMatrixSetObjectType(A12, HYPRE_PARCSR);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros in each row of A12
    // (which consists of the rows in selectedList and the constraints)
    //------------------------------------------------------------------

    rowCount = A12StartRow;
    rowIndex = 0;

    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )  
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          newRowSize = 0;
          for (j = 0; j < rowSize; j++)  
          {
             if ( colVal[j] != 0.0 )
             {
                colIndex = colInd[j];
                for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                if ( procIndex == numProcs_ ) 
                   ubound = globalNRows - 
                            (globalNConstr - ProcNConstr[numProcs_-1]);
                else
                   ubound = ProcNRows[procIndex] - 
                            (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
                procIndex--;
                if ( colIndex >= ubound ) newRowSize++; 
                else
                {
                   if (hypre_BinarySearch(globalSelectedList,colIndex, 
                                                    globalNSelected) >= 0)
                      newRowSize++;
                }
             }
          }
          A12MatSize[rowIndex++] = newRowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          rowCount++;
       }
    }
 
    //------------------------------------------------------------------
    // after fetching the row sizes, set up A12 with such sizes
    //------------------------------------------------------------------

    nnzA12 = 0;
    for ( i = 0; i < A12NRows; i++ ) nnzA12 += A12MatSize[i];
    ierr  = HYPRE_IJMatrixSetRowSizes(A12, A12MatSize);
    ierr += HYPRE_IJMatrixInitialize(A12);
    assert(!ierr);
    delete [] A12MatSize;

    //------------------------------------------------------------------
    // load the A12 matrix 
    //------------------------------------------------------------------

    rowCount = A12StartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList, i, nSelected); 
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr, i, &rowSize, &colInd, &colVal);
          newRowSize = 0;
          newColInd  = new int[rowSize];
          newColVal  = new double[rowSize];
          for (j = 0; j < rowSize; j++)  
          {
             colIndex = colInd[j];
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows-(globalNConstr-ProcNConstr[numProcs_-1]);
             else
                ubound = ProcNRows[procIndex] - 
                         (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
             procIndex--;
             if ( colIndex >= ubound ) { 
                if ( procIndex != numProcs_ - 1 ) 
                {
                   newColInd[newRowSize] = colInd[j] - ubound + 
                                           ProcNConstr[procIndex] +
                                           ProcNConstr[procIndex+1];
                }
                else 
                {
                   newColInd[newRowSize] = colInd[j] - ubound + 
                                           ProcNConstr[procIndex] +
                                           globalNConstr;
                }
                if ( newColInd[newRowSize] < 0 || 
                     newColInd[newRowSize] >= A12GlobalNCols )
                {
                   if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                   {
                      printf("%4d : SlideReductionC WARNING - A12 col index ",
                             mypid_);
                      printf("out of range %d %d(%d)\n", i, 
                              newColInd[newRowSize], A12GlobalNCols);
                   }
                }
                newColVal[newRowSize++] = colVal[j];
             } else
             {
                searchIndex = HYPRE_LSI_Search(globalSelectedList,colInd[j],
                                               globalNSelected);
                if ( searchIndex >= 0) 
                {
                   searchIndex = globalSelectedListAux[searchIndex];
                   newColInd[newRowSize] = searchIndex + 
                                           ProcNConstr[procIndex]; 
                   if ( newColInd[newRowSize] < 0 || 
                        newColInd[newRowSize] >= A12GlobalNCols )
                   {
                      if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                      {
                         printf("%4d : SlideReductionC WARNING - \n",mypid_);
                         printf("      A12(%d,%d,%d))\n", i, 
                                newColInd[newRowSize], A12GlobalNCols);
                      }
                   }
                   newColVal[newRowSize++] = colVal[j];
                }
             }
          }
          ierr = HYPRE_IJMatrixSetValues(A12, 1, &newRowSize, 
			(const int *) &rowCount, (const int *) newColInd, 
			(const double *) newColVal);
          assert(!ierr);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);

          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
    }
    MPI_Allreduce(&nnzA12,&ncnt,1,MPI_INT,MPI_SUM,comm_);
    if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 ))
       printf("       SlideReductionC : NNZ of A12 = %d\n", ncnt);

    //------------------------------------------------------------------
    // assemble the A12 matrix 
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixAssemble(A12);
    assert( !ierr );
    HYPRE_IJMatrixGetObject(A12, (void **) &A12_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       MPI_Barrier(comm_);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             for (i=A12StartRow;i<A12StartRow+A12NRows;i++)
             {
                printf("%d : A12 ROW %d\n", mypid_, i);
                HYPRE_ParCSRMatrixGetRow(A12_csr,i,&rowSize,&colInd,&colVal);
                //qsort1(colInd, colVal, 0, rowSize-1);
                for ( j = 0; j < rowSize; j++ )
                   if ( colVal[j] != 0.0 )
                      printf(" A12 %d %d %20.13e\n", i, colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(A12_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(comm_);
          ncnt++;
       }
    }

    //------------------------------------------------------------------
    // form reducedB_ = A12 * f2hat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJVectorCreate(comm_, reducedAStartRow, 
			reducedAStartRow+newNRows-1, &reducedB_);
    ierr += HYPRE_IJVectorSetObjectType(reducedB_, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(reducedB_);
    ierr += HYPRE_IJVectorAssemble(reducedB_);
    assert( !ierr );

    HYPRE_IJVectorGetObject(reducedB_, (void **) &reducedB_csr);
    HYPRE_ParCSRMatrixMatvec( -1.0, A12_csr, f2hat_csr, 0.0, reducedB_csr );
    HYPRE_IJMatrixDestroy(A12); 
    HYPRE_IJVectorDestroy(f2hat); 

    //------------------------------------------------------------------
    // finally form reducedB = f1 - f2til
    //------------------------------------------------------------------

    rowCount = reducedAStartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       if ( hypre_BinarySearch(selectedList, i, nSelected) < 0 ) 
       {
          HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
          HYPRE_IJVectorAddToValues(reducedB_, 1, (const int *) &rowCount, 
			(const double *) &ddata);
          rowCount++;
       }
    }

    //******************************************************************
    // set up the system with the new matrix
    //------------------------------------------------------------------

    reducedA_ = reducedA;
    ierr = HYPRE_IJVectorCreate(comm_, reducedAStartRow,
			reducedAStartRow+newNRows-1, &reducedX_);
    ierr = HYPRE_IJVectorSetObjectType(reducedX_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(reducedX_);
    ierr = HYPRE_IJVectorAssemble(reducedX_);
    assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm_, reducedAStartRow,
			reducedAStartRow+newNRows-1, &reducedR_);
    ierr = HYPRE_IJVectorSetObjectType(reducedR_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(reducedR_);
    ierr = HYPRE_IJVectorAssemble(reducedR_);
    assert(!ierr);
}

//*****************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//
// Additional assumptions are :
//
//    - a given slave equation and the corresponding constraint equation
//      reside in the same processor
//    - constraint equations are given at the end of the local matrix
//      (hence given by EndRow_-nConstr to EndRow_)
//    - each processor gets a contiguous block of equations, and processor
//      i+1 has equation numbers higher than those of processor i
//-----------------------------------------------------------------------------
// This version replaces the selected slave equation with an identity row
//-----------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSlideReducedSystem2()
{
    int    i, j, nRows, globalNRows, colIndex;
    int    globalNConstr, globalNSelected, *globalSelectedList;
    int    *globalSelectedListAux;
    int    nSelected, *tempList, reducedAStartRow;
    int    searchIndex, procIndex, A21StartRow, A12StartRow, A12NRows;
    int    rowSize, *colInd, A21NRows, A21GlobalNRows;
    int    A21NCols, A21GlobalNCols, rowCount, maxRowSize, newEndRow;
    int    A12NCols, A12GlobalNCols;
    int    *A21MatSize, rowIndex, *A12MatSize, A12GlobalNRows;
    int    *newColInd, diagCount, newRowSize, ierr;
    int    invA22NRows, invA22GlobalNRows, invA22NCols, invA22GlobalNCols;
    int    *invA22MatSize, newNRows, newColIndex;
    int    *colInd2, ncnt, ubound, one=1;
    int    rowSize2, *recvCntArray, *displArray, ncnt2, isAConstr;
    int    StartRow, EndRow, *reducedAMatSize;
    int    *ProcNRows, *ProcNConstr, nnzA21, nnzA12;
    int    A21StartCol;
    double *colVal, *colVal2, *newColVal, *diagonal;
    double *extDiagonal, ddata;
    HYPRE_IJMatrix     A12, A21, invA22, reducedA;
    HYPRE_ParCSRMatrix A_csr, A12_csr, A21_csr, invA22_csr, RAP_csr;
    HYPRE_ParCSRMatrix reducedA_csr;
    HYPRE_IJVector     f2, f2hat;
    HYPRE_ParVector    f2_csr, f2hat_csr, reducedB_csr;

    //******************************************************************
    // fetch local matrix information
    //------------------------------------------------------------------

    if ( mypid_ == 0 && (HYOutputLevel_ & HYFEI_SLIDEREDUCE1) )
       printf("%4d : SlideReduction2 begins....\n",mypid_);
    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    nRows    = localEndRow_ - localStartRow_ + 1;
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction2 : StartRow/EndRow = %d %d\n",mypid_,
                                       StartRow,EndRow);

    //******************************************************************
    // construct local and global information about where the constraints
    // are (this is given by user or searched within this code)
    //------------------------------------------------------------------

    HYPRE_IJMatrixGetObject(HYA_, (void **) &A_csr);

    //------------------------------------------------------------------
    // search the entire local matrix to find where the constraint
    // equations are, if not already given
    // (The constraint equations are assumed to be at the end of the
    //  matrix) ==> nConstraints, globalNConstr
    //------------------------------------------------------------------
    
    MPI_Allreduce(&nConstraints_,&globalNConstr,1,MPI_INT,MPI_SUM,comm_);
    if ( globalNConstr == 0 )
    {
       for ( i = EndRow; i >= StartRow; i-- ) 
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          isAConstr = 1;
          for (j = 0; j < rowSize; j++) 
             if (colInd[j] == i && colVal[j] != 0.0) {isAConstr = 0; break;}
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          if ( isAConstr ) nConstraints_++;
          else             break;
       }
    }
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction2 : no. constr = %d\n", mypid_,
              nConstraints_);

    MPI_Allreduce(&nConstraints_, &globalNConstr, 1, MPI_INT, MPI_SUM, comm_);
    if ( globalNConstr == 0 ) return;

    //------------------------------------------------------------------
    // get information about nRows from all processors, and then
    // compute the base NRows on each processor
    // (This is needed later on for column index conversion)
    // ==> ProcNRows, globalNRows
    //------------------------------------------------------------------
 
    ProcNRows   = new int[numProcs_];
    tempList    = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nRows;
    MPI_Allreduce(tempList, ProcNRows, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    globalNRows = 0;
    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ ) 
    {
       globalNRows   += ProcNRows[i];
       ncnt2          = ProcNRows[i];
       ProcNRows[i]   = ncnt;
       ncnt          += ncnt2;
    }
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction2 : localNRows = %d\n", mypid_, nRows);

    //------------------------------------------------------------------
    // compose a global array marking where the constraint equations are,
    // then compute the base nConstraints on each processor
    // (This is needed later on for column index conversion)
    // ==> ProcNConstr, globalNConstr
    //------------------------------------------------------------------
    
    globalNConstr = 0;
    tempList    = new int[numProcs_];
    ProcNConstr = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nConstraints_;
    MPI_Allreduce(tempList,ProcNConstr,numProcs_,MPI_INT,MPI_SUM,comm_);
    delete [] tempList;
    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ ) 
    {
       globalNConstr += ProcNConstr[i];
       ncnt2          = ProcNConstr[i];
       ProcNConstr[i] = ncnt;
       ncnt          += ncnt2;
    }
#ifdef HAVE_MLI
    if ( HYPreconID_ == HYMLI )
       HYPRE_LSI_MLIAdjustNodeEqnMap(HYPrecon_, ProcNRows, ProcNConstr);
#endif
   
    //******************************************************************
    // compose the local and global selected node lists
    //------------------------------------------------------------------

    if ( selectedList_    != NULL ) delete [] selectedList_;
    if ( selectedListAux_ != NULL ) delete [] selectedListAux_;
    nSelected = nConstraints_;
    if ( nConstraints_ > 0 ) 
    {
       selectedList_ = new int[nConstraints_];
       selectedListAux_ = new int[nConstraints_];
    }
    else selectedList_ = selectedListAux_ = NULL;
    globalNSelected = globalNConstr;
    if (globalNSelected > 0) 
    {
       globalSelectedList = new int[globalNSelected];
       globalSelectedListAux = new int[globalNSelected];
    }
    else globalSelectedList = globalSelectedListAux = NULL;
   
    buildSlideReducedSystemPartA(ProcNRows,ProcNConstr,globalNRows, 
                                 globalNSelected,globalSelectedList, 
                                 globalSelectedListAux);

    //******************************************************************
    // construct A21
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of A21
    //------------------------------------------------------------------

    A21NRows       = 2 * nConstraints_;
    A21NCols       = nRows - nConstraints_;
    A21GlobalNRows = 2 * globalNConstr;
    A21GlobalNCols = globalNRows - globalNConstr;
    A21StartRow    = 2 * ProcNConstr[mypid_];
    A21StartCol    = ProcNRows[mypid_] - ProcNConstr[mypid_];

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
    {
       printf("%4d : SlideReduction2 : A21StartRow  = %d\n", mypid_,
                                       A21StartRow);
       printf("%4d : SlideReduction2 : A21GlobalDim = %d %d\n", mypid_, 
                                       A21GlobalNRows, A21GlobalNCols);
       printf("%4d : SlideReduction2 : A21LocalDim  = %d %d\n",mypid_,
                                       A21NRows, A21NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A21
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_, A21StartRow, A21StartRow+A21NRows-1,
				 A21StartCol, A21StartCol+A21NCols-1, &A21);
    ierr += HYPRE_IJMatrixSetObjectType(A21, HYPRE_PARCSR);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros in the first nConstraint row of A21
    // (which consists of the rows in selectedList), the nnz will
    // be reduced by excluding the constraint and selected slave columns
    //------------------------------------------------------------------

    rowCount   = 0;
    maxRowSize = 0;
    newEndRow  = EndRow - nConstraints_;
    A21MatSize = new int[A21NRows];

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux_[j] == i ) 
          {
             rowIndex = selectedList_[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);

       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 ) 
          {
             if (colIndex <= newEndRow || colIndex >= localEndRow_)
             {
                if (colIndex >= StartRow && colIndex <= newEndRow )
                   searchIndex = hypre_BinarySearch(selectedList_,colIndex, 
                                                    nSelected);
                else 
                   searchIndex = hypre_BinarySearch(globalSelectedList,
                                       colIndex, globalNSelected);
                if (searchIndex < 0 ) newRowSize++;
             }
          }
       }
       A21MatSize[rowCount] = newRowSize;
       maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowCount++;
    }

    //------------------------------------------------------------------
    // compute the number of nonzeros in the second nConstraint row of A21
    // (which consists of the rows in constraint equations), the nnz will
    // be reduced by excluding the selected slave columns only (since the
    // entries corresponding to the constraint columns are 0, and since
    // the selected matrix is a diagonal matrix, there is no need to 
    // search for slave equations in the off-processor list)
    //------------------------------------------------------------------

    rowCount = nSelected;
    for ( i = EndRow-nConstraints_+1; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
             if (colIndex <= newEndRow || colIndex >= localEndRow_)
             {
                if (colIndex >= StartRow && colIndex <= newEndRow )
                   searchIndex = hypre_BinarySearch(selectedList_,colIndex, 
                                                    nSelected);
                else 
                   searchIndex = hypre_BinarySearch(globalSelectedList,
                                           colIndex, globalNSelected);
                if ( searchIndex < 0 ) newRowSize++;
             }
          }
       }
       A21MatSize[rowCount] = newRowSize;
       maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowCount++;
    }
    nnzA21 = 0;
    for ( i = 0; i < 2*nConstraints_; i++ ) nnzA21 += A21MatSize[i];

    //------------------------------------------------------------------
    // after fetching the row sizes, set up A21 with such sizes
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(A21, A21MatSize);
    ierr += HYPRE_IJMatrixInitialize(A21);
    assert(!ierr);
    delete [] A21MatSize;

    //------------------------------------------------------------------
    // next load the first nConstraint row to A21 extracted from A
    // (at the same time, the D block is saved for future use)
    //------------------------------------------------------------------

    rowCount  = A21StartRow;
    if ( nConstraints_ > 0 ) diagonal = new double[nConstraints_];
    else                     diagonal = NULL;
    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];

    diagCount = 0;
    for ( i = 0; i < nSelected; i++ )
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux_[j] == i ) 
          {
             rowIndex = selectedList_[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];

             if (colIndex <= newEndRow || colIndex >= localEndRow_) 
             {
                searchIndex = hypre_BinarySearch(globalSelectedList,
                                       colIndex, globalNSelected);
                if ( searchIndex < 0 ) 
                {
                   for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                      if ( ProcNRows[procIndex] > colIndex ) break;
                   procIndex--;
                   newColIndex = colInd[j] - ProcNConstr[procIndex];
                   newColInd[newRowSize]   = newColIndex;
                   newColVal[newRowSize++] = colVal[j];
                   if ( newColIndex < 0 || newColIndex >= A21GlobalNCols )
                   {
                      if (HYOutputLevel_ & HYFEI_SLIDEREDUCE1)
                      {
                         printf("%4d : SlideReduction2 WARNING - ",mypid_);
                         printf(" A21(%d,%d(%d))\n", rowCount, 
                                colIndex, A21GlobalNCols);
                      } 
                   } 
                   if ( newRowSize > maxRowSize+1 ) 
                   {
                      if (HYOutputLevel_ & HYFEI_SLIDEREDUCE1)
                      {
                         printf("%4d : SlideReduction2 : WARNING - ",mypid_);
                         printf("cross array boundary(1).\n");
                      }
                   }
                }
             }

             //---------------------------------------------------------
             // slave equations should only have one nonzeros 
             // corresponding to the D in A22
             //---------------------------------------------------------

             else if ( colIndex > newEndRow && colIndex <= EndRow ) 
             {
                if ( colVal[j] != 0.0 ) diagonal[diagCount++] = colVal[j];
                if ( habs(colVal[j]) < 1.0E-8 )
                {
                   if (HYOutputLevel_ & HYFEI_SLIDEREDUCE1)
                   {
                      printf("%4d : SlideReduction2 WARNING : large ",mypid_);
                      printf("entry in invA22\n");
                   }
                }
             }
          } 
       }

       HYPRE_IJMatrixSetValues(A21, 1, &newRowSize, (const int *) &rowCount,
		(const int *) newColInd, (const double *) newColVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       if ( diagCount != (i+1) )
       {
          printf("%4d : SlideReduction2 ERROR (3) : %d %d.\n", mypid_,
                  diagCount,i+1);
          exit(1);
       }
       rowCount++;
    }

    //------------------------------------------------------------------
    // send the diagonal to each processor that needs them
    //------------------------------------------------------------------

    recvCntArray = new int[numProcs_];
    displArray   = new int[numProcs_];
    MPI_Allgather(&diagCount, 1, MPI_INT, recvCntArray, 1, MPI_INT, comm_);
    displArray[0] = 0;
    for ( i = 1; i < numProcs_; i++ ) 
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    ncnt = displArray[numProcs_-1] + recvCntArray[numProcs_-1];
    if ( ncnt > 0 ) extDiagonal = new double[ncnt];
    else            extDiagonal = NULL;
    MPI_Allgatherv(diagonal, diagCount, MPI_DOUBLE, extDiagonal,
                   recvCntArray, displArray, MPI_DOUBLE, comm_);
    diagCount = ncnt;
    delete [] recvCntArray;
    delete [] displArray;
    if ( diagonal != NULL ) delete [] diagonal;

    //------------------------------------------------------------------
    // next load the second nConstraint rows to A21 extracted from A
    // (assume the constraint-constraint block is 0 )
    //------------------------------------------------------------------

    for ( i = EndRow-nConstraints_+1; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 )
          {
             if (colIndex <= newEndRow || colIndex >= localEndRow_)
             {
                if (colIndex >= StartRow && colIndex <= newEndRow )
                   searchIndex = hypre_BinarySearch(selectedList_,colIndex, 
                                                    nSelected);
                else 
                   searchIndex = hypre_BinarySearch(globalSelectedList,
                                           colIndex, globalNSelected);

                if ( searchIndex < 0 ) 
                {
                   for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                      if ( ProcNRows[procIndex] > colIndex ) break;
                   procIndex--;
                   newColIndex = colInd[j] - ProcNConstr[procIndex];
                   newColInd[newRowSize]   = newColIndex;
                   newColVal[newRowSize++] = colVal[j];
                   if ( newColIndex < 0 || newColIndex >= A21GlobalNCols )
                   {
                      if (HYOutputLevel_ & HYFEI_SLIDEREDUCE1)
                      {
                         printf("%4d : SlideReduction2 WARNING - ",mypid_);
                         printf(" A21(%d,%d(%d))\n", rowCount, 
                                colIndex, A21GlobalNCols);
                      } 
                   } 
                   if ( newRowSize > maxRowSize+1 ) 
                   {
                      if (HYOutputLevel_ & HYFEI_SLIDEREDUCE1)
                      {
                         printf("%4d : SlideReductionWARNING : ",mypid_);
                         printf("crossing array boundary(2).\n");
                      }
                   }
                }
             }
          } 
       }
       if ( newRowSize == 0 && (HYOutputLevel_ & HYFEI_SLIDEREDUCE1))
          printf("%4d : SlideReduction2 WARNING : loading all 0 to A21\n",
                 mypid_);
       HYPRE_IJMatrixSetValues(A21, 1, &newRowSize, (const int *) &rowCount,
		(const int *) newColInd, (const double *) newColVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowCount++;
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix and sanitize
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(A21);

    HYPRE_IJMatrixGetObject(A21, (void **) &A21_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A21_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       ncnt = 0;
       MPI_Barrier(comm_);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d : SlideReduction2 : matrix A21 assembled %d.\n",
                                        mypid_,A21StartRow);
             fflush(stdout);
             for ( i = A21StartRow; i < A21StartRow+2*nConstraints_; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(A21_csr,i,&rowSize,&colInd,&colVal);
                printf("A21 ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(A21_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(comm_);
       }
    }

    //******************************************************************
    // construct invA22
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of invA22
    //------------------------------------------------------------------

    invA22NRows       = A21NRows;
    invA22NCols       = invA22NRows;
    invA22GlobalNRows = A21GlobalNRows;
    invA22GlobalNCols = invA22GlobalNRows;
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
    {
       printf("%4d : SlideReduction2 - A22GlobalDim = %d %d\n", mypid_, 
                        invA22GlobalNRows, invA22GlobalNCols);
       printf("%4d : SlideReduction2 - A22LocalDim  = %d %d\n", mypid_, 
                        invA22NRows, invA22NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A22
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixCreate(comm_, A21StartRow, A21StartRow+invA22NRows-1,
                           A21StartRow, A21StartRow+invA22NCols-1, &invA22);
    ierr += HYPRE_IJMatrixSetObjectType(invA22, HYPRE_PARCSR);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the no. of nonzeros in the first nConstraint row of invA22
    //------------------------------------------------------------------

    maxRowSize  = 0;
    invA22MatSize = new int[invA22NRows];
    for ( i = 0; i < nConstraints_; i++ ) invA22MatSize[i] = 1;

    //------------------------------------------------------------------
    // compute the number of nonzeros in the second nConstraints row of 
    // invA22 (consisting of [D and A22 block])
    //------------------------------------------------------------------

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux_[j] == i ) 
          {
             rowIndex = selectedList_[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       rowSize2 = 1;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 ) 
          {
             if ( colIndex >= StartRow && colIndex <= newEndRow ) 
             {
	        searchIndex = hypre_BinarySearch(selectedList_, colIndex, 
                                                 nSelected); 
                if ( searchIndex >= 0 ) rowSize2++;
             } 
             else if ( colIndex < StartRow || colIndex > EndRow ) 
             {
	        searchIndex = hypre_BinarySearch(globalSelectedList,colIndex, 
                                                 globalNSelected); 
                if ( searchIndex >= 0 ) rowSize2++;
             }
          }
       }
       invA22MatSize[nConstraints_+i] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }

    //------------------------------------------------------------------
    // after fetching the row sizes, set up invA22 with such sizes
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(invA22, invA22MatSize);
    ierr += HYPRE_IJMatrixInitialize(invA22);
    assert(!ierr);
    delete [] invA22MatSize;

    //------------------------------------------------------------------
    // next load the first nConstraints_ row to invA22 extracted from A
    // (that is, the D block)
    //------------------------------------------------------------------

    maxRowSize++;
    newColInd = new int[maxRowSize];
    newColVal = new double[maxRowSize];

    for ( i = 0; i < diagCount; i++ ) extDiagonal[i] = 1.0 / extDiagonal[i];
    for ( i = 0; i < nConstraints_; i++ ) 
    {
       newColInd[0] = A21StartRow + nConstraints_ + i; 
       rowIndex     = A21StartRow + i;
       if ( newColInd[0] < 0 || newColInd[0] >= invA22GlobalNCols )
       {
          if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
             printf("%4d : SlideReduction2 WARNING - A22(%d,%d(%d))\n", 
                    mypid_, rowIndex, newColInd[0], invA22GlobalNCols);
       } 
       newColVal[0] = extDiagonal[A21StartRow/2+i];
       ierr = HYPRE_IJMatrixSetValues(invA22, 1, &one, (const int *) &rowIndex,
		(const int *) newColInd, (const double *) newColVal);
       assert(!ierr);
    }

    //------------------------------------------------------------------
    // next load the second nConstraints_ rows to A22 extracted from A
    //------------------------------------------------------------------

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux_[j] == i ) 
          {
             rowIndex = selectedList_[j]; 
             break;
          }
       }
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 1;
       newColInd[0] = A21StartRow + i;
       newColVal[0] = extDiagonal[A21StartRow/2+i]; 
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
          if ( colVal[j] != 0.0 )
          {
             searchIndex = hypre_BinarySearch(globalSelectedList,
                                              colIndex,globalNSelected);
             if ( searchIndex >= 0 ) 
             {
                searchIndex = globalSelectedListAux[searchIndex];
                for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                if ( procIndex == numProcs_ )
                   newColInd[newRowSize] = searchIndex + globalNConstr; 
                else
                   newColInd[newRowSize] = searchIndex+ProcNConstr[procIndex]; 
                if ( newColInd[newRowSize] < 0 || 
                     newColInd[newRowSize] >= invA22GlobalNCols )
                {
                   if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                      printf("%4d : SlideReduction2 WARNING - A22(%d,%d,%d)\n",
                             mypid_, rowCount, newColInd[newRowSize], 
                             invA22GlobalNCols);
                } 
                newColVal[newRowSize++] = - extDiagonal[A21StartRow/2+i] * 
                                        colVal[j] * extDiagonal[searchIndex];
                if ( newRowSize > maxRowSize )
                {
                   if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                   {
                      printf("%4d : SlideReduction2 WARNING - ",mypid_);
                      printf("passing array boundary(3).\n");
                   }
                }
      	     } 
	  } 
       }
       rowCount = A21StartRow + nConstraints_ + i;
       ierr = HYPRE_IJMatrixSetValues(invA22, 1, &newRowSize, 
		(const int *) &rowCount, (const int *) newColInd, 
		(const double *) newColVal);
       assert(!ierr);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }
    delete [] newColInd;
    delete [] newColVal;
    delete [] extDiagonal;

    //------------------------------------------------------------------
    // finally assemble the matrix and sanitize
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(invA22);
    HYPRE_IJMatrixGetObject(invA22, (void **) &invA22_csr);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) invA22_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       ncnt = 0;
       MPI_Barrier(comm_);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d : SlideReduction - invA22 \n", mypid_);
             for ( i = A21StartRow; i < A21StartRow+2*nConstraints_; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(invA22_csr,i,&rowSize,&colInd,
                                         &colVal);
                printf("invA22 ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(invA22_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(comm_);
          ncnt++;
       }
    }

    //******************************************************************
    // perform the triple matrix product
    //------------------------------------------------------------------

    HYPRE_IJMatrixGetObject(A21, (void **) &A21_csr);
    HYPRE_IJMatrixGetObject(invA22, (void **) &invA22_csr);
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction2 - Triple matrix product starts\n",mypid_);

    hypre_BoomerAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix *) invA22_csr,
                                     (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix **) &RAP_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction2 - Triple matrix product ends\n", mypid_);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       MPI_Barrier(comm_);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             for ( i = A21StartRow; i < A21StartRow+A21NCols; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(RAP_csr,i,&rowSize,&colInd, &colVal);
                printf("RAP ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(RAP_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
          }
          MPI_Barrier(comm_);
          ncnt++;
       }
    }

    //******************************************************************
    // finally formed the Schur complement reduced system by
    // extracting the A11 part of A and subtracting the RAP
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // first calculate the dimension of the reduced matrix
    //------------------------------------------------------------------

    newNRows = nRows - nConstraints_;
    ierr  = HYPRE_IJMatrixCreate(comm_, A21StartCol, A21StartCol+newNRows-1,
                           A21StartCol, A21StartCol+newNRows-1, &reducedA);
    ierr += HYPRE_IJMatrixSetObjectType(reducedA, HYPRE_PARCSR);
    assert(!ierr);

    //------------------------------------------------------------------
    // set up reducedA with proper sizes
    //------------------------------------------------------------------

    reducedAMatSize  = new int[newNRows];
    reducedAStartRow = ProcNRows[mypid_] - ProcNConstr[mypid_];
    rowCount = reducedAStartRow;
    rowIndex = 0;

    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList_, i, nSelected); 
       if ( searchIndex < 0 )  
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          ierr = HYPRE_ParCSRMatrixGetRow(RAP_csr,rowCount,&rowSize2,
                                          &colInd2, &colVal2);
          assert( !ierr );
          newRowSize = rowSize + rowSize2;
          newColInd = new int[newRowSize];
          for (j = 0; j < rowSize; j++)  newColInd[j] = colInd[j]; 
          for (j = 0; j < rowSize2; j++) newColInd[rowSize+j] = colInd2[j];
          qsort0(newColInd, 0, newRowSize-1);
          ncnt = 0;
          for ( j = 0; j < newRowSize; j++ ) 
          {
             if ( newColInd[j] != newColInd[ncnt] ) 
             {
                ncnt++;
                newColInd[ncnt] = newColInd[j];
             }  
          }
          reducedAMatSize[rowIndex++] = ncnt;
         
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          ierr = HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowCount,&rowSize2,
                                              &colInd2,&colVal2);
          delete [] newColInd;
          assert( !ierr );
          rowCount++;
       }
       else
       {
          reducedAMatSize[rowIndex++] = 1;
          rowCount++;
       }
    }

    //------------------------------------------------------------------
    // create a matrix context for reducedA
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixSetRowSizes(reducedA, reducedAMatSize);
    ierr += HYPRE_IJMatrixInitialize(reducedA);
    assert(!ierr);
    delete [] reducedAMatSize;

    //------------------------------------------------------------------
    // load the reducedA matrix 
    //------------------------------------------------------------------

    rowCount = reducedAStartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList_, i, nSelected); 
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr, i, &rowSize, &colInd, &colVal);
          HYPRE_ParCSRMatrixGetRow(RAP_csr,rowCount,&rowSize2,&colInd2,
                                   &colVal2);
          newRowSize = rowSize + rowSize2;
          newColInd  = new int[newRowSize];
          newColVal  = new double[newRowSize];
          ncnt       = 0;
                  
          for ( j = 0; j < rowSize; j++ ) 
          {
             colIndex = colInd[j];
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows-(globalNConstr-ProcNConstr[numProcs_-1]);
             else
                ubound = ProcNRows[procIndex] - 
                         (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
             procIndex--;
             if ( colIndex < ubound ) 
             {
                if ( colIndex >= StartRow && colIndex <= EndRow )
                   searchIndex = HYPRE_LSI_Search(selectedList_,colIndex, 
                                                  nSelected); 
                else
                   searchIndex = HYPRE_LSI_Search(globalSelectedList,colIndex, 
                                                  globalNSelected); 

                if ( searchIndex < 0 ) 
                {
                   newColInd[ncnt] = colIndex - ProcNConstr[procIndex];
                   newColVal[ncnt++] = colVal[j]; 
                }
             }
          }
          for ( j = 0; j < rowSize2; j++ ) 
          {
             newColInd[ncnt+j] = colInd2[j]; 
             newColVal[ncnt+j] = - colVal2[j]; 
          }
          newRowSize = ncnt + rowSize2;
          qsort1(newColInd, newColVal, 0, newRowSize-1);
          ncnt = 0;
          for ( j = 0; j < newRowSize; j++ ) 
          {
             if ( j != ncnt && newColInd[j] == newColInd[ncnt] ) 
                newColVal[ncnt] += newColVal[j];
             else if ( newColInd[j] != newColInd[ncnt] ) 
             {
                ncnt++;
                newColVal[ncnt] = newColVal[j];
                newColInd[ncnt] = newColInd[j];
             }  
          } 
          newRowSize = ncnt + 1;
          ierr = HYPRE_IJMatrixSetValues(reducedA, 1, &newRowSize, 
			(const int *) &rowCount, (const int *) newColInd, 
			(const double *) newColVal);
          assert(!ierr);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_ParCSRMatrixRestoreRow(RAP_csr,rowCount,&rowSize2,&colInd2,
                                       &colVal2);
          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
       else
       {
          newRowSize = 1;
          newColInd  = new int[newRowSize];
          newColVal  = new double[newRowSize];
          newColInd[0] = rowCount;
          newColVal[0] = 1.0;
          ierr = HYPRE_IJMatrixSetValues(reducedA, 1, &newRowSize, 
			(const int *) &rowCount, (const int *) newColInd, 
			(const double *) newColVal);
          assert(!ierr);
          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
    }

    //------------------------------------------------------------------
    // assemble the reduced matrix
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(reducedA);
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction2 - reducedA - StartRow = %d\n", 
                                       mypid_, reducedAStartRow);

    HYPRE_IJMatrixGetObject(reducedA, (void **) &reducedA_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       MPI_Barrier(comm_);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             for ( i = reducedAStartRow; 
                   i < reducedAStartRow+nRows-2*nConstraints_; i++ )
             {
                printf("%d : reducedA ROW %d\n", mypid_, i);
                ierr = HYPRE_ParCSRMatrixGetRow(reducedA_csr,i,&rowSize,
                                                &colInd,&colVal);
                //qsort1(colInd, colVal, 0, rowSize-1);
                for ( j = 0; j < rowSize; j++ )
                   if ( colVal[j] != 0.0 )
                      printf("%4d %4d %20.13e\n", i+1, colInd[j]+1, colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(reducedA_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(comm_);
          ncnt++;
       }
    }

    // *****************************************************************
    // form modified right hand side  (f1 = f1 - A12*invA22*f2)
    // *****************************************************************

    // *****************************************************************
    // form f2hat = invA22 * f2
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, A21StartRow, A21StartRow+A21NRows-1, &f2);
    HYPRE_IJVectorSetObjectType(f2, HYPRE_PARCSR);
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction2 - A21 dims = %d %d %d\n", mypid_, 
               A21StartRow, A21NRows, A21GlobalNRows);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorAssemble(f2);
    assert(!ierr);

    HYPRE_IJVectorCreate(comm_, A21StartRow, A21StartRow+A21NRows-1, &f2hat);
    HYPRE_IJVectorSetObjectType(f2hat, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    assert(!ierr);

    colInd = new int[nSelected*2];
    colVal = new double[nSelected*2];

    for ( i = 0; i < nSelected; i++ ) 
    {
       for ( j = 0; j < nSelected; j++ ) 
       {
          if ( selectedListAux_[j] == i ) 
          {
             colInd[i] = selectedList_[j];
             break;
          }
       }
       if ( colInd[i] < 0 )
       {
          printf("%4d : SlideReduction2 ERROR - out of range %d\n", mypid_,
                  colInd[i]);
          exit(1);
       }
    }
    for ( i = 0; i < nSelected; i++ ) 
    {
       colInd[nSelected+i] = EndRow - nConstraints_ + i + 1;
    }
    HYPRE_IJVectorGetValues(HYb_, 2*nSelected, colInd, colVal);
    for ( i = 0; i < nSelected*2; i++ ) colInd[i] = A21StartRow + i;
    ierr = HYPRE_IJVectorSetValues(f2, 2*nSelected, (const int *) colInd, 
			(const double *) colVal);
    assert( !ierr );
    HYPRE_IJVectorGetObject(f2, (void **) &f2_csr);
    HYPRE_IJVectorGetObject(f2hat, (void **) &f2hat_csr);
    HYPRE_ParCSRMatrixMatvec( 1.0, invA22_csr, f2_csr, 0.0, f2hat_csr );
    delete [] colVal;
    delete [] colInd;
    HYPRE_IJVectorDestroy(f2); 

    // *****************************************************************
    // set up A12 with proper sizes before forming f2til = A12 * f2hat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of A12
    //------------------------------------------------------------------

    A12NRows       = A21NCols;
    A12NCols       = A21NRows;
    A12GlobalNRows = A21GlobalNCols;
    A12GlobalNCols = A21GlobalNRows;
    A12MatSize     = new int[A12NRows];
    A12StartRow    = ProcNRows[mypid_] - ProcNConstr[mypid_];
    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
    {
       printf("%4d : SlideReduction2 - A12GlobalDim = %d %d\n", mypid_, 
                        A12GlobalNRows, A12GlobalNCols);
       printf("%4d : SlideReduction2 - A12LocalDim  = %d %d\n", mypid_, 
                        A12NRows, A12NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A12
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_, A21StartCol, A21StartCol+A12NRows-1,
				 A21StartRow, A21StartRow+A12NCols-1, &A12);
    ierr += HYPRE_IJMatrixSetObjectType(A12, HYPRE_PARCSR);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros in each row of A12
    // (which consists of the rows in selectedList and the constraints)
    //------------------------------------------------------------------

    rowCount = A12StartRow;
    rowIndex = 0;
    nnzA12 = 0;

    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList_, i, nSelected); 
       if ( searchIndex < 0 )  
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          newRowSize = 0;
          for (j = 0; j < rowSize; j++)  
          {
             if ( colVal[j] != 0.0 )
             {
                colIndex = colInd[j];
                for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                if ( procIndex == numProcs_ ) 
                   ubound = globalNRows -
                            (globalNConstr-ProcNConstr[numProcs_-1]);
                else
                   ubound = ProcNRows[procIndex] - 
                            (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
                procIndex--;
                if ( colIndex >= ubound ) newRowSize++; 
                else if (colIndex >= StartRow && colIndex <= EndRow)
                {
                   if (hypre_BinarySearch(selectedList_,colIndex,nSelected)>=0)
                      newRowSize++;
                }
                else
                {
                   if (hypre_BinarySearch(globalSelectedList,colIndex, 
                                                    globalNSelected) >= 0)
                      newRowSize++;
                }
             }
          }
          A12MatSize[rowIndex++] = newRowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          rowCount++;
       }
       else
       {
          A12MatSize[rowIndex++] = 1;
          rowCount++;
          nnzA12--;
       }
    }
 
    //------------------------------------------------------------------
    // after fetching the row sizes, set up A12 with such sizes
    //------------------------------------------------------------------

    for ( i = 0; i < A12NRows; i++ ) nnzA12 += A12MatSize[i];
    ierr  = HYPRE_IJMatrixSetRowSizes(A12, A12MatSize);
    ierr += HYPRE_IJMatrixInitialize(A12);
    assert(!ierr);
    delete [] A12MatSize;

    //------------------------------------------------------------------
    // load the A12 matrix 
    //------------------------------------------------------------------

    rowCount = A12StartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(selectedList_, i, nSelected); 
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr, i, &rowSize, &colInd, &colVal);
          newRowSize = 0;
          newColInd  = new int[rowSize];
          newColVal  = new double[rowSize];
          for (j = 0; j < rowSize; j++)  
          {
             colIndex = colInd[j];
             if ( colVal[j] != 0.0 )
             {
                for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                if ( procIndex == numProcs_ ) 
                   ubound = globalNRows -
                            (globalNConstr - ProcNConstr[numProcs_-1]);
                else
                   ubound = ProcNRows[procIndex] - 
                            (ProcNConstr[procIndex]-ProcNConstr[procIndex-1]);
                procIndex--;
                if ( colIndex >= ubound ) { 
                   if ( procIndex != numProcs_ - 1 ) 
                   {
                      newColInd[newRowSize] = colInd[j] - ubound + 
                                              ProcNConstr[procIndex] +
                                              ProcNConstr[procIndex+1];
                   }
                   else 
                   {
                      newColInd[newRowSize] = colInd[j] - ubound + 
                                              ProcNConstr[procIndex] +
                                              globalNConstr;
                   }
                   if ( newColInd[newRowSize] < 0 || 
                        newColInd[newRowSize] >= A12GlobalNCols )
                   {
                      if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                      {
                         printf("%4d : SlideReduction WARNING - A12 col index",
                                mypid_);
                         printf(" out of range %d %d(%d)\n", i, 
                              newColInd[newRowSize], A12GlobalNCols);
                      }
                   }
                   newColVal[newRowSize++] = colVal[j];
                } 
                else
                {
                   if ( colInd[j] >= StartRow && colInd[j] <= EndRow ) 
                   {
                      searchIndex = HYPRE_LSI_Search(selectedList_,colInd[j],
                                                     nSelected);
                      if ( searchIndex >= 0 ) 
                         searchIndex = selectedListAux_[searchIndex] + 
                                       ProcNConstr[mypid_];
                   } else {
                      searchIndex = HYPRE_LSI_Search(globalSelectedList,
                                          colInd[j], globalNSelected);
                      if ( searchIndex >= 0 ) 
                         searchIndex = globalSelectedListAux[searchIndex]; 
                   }
                   if ( searchIndex >= 0) 
                   {
                      newColInd[newRowSize] = searchIndex + 
                                              ProcNConstr[procIndex]; 
                      if ( newColInd[newRowSize] < 0 || 
                           newColInd[newRowSize] >= A12GlobalNCols )
                      {
                         if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
                         {
                            printf("%4d : SlideReduction WARNING - A12 col ",
                                   mypid_);
                            printf(" index out of range %d %d(%d)\n",i,
                                newColInd[newRowSize], A12GlobalNCols);
                         }
                      }
                      newColVal[newRowSize++] = colVal[j];
                   }
                }
             }
          }
          ierr = HYPRE_IJMatrixSetValues(A12, 1, &newRowSize, 
			(const int *) &rowCount, (const int *) newColInd, 
			(const double *) newColVal);
          assert(!ierr);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
       else
       {
          newRowSize = 1;
          newColInd  = new int[newRowSize];
          newColVal  = new double[newRowSize];
          newColInd[0] = A21StartRow;
          newColVal[0] = 0.0;
          ierr = HYPRE_IJMatrixSetValues(A12, 1, &newRowSize, 
			(const int *) &rowCount, (const int *) newColInd, 
			(const double *) newColVal);
          assert(!ierr);
          rowCount++;
          delete [] newColInd;
          delete [] newColVal;
       }
    }

    //------------------------------------------------------------------
    // assemble the A12 matrix 
    //------------------------------------------------------------------

    ierr = HYPRE_IJMatrixAssemble(A12);
    assert( !ierr );
    HYPRE_IJMatrixGetObject(A12, (void **) &A12_csr);

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE3 )
    {
       MPI_Barrier(comm_);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             for (i=A12StartRow;i<A12StartRow+A12NRows;i++)
             {
                printf("%d : A12 ROW %d\n", mypid_, i+1);
                HYPRE_ParCSRMatrixGetRow(A12_csr,i,&rowSize,&colInd,&colVal);
                //qsort1(colInd, colVal, 0, rowSize-1);
                for ( j = 0; j < rowSize; j++ )
                   if ( colVal[j] != 0.0 )
                      printf(" A12 %d %d %20.13e\n",i+1,colInd[j]+1,colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(A12_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(comm_);
          ncnt++;
       }
    }

    //------------------------------------------------------------------
    // form reducedB_ = A12 * f2hat
    //------------------------------------------------------------------

    if ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 )
       printf("%4d : SlideReduction2 - form reduced right hand side\n",mypid_);
    ierr  = HYPRE_IJVectorCreate(comm_, reducedAStartRow,
		reducedAStartRow+newNRows-1, &reducedB_);
    ierr += HYPRE_IJVectorSetObjectType(reducedB_, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorInitialize(reducedB_);
    ierr += HYPRE_IJVectorAssemble(reducedB_);
    assert( !ierr );
    HYPRE_IJVectorGetObject(reducedB_, (void **) &reducedB_csr);

    HYPRE_ParCSRMatrixMatvec( -1.0, A12_csr, f2hat_csr, 0.0, reducedB_csr );
    HYPRE_IJMatrixDestroy(A12); 
    HYPRE_IJVectorDestroy(f2hat); 

    //------------------------------------------------------------------
    // finally form reducedB = f1 - f2til
    //------------------------------------------------------------------

    rowCount = reducedAStartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       if ( hypre_BinarySearch(selectedList_, i, nSelected) < 0 ) 
       {
          HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
          HYPRE_IJVectorAddToValues(reducedB_, 1, (const int *) &rowCount,
                                             (const double *) &ddata);
          rowCount++;
       }
       else
       {
          ddata = 0.0;
          HYPRE_IJVectorSetValues(reducedB_, 1, (const int *) &rowCount,
                                           (const double *) &ddata);
          rowCount++;
       }
    }

    //******************************************************************
    // set up the system with the new matrix
    //------------------------------------------------------------------

    reducedA_ = reducedA;
    ierr = HYPRE_IJVectorCreate(comm_, reducedAStartRow,
		reducedAStartRow+newNRows-1, &reducedX_);
    ierr = HYPRE_IJVectorSetObjectType(reducedX_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(reducedX_);
    ierr = HYPRE_IJVectorAssemble(reducedX_);
    assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm_, reducedAStartRow,
		reducedAStartRow+newNRows-1, &reducedR_);
    ierr = HYPRE_IJVectorSetObjectType(reducedR_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorInitialize(reducedR_);
    ierr = HYPRE_IJVectorAssemble(reducedR_);
    assert(!ierr);

    currA_ = reducedA_;
    currB_ = reducedB_;
    currR_ = reducedR_;
    currX_ = reducedX_;

    //******************************************************************
    // save A21 and invA22 for solution recovery
    //------------------------------------------------------------------

    HYA21_    = A21; 
    HYinvA22_ = invA22; 

    //------------------------------------------------------------------
    // final clean up
    //------------------------------------------------------------------

    if (globalSelectedList != NULL) delete [] globalSelectedList;
    if (globalSelectedListAux != NULL) delete [] globalSelectedListAux;
    if (ProcNRows != NULL) delete [] ProcNRows;
    if (ProcNConstr != NULL) delete [] ProcNConstr;

    HYPRE_ParCSRMatrixDestroy(RAP_csr);
    if ( colIndices_ != NULL )
    {
       for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
          if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
       delete [] colIndices_;
       colIndices_ = NULL;
    }
    if ( colValues_ != NULL )
    {
       for ( j = 0; j < localEndRow_-localStartRow_+1; j++ )
          if ( colValues_[j] != NULL ) delete [] colValues_[j];
       delete [] colValues_;
       colValues_ = NULL;
       if ( rowLengths_ != NULL ) 
       {
          delete [] rowLengths_;
          rowLengths_ = NULL;
       }
    }

    //------------------------------------------------------------------
    // checking 
    //------------------------------------------------------------------

    MPI_Allreduce(&nnzA12,&ncnt,1,MPI_INT,MPI_SUM,comm_);
    if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1) )
       printf("       SlideReduction2 - NNZ of A12 = %d\n", ncnt);
    MPI_Allreduce(&nnzA21,&ncnt,1,MPI_INT,MPI_SUM,comm_);
    if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1) )
       printf("       SlideReduction2 - NNZ of A21 = %d\n", ncnt);
}

//*****************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//-----------------------------------------------------------------------------

double HYPRE_LinSysCore::buildSlideReducedSoln()
{
    int                i, j, *int_array, *gint_array, x2NRows, x2GlobalNRows;
    int                ierr, rowNum, startRow, startRow2, index, localNRows;
    double             ddata, rnorm;
    HYPRE_ParCSRMatrix A_csr, A21_csr, A22_csr;
    HYPRE_ParVector    x_csr, x2_csr, r_csr, b_csr;
    HYPRE_IJVector     R1, x2; 
       
    if ( HYA21_ == NULL || HYinvA22_ == NULL )
    {
       printf("buildSlideReducedSoln WARNING : A21 or A22 absent.\n");
       return (0.0);
    }
    else
    {
       //------------------------------------------------------------------
       // compute A21 * sol
       //------------------------------------------------------------------

       x2NRows = 2 * nConstraints_;
       int_array = new int[numProcs_];
       gint_array = new int[numProcs_];
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
       ierr = HYPRE_IJVectorCreate(comm_, startRow, startRow+x2NRows-1, &R1);
       ierr = HYPRE_IJVectorSetObjectType(R1, HYPRE_PARCSR);
       ierr = HYPRE_IJVectorInitialize(R1);
       ierr = HYPRE_IJVectorAssemble(R1);
       assert(!ierr);
       HYPRE_IJMatrixGetObject(HYA21_, (void **) &A21_csr);
       HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
       HYPRE_IJVectorGetObject(R1, (void **) &r_csr);
       HYPRE_ParCSRMatrixMatvec( -1.0, A21_csr, x_csr, 0.0, r_csr );

       //------------------------------------------------------------------
       // f2 - A21 * sol
       //------------------------------------------------------------------

       for ( i = 0; i < nConstraints_; i++ )
       {
          for ( j = 0; j < nConstraints_; j++ ) 
          {
             if ( selectedListAux_[j] == i ) 
             {
                index = selectedList_[j]; 
                break;
             }
          }
          HYPRE_IJVectorGetValues(HYb_, 1, &index, &ddata);
          HYPRE_IJVectorAddToValues(R1, 1, (const int *) &rowNum,
			(const double *) &ddata);
          rowNum++;
       }
       for ( i = localEndRow_-nConstraints_; i < localEndRow_; i++ )
       {
          HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
          HYPRE_IJVectorAddToValues(R1, 1, (const int *) &rowNum,
			(const double *) &ddata);
          rowNum++;
       } 

       //-------------------------------------------------------------
       // inv(A22) * (f2 - A21 * sol)
       //-------------------------------------------------------------

       ierr = HYPRE_IJVectorCreate(comm_, startRow, startRow+x2NRows-1, &x2);
       ierr = HYPRE_IJVectorSetObjectType(x2, HYPRE_PARCSR);
       ierr = HYPRE_IJVectorInitialize(x2);
       ierr = HYPRE_IJVectorAssemble(x2);
       assert(!ierr);
       HYPRE_IJMatrixGetObject(HYinvA22_, (void **) &A22_csr);
       HYPRE_IJVectorGetObject(R1, (void **) &r_csr);
       HYPRE_IJVectorGetObject(x2, (void **) &x2_csr);
       HYPRE_ParCSRMatrixMatvec( 1.0, A22_csr, r_csr, 0.0, x2_csr );

       //-------------------------------------------------------------
       // inject final solution to the solution vector
       //-------------------------------------------------------------

       localNRows = localEndRow_ - localStartRow_ + 1 - 2 * nConstraints_;
       rowNum = localStartRow_ - 1;
       for ( i = startRow2; i < startRow2+localNRows; i++ )
       {
          HYPRE_IJVectorGetValues(reducedX_, 1, &i, &ddata);
          while (HYPRE_LSI_Search(selectedList_,rowNum,nConstraints_)>=0)
             rowNum++;
          HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &rowNum,
			(const double *) &ddata);
          rowNum++;
       }
       for ( i = 0; i < nConstraints_; i++ )
       {
          for ( j = 0; j < nConstraints_; j++ ) 
          {
             if ( selectedListAux_[j] == i ) 
             {
                index = selectedList_[j]; 
                break;
             }
          }
          j = i + startRow; 
          HYPRE_IJVectorGetValues(x2, 1, &j, &ddata);
          HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &index,
			(const double *) &ddata);
       }
       for ( i = nConstraints_; i < 2*nConstraints_; i++ )
       {
          j = startRow + i;
          HYPRE_IJVectorGetValues(x2, 1, &j, &ddata);
          index = localEndRow_ - 2 * nConstraints_ + i;
          HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &index,
			(const double *) &ddata);
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
       if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 ))
          printf("buildSlideReducedSoln::final residual norm = %e\n", rnorm);
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
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//-----------------------------------------------------------------------------

double HYPRE_LinSysCore::buildSlideReducedSoln2()
{
    int                i, j, *int_array, *gint_array, x2NRows, x2GlobalNRows;
    int                ierr, rowNum, startRow, startRow2, index, localNRows;
    int                index2;
    double             ddata, rnorm;
    HYPRE_ParCSRMatrix A_csr, A21_csr, A22_csr;
    HYPRE_ParVector    x_csr, x2_csr, r_csr, b_csr;
    HYPRE_IJVector     R1, x2; 
       
    if ( HYA21_ == NULL || HYinvA22_ == NULL )
    {
       printf("buildSlideReducedSoln2 WARNING : A21 or A22 absent.\n");
       return (0.0);
    }
    else
    {
       //------------------------------------------------------------------
       // compute A21 * sol
       //------------------------------------------------------------------

       x2NRows = 2 * nConstraints_;
       int_array = new int[numProcs_];
       gint_array = new int[numProcs_];
       for ( i = 0; i < numProcs_; i++ ) int_array[i] = 0;
       int_array[mypid_] = x2NRows;
       MPI_Allreduce(int_array,gint_array,numProcs_,MPI_INT,MPI_SUM,comm_);
       x2GlobalNRows = 0;
       for ( i = 0; i < numProcs_; i++ ) x2GlobalNRows += gint_array[i];
       rowNum = 0;
       for ( i = 0; i < mypid_; i++ ) rowNum += gint_array[i];
       startRow = rowNum;
       startRow2 = localStartRow_ - 1 - rowNum/2;
       delete [] int_array;
       delete [] gint_array;
       ierr = HYPRE_IJVectorCreate(comm_, startRow, startRow+x2NRows-1, &R1);
       ierr = HYPRE_IJVectorSetObjectType(R1, HYPRE_PARCSR);
       ierr = HYPRE_IJVectorInitialize(R1);
       ierr = HYPRE_IJVectorAssemble(R1);
       assert(!ierr);
       HYPRE_IJMatrixGetObject(HYA21_, (void **) &A21_csr);
       HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
       HYPRE_IJVectorGetObject(R1, (void **) &r_csr);
       HYPRE_ParCSRMatrixMatvec( -1.0, A21_csr, x_csr, 0.0, r_csr );

       //------------------------------------------------------------------
       // f2 - A21 * sol
       //------------------------------------------------------------------

       for ( i = 0; i < nConstraints_; i++ )
       {
          for ( j = 0; j < nConstraints_; j++ ) 
          {
             if ( selectedListAux_[j] == i ) 
             {
                index = selectedList_[j]; 
                break;
             }
          }
          HYPRE_IJVectorGetValues(HYb_, 1, &index, &ddata);
          HYPRE_IJVectorAddToValues(R1, 1, (const int *) &rowNum,
			(const double *) &ddata);
          rowNum++;
       }
       for ( i = localEndRow_-nConstraints_; i < localEndRow_; i++ )
       {
          HYPRE_IJVectorGetValues(HYb_, 1, &i, &ddata);
          HYPRE_IJVectorAddToValues(R1, 1, (const int *) &rowNum,
			(const double *) &ddata);
          rowNum++;
       } 

       //-------------------------------------------------------------
       // inv(A22) * (f2 - A21 * sol)
       //-------------------------------------------------------------

       ierr = HYPRE_IJVectorCreate(comm_, startRow, startRow+x2NRows-1, &x2);
       ierr = HYPRE_IJVectorSetObjectType(x2, HYPRE_PARCSR);
       ierr = HYPRE_IJVectorInitialize(x2);
       ierr = HYPRE_IJVectorAssemble(x2);
       assert(!ierr);
       HYPRE_IJMatrixGetObject(HYinvA22_, (void **) &A22_csr );
       HYPRE_IJVectorGetObject(R1, (void **) &r_csr );
       HYPRE_IJVectorGetObject(x2, (void **) &x2_csr );
       HYPRE_ParCSRMatrixMatvec( 1.0, A22_csr, r_csr, 0.0, x2_csr );

       //-------------------------------------------------------------
       // inject final solution to the solution vector
       //-------------------------------------------------------------

       localNRows = localEndRow_ - localStartRow_ + 1 - nConstraints_;
       for ( i = 0; i < localNRows; i++ )
       {
          index = startRow2 + i;
          HYPRE_IJVectorGetValues(reducedX_, 1, &index, &ddata);
          index2 = localStartRow_ - 1 + i;
          HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &index2,
			(const double *) &ddata);
       }
       for ( i = 0; i < nConstraints_; i++ )
       {
          for ( j = 0; j < nConstraints_; j++ ) 
          {
             if ( selectedListAux_[j] == i ) 
             {
                index = selectedList_[j]; 
                break;
             }
          }
          j = i + startRow; 
          HYPRE_IJVectorGetValues(x2, 1, &j, &ddata);
          HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &index,
			(const double *) &ddata);
       }
       for ( i = nConstraints_; i < 2*nConstraints_; i++ )
       {
          j = startRow + i;
          HYPRE_IJVectorGetValues(x2, 1, &j, &ddata);
          index = localEndRow_ - 2 * nConstraints_ + i;
          HYPRE_IJVectorSetValues(HYx_, 1, (const int *) &index,
			(const double *) &ddata);
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
       if ( mypid_ == 0 && ( HYOutputLevel_ & HYFEI_SLIDEREDUCE1 ))
          printf("buildSlideReducedSoln::final residual norm = %e\n", rnorm);
    } 
    currX_ = HYx_;

    //****************************************************************
    // clean up
    //----------------------------------------------------------------

    HYPRE_IJVectorDestroy(R1); 
    HYPRE_IJVectorDestroy(x2); 
    return rnorm;
}

