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
#include "base/Data.h"
#include "base/basicTypes.h"
#include "base/Utils.h"
#include "base/LinearSystemCore.h"
#include "HYPRE_LinSysCore.h"

#define abs(x) (((x) > 0.0) ? x : -(x))

//---------------------------------------------------------------------------
// parcsr_matrix_vector.h is put here instead of in HYPRE_LinSysCore.h 
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "parcsr_matrix_vector/parcsr_matrix_vector.h"

//---------------------------------------------------------------------------
// These are external functions needed internally here
//---------------------------------------------------------------------------

extern "C" 
{
   int hypre_ParAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix**);
   void qsort0(int *, int, int);
   void qsort1(int *, double *, int, int);

}

//***************************************************************************
// HYFEI_BinarySearch - this is a modification of hypre_BinarySearch
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::HYFEI_BinarySearch(int *list,int value,int list_length)
{
   int low, high, m;
   int not_found = 1;

   low = 0;
   high = list_length-1;
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (value < list[m])
      {
         high = m - 1;
      }
      else if (value > list[m])
      {
        low = m + 1;
      }
      else
      {
        not_found = 0;
        return m;
      }
   }
   return -(low+1);
}

//******************************************************************************
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
//------------------------------------------------------------------------------
// This first draft does not try to reduce the number of columns for
// A21 so that the triple matrix product operations will result in a
// NxN matrix.  This can be inefficient and needs to be studied further.
// However, this approach simplifies the searching and remapping. 
//------------------------------------------------------------------------------

void HYPRE_LinSysCore::buildReducedSystem()
{
    int    j, k, nRows, globalNRows, colIndex, nSlaves;
    int    globalNConstr, globalNSelected, *globalSelectedList;
    int    *globalSelectedListAux, *selectedListAux;
    int    nSelected, *tempList, i, reducedAStartRow;
    int    searchIndex, procIndex, A21StartRow, A12StartRow, A12NRows;
    int    rowSize, *colInd, A21NRows, A21GlobalNRows;
    int    A21NCols, A21GlobalNCols, rowCount, maxRowSize, newEndRow;
    int    A12NCols, A12GlobalNCols, *constrListAux;
    int    *A21MatSize, rowIndex, *A12MatSize, A12GlobalNRows;
    int    *newColInd, diagCount, newRowSize, ierr;
    int    invA22NRows, invA22GlobalNRows, invA22NCols, invA22GlobalNCols;
    int    *invA22MatSize, newNRows, newGlobalNRows;
    int    *colInd2, *selectedList, ncnt, ubound;
    int    rowSize2, *recvCntArray, *displArray, ncnt2;
    int    StartRow, EndRow, *reducedAMatSize;
    int    *ProcNRows, *ProcNConstr;

    double searchValue, *colVal, *colVal2, *newColVal, *diagonal;
    double *extDiagonal, *dble_array, ddata;

    HYPRE_IJMatrix     A12, A21, invA22, reducedA;
    HYPRE_ParCSRMatrix A_csr, A12_csr, A21_csr, invA22_csr, RAP_csr;
    HYPRE_ParCSRMatrix reducedA_csr;
    HYPRE_IJVector     f2, f2hat;
    HYPRE_ParVector    f2_csr, f2hat_csr, reducedB_csr;

    //******************************************************************
    // initial set up 
    //------------------------------------------------------------------

    if ( mypid_ == 0 ) printf("%4d buildReducedSystem activated.\n",mypid_);
    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : StartRow/EndRow = %d %d\n",mypid_,
                                        StartRow,EndRow);
    }

    //******************************************************************
    // construct local and global information about where the constraints
    // are (this is given by user or searched within this code)
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // get the CSR matrix for A
    //------------------------------------------------------------------

    A_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);

    //------------------------------------------------------------------
    // search the entire local matrix to find where the constraint
    // equations are, if not already given
    //------------------------------------------------------------------
    
    MPI_Allreduce(&nConstraints_,&globalNConstr,1,MPI_INT,MPI_SUM,comm_);
    if ( globalNConstr == 0 )
    {
       for ( i = EndRow; i >= StartRow; i-- ) 
       {
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          for (j = 0; j < rowSize; j++) 
          {
             if ( colInd[j] == i && colVal[j] != 0.0 ) break;
          }
          ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          assert(!ierr);
          if ( j < rowSize ) nConstraints_++;
          else               break;
       }
    }
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : no. constr = %d\n",mypid_,nConstraints_);
    }

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
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : localNRows = %d\n", mypid_, nRows);
    }

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
    else
    {
       globalSelectedList = NULL;
       globalSelectedListAux = NULL;
    }
    nSelected = nConstraints_;
    if ( nConstraints_ > 0 ) 
    {
       selectedList = new int[nConstraints_];
       selectedListAux = new int[nConstraints_];
    }
    else 
    {
       selectedList = NULL;
       selectedListAux = NULL;
    }
   
    //------------------------------------------------------------------
    // compose candidate slave list (if not given already)
    //------------------------------------------------------------------

    if ( nConstraints_ > 0 && constrList_ == NULL )
    {
       constrList_   = new int[EndRow-nConstraints_-StartRow+1];
       constrListAux = new int[EndRow-nConstraints_-StartRow+1];
       nSlaves = 0;

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
          if ( HYOutputLevel_ > 2 )
          {
             if ( j == rowSize && ncnt == 1 ) 
                printf("%d buildReducedSystem : slave candidate %d = %d(%d)\n", 
                        mypid_, nSlaves-1, i, constrListAux[nSlaves-1]);
          }
       }
       if ( HYOutputLevel_ > 1 )
       {
          printf("%d buildReducedSystem : nSlave Candidate, nConstr = %d %d\n",
                 mypid_,nSlaves, nConstraints_);
       }
    }
    else
    {
       if ( mypid_ == 0 )
          printf("%4d buildReducedSystem WARNING : HARDWIRED TO 3 DOF/NODE.\n",
                  mypid_);
       constrListAux = new int[EndRow-nConstraints_-StartRow+1];
       nSlaves = 3 * nConstraints_;;
       for ( i = 0; i < 3*nConstraints_; i++ ) 
       {
          rowIndex = constrList_[i]; 
          if ( rowIndex < localStartRow_-1 || rowIndex >= localEndRow_)
          {
             printf("%4d buildReducedSystem : slave %d not on my proc\n",
                    mypid_, rowIndex, localStartRow_-1, localEndRow_);
             exit(1);
          }
          ierr = HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
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
             if ( colIndex >= ubound && colVal[j] != 0.0 ) 
             {
                ncnt++;
                searchIndex = colIndex;
             }
             if ( ncnt > 1 ) break;
          }
          HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
          if ( j == rowSize && ncnt == 1 ) constrListAux[i] = searchIndex;
          else                             constrListAux[i] = -1;
          if ( HYOutputLevel_ > 1 )
          {
             if ( j == rowSize && ncnt == 1 ) 
                printf("%4d buildReducedSystem : slave,constr pair = %d %d\n",
                        mypid_, constrList_[i], constrListAux[i]);
          }
       }
    }   
    if ( HYOutputLevel_ > 1 )
    {
       printf("%d buildReducedSystem : nSlave Candidate, nConstr = %d %d\n",
              mypid_,nSlaves, nConstraints_);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //---------------------------------------------------------------------
    // search the constraint equations for the selected nodes
    // (search for candidates column index with maximum magnitude)
    //---------------------------------------------------------------------
    
    nSelected = 0;
    rowIndex = -1;

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
                 if ( abs(colVal[j]) > searchValue )
                 {
                    if (i != constrListAux[colIndex]) 
                    {
                       printf("%4d buildReducedSystem WARNING : slave %d",
                               mypid_, colInd[j]);
                       printf(" candidate does not have constr %d\n", i);
                    }
                    searchValue = abs(colVal[j]);
                    searchIndex = colInd[j];
                 }
             }
          }
       } 
       if ( searchIndex >= 0 )
       {
          selectedList[nSelected++] = searchIndex;
          if ( HYOutputLevel_ > 1 )
          {
             printf("%4d buildReducedSystem : constraint %4d <=> slave %d\n",
                    mypid_,i,searchIndex);
          }
       } else 
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
             printf("%4d buildReducedSystem ERROR : constraint number",mypid_);
             printf(" cannot be found for row %d\n", rowIndex);
             for (j = 0; j < rowSize; j++) 
             {
                printf("ROW %4d COL = %d VAL = %e\n",rowIndex,colInd[j],colVal[j]);
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
          MPI_Barrier(MPI_COMM_WORLD);
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
    qsort1(selectedList, dble_array, 0, nSelected-1);
    for (i = 1; i < nSelected; i++) 
    {
       if ( selectedList[i] == selectedList[i-1] )
       {
          printf("%4d buildReducedSystem : repeated selected nodes %d \n", 
                 mypid_, selectedList[i]);
          exit(1);
       }
    }
    for (i = 0; i < nSelected; i++) selectedListAux[i] = (int) dble_array[i];
    delete [] dble_array;
    
    recvCntArray = new int[numProcs_];
    displArray   = new int[numProcs_];
    MPI_Allgather(&nSelected, 1, MPI_INT,recvCntArray, 1,MPI_INT, comm_);
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

    if ( HYOutputLevel_ > 1 )
    {
       for ( i = 0; i < nSelected; i++ )
          printf("%4d buildReducedSystem : selectedList %d = %d(%d)\n",mypid_,
                 i,selectedList[i],selectedListAux[i]);
    }
 
    //******************************************************************
    // construct A21
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of A21
    //------------------------------------------------------------------

    A21NRows       = 2 * nConstraints_;
    A21NCols       = nRows - 2 * nConstraints_;
    A21GlobalNRows = 2 * globalNConstr;
    A21GlobalNCols = globalNRows - 2 * globalNConstr;
    A21StartRow    = 2 * ProcNConstr[mypid_];

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : A21StartRow  = %d\n",mypid_,A21StartRow);
       printf("%4d buildReducedSystem : A21GlobalDim = %d %d\n", mypid_, 
                                        A21GlobalNRows, A21GlobalNCols);
       printf("%4d buildReducedSystem : A21LocalDim  = %d %d\n",mypid_,
                                        A21NRows, A21NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A21
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&A21,A21GlobalNRows,A21GlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(A21, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(A21, A21NRows, A21NCols);
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
	  searchIndex = hypre_BinarySearch(globalSelectedList,colIndex, 
                                           globalNSelected);
          if (searchIndex < 0 && 
              (colIndex <= newEndRow || colIndex > localEndRow_)) rowSize2++;
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
	     searchIndex = hypre_BinarySearch(selectedList,colIndex,nSelected); 
             if ( searchIndex < 0 ) rowSize2++;
          }
       }
       A21MatSize[rowCount] = rowSize2;
       maxRowSize = ( rowSize2 > maxRowSize ) ? rowSize2 : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowCount++;
    }

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
    else                    diagonal = NULL;
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
             if (colIndex <= newEndRow || colIndex > localEndRow_) 
             {
	        searchIndex = HYFEI_BinarySearch(globalSelectedList,colIndex, 
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
                      printf("%4d buildReducedSystem WARNING : A21 ", mypid_);
                      printf("out of range %d - %d (%d)\n", rowCount, colIndex, 
                              A21GlobalNCols);
                   } 
                   if ( newRowSize > maxRowSize+1 ) 
                   {
                      printf("%4d buildReducedSystem : WARNING - ",mypid_);
                      printf("passing array boundary(1).\n");
                   }
                }
             }
             else if ( colIndex > newEndRow && colIndex <= EndRow ) 
             {
                if ( colVal[j] != 0.0 ) diagonal[diagCount++] = colVal[j];
                if ( abs(colVal[j]) < 1.0E-8 )
                {
                   printf("%4d buildReducedSystem WARNING : large entry ",mypid_);
                   printf("in invA22\n");
                }
             }
          } 
       }

       HYPRE_IJMatrixInsertRow(A21,newRowSize,rowCount,newColInd,newColVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       if ( diagCount != (i+1) )
       {
          printf("%4d buildReducedSystem ERROR (3) : %d %d.\n", mypid_,
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
	  searchIndex = HYFEI_BinarySearch(globalSelectedList,colIndex,
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
                printf("%4d buildReducedSystem WARNING : A21(%d,%d) out of range\n",
                       mypid_, rowCount, colIndex, A21GlobalNCols);
             } 
             if ( newRowSize > maxRowSize+1 ) 
             {
                printf("%4d : buildReducedSystem WARNING : ",mypid_);
                printf("passing array boundary(2).\n");
             }
          } 
       }
       HYPRE_IJMatrixInsertRow(A21,newRowSize,rowCount,newColInd,newColVal);
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       rowCount++;
    }
    delete [] newColInd;
    delete [] newColVal;

    //------------------------------------------------------------------
    // finally assemble the matrix and sanitize
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(A21);
    A21_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(A21);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A21_csr);

    if ( HYOutputLevel_ > 3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d buildReducedSystem : matrix A21 assembled %d.\n",
                                        mypid_,A21StartRow);
             fflush(stdout);
             for ( i = A21StartRow; i < A21StartRow+2*nConstraints_; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(A21_csr,i,&rowSize,&colInd,&colVal);
                printf("A21 ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(A21_csr,i,&rowSize,&colInd,&colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(MPI_COMM_WORLD);
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
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : A22GlobalDim = %d %d\n", mypid_, 
                        invA22GlobalNRows, invA22GlobalNCols);
       printf("%4d buildReducedSystem : A22LocalDim  = %d %d\n", mypid_, 
                        invA22NRows, invA22NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A22
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&invA22,invA22GlobalNRows,
                                 invA22GlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(invA22, HYPRE_PARCSR);
    ierr += HYPRE_IJMatrixSetLocalSize(invA22, invA22NRows, invA22NCols);
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
    for ( i = 0; i < nConstraints_; i++ ) {
       newColInd[0] = A21StartRow + nConstraints_ + i; 
       rowIndex     = A21StartRow + i;
       if ( newColInd[0] < 0 || newColInd[0] >= invA22GlobalNCols )
       {
          printf("%4d buildReducedSystem WARNING : A22 out of range %d, %d (%d)\n", 
                 mypid_, rowIndex, newColInd[0], invA22GlobalNCols);
       } 
       newColVal[0] = extDiagonal[A21StartRow/2+i];
       ierr = HYPRE_IJMatrixInsertRow(invA22,1,rowIndex,newColInd,newColVal);
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
                   printf("%4d buildReducedSystem WARNING : A22 out of range",
                          mypid_);
                   printf(" %d - %d (%d)\n", rowCount, newColInd[newRowSize], 
                          invA22GlobalNCols);
                } 
                newColVal[newRowSize++] = - extDiagonal[A21StartRow/2+i] * 
                                        colVal[j] * extDiagonal[searchIndex];
                if ( newRowSize > maxRowSize )
                {
                   printf("%4d buildReducedSystem : WARNING - ",mypid_);
                   printf("passing array boundary(3).\n");
                }
      	     } 
	  } 
       }
       rowCount = A21StartRow + nConstraints_ + i;
       ierr = HYPRE_IJMatrixInsertRow(invA22, newRowSize, rowCount, 
                                      newColInd, newColVal);
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
    invA22_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(invA22);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) invA22_csr);

    if ( HYOutputLevel_ > 3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d buildReducedSystem : invA22 \n", mypid_);
             for ( i = A21StartRow; i < A21StartRow+2*nConstraints_; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(invA22_csr,i,&rowSize,&colInd,&colVal);
                printf("invA22 ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(invA22_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    //******************************************************************
    // perform the triple matrix product
    //------------------------------------------------------------------

    A21_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(A21);
    invA22_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(invA22);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : Triple matrix product starts\n",mypid_);
    }
    hypre_ParAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix *) invA22_csr,
                                     (hypre_ParCSRMatrix *) A21_csr,
                                     (hypre_ParCSRMatrix **) &RAP_csr);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : Triple matrix product ends\n", mypid_);
    }

    if ( HYOutputLevel_ > 3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
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
                HYPRE_ParCSRMatrixRestoreRow(RAP_csr,i,&rowSize,&colInd,&colVal);
             }
          }
          MPI_Barrier(MPI_COMM_WORLD);
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

    newNRows       = nRows - 2 * nConstraints_;
    newGlobalNRows = globalNRows - 2 * globalNConstr;
    ierr  = HYPRE_IJMatrixCreate(comm_,&reducedA,
                                 newGlobalNRows,newGlobalNRows);
    ierr += HYPRE_IJMatrixSetLocalStorageType(reducedA, HYPRE_PARCSR);
    ierr += HYPRE_IJMatrixSetLocalSize(reducedA, newNRows, newNRows);
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
                searchIndex = HYFEI_BinarySearch(globalSelectedList,colIndex, 
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
          // translate the newColInd
          ierr = HYPRE_IJMatrixInsertRow(reducedA, newRowSize, rowCount,
                                        newColInd, newColVal);
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
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : FINAL - reducedAStartRow = %d\n", 
                                       mypid_, reducedAStartRow);
    }

    reducedA_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(reducedA);

    if ( HYOutputLevel_ > 3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             printf("====================================================\n");
             for (i=reducedAStartRow;i<reducedAStartRow+nRows-2*nConstraints_;i++)
             {
                printf("%d : reducedA ROW %d\n", mypid_, i);
                ierr = HYPRE_ParCSRMatrixGetRow(reducedA_csr,i,&rowSize,&colInd,
                                                &colVal);
                //qsort1(colInd, colVal, 0, rowSize-1);
                for ( j = 0; j < rowSize; j++ )
                   if ( colVal[j] != 0.0 )
                      printf("%4d %4d %20.13e\n", i+1, colInd[j]+1, colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(reducedA_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    // *****************************************************************
    // form modified right hand side  (f1 = f1 - A12*invA22*f2)
    // *****************************************************************

    // *****************************************************************
    // form f2hat = invA22 * f2
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, &f2, A21GlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2, HYPRE_PARCSR);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : A21 dims = %d %d %d\n", mypid_, 
               A21StartRow, A21NRows, A21GlobalNRows);
    }
    ierr =  HYPRE_IJVectorSetLocalPartitioning(f2,A21StartRow,
                                               A21StartRow+A21NRows);
    ierr += HYPRE_IJVectorAssemble(f2);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2);
    //assert(!ierr);

    HYPRE_IJVectorCreate(comm_, &f2hat, A21GlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2hat, HYPRE_PARCSR);
    ierr =  HYPRE_IJVectorSetLocalPartitioning(f2hat,A21StartRow,
                                                  A21StartRow+A21NRows);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2hat);
    //assert(!ierr);

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
          printf("%4d buildReducedSystem ERROR : out of range %d\n", mypid_,
                  colInd[i]);
          exit(1);
       }
    }
    for ( i = 0; i < nSelected; i++ ) 
    {
       colInd[nSelected+i] = EndRow - nConstraints_ + i + 1;
    }
    HYPRE_IJVectorGetLocalComponents(HYb_, 2*nSelected, colInd,NULL,colVal);
    for ( i = 0; i < nSelected*2; i++ ) colInd[i] = A21StartRow + i;
    ierr = HYPRE_IJVectorSetLocalComponents(f2, 2*nSelected, colInd,
                                            NULL, colVal);
    assert( !ierr );
    f2_csr     = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f2);
    f2hat_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f2hat);
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
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildReducedSystem : A12GlobalDim = %d %d\n", mypid_, 
                        A12GlobalNRows, A12GlobalNCols);
       printf("%4d buildReducedSystem : A12LocalDim  = %d %d\n", mypid_, 
                        A12NRows, A12NCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for A12
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&A12,A12GlobalNRows,A12GlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(A12, HYPRE_PARCSR);
    ierr += HYPRE_IJMatrixSetLocalSize(A12, A12NRows, A12NCols);
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
             colIndex = colInd[j];
             for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                if ( ProcNRows[procIndex] > colIndex ) break;
             if ( procIndex == numProcs_ ) 
                ubound = globalNRows-(globalNConstr-ProcNConstr[numProcs_-1]);
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
          A12MatSize[rowIndex++] = newRowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          rowCount++;
       }
    }
 
    //------------------------------------------------------------------
    // after fetching the row sizes, set up A12 with such sizes
    //------------------------------------------------------------------

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
                   printf("%4d buildReducedSystem WARNING : A12 col index out ",
                          mypid_);
                   printf("of range %d %d(%d)\n", mypid_, i, 
                           newColInd[newRowSize], A12GlobalNCols);
                }
                newColVal[newRowSize++] = colVal[j];
             } else
             {
                searchIndex = HYFEI_BinarySearch(globalSelectedList,colInd[j],
                                                 globalNSelected);
                if ( searchIndex >= 0) 
                {
                   searchIndex = globalSelectedListAux[searchIndex];
                   newColInd[newRowSize] = searchIndex + 
                                           ProcNConstr[procIndex]; 
                   if ( newColInd[newRowSize] < 0 || 
                        newColInd[newRowSize] >= A12GlobalNCols )
                   {
                      printf("%4d buildReducedSystem WARNING : A12 col index ",
                             mypid_);
                      printf("out of range %d %d(%d)\n", mypid_, i, 
                             newColInd[newRowSize], A12GlobalNCols);
                   }
                   newColVal[newRowSize++] = colVal[j];
                }
             }
          }
          ierr = HYPRE_IJMatrixInsertRow(A12, newRowSize, rowCount,
                                         newColInd, newColVal);
          assert(!ierr);
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);

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
    A12_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(A12);

    if ( HYOutputLevel_ > 3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
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
                      printf(" A12 %d %d %20.13e\n", i+1, colInd[j]+1, colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(A12_csr,i,&rowSize,&colInd,
                                             &colVal);
             }
             printf("====================================================\n");
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    //------------------------------------------------------------------
    // form reducedB_ = A12 * f2hat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJVectorCreate(comm_, &reducedB_, newGlobalNRows);
    ierr += HYPRE_IJVectorSetLocalStorageType(reducedB_, HYPRE_PARCSR);
    ierr += HYPRE_IJVectorSetLocalPartitioning(reducedB_,reducedAStartRow,
                                               reducedAStartRow+newNRows);
    ierr += HYPRE_IJVectorAssemble(reducedB_);
    ierr += HYPRE_IJVectorInitialize(reducedB_);
    ierr += HYPRE_IJVectorZeroLocalComponents(reducedB_);
    assert( !ierr );

    reducedB_csr = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(reducedB_);
    HYPRE_ParCSRMatrixMatvec( -1.0, A12_csr, f2hat_csr, 0.0, reducedB_csr );
    HYPRE_IJMatrixDestroy(A12); 
    HYPRE_IJVectorDestroy(f2hat); 
    //for ( i = reducedAStartRow; i < reducedAStartRow+newNRows; i++ ) 
    //{
    //   HYPRE_IJVectorGetLocalComponents(reducedB_, 1, &i, NULL, &ddata);
    //   printf("A12 * invA22 * f2 %d = %e\n", i, ddata);
    //}

    //------------------------------------------------------------------
    // finally form reducedB = f1 - f2til
    //------------------------------------------------------------------

    rowCount = reducedAStartRow;
    for ( i = StartRow; i <= newEndRow; i++ ) 
    {
       if ( hypre_BinarySearch(selectedList, i, nSelected) < 0 ) 
       {
          HYPRE_IJVectorGetLocalComponents(HYb_, 1, &i, NULL, &ddata);
          HYPRE_IJVectorAddToLocalComponents(reducedB_,1,&rowCount,NULL,
                                             &ddata);
          HYPRE_IJVectorGetLocalComponents(reducedB_,1,&rowCount,NULL, 
                                           &searchValue);
          rowCount++;
       }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //for ( i = reducedAStartRow; 
    //      i < reducedAStartRow+EndRow-StartRow+1-2*nConstraints_; i++ ) 
    //{
    //   HYPRE_IJVectorGetLocalComponents(reducedB_, 1, &i, NULL, &ddata);
    //   printf("RHS(2) %d = %e\n", i, ddata);
    //}

    //******************************************************************
    // set up the system with the new matrix
    //------------------------------------------------------------------

    reducedA_ = reducedA;
    ierr = HYPRE_IJVectorCreate(comm_, &reducedX_, newGlobalNRows);
    ierr = HYPRE_IJVectorSetLocalStorageType(reducedX_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(reducedX_,reducedAStartRow,
                                              reducedAStartRow+newNRows);
    ierr = HYPRE_IJVectorAssemble(reducedX_);
    ierr = HYPRE_IJVectorInitialize(reducedX_);
    ierr = HYPRE_IJVectorZeroLocalComponents(reducedX_);
    assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm_, &reducedR_, newGlobalNRows);
    ierr = HYPRE_IJVectorSetLocalStorageType(reducedR_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(reducedR_,reducedAStartRow,
                                              reducedAStartRow+newNRows);
    ierr = HYPRE_IJVectorAssemble(reducedR_);
    ierr = HYPRE_IJVectorInitialize(reducedR_);
    ierr = HYPRE_IJVectorZeroLocalComponents(reducedR_);
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
    systemReduced_ = 1;
    selectedList_ = selectedList;
    selectedListAux_ = selectedListAux;

    //------------------------------------------------------------------
    // final clean up
    //------------------------------------------------------------------

    delete [] globalSelectedList;
    delete [] globalSelectedListAux;
    delete [] ProcNRows;
    delete [] ProcNConstr;

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

