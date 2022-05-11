/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include <stdio.h>
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "mli_matrix.h"
#include "mli_matrix_misc.h"
#include "mli_utils.h"

/***************************************************************************
 * get the external rows of B in order to multiply A * B
 *--------------------------------------------------------------------------*/

void MLI_Matrix_MatMatMult( MLI_Matrix *Amat, MLI_Matrix *Bmat,
                            MLI_Matrix **Cmat)
{
   int    ir, ic, is, ia, ia2, ib, index, length, offset, iTemp;
   int    *iArray, ibegin, sortFlag, tempCnt, nprocs, mypid;
   int    BExtNumUniqueCols, BExtNRows, *BExtRowLengs, *BExtCols, BExtNnz;
   int    *extColList, *extColListAux;
   int    *BRowStarts, *BColStarts, BNRows, BNCols, BStartCol, BEndCol;
   int    *ARowStarts, *AColStarts, ANRows, ANCols;
   int    *ADiagIA, *AOffdIA, *ADiagJA, *AOffdJA, *CDiagIA, *CDiagJA;
   int    *BDiagIA, *BOffdIA, *BDiagJA, *BOffdJA, *COffdIA, *COffdJA;
   int    *diagCols, *BColMap, BColMapInd, iTempDiag, iTempOffd;
   int    mergeSortNList, **mergeSortList2D, **mergeSortAuxs;
   int    *mergeSortList, *mergeSortLengs, *extDiagListAux;
   int    CNRows, CNCols, *CDiagReg, *COffdReg, COffdNCols;
   int    *CRowStarts, *CColStarts, CExtNCols, colIndA, colIndB;
   int    *CColMap, *CColMapAux, CDiagNnz, COffdNnz;
   double *BDiagAA, *BOffdAA, *ADiagAA, *AOffdAA, dTemp;
   double *BExtVals, *CDiagAA, dTempA, dTempB;
   double *COffdAA;
   char   paramString[50];
   MPI_Comm            mpiComm;
   MLI_Function        *funcPtr;
   hypre_CSRMatrix     *BDiag, *BOffd, *ADiag, *AOffd, *CDiag, *COffd;
   hypre_ParCSRMatrix  *hypreA, *hypreB, *hypreC;

   /* -----------------------------------------------------------------------
    * check to make sure both matrices are ParCSR matrices
    * ----------------------------------------------------------------------*/

   if ( strcmp(Amat->getName(),"HYPRE_ParCSR") ||
        strcmp(Bmat->getName(),"HYPRE_ParCSR") )
   {
      printf("MLI_Matrix_MatMatMult ERROR - matrix has invalid type.\n");
      exit(1);
   }
   hypreA  = (hypre_ParCSRMatrix *) Amat->getMatrix();
   hypreB  = (hypre_ParCSRMatrix *) Bmat->getMatrix();
   mpiComm = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_size(mpiComm, &nprocs);
   MPI_Comm_rank(mpiComm, &mypid);

   /* -----------------------------------------------------------------------
    * Get external rows of B (BExtRowLengs has been allocated 1 longer than
    *     BExtNRows in the GetExtRows function)
    * Extract the diagonal indices into arrays diagCols
    * ----------------------------------------------------------------------*/

   MLI_Matrix_GetExtRows(Amat, Bmat, &BExtNRows, &BExtRowLengs, &BExtCols,
                         &BExtVals);
   tempCnt = 0;
   for ( ir = 0; ir < BExtNRows*2; ir++ )
   {
      iTemp = BExtRowLengs[ir];
      BExtRowLengs[ir] = tempCnt;
      tempCnt += iTemp;
   }
   if ( BExtNRows > 0 )
   {
      BExtRowLengs[BExtNRows*2] = tempCnt;
      for ( ir = 0; ir < BExtNRows*2; ir++ )
      {
         ibegin = BExtRowLengs[ir] + 1;
         if ( ir % 2 == 0 ) ibegin++;
         sortFlag = 0;
         for ( ic = ibegin; ic < BExtRowLengs[ir+1]; ic++ )
            if ( BExtCols[ic] < BExtCols[ic-1] ) {sortFlag=1; break;}
         if ( sortFlag )
         {
            ibegin = BExtRowLengs[ir];
            if ( ir % 2 == 0 ) ibegin++;
            length = BExtRowLengs[ir+1] - ibegin;
            MLI_Utils_IntQSort2a(&(BExtCols[ibegin]), &(BExtVals[ibegin]),
                                 0, length-1);
         }
      }
      diagCols = new int[BExtNRows];
      for ( ir = 0; ir < BExtNRows; ir++ )
         diagCols[ir] = BExtCols[BExtRowLengs[ir*2]];
      MLI_Utils_IntQSort2(diagCols, NULL, 0, BExtNRows-1);
   }

   /* -----------------------------------------------------------------------
    * Compile a list of unique column numbers in BExt.  This is done by
    * exploiting the orderings in the BExtCols in each external row by
    * a merge sort.  At the end, the unique column numbers are given in
    * extColList and the extColListAux array will map the column number of
    * BExtCols[j] to its offset in extColList.
    * ----------------------------------------------------------------------*/

   if ( BExtNRows == 0 ) BExtNnz = BExtNumUniqueCols = 0;
   else
   {
      BExtNnz = BExtRowLengs[BExtNRows*2];
      extColList     = new int[BExtNnz];
      extColListAux  = new int[BExtNnz];
      extDiagListAux = new int[BExtNRows];
      for ( ir = 0; ir < BExtNnz; ir++ ) extColList[ir] = BExtCols[ir];
      for ( ir = 0; ir < BExtNnz; ir++ ) extColListAux[ir] = -1;
      mergeSortNList  = BExtNRows * 2 + 1;
      mergeSortList2D = new int*[mergeSortNList];
      mergeSortAuxs   = new int*[mergeSortNList];
      mergeSortLengs  = new int[mergeSortNList];
      for ( is = 0; is < BExtNRows*2; is++ )
      {
         if ( is % 2 == 0 )
         {
            mergeSortList2D[is] = &(extColList[BExtRowLengs[is]+1]);
            mergeSortAuxs[is]   = &(extColListAux[BExtRowLengs[is]+1]);
            mergeSortLengs[is]  = BExtRowLengs[is+1] - BExtRowLengs[is] - 1;
         }
         else
         {
            mergeSortList2D[is] = &(extColList[BExtRowLengs[is]]);
            mergeSortAuxs[is]   = &(extColListAux[BExtRowLengs[is]]);
            mergeSortLengs[is]  = BExtRowLengs[is+1] - BExtRowLengs[is];
         }
      }
      for ( ir = 0; ir < BExtNRows; ir++ )
         extDiagListAux[ir] = extColListAux[BExtRowLengs[ir*2]];
      mergeSortList2D[BExtNRows*2] = diagCols;
      mergeSortAuxs[BExtNRows*2]   = extDiagListAux;
      mergeSortLengs[BExtNRows*2]  = BExtNRows;
      MLI_Utils_IntMergeSort(mergeSortNList, mergeSortLengs,
                mergeSortList2D, mergeSortAuxs, &BExtNumUniqueCols,
                &mergeSortList);

      for ( ir = 0; ir < BExtNRows; ir++ )
         extColListAux[BExtRowLengs[ir*2]] = extDiagListAux[ir];
      delete [] mergeSortList2D;
      delete [] mergeSortAuxs;
      delete [] mergeSortLengs;
      delete [] extDiagListAux;
      delete [] extColList;
      delete [] diagCols;
      if ( BExtNumUniqueCols > 0 ) extColList = new int[BExtNumUniqueCols];
      else                         extColList = NULL;
      for ( ir = 0; ir < BExtNumUniqueCols; ir++ )
         extColList[ir] = mergeSortList[ir];
      free( mergeSortList );
   }

   /* -----------------------------------------------------------------------
    * Next prune the internal columns (to my proc) from this list (by setting
    * the colum index to its ones-complement), since they have already been
    * included elsewhere
    * ----------------------------------------------------------------------*/

   BColStarts = hypre_ParCSRMatrixColStarts(hypreB);
   BStartCol  = BColStarts[mypid];
   BEndCol    = BColStarts[mypid+1] - 1;
   for ( ir = 0; ir < BExtNumUniqueCols; ir++ )
   {
      if ( extColList[ir] >= BStartCol && extColList[ir] <= BEndCol )
         extColList[ir] = - (extColList[ir] - BStartCol) - 1;
   }

   /* -----------------------------------------------------------------------
    * Next prune the external columns by eliminating all columns already
    * present in the BColMap list, which is assumed ordered
    * ----------------------------------------------------------------------*/

   BOffd      = hypre_ParCSRMatrixOffd(hypreB);
   BColMap    = hypre_ParCSRMatrixColMapOffd(hypreB);
   BColMapInd = 0;
   BNCols     = BColStarts[mypid+1] - BColStarts[mypid];
   for ( ir = 0; ir < BExtNumUniqueCols; ir++ )
   {
      if ( extColList[ir] >= 0 )
      {
         while (BColMapInd<BExtNRows && BColMap[BColMapInd]<extColList[ir])
            BColMapInd++;
         if (BColMapInd<BExtNRows && extColList[ir]==BColMap[BColMapInd])
         {
            extColList[ir] = - (BColMapInd + BNCols) - 1;
            BColMapInd++;
         }
      }
   }

   /* -----------------------------------------------------------------------
    * Compute the number of columns in the extended B local matrix
    * (CExtNCols as a sum of BNCols, BExtNRows, and BExtNumUniqueCols)
    * ----------------------------------------------------------------------*/

   if ( BExtNumUniqueCols > 0 ) iArray  = new int[BExtNumUniqueCols];
   tempCnt = 0;
   for ( ir = 0; ir < BExtNumUniqueCols; ir++ )
   {
      if ( extColList[ir] >= 0 ) iArray[ir] = tempCnt++;
      else                       iArray[ir] = -1;
   }
   for ( ir = 0; ir < BExtNnz; ir++ )
   {
      index = extColListAux[ir];
      iTemp = extColList[index];
      if ( iTemp < 0 ) BExtCols[ir] = - iTemp - 1;
      else             BExtCols[ir] = iArray[index] + BNCols + BExtNRows;
   }
   if ( BExtNumUniqueCols > 0 ) delete [] iArray;
   tempCnt = BExtNumUniqueCols;
   BExtNumUniqueCols = 0;
   for ( ir = 0; ir < tempCnt; ir++ )
   {
      if ( extColList[ir] >= 0 )
         extColList[BExtNumUniqueCols++] = extColList[ir];
   }
   if ( BExtNRows > 0 ) delete [] extColListAux;
   CExtNCols = BNCols + BExtNRows + BExtNumUniqueCols;

   /* -----------------------------------------------------------------------
    * fetch information about matrix A and B
    * ----------------------------------------------------------------------*/

   if (!hypre_ParCSRMatrixCommPkg(hypreA)) hypre_MatvecCommPkgCreate(hypreA);
   ADiag      = hypre_ParCSRMatrixDiag(hypreA);
   ADiagIA    = hypre_CSRMatrixI(ADiag);
   ADiagJA    = hypre_CSRMatrixJ(ADiag);
   ADiagAA    = hypre_CSRMatrixData(ADiag);
   AOffd      = hypre_ParCSRMatrixOffd(hypreA);
   AOffdIA    = hypre_CSRMatrixI(AOffd);
   AOffdJA    = hypre_CSRMatrixJ(AOffd);
   AOffdAA    = hypre_CSRMatrixData(AOffd);
   ARowStarts = hypre_ParCSRMatrixRowStarts(hypreA);
   AColStarts = hypre_ParCSRMatrixColStarts(hypreA);
   ANRows     = ARowStarts[mypid+1] - ARowStarts[mypid];
   ANCols     = AColStarts[mypid+1] - AColStarts[mypid];
   if (!hypre_ParCSRMatrixCommPkg(hypreB)) hypre_MatvecCommPkgCreate(hypreB);
   BDiag      = hypre_ParCSRMatrixDiag(hypreB);
   BDiagIA    = hypre_CSRMatrixI(BDiag);
   BDiagJA    = hypre_CSRMatrixJ(BDiag);
   BDiagAA    = hypre_CSRMatrixData(BDiag);
   BOffd      = hypre_ParCSRMatrixOffd(hypreB);
   BOffdIA    = hypre_CSRMatrixI(BOffd);
   BOffdJA    = hypre_CSRMatrixJ(BOffd);
   BOffdAA    = hypre_CSRMatrixData(BOffd);
   BRowStarts = hypre_ParCSRMatrixRowStarts(hypreB);
   BNRows     = BRowStarts[mypid+1] - BRowStarts[mypid];

   /* -----------------------------------------------------------------------
    * matrix matrix multiply - first compute sizes of each row in C
    * ----------------------------------------------------------------------*/

   CNRows   = ANRows;
   CNCols   = BNCols;
   CDiagNnz = COffdNnz = 0;
   CDiagReg = new int[CNRows];
   if ( CExtNCols > 0 ) COffdReg = new int[CExtNCols];
   for ( ib = 0; ib < CNRows; ib++ ) CDiagReg[ib] = -1;
   for ( ib = 0; ib < CExtNCols; ib++ ) COffdReg[ib] = -1;
   for ( ia = 0; ia < ANRows; ia++ )
   {
      for ( ia2 = ADiagIA[ia]; ia2 < ADiagIA[ia+1]; ia2++ )
      {
         colIndA = ADiagJA[ia2];
         if ( colIndA < BNRows )
         {
            for ( ib = BDiagIA[colIndA]; ib < BDiagIA[colIndA+1]; ib++ )
            {
               colIndB = BDiagJA[ib];
               if ( CDiagReg[colIndB] != ia )
               {
                  CDiagReg[colIndB] = ia;
                  CDiagNnz++;
               }
            }
            for ( ib = BOffdIA[colIndA]; ib < BOffdIA[colIndA+1]; ib++ )
            {
               colIndB = BOffdJA[ib] + BNCols;
               if ( COffdReg[colIndB] != ia )
               {
                  COffdReg[colIndB] = ia;
                  COffdNnz++;
               }
            }
         }
         else
         {
            index = colIndA - BNRows;
            for (ib=BExtRowLengs[2*index]; ib<BExtRowLengs[2*(index+1)]; ib++)
            {
               colIndB = BExtCols[ib];
               if ( colIndB < CNCols )
               {
                  if ( CDiagReg[colIndB] != ia ) CDiagNnz++;
               }
               else
               {
                  if ( CDiagReg[colIndB] != ia ) COffdNnz++;
               }
            }
         }
      }
      for ( ia2 = AOffdIA[ia]; ia2 < AOffdIA[ia+1]; ia2++ )
      {
         colIndA = AOffdJA[ia2] + ANCols;
         if ( colIndA < BNRows )
         {
            for ( ib = BDiagIA[colIndA]; ib < BDiagIA[colIndA+1]; ib++ )
            {
               colIndB = BDiagJA[ib];
               if ( CDiagReg[colIndB] != ia )
               {
                  CDiagReg[colIndB] = ia;
                  CDiagNnz++;
               }
            }
            for ( ib = BOffdIA[colIndA]; ib < BOffdIA[colIndA+1]; ib++ )
            {
               colIndB = BOffdJA[ib] + BNCols;
               if ( COffdReg[colIndB] != ia )
               {
                  COffdReg[colIndB] = ia;
                  COffdNnz++;
               }
            }
         }
         else
         {
            index = colIndA - BNRows;
            for (ib=BExtRowLengs[2*index]; ib<BExtRowLengs[2*(index+1)]; ib++)
            {
               colIndB = BExtCols[ib];
               if ( colIndB < CNCols )
               {
                  if ( CDiagReg[colIndB] != ia ) CDiagNnz++;
               }
               else
               {
                  if ( COffdReg[colIndB] != ia ) COffdNnz++;
               }
            }
         }
      }
   }

   /* -----------------------------------------------------------------------
    * matrix matrix multiply - perform the actual multiplication
    * ----------------------------------------------------------------------*/

   CDiagIA = hypre_TAlloc(int,  (CNRows+1) , HYPRE_MEMORY_HOST);
   CDiagJA = hypre_TAlloc(int,  CDiagNnz , HYPRE_MEMORY_HOST);
   CDiagAA = hypre_TAlloc(double,  CDiagNnz , HYPRE_MEMORY_HOST);
   COffdIA = hypre_TAlloc(int,  (CNRows+1) , HYPRE_MEMORY_HOST);
   if ( COffdNnz > 0 )
   {
      COffdJA = hypre_TAlloc(int,  COffdNnz , HYPRE_MEMORY_HOST);
      COffdAA = hypre_TAlloc(double,  COffdNnz , HYPRE_MEMORY_HOST);
   }
   else
   {
      COffdJA = NULL;
      COffdAA = NULL;
   }
   COffdNCols = CExtNCols - CNCols;
   CDiagNnz = COffdNnz = 0;
   for ( ib = 0; ib < CNRows; ib++ ) CDiagReg[ib] = -1;
   for ( ib = 0; ib < CExtNCols; ib++ ) COffdReg[ib] = -1;
   CColMap = NULL;
   if (COffdNCols > 0) CColMap = hypre_TAlloc(int, COffdNCols , HYPRE_MEMORY_HOST);
   for ( ia = 0; ia < BExtNRows; ia++ ) CColMap[ia] = BColMap[ia];
   for ( ia = BExtNRows; ia < COffdNCols; ia++ )
      CColMap[ia] = extColList[ia-BExtNRows];
   if ( COffdNCols > 0 ) CColMapAux = new int[COffdNCols];
   for ( ia = 0; ia < COffdNCols; ia++ ) CColMapAux[ia] = ia;
   MLI_Utils_IntQSort2(CColMap, CColMapAux, 0, COffdNCols-1);
   iArray = CColMapAux;
   if ( COffdNCols > 0 ) CColMapAux = new int[COffdNCols];
   for ( ia = 0; ia < COffdNCols; ia++ )
      CColMapAux[iArray[ia]] = ia;
   if ( COffdNCols > 0 ) delete [] iArray;

   CDiagIA[0] = COffdIA[0] = 0;
   for ( ia = 0; ia < ANRows; ia++ )
   {
      iTempDiag = CDiagNnz;
      iTempOffd = COffdNnz;
      for ( ia2 = ADiagIA[ia]; ia2 < ADiagIA[ia+1]; ia2++ )
      {
         colIndA = ADiagJA[ia2];
         dTempA  = ADiagAA[ia2];
         if ( colIndA < BNRows )
         {
            for ( ib = BDiagIA[colIndA]; ib < BDiagIA[colIndA+1]; ib++ )
            {
               colIndB = BDiagJA[ib];
               dTempB  = BDiagAA[ib];
               offset  = CDiagReg[colIndB];
               if ( offset < iTempDiag )
               {
                  CDiagReg[colIndB] = CDiagNnz;
                  CDiagJA[CDiagNnz] = colIndB;
                  CDiagAA[CDiagNnz++] = dTempA * dTempB;
               }
               else CDiagAA[offset] += dTempA * dTempB;
            }
            for ( ib = BOffdIA[colIndA]; ib < BOffdIA[colIndA+1]; ib++ )
            {
               colIndB = BOffdJA[ib] + BNCols;
               dTempB  = BOffdAA[ib];
               offset  = COffdReg[colIndB];
               if ( offset < iTempOffd )
               {
                  COffdReg[colIndB] = COffdNnz;
                  COffdJA[COffdNnz] = CColMapAux[colIndB-BNCols];
                  COffdAA[COffdNnz++] = dTempA * dTempB;
               }
               else COffdAA[offset] += dTempA * dTempB;
            }
         }
         else
         {
            index = colIndA - BNRows;
            for (ib=BExtRowLengs[2*index];ib<BExtRowLengs[2*(index+1)];ib++)
            {
               colIndB = BExtCols[ib];
               dTempB  = BOffdAA[ib];
               if ( colIndB < CNCols )
               {
                  offset  = CDiagReg[colIndB];
                  if ( offset < iTempDiag )
                  {
                     CDiagReg[colIndB] = CDiagNnz;
                     CDiagJA[CDiagNnz] = colIndB;
                     CDiagAA[CDiagNnz++] = dTempA * dTempB;
                  }
                  else CDiagAA[offset] += dTempA * dTempB;
               }
               else
               {
                  offset = COffdReg[colIndB];
                  if ( offset < iTempOffd )
                  {
                     COffdReg[colIndB] = COffdNnz;
                     COffdJA[COffdNnz] = CColMapAux[colIndB-CNCols];
                     COffdAA[COffdNnz++] = dTempA * dTempB;
                  }
                  else COffdAA[offset] += dTempA * dTempB;
               }
            }
         }
      }
      for ( ia2 = AOffdIA[ia]; ia2 < AOffdIA[ia+1]; ia2++ )
      {
         colIndA = AOffdJA[ia2] + ANCols;
         dTempA  = AOffdAA[ia2];
         if ( colIndA < BNRows )
         {
            for ( ib = BDiagIA[colIndA]; ib < BDiagIA[colIndA+1]; ib++ )
            {
               colIndB = BDiagJA[ib];
               dTempB  = BDiagAA[ib];
               offset  = CDiagReg[colIndB];
               if ( offset < iTempDiag )
               {
                  CDiagReg[colIndB]   = CDiagNnz;
                  CDiagJA[CDiagNnz]   = colIndB;
                  CDiagAA[CDiagNnz++] = dTempA * dTempB;
               }
               else CDiagAA[offset] += dTempA * dTempB;
            }
            for ( ib = BOffdIA[colIndA]; ib < BOffdIA[colIndA+1]; ib++ )
            {
               colIndB = BOffdJA[ib] + BNCols;
               dTempB  = BOffdAA[ib];
               offset  = COffdReg[colIndB];
               if ( offset < iTemp )
               {
                  COffdReg[colIndB]   = COffdNnz;
                  COffdJA[COffdNnz]   = CColMapAux[colIndB-BNCols];
                  COffdAA[COffdNnz++] = dTempA * dTempB;
               }
               else COffdAA[offset] += dTempA * dTempB;
            }
         }
         else
         {
            index = colIndA - BNRows;
            for (ib=BExtRowLengs[2*index];ib<BExtRowLengs[2*(index+1)];ib++)
            {
               colIndB = BExtCols[ib];
               dTempB  = BExtVals[ib];
               if ( colIndB < CNCols )
               {
                  offset  = CDiagReg[colIndB];
                  if ( offset < iTempDiag )
                  {
                     CDiagReg[colIndB]   = CDiagNnz;
                     CDiagJA[CDiagNnz]   = colIndB;
                     CDiagAA[CDiagNnz++] = dTempA * dTempB;
                  }
                  else CDiagAA[offset] += dTempA * dTempB;
               }
               else
               {
                  offset  = COffdReg[colIndB];
                  if ( offset < iTempOffd )
                  {
                     COffdReg[colIndB]   = COffdNnz;
                     COffdJA[COffdNnz]   = CColMapAux[colIndB-CNCols];
                     COffdAA[COffdNnz++] = dTempA * dTempB;
                  }
                  else COffdAA[offset] += dTempA * dTempB;
               }
            }
         }
      }
      CDiagIA[ia+1] = CDiagNnz;
      COffdIA[ia+1] = COffdNnz;
   }
   if ( CNRows    > 0 )  delete [] CDiagReg;
   if ( CExtNCols > 0 )  delete [] COffdReg;
   if ( COffdNCols > 0 ) delete [] CColMapAux;

   /* -----------------------------------------------------------------------
    * move the diagonal entry to the beginning of the row
    * ----------------------------------------------------------------------*/

   for ( ia = 0; ia < CNRows; ia++ )
   {
      iTemp = -1;
      for ( ia2 = CDiagIA[ia]; ia2 < CDiagIA[ia+1]; ia2++ )
      {
         if ( CDiagJA[ia2] == ia )
         {
            iTemp = CDiagJA[ia2];
            dTemp = CDiagAA[ia2];
            break;
         }
      }
      if ( iTemp >= 0 )
      {
         for ( ib = ia2; ib > CDiagIA[ia]; ib-- )
         {
            CDiagJA[ib] = CDiagJA[ib-1];
            CDiagAA[ib] = CDiagAA[ib-1];
         }
         CDiagJA[CDiagIA[ia]] = iTemp;
         CDiagAA[CDiagIA[ia]] = dTemp;
      }
   }

   /* -----------------------------------------------------------------------
    * finally form HYPRE_ParCSRMatrix for the product
    * ----------------------------------------------------------------------*/

#if 0
   if ( mypid == 1 )
   {
      for ( ia = 0; ia < CNRows; ia++ )
      {
         for ( ia2 = CDiagIA[ia]; ia2 < CDiagIA[ia+1]; ia2++ )
            printf("%d : CDiag %5d = %5d %e\n",mypid,ia,CDiagJA[ia2],
                   CDiagAA[ia2]);
         for ( ia2 = COffdIA[ia]; ia2 < COffdIA[ia+1]; ia2++ )
            printf("%d : COffd %5d = %5d %e\n",mypid,ia,COffdJA[ia2],
                   COffdAA[ia2]);
      }
   }
#endif

   CRowStarts = hypre_TAlloc(int,  (nprocs+1) , HYPRE_MEMORY_HOST);
   CColStarts = hypre_TAlloc(int,  (nprocs+1) , HYPRE_MEMORY_HOST);
   for ( ia = 0; ia <= nprocs; ia++ ) CRowStarts[ia] = ARowStarts[ia];
   for ( ia = 0; ia <= nprocs; ia++ ) CColStarts[ia] = BColStarts[ia];
   hypreC = hypre_ParCSRMatrixCreate(mpiComm, CNRows, CNCols, CRowStarts,
                   CColStarts, COffdNCols, CDiagNnz, COffdNnz);
   CDiag = hypre_ParCSRMatrixDiag(hypreC);
   hypre_CSRMatrixData(CDiag) = CDiagAA;
   hypre_CSRMatrixI(CDiag) = CDiagIA;
   hypre_CSRMatrixJ(CDiag) = CDiagJA;

   COffd = hypre_ParCSRMatrixOffd(hypreC);
   hypre_CSRMatrixI(COffd) = COffdIA;
   if ( COffdNCols > 0 )
   {
      hypre_CSRMatrixJ(COffd) = COffdJA;
      hypre_CSRMatrixData(COffd) = COffdAA;
      hypre_ParCSRMatrixColMapOffd(hypreC) = CColMap;
   }

   sprintf(paramString, "HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   (*Cmat) = new MLI_Matrix((void *) hypreC, paramString, funcPtr);
   delete funcPtr;
}

/***************************************************************************
 * get the external rows of B in order to multiply A * B
 * (modified so that extRowLengs has 2 numbers for each row, one for
 *  the diagonal part, and the other for the off-diagonal part.  This is
 *  done to optimize the code in order each part is sorted.)
 *--------------------------------------------------------------------------*/

void MLI_Matrix_GetExtRows( MLI_Matrix *Amat, MLI_Matrix *Bmat, int *extNRowsP,
                       int **extRowLengsP, int **extColsP, double **extValsP)
{
   hypre_ParCSRMatrix  *hypreA, *hypreB;
   hypre_ParCSRCommPkg *commPkg;
   hypre_CSRMatrix     *BDiag, *BOffd;
   int                 nprocs, mypid, nSends, *sendProcs, *sendStarts;
   int                 *sendMap, nRecvs, *recvProcs, *recvStarts;
   int                 ip, jp, kp, rowIndex, length;
   int                 recvNRows, sendNRows, totalSendNnz, totalRecvNnz;
   int                 curNnz, sendIndex, proc;
   int                 ir, offset, upper, requestCnt, *recvCols, *iSendBuf;
   int                 *BDiagIA, *BOffdIA, *BDiagJA, *BOffdJA, *colMapOffd;
#if 0
   int                 *BRowStarts;
#endif
   int                 *BColStarts;
   int                 *recvRowLengs, BStartCol;
   double              *BDiagAA, *BOffdAA, *recvVals, *dSendBuf;
   MPI_Request         *requests;
   MPI_Status          *statuses;
   MPI_Comm            mpiComm;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Matrix_GetExtRows begins... \n");
#endif

   /* -----------------------------------------------------------------------
    * fetch HYPRE matrices and machine information
    * ----------------------------------------------------------------------*/

   hypreA     = (hypre_ParCSRMatrix *) Amat->getMatrix();
   hypreB     = (hypre_ParCSRMatrix *) Bmat->getMatrix();
   mpiComm    = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_size(mpiComm, &nprocs);
   MPI_Comm_rank(mpiComm, &mypid);
#if 0
   BRowStarts = hypre_ParCSRMatrixRowStarts(hypreB);
#endif
   BColStarts = hypre_ParCSRMatrixColStarts(hypreB);
   BStartCol  = BColStarts[mypid];
   if ( nprocs == 1 )
   {
      (*extRowLengsP) = NULL;
      (*extColsP)     = NULL;
      (*extValsP)     = NULL;
      (*extNRowsP)    = 0;
      return;
   }

   /* -----------------------------------------------------------------------
    * fetch communication information
    * ----------------------------------------------------------------------*/

   commPkg = hypre_ParCSRMatrixCommPkg(hypreA);
   if ( !commPkg ) hypre_MatvecCommPkgCreate(hypreA);
   commPkg    = hypre_ParCSRMatrixCommPkg(hypreA);
   nSends     = hypre_ParCSRCommPkgNumSends(commPkg);
   sendProcs  = hypre_ParCSRCommPkgSendProcs(commPkg);
   sendStarts = hypre_ParCSRCommPkgSendMapStarts(commPkg);
   sendMap    = hypre_ParCSRCommPkgSendMapElmts(commPkg);
   nRecvs     = hypre_ParCSRCommPkgNumRecvs(commPkg);
   recvProcs  = hypre_ParCSRCommPkgRecvProcs(commPkg);
   recvStarts = hypre_ParCSRCommPkgRecvVecStarts(commPkg);
   recvNRows  = recvStarts[nRecvs];
   sendNRows  = sendStarts[nSends];
   if ( nRecvs + nSends > 0 ) requests = new MPI_Request[nRecvs+nSends];

   /* -----------------------------------------------------------------------
    * fetch the local B matrix
    * ----------------------------------------------------------------------*/

   colMapOffd = hypre_ParCSRMatrixColMapOffd(hypreB);
   BDiag   = hypre_ParCSRMatrixDiag(hypreB);
   BDiagIA = hypre_CSRMatrixI(BDiag);
   BDiagJA = hypre_CSRMatrixJ(BDiag);
   BDiagAA = hypre_CSRMatrixData(BDiag);
   BOffd   = hypre_ParCSRMatrixOffd(hypreB);
   BOffdIA = hypre_CSRMatrixI(BOffd);
   BOffdJA = hypre_CSRMatrixJ(BOffd);
   BOffdAA = hypre_CSRMatrixData(BOffd);

   /* -----------------------------------------------------------------------
    * construct external row lengths (recvRowLengs)
    * ----------------------------------------------------------------------*/

   if ( recvNRows > 0 ) recvRowLengs = new int[2*recvNRows+1];
   else                 recvRowLengs = NULL;
   requestCnt = 0;
   for ( ip = 0; ip < nRecvs; ip++ )
   {
      proc   = recvProcs[ip];
      offset = recvStarts[ip];
      length = recvStarts[ip+1] - offset;
      MPI_Irecv(&(recvRowLengs[offset*2]), length*2, MPI_INT, proc, 27027,
                mpiComm, &requests[requestCnt++]);
   }
   if ( sendNRows > 0 ) iSendBuf = new int[sendNRows*2];
   sendIndex = totalSendNnz = 0;
   for ( ip = 0; ip < nSends; ip++ )
   {
      proc   = sendProcs[ip];
      offset = sendStarts[ip];
      upper  = sendStarts[ip+1];
      length = upper - offset;
      for ( jp = offset; jp < upper; jp++ )
      {
         //rowIndex = sendMap[jp] + BStartRow;
         //hypre_ParCSRMatrixGetRow(hypreB,rowIndex,&rowLeng,NULL,NULL);
         rowIndex = sendMap[jp];
         iSendBuf[sendIndex++] = BDiagIA[rowIndex+1] - BDiagIA[rowIndex];
         iSendBuf[sendIndex++] = BOffdIA[rowIndex+1] - BOffdIA[rowIndex];
         totalSendNnz += iSendBuf[sendIndex-1] + iSendBuf[sendIndex-2];
         //hypre_ParCSRMatrixRestoreRow(hypreB,rowIndex,&rowLeng,NULL,NULL);
      }
      MPI_Isend(&iSendBuf[offset*2], length*2, MPI_INT, proc, 27027, mpiComm,
                &requests[requestCnt++]);
   }
   statuses = new MPI_Status[requestCnt];
   MPI_Waitall( requestCnt, requests, statuses );
   if ( sendNRows > 0 ) delete [] iSendBuf;

   /*-----------------------------------------------------------------
    * construct offCols
    *-----------------------------------------------------------------*/

   totalRecvNnz = 0;
   for ( ir = 0; ir < recvNRows*2; ir++) totalRecvNnz += recvRowLengs[ir];
   if ( totalRecvNnz > 0 )
   {
      recvCols = new int[totalRecvNnz];
      recvVals = new double[totalRecvNnz];
   }
   requestCnt = totalRecvNnz = 0;
   for ( ip = 0; ip < nRecvs; ip++ )
   {
      proc   = recvProcs[ip];
      offset = recvStarts[ip];
      length = recvStarts[ip+1] - offset;
      curNnz = 0;
      for ( jp = 0; jp < length*2; jp++ ) curNnz += recvRowLengs[offset*2+jp];
      MPI_Irecv(&recvCols[totalRecvNnz], curNnz, MPI_INT, proc, 27028, mpiComm,
                &requests[requestCnt++]);
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) iSendBuf = new int[totalSendNnz];
   sendIndex = totalSendNnz = 0;
   for ( ip = 0; ip < nSends; ip++ )
   {
      proc   = sendProcs[ip];
      offset = sendStarts[ip];
      upper  = sendStarts[ip+1];
      length = upper - offset;
      curNnz = totalSendNnz;
      for ( jp = offset; jp < upper; jp++ )
      {
         //rowIndex = sendMap[jp] + BStartRow;
         //hypre_ParCSRMatrixGetRow(hypreB,rowIndex,&rowLeng,&cols,NULL);
         //for ( kp = 0;  kp < rowLeng;  kp++ )
         //   iSendBuf[curNnz++] = cols[kp];
         //hypre_ParCSRMatrixRestoreRow(hypreB,rowIndex,&rowLeng,&cols,NULL);
         rowIndex = sendMap[jp];
         for ( kp = BDiagIA[rowIndex];  kp < BDiagIA[rowIndex+1]; kp++ )
            iSendBuf[curNnz++] = BDiagJA[kp] + BStartCol;
         for ( kp = BOffdIA[rowIndex];  kp < BOffdIA[rowIndex+1]; kp++ )
            iSendBuf[curNnz++] = colMapOffd[BOffdJA[kp]];
      }
      curNnz -= totalSendNnz;
      MPI_Isend(&iSendBuf[totalSendNnz], curNnz, MPI_INT, proc, 27028, mpiComm,
                &requests[requestCnt++]);
      totalSendNnz += curNnz;
   }
   MPI_Waitall( requestCnt, requests, statuses );
   if ( totalSendNnz > 0 ) delete [] iSendBuf;

   /*-----------------------------------------------------------------
    * construct offVals
    *-----------------------------------------------------------------*/

   requestCnt = totalRecvNnz = 0;
   for ( ip = 0; ip < nRecvs; ip++ )
   {
      proc   = recvProcs[ip];
      offset = recvStarts[ip];
      length = recvStarts[ip+1] - offset;
      curNnz = 0;
      for (jp = 0; jp < length*2; jp++) curNnz += recvRowLengs[offset*2+jp];
      MPI_Irecv(&recvVals[totalRecvNnz], curNnz, MPI_DOUBLE, proc, 27029,
                mpiComm, &requests[requestCnt++]);
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) dSendBuf = new double[totalSendNnz];
   sendIndex = totalSendNnz = 0;
   for ( ip = 0; ip < nSends; ip++ )
   {
      proc   = sendProcs[ip];
      offset = sendStarts[ip];
      upper  = sendStarts[ip+1];
      length = upper - offset;
      curNnz = totalSendNnz;
      for ( jp = offset; jp < upper; jp++ )
      {
         //rowIndex = sendMap[jp] + BStartRow;
         //hypre_ParCSRMatrixGetRow(hypreB,rowIndex,&rowLeng,NULL,&vals);
         //for ( kp = 0;  kp < rowLeng;  kp++ )
         //   dSendBuf[curNnz++] = vals[kp];
         //hypre_ParCSRMatrixRestoreRow(hypreB,rowIndex,&rowLeng,NULL,&vals);
         rowIndex = sendMap[jp];
         for ( kp = BDiagIA[rowIndex];  kp < BDiagIA[rowIndex+1]; kp++ )
            dSendBuf[curNnz++] = BDiagAA[kp];
         for ( kp = BOffdIA[rowIndex];  kp < BOffdIA[rowIndex+1]; kp++ )
            dSendBuf[curNnz++] = BOffdAA[kp];
      }
      curNnz -= totalSendNnz;
      MPI_Isend(&(dSendBuf[totalSendNnz]), curNnz, MPI_DOUBLE, proc, 27029,
                mpiComm, &requests[requestCnt++]);
      totalSendNnz += curNnz;
   }
   MPI_Waitall( requestCnt, requests, statuses );
   if ( totalSendNnz > 0 ) delete [] dSendBuf;
   if ( nRecvs + nSends > 0 ) delete [] requests;
   if ( nRecvs + nSends > 0 ) delete [] statuses;

   /* -----------------------------------------------------------------------
    * diagnostics
    * ----------------------------------------------------------------------*/

   (*extRowLengsP) = recvRowLengs;
   (*extColsP)     = recvCols;
   (*extValsP)     = recvVals;
   (*extNRowsP)    = recvNRows;

#if 0
   BRowStarts = hypre_ParCSRMatrixRowStarts(hypreB);
   totalRecvNnz = 0;
   if ( mypid == 0 )
   {
      for ( ip = 0; ip < nprocs; ip++ )
         printf("processor has range %d %d\n",BRowStarts[ip],BRowStarts[ip+1]);
      for ( ip = 0; ip < nprocs; ip++ )
      {
         offset = recvStarts[ip];
         length = recvStarts[ip+1] - offset;
         curNnz = 0;
         for (jp = 0; jp < length*2; jp++) curNnz += recvRowLengs[offset*2+jp];
         for (jp = 0; jp < curNnz; jp++)
         {
            printf("%d : recvData = %5d %e\n", mypid, recvCols[totalRecvNnz],
                   recvVals[totalRecvNnz]);
            totalRecvNnz++;
         }
      }
   }
#endif
}

