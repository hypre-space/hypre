/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "seq_mv/seq_mv.h"

#include "HYPRE_FEI.h"

extern void hypre_qsort0(int*, int, int);
extern void hypre_qsort1(int*, double*, int, int);

#define habs(x) ((x) > 0.0 ? x : -(x))

/***************************************************************************/
/* reading a matrix from a file in ija format (first row : nrows, nnz)     */
/* (read by a single processor)                                            */
/*-------------------------------------------------------------------------*/

void HYPRE_LSI_Get_IJAMatrixFromFile(double **val, int **ia,
     int **ja, int *N, double **rhs, char *matfile, char *rhsfile)
{
    int    i, j, Nrows, nnz, icount, rowindex, colindex, curr_row;
    int    k, m, *mat_ia, *mat_ja, ncnt, rnum;
    double dtemp, *mat_a, value, *rhs_local;
    FILE   *fp;

    /*------------------------------------------------------------------*/
    /* read matrix file                                                 */
    /*------------------------------------------------------------------*/

    printf("Reading matrix file = %s \n", matfile );
    fp = fopen( matfile, "r" );
    if ( fp == NULL ) {
       printf("Error : file open error (filename=%s).\n", matfile);
       exit(1);
    }
    fscanf(fp, "%d %d", &Nrows, &nnz);
    if ( Nrows <= 0 || nnz <= 0 ) {
       printf("Error : nrows,nnz = %d %d\n", Nrows, nnz);
       exit(1);
    }
    mat_ia = hypre_TAlloc(int, (Nrows+1) , HYPRE_MEMORY_HOST);
    mat_ja = hypre_TAlloc(int,  nnz , HYPRE_MEMORY_HOST);
    mat_a  = hypre_TAlloc(double,  nnz , HYPRE_MEMORY_HOST);
    mat_ia[0] = 0;

    curr_row = 0;
    icount   = 0;
    for ( i = 0; i < nnz; i++ ) {
       fscanf(fp, "%d %d %lg", &rowindex, &colindex, &value);
       rowindex--;
       colindex--;
       if ( rowindex != curr_row ) mat_ia[++curr_row] = icount;
       if ( rowindex < 0 || rowindex >= Nrows )
          printf("Error reading row %d (curr_row = %d)\n", rowindex, curr_row);
       if ( colindex < 0 || colindex >= Nrows )
          printf("Error reading col %d (rowindex = %d)\n", colindex, rowindex);
         /*if ( value != 0.0 ) {*/
          mat_ja[icount] = colindex;
          mat_a[icount++]  = value;
         /*}*/
    }
    fclose(fp);
    for ( i = curr_row+1; i <= Nrows; i++ ) mat_ia[i] = icount;
    (*val) = mat_a;
    (*ia)  = mat_ia;
    (*ja)  = mat_ja;
    (*N) = Nrows;
    printf("matrix has %6d rows and %7d nonzeros\n", Nrows, mat_ia[Nrows]);

    /*------------------------------------------------------------------*/
    /* read rhs file                                                    */
    /*------------------------------------------------------------------*/

    printf("reading rhs file = %s \n", rhsfile );
    fp = fopen( rhsfile, "r" );
    if ( fp == NULL ) {
       printf("Error : file open error (filename=%s).\n", rhsfile);
       exit(1);
    }
    fscanf(fp, "%d", &ncnt);
    if ( ncnt <= 0 || ncnt != Nrows) {
       printf("Error : nrows = %d \n", ncnt);
       exit(1);
    }
    fflush(stdout);
    rhs_local  = hypre_TAlloc(double,  Nrows , HYPRE_MEMORY_HOST);
    m = 0;
    for ( k = 0; k < ncnt; k++ ) {
       fscanf(fp, "%d %lg", &rnum, &dtemp);
       rhs_local[rnum-1] = dtemp; m++;
    }
    fflush(stdout);
    ncnt = m;
    fclose(fp);
    (*rhs) = rhs_local;
    printf("reading rhs done \n");
    for ( i = 0; i < Nrows; i++ ) {
       for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
          mat_ja[j]++;
    }
    printf("returning from reading matrix\n");
}


/***************************************************************************/
/* HYPRE_LSI_Search - this is a modification of hypre_BinarySearch         */
/*-------------------------------------------------------------------------*/

int HYPRE_LSI_Search(int *list,int value,int list_length)
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

/* ************************************************************************ */
/* Given a sorted list of indices and the key, find the position of the     */
/* key in the list.  If not found, return the index of the position         */
/* corresponding to where it would have been stored.                        */
/* (borrowed from the search routine in ML)                                 */
/* ------------------------------------------------------------------------ */

int HYPRE_LSI_Search2(int key, int nlist, int *list)
{
   int  nfirst, nlast, nmid, found, index;

   if (nlist <= 0) return -1;
   nfirst = 0;
   nlast  = nlist-1;
   if (key > list[nlast])  return -(nlast+1);
   if (key < list[nfirst]) return -(nfirst+1);
   found = 0;
   while ((found == 0) && ((nlast-nfirst)>1)) {
      nmid = (nfirst + nlast) / 2;
      if (key == list[nmid])     {index  = nmid; found = 1;}
      else if (key > list[nmid])  nfirst = nmid;
      else                        nlast  = nmid;
   }
   if (found == 1)               return index;
   else if (key == list[nfirst]) return nfirst;
   else if (key == list[nlast])  return nlast;
   else                          return -(nfirst+1);
}

/* ************************************************************************ */
/* this function extracts the matrix in a CSR format                        */
/* ------------------------------------------------------------------------ */

int HYPRE_LSI_GetParCSRMatrix(HYPRE_IJMatrix Amat, int nrows, int nnz,
                              int *ia_ptr, int *ja_ptr, double *a_ptr)
{
    int                nz, i, j, ierr, rowSize, *colInd, nz_ptr, *colInd2;
    int                firstNnz;
    double             *colVal, *colVal2;
    HYPRE_ParCSRMatrix A_csr;

    nz        = 0;
    nz_ptr    = 0;
    ia_ptr[0] = nz_ptr;

    /* ---old_IJ----------------------------------------------------------- */
    /*A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(Amat);*/
    /* ---new_IJ----------------------------------------------------------- */
    HYPRE_IJMatrixGetObject(Amat, (void**) &A_csr);
    /* -------------------------------------------------------------------- */

    for ( i = 0; i < nrows; i++ )
    {
       ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       hypre_assert(!ierr);
       colInd2 = hypre_TAlloc(int, rowSize , HYPRE_MEMORY_HOST);
       colVal2 = hypre_TAlloc(double, rowSize , HYPRE_MEMORY_HOST);
       for ( j = 0; j < rowSize; j++ )
       {
          colInd2[j] = colInd[j];
          colVal2[j] = colVal[j];
       }
       hypre_qsort1(colInd2, colVal2, 0, rowSize-1);
       for ( j = 0; j < rowSize-1; j++ )
          if ( colInd2[j] == colInd2[j+1] )
             printf("HYPRE_LSI_GetParCSRMatrix-duplicate colind at row %d \n",i);

       firstNnz = 0;
       for ( j = 0; j < rowSize; j++ )
       {
          if ( colVal2[j] != 0.0 )
          {
             if (nz_ptr > 0 && firstNnz > 0 && colInd2[j] == ja_ptr[nz_ptr-1])
             {
                a_ptr[nz_ptr-1] += colVal2[j];
                printf("HYPRE_LSI_GetParCSRMatrix:: repeated col in row %d\n",i);
             }
             else
             {
                ja_ptr[nz_ptr] = colInd2[j];
                a_ptr[nz_ptr++]  = colVal2[j];
                if ( nz_ptr > nnz )
                {
                   printf("HYPRE_LSI_GetParCSRMatrix Error (1) - %d %d.\n",i,
                          nrows);
                   exit(1);
                }
                firstNnz++;
             }
          } else nz++;
       }
       hypre_TFree(colInd2, HYPRE_MEMORY_HOST);
       hypre_TFree(colVal2, HYPRE_MEMORY_HOST);
       ia_ptr[i+1] = nz_ptr;
       ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       hypre_assert(!ierr);
    }
    /*
    if ( nnz != nz_ptr )
    {
       printf("HYPRE_LSI_GetParCSRMatrix note : matrix sparsity has been \n");
       printf("      changed since matConfigure - %d > %d ?\n", nnz, nz_ptr);
       printf("      number of zeros            = %d \n", nz );
    }
    */
    return nz_ptr;
}

/* ******************************************************************** */
/* sort integers                                                        */
/* -------------------------------------------------------------------- */

void HYPRE_LSI_qsort1a( int *ilist, int *ilist2, int left, int right)
{
   int i, last, mid, itemp;

   if (left >= right) return;
   mid          = (left + right) / 2;
   itemp        = ilist[left];
   ilist[left]  = ilist[mid];
   ilist[mid]   = itemp;
   itemp        = ilist2[left];
   ilist2[left] = ilist2[mid];
   ilist2[mid]  = itemp;
   last         = left;
   for (i = left+1; i <= right; i++)
   {
      if (ilist[i] < ilist[left])
      {
         last++;
         itemp        = ilist[last];
         ilist[last]  = ilist[i];
         ilist[i]     = itemp;
         itemp        = ilist2[last];
         ilist2[last] = ilist2[i];
         ilist2[i]    = itemp;
      }
   }
   itemp        = ilist[left];
   ilist[left]  = ilist[last];
   ilist[last]  = itemp;
   itemp        = ilist2[left];
   ilist2[left] = ilist2[last];
   ilist2[last] = itemp;
   HYPRE_LSI_qsort1a(ilist, ilist2, left, last-1);
   HYPRE_LSI_qsort1a(ilist, ilist2, last+1, right);
}

/* ******************************************************************** */
/* sort a given list in increasing order                                */
/* -------------------------------------------------------------------- */

int HYPRE_LSI_SplitDSort2(double *dlist, int nlist, int *ilist, int limit)
{
   int    itemp, *iarray1, *iarray2, count1, count2, i;
   double dtemp, *darray1, *darray2;

   if ( nlist <= 1 ) return 0;
   if ( nlist == 2 )
   {
      if ( dlist[0] < dlist[1] )
      {
         dtemp = dlist[0]; dlist[0] = dlist[1]; dlist[1] = dtemp;
         itemp = ilist[0]; ilist[0] = ilist[1]; ilist[1] = itemp;
      }
      return 0;
   }
   count1 = 0;
   count2 = 0;
   iarray1 = hypre_TAlloc(int,  2 * nlist , HYPRE_MEMORY_HOST);
   iarray2 = iarray1 + nlist;
   darray1 = hypre_TAlloc(double,  2 * nlist , HYPRE_MEMORY_HOST);
   darray2 = darray1 + nlist;

   if ( darray2 == NULL )
   {
      printf("ERROR : malloc\n");
      exit(1);
   }
   dtemp  = dlist[0];
   itemp  = ilist[0];
   for ( i = 1; i < nlist; i++ )
   {
      if (dlist[i] >= dtemp  )
      {
         darray1[count1] = dlist[i];
         iarray1[count1++] = ilist[i];
      }
      else
      {
         darray2[count2] = dlist[i];
         iarray2[count2++] = ilist[i];
      }
   }
   dlist[count1] = dtemp;
   ilist[count1] = itemp;
   for ( i = 0; i < count1; i++ )
   {
      dlist[i] = darray1[i];
      ilist[i] = iarray1[i];
   }
   for ( i = 0; i < count2; i++ )
   {
      dlist[count1+1+i] = darray2[i];
      ilist[count1+1+i] = iarray2[i];
   }
   hypre_TFree(darray1, HYPRE_MEMORY_HOST);
   hypre_TFree(iarray1, HYPRE_MEMORY_HOST);
   if ( count1+1 == limit ) return 0;
   else if ( count1+1 < limit )
      HYPRE_LSI_SplitDSort2(&(dlist[count1+1]),count2,&(ilist[count1+1]),
                     limit-count1-1);
   else
      HYPRE_LSI_SplitDSort2( dlist, count1, ilist, limit );
   return 0;
}

/* ******************************************************************** */
/* sort a given list in increasing order                                */
/* -------------------------------------------------------------------- */

int HYPRE_LSI_SplitDSort(double *dlist, int nlist, int *ilist, int limit)
{
   int    i, first, last, itemp, cur_index;
   double dtemp, cur_val;

   if ( nlist <= 1 ) return 0;
   if ( nlist == 2 )
   {
      if ( dlist[0] < dlist[1] )
      {
         dtemp = dlist[0]; dlist[0] = dlist[1]; dlist[1] = dtemp;
         itemp = ilist[0]; ilist[0] = ilist[1]; ilist[1] = itemp;
      }
      return 0;
   }

   first = 0;
   last  = nlist - 1;

   do
   {
      cur_index = first;
      cur_val = dlist[cur_index];

      for ( i = first+1; i <= last; i++ )
      {
         if ( dlist[i] > cur_val )
         {
            cur_index++;
            itemp = ilist[cur_index];
            ilist[cur_index] = ilist[i];
            ilist[i] = itemp;
            dtemp = dlist[cur_index];
            dlist[cur_index] = dlist[i];
            dlist[i] = dtemp;
         }
      }
      itemp = ilist[cur_index];
      ilist[cur_index] = ilist[first];
      ilist[first] = itemp;
      dtemp = dlist[cur_index];
      dlist[cur_index] = dlist[first];
      dlist[first] = dtemp;

      if ( cur_index > limit ) last = cur_index - 1;
      else if ( cur_index < limit ) first = cur_index + 1;
   } while ( cur_index != limit );

   return 0;
}

/* ******************************************************************** */
/* copy from one vector to another (identity preconditioning)           */
/* -------------------------------------------------------------------- */

int HYPRE_LSI_SolveIdentity(HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                            HYPRE_ParVector b, HYPRE_ParVector x)
{
   (void) solver;
   (void) Amat;
   HYPRE_ParVectorCopy( b, x );
   return 0;
}

/* ******************************************************************** */
/* Cuthill McKee reordering algorithm                                   */
/* -------------------------------------------------------------------- */

int HYPRE_LSI_Cuthill(int n, int *ia, int *ja, double *aa, int *order_array,
                      int *reorder_array)
{
   int    nnz, *nz_array, cnt, i, j, *tag_array, *queue, nqueue, qhead;
   int    root, norder, mindeg, *ia2, *ja2;
   double *aa2;

   nz_array = hypre_TAlloc(int,  n , HYPRE_MEMORY_HOST);
   nnz      = ia[n];
   for ( i = 0; i < n; i++ ) nz_array[i] = ia[i+1] - ia[i];
   tag_array = hypre_TAlloc(int,  n , HYPRE_MEMORY_HOST);
   queue     = hypre_TAlloc(int,  n , HYPRE_MEMORY_HOST);
   for ( i = 0; i < n; i++ ) tag_array[i] = 0;
   norder = 0;
   mindeg = 10000000;
   root   = -1;
   for ( i = 0; i < n; i++ )
   {
      if ( nz_array[i] == 1 )
      {
         tag_array[i] = 1;
         order_array[norder++] = i;
         reorder_array[i] = norder-1;
      }
      else if ( nz_array[i] < mindeg )
      {
         mindeg = nz_array[i];
         root = i;
      }
   }
   if ( root == -1 )
   {
      printf("HYPRE_LSI_Cuthill ERROR : Amat is diagonal\n");
      exit(1);
   }
   nqueue = 0;
   queue[nqueue++] = root;
   qhead = 0;
   tag_array[root] = 1;
   while ( qhead < nqueue )
   {
      root = queue[qhead++];
      order_array[norder++] = root;
      reorder_array[root] = norder - 1;
      for ( j = ia[root]; j < ia[root+1]; j++ )
      {
         if ( tag_array[ja[j]] == 0 )
         {
            tag_array[ja[j]] = 1;
            queue[nqueue++] = ja[j];
         }
      }
      if ( qhead == nqueue && norder < n )
         for ( j = 0; j < n; j++ )
            if ( tag_array[j] == 0 ) queue[nqueue++] = j;
   }
   ia2 = hypre_TAlloc(int,  (n+1) , HYPRE_MEMORY_HOST);
   ja2 = hypre_TAlloc(int,  nnz , HYPRE_MEMORY_HOST);
   aa2 = hypre_TAlloc(double,  nnz , HYPRE_MEMORY_HOST);
   ia2[0] = 0;
   nnz = 0;
   for ( i = 0; i < n; i++ )
   {
      cnt = order_array[i];
      for ( j = ia[cnt]; j < ia[cnt+1]; j++ )
      {
         ja2[nnz] = ja[j];
         aa2[nnz++] = aa[j];
      }
      ia2[i+1] = nnz;
   }
   for ( i = 0; i < nnz; i++ ) ja[i] = reorder_array[ja2[i]];
   for ( i = 0; i < nnz; i++ ) aa[i] = aa2[i];
   for ( i = 0; i <= n; i++ )  ia[i] = ia2[i];
   hypre_TFree(ia2, HYPRE_MEMORY_HOST);
   hypre_TFree(ja2, HYPRE_MEMORY_HOST);
   hypre_TFree(aa2, HYPRE_MEMORY_HOST);
   hypre_TFree(nz_array, HYPRE_MEMORY_HOST);
   hypre_TFree(tag_array, HYPRE_MEMORY_HOST);
   hypre_TFree(queue, HYPRE_MEMORY_HOST);
   return 0;
}

/* ******************************************************************** */
/* matrix of a dense matrix                                             */
/* -------------------------------------------------------------------- */

int HYPRE_LSI_MatrixInverse( double **Amat, int ndim, double ***Cmat )
{
   int    i, j, k;
   double denom, **Bmat, dmax;

   (*Cmat) = NULL;
   if ( ndim == 1 )
   {
      if ( habs(Amat[0][0]) <= 1.0e-16 ) return -1;
      Bmat = hypre_TAlloc(double*,  ndim , HYPRE_MEMORY_HOST);
      for ( i = 0; i < ndim; i++ )
         Bmat[i] = hypre_TAlloc(double,  ndim , HYPRE_MEMORY_HOST);
      Bmat[0][0] = 1.0 / Amat[0][0];
      (*Cmat) = Bmat;
      return 0;
   }
   if ( ndim == 2 )
   {
      denom = Amat[0][0] * Amat[1][1] - Amat[0][1] * Amat[1][0];
      if ( habs( denom ) <= 1.0e-16 ) return -1;
      Bmat = hypre_TAlloc(double*,  ndim , HYPRE_MEMORY_HOST);
      for ( i = 0; i < ndim; i++ )
         Bmat[i] = hypre_TAlloc(double,  ndim , HYPRE_MEMORY_HOST);
      Bmat[0][0] = Amat[1][1] / denom;
      Bmat[1][1] = Amat[0][0] / denom;
      Bmat[0][1] = - ( Amat[0][1] / denom );
      Bmat[1][0] = - ( Amat[1][0] / denom );
      (*Cmat) = Bmat;
      return 0;
   }
   else
   {
      Bmat = hypre_TAlloc(double*,  ndim , HYPRE_MEMORY_HOST);
      for ( i = 0; i < ndim; i++ )
      {
         Bmat[i] = hypre_TAlloc(double,  ndim , HYPRE_MEMORY_HOST);
         for ( j = 0; j < ndim; j++ ) Bmat[i][j] = 0.0;
         Bmat[i][i] = 1.0;
      }
      for ( i = 1; i < ndim; i++ )
      {
         for ( j = 0; j < i; j++ )
         {
            if ( habs(Amat[j][j]) < 1.0e-16 ) return -1;
            denom = Amat[i][j] / Amat[j][j];
            for ( k = 0; k < ndim; k++ )
            {
               Amat[i][k] -= denom * Amat[j][k];
               Bmat[i][k] -= denom * Bmat[j][k];
            }
         }
      }
      for ( i = ndim-2; i >= 0; i-- )
      {
         for ( j = ndim-1; j >= i+1; j-- )
         {
            if ( habs(Amat[j][j]) < 1.0e-16 ) return -1;
            denom = Amat[i][j] / Amat[j][j];
            for ( k = 0; k < ndim; k++ )
            {
               Amat[i][k] -= denom * Amat[j][k];
               Bmat[i][k] -= denom * Bmat[j][k];
            }
         }
      }
      for ( i = 0; i < ndim; i++ )
      {
         denom = Amat[i][i];
         if ( habs(denom) < 1.0e-16 ) return -1;
         for ( j = 0; j < ndim; j++ ) Bmat[i][j] /= denom;
      }

      for ( i = 0; i < ndim; i++ )
         for ( j = 0; j < ndim; j++ )
            if ( habs(Bmat[i][j]) < 1.0e-17 ) Bmat[i][j] = 0.0;
      dmax = 0.0;
      for ( i = 0; i < ndim; i++ )
      {
         for ( j = 0; j < ndim; j++ )
            if ( habs(Bmat[i][j]) > dmax ) dmax = habs(Bmat[i][j]);
/*
         for ( j = 0; j < ndim; j++ )
            if ( habs(Bmat[i][j]/dmax) < 1.0e-15 ) Bmat[i][j] = 0.0;
*/
      }
      (*Cmat) = Bmat;
      if ( dmax > 1.0e6 ) return 1;
      else                return 0;
   }
}

/* ******************************************************************** */
/* find the separators of a matrix                                      */
/* -------------------------------------------------------------------- */

int HYPRE_LSI_PartitionMatrix( int nRows, int startRow, int *rowLengths,
                               int **colIndices, double **colValues,
                               int *nLabels, int **labels)
{
   int irow, rowCnt, labelNum, *localLabels, actualNRows;
   int jcol, root, indHead, indTail, *indSet, index;

   /*----------------------------------------------------------------*/
   /* search for constraint rows                                     */
   /*----------------------------------------------------------------*/

   for ( irow = nRows-1; irow >= 0; irow-- )
   {
      index = irow + startRow;
      for ( jcol = 0; jcol < rowLengths[irow]; jcol++ )
         if (colIndices[irow][jcol] == index && colValues[irow][jcol] != 0.0)
            break;
      if ( jcol != rowLengths[irow] ) break;
   }
   (*nLabels) = actualNRows = irow + 1;

   /*----------------------------------------------------------------*/
   /* search for constraint rows                                     */
   /*----------------------------------------------------------------*/

   localLabels = hypre_TAlloc(int,  actualNRows , HYPRE_MEMORY_HOST);
   for ( irow = 0; irow < actualNRows; irow++ ) localLabels[irow] = -1;
   indSet = hypre_TAlloc(int,  actualNRows , HYPRE_MEMORY_HOST);

   labelNum = 0;
   rowCnt   = actualNRows;

   while ( rowCnt > 0 )
   {
      root = -1;
      for ( irow = 0; irow < actualNRows; irow++ )
         if ( localLabels[irow] == -1 ) {root = irow; break;}
      if ( root == -1 )
      {
         printf("HYPRE_LSI_PartitionMatrix : something wrong.\n");
         exit(1);
      }
      indHead = indTail = 0;
      localLabels[root] = labelNum;
      rowCnt--;
      for ( jcol = 0; jcol < rowLengths[root]; jcol++ )
      {
         index = colIndices[root][jcol] - startRow;
         if ( index >= 0 && index < actualNRows && localLabels[index] < 0 )
         {
            indSet[indTail++] = index;
            localLabels[index] = labelNum;
         }
      }
      while ( (indTail - indHead) > 0 )
      {
         root = indSet[indHead++];
         rowCnt--;
         for ( jcol = 0; jcol < rowLengths[root]; jcol++ )
         {
            index = colIndices[root][jcol] - startRow;
            if ( index >= 0 && index < actualNRows && localLabels[index] < 0 )
            {
               indSet[indTail++] = index;
               localLabels[index] = labelNum;
            }
         }
      }
      labelNum++;
   }
   if ( labelNum > 4 )
   {
      printf("HYPRE_LSI_PartitionMatrix : number of labels %d too large.\n",
             labelNum+1);
      hypre_TFree(localLabels, HYPRE_MEMORY_HOST);
      (*nLabels) = 0;
      (*labels)  = NULL;
   }
   else
   {
      printf("HYPRE_LSI_PartitionMatrix : number of labels = %d.\n",
             labelNum);
      (*labels)  = localLabels;
   }
   hypre_TFree(indSet, HYPRE_MEMORY_HOST);
   return 0;
}

