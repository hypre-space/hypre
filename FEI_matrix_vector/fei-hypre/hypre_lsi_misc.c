/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "utilities/utilities.h"
#include "IJ_matrix_vector/HYPRE_IJ_mv.h"
#include "parcsr_matrix_vector/parcsr_matrix_vector.h"

extern void qsort1(int*, double*, int, int);

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
    mat_ia = (int *) malloc((Nrows+1) * sizeof(int));
    mat_ja = (int *) malloc( nnz * sizeof(int));
    mat_a  = (double *) malloc( nnz * sizeof(double));
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
    rhs_local  = (double *) malloc( Nrows * sizeof(double));
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
/* sort a double array                                                      */
/* (borrowed from the search routine in ML)                                 */
/* ------------------------------------------------------------------------ */

void HYPRE_LSI_DSort(double dlist[], int N, int list2[])
{
   int    l, r, j, i, flag, RR2;
   double dRR, dK;

   if (N <= 1) return;

   l    = N / 2 + 1;
   r    = N - 1;
   l    = l - 1;
   dRR  = dlist[l - 1];
   dK   = dlist[l - 1];

   if (list2 != NULL) {
      RR2 = list2[l - 1];
      while (r != 0) {
         j = l;
         flag = 1;

         while (flag == 1) {
            i = j;
            j = j + j;

            if (j > r + 1)
               flag = 0;
            else {
               if (j < r + 1)
                  if (dlist[j] > dlist[j - 1]) j = j + 1;
 
               if (dlist[j - 1] > dK) {
                  dlist[ i - 1] = dlist[ j - 1];
                  list2[i - 1] = list2[j - 1];
               }
               else {
                  flag = 0;
               }
            }
         }
         dlist[ i - 1] = dRR;
         list2[i - 1] = RR2;

         if (l == 1) {
            dRR  = dlist [r];
            RR2 = list2[r];
            dK = dlist[r];
            dlist[r ] = dlist[0];
            list2[r] = list2[0];
            r = r - 1;
          }
          else {
             l   = l - 1;
             dRR  = dlist[ l - 1];
             RR2 = list2[l - 1];
             dK   = dlist[l - 1];
          }
       }
       dlist[ 0] = dRR;
       list2[0] = RR2;
   }
   else {
      while (r != 0) {
         j = l;
         flag = 1;
         while (flag == 1) {
            i = j;
            j = j + j;
            if (j > r + 1)
               flag = 0;
            else {
               if (j < r + 1)
                  if (dlist[j] > dlist[j - 1]) j = j + 1;
               if (dlist[j - 1] > dK) {
                  dlist[ i - 1] = dlist[ j - 1];
               }
               else {
                  flag = 0;
               }
            }
         }
         dlist[ i - 1] = dRR;
         if (l == 1) {
            dRR  = dlist [r];
            dK = dlist[r];
            dlist[r ] = dlist[0];
            r = r - 1;
         }
         else {
            l   = l - 1;
            dRR  = dlist[ l - 1];
            dK   = dlist[l - 1];
         }
      }
      dlist[ 0] = dRR;
   }
}

/* ************************************************************************ */
/* this function extracts the matrix in a CSR format                        */
/* ------------------------------------------------------------------------ */

int getMatrixCSR(HYPRE_IJMatrix Amat, int nrows, int nnz, int *ia_ptr, 
                 int *ja_ptr, double *a_ptr) 
{
    int                nz, i, j, ierr, rowSize, *colInd, nz_ptr, *colInd2;
    int                firstNnz;
    double             *colVal, *colVal2;
    HYPRE_ParCSRMatrix A_csr;

    nz        = 0;
    nz_ptr    = 0;
    ia_ptr[0] = nz_ptr;
    A_csr  = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(Amat);
    for ( i = 0; i < nrows; i++ )
    {
       ierr = HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       assert(!ierr);
       colInd2 = (int *)    malloc(rowSize * sizeof(int));
       colVal2 = (double *) malloc(rowSize * sizeof(double));
       for ( j = 0; j < rowSize; j++ )
       {
          colInd2[j] = colInd[j];
          colVal2[j] = colVal[j];
       }
       qsort1(colInd2, colVal2, 0, rowSize-1);
       for ( j = 0; j < rowSize-1; j++ )
          if ( colInd2[j] == colInd2[j+1] )
             printf("getMatrixCSR - duplicate colind at row %d \n",i);

       firstNnz = 0;
       for ( j = 0; j < rowSize; j++ )
       {
          if ( colVal2[j] != 0.0 )
          {
             if (nz_ptr > 0 && firstNnz > 0 && colInd2[j] == ja_ptr[nz_ptr-1]) 
             {
                a_ptr[nz_ptr-1] += colVal2[j];
                printf("getMatrixCSR :: repeated col in row %d\n", i);
             }
             else
             { 
                ja_ptr[nz_ptr] = colInd2[j];
                a_ptr[nz_ptr++]  = colVal2[j];
                if ( nz_ptr > nnz )
                {
                   printf("getMatrixCSR error (1) - %d %d.\n",i, nrows);
                   exit(1);
                }
                firstNnz++;
             }
          } else nz++;
       }
       free( colInd2 );
       free( colVal2 );
       ia_ptr[i+1] = nz_ptr;
       ierr = HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       assert(!ierr);
    }   
    if ( nnz != nz_ptr )
    {
       printf("getMatrixCSR note : matrix sparsity has been changed since\n");
       printf("             matConfigure - %d > %d ?\n", nnz, nz_ptr);
       printf("             number of zeros            = %d \n", nz );
    }
    return nz_ptr;
}

/* ******************************************************************** */
/* taken from ML                                                        */
/* -------------------------------------------------------------------- */

void HYPRE_LSI_Sort(int list[], int N, int list2[], double list3[])
{

  /* local variables */

  int    l, r, RR, K, j, i, flag;
  int    RR2;
  double RR3;

  /*********************** execution begins ******************************/

  if (N <= 1) return;

  l   = N / 2 + 1;
  r   = N - 1;
  l   = l - 1;
  RR  = list[l - 1];
  K   = list[l - 1];

  if ((list2 != NULL) && (list3 != NULL)) {
    RR2 = list2[l - 1];
    RR3 = list3[l - 1];
    while (r != 0) {
      j = l;
      flag = 1;

      while (flag == 1) {
        i = j;
        j = j + j;

        if (j > r + 1)
          flag = 0;
        else {
          if (j < r + 1)
            if (list[j] > list[j - 1]) j = j + 1;

          if (list[j - 1] > K) {
            list[ i - 1] = list[ j - 1];
            list2[i - 1] = list2[j - 1];
            list3[i - 1] = list3[j - 1];
          }
          else {
            flag = 0;
          }
        }
      }

      list[ i - 1] = RR;
      list2[i - 1] = RR2;
      list3[i - 1] = RR3;

      if (l == 1) {
        RR  = list [r];
        RR2 = list2[r];
        RR3 = list3[r];

        K = list[r];
        list[r ] = list[0];
        list2[r] = list2[0];
        list3[r] = list3[0];
        r = r - 1;
      }
      else {
        l   = l - 1;
        RR  = list[ l - 1];
        RR2 = list2[l - 1];
        RR3 = list3[l - 1];
        K   = list[l - 1];
      }
    }

    list[ 0] = RR;
    list2[0] = RR2;
    list3[0] = RR3;
  }
  else if (list2 != NULL) {
    RR2 = list2[l - 1];
    while (r != 0) {
      j = l;
      flag = 1;

      while (flag == 1) {
        i = j;
        j = j + j;

        if (j > r + 1)
          flag = 0;
        else {
          if (j < r + 1)
            if (list[j] > list[j - 1]) j = j + 1;

          if (list[j - 1] > K) {
            list[ i - 1] = list[ j - 1];
            list2[i - 1] = list2[j - 1];
          }
          else {
            flag = 0;
          }
        }
      }

      list[ i - 1] = RR;
      list2[i - 1] = RR2;

      if (l == 1) {
        RR  = list [r];
        RR2 = list2[r];

        K = list[r];
        list[r ] = list[0];
        list2[r] = list2[0];
        r = r - 1;
      }
      else {
        l   = l - 1;
        RR  = list[ l - 1];
        RR2 = list2[l - 1];
        K   = list[l - 1];
      }
    }

    list[ 0] = RR;
    list2[0] = RR2;
  }
  else if (list3 != NULL) {
    RR3 = list3[l - 1];
    while (r != 0) {
      j = l;
      flag = 1;

      while (flag == 1) {
        i = j;
        j = j + j;

        if (j > r + 1)
          flag = 0;
        else {
          if (j < r + 1)
            if (list[j] > list[j - 1]) j = j + 1;

          if (list[j - 1] > K) {
            list[ i - 1] = list[ j - 1];
            list3[i - 1] = list3[j - 1];
          }
          else {
            flag = 0;
          }
        }
      }

      list[ i - 1] = RR;
      list3[i - 1] = RR3;

      if (l == 1) {
        RR  = list [r];
        RR3 = list3[r];

        K = list[r];
        list[r ] = list[0];
        list3[r] = list3[0];
        r = r - 1;
      }
      else {
        l   = l - 1;
        RR  = list[ l - 1];
        RR3 = list3[l - 1];
        K   = list[l - 1];
      }
    }

    list[ 0] = RR;
    list3[0] = RR3;

  }
  else {
    while (r != 0) {
      j = l;
      flag = 1;

      while (flag == 1) {
        i = j;
        j = j + j;

        if (j > r + 1)
          flag = 0;
        else {
          if (j < r + 1)
            if (list[j] > list[j - 1]) j = j + 1;

          if (list[j - 1] > K) {
            list[ i - 1] = list[ j - 1];
          }
          else {
            flag = 0;
          }
        }
      }

      list[ i - 1] = RR;

      if (l == 1) {
        RR  = list [r];

        K = list[r];
        list[r ] = list[0];
        r = r - 1;
      }
      else {
        l   = l - 1;
        RR  = list[ l - 1];
        K   = list[l - 1];
      }
    }

    list[ 0] = RR;
  }

}

/* ******************************************************************** */
/* sort a given list in increasing order                                */
/* taken from ML                                                        */
/* -------------------------------------------------------------------- */

int HYPRE_LSI_SplitDSort(double *dlist, int nlist, int *ilist, int limit)
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
   iarray1 = (int *)   malloc( 2 * nlist * sizeof(int) );
   iarray2 = iarray1 + nlist;
   darray1 = (double*) malloc( 2 * nlist * sizeof(double) );
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
   free( darray1 );
   free( iarray1 );
   free( darray2 );
   free( iarray2 );
   if ( count1+1 == limit ) return 0;
   else if ( count1+1 < limit )
      HYPRE_LSI_SplitDSort(&(dlist[count1+1]),count2,&(ilist[count1+1]),
                     limit-count1-1);
   else
      HYPRE_LSI_SplitDSort( dlist, count1, ilist, limit );
   return 0;
}

