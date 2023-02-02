/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * tests for various function in this directory
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mli_utils.h"

extern int mli_computespectrum_(int *,int *,double *, double *, int *,
                                double *, double *, double *, int *);
void testEigen();
void testMergeSort();

/******************************************************************************
 * main program
 *****************************************************************************/

main()
{
   int test=2;

   switch (test)
   {
      case 1 : testEigen();
      case 2 : testMergeSort();
   }
}

/******************************************************************************
 * test the Fortan functin for computing eigenvalues
 *---------------------------------------------------------------------------*/

void testEigen()
{
   int    i, mDim=24, ierr, matz=0;
   double *matrix, *evalues, *evectors, *daux1, *daux2;
   FILE   *fp;

   matrix = hypre_TAlloc(double,  mDim * mDim , HYPRE_MEMORY_HOST);
   fp = fopen("test.m", "r");
   if ( fp == NULL )
   {
      printf("testEigen ERROR : file not found.\n");
      exit(1);
   }
   for ( i = 0; i < mDim*mDim; i++ ) fscanf(fp, "%lg", &(matrix[i]));
   evectors = hypre_TAlloc(double,  mDim * mDim , HYPRE_MEMORY_HOST);
   evalues  = hypre_TAlloc(double,  mDim , HYPRE_MEMORY_HOST);
   daux1    = hypre_TAlloc(double,  mDim , HYPRE_MEMORY_HOST);
   daux2    = hypre_TAlloc(double,  mDim , HYPRE_MEMORY_HOST);
   mli_computespectrum_(&mDim, &mDim, matrix, evalues, &matz, evectors,
                        daux1, daux2, &ierr);
   for ( i = 0; i < mDim; i++ ) printf("eigenvalue = %e\n", evalues[i]);
   hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   hypre_TFree(evectors, HYPRE_MEMORY_HOST);
   hypre_TFree(evalues, HYPRE_MEMORY_HOST);
   hypre_TFree(daux1, HYPRE_MEMORY_HOST);
   hypre_TFree(daux2, HYPRE_MEMORY_HOST);
}

/******************************************************************************
 * test merge sort utility
 *---------------------------------------------------------------------------*/

void testMergeSort()
{
   int i, j, nlist=7, maxLeng=20, **list, **list2, *listLengs;
   int newNList, *newList, *checkList, checkN, checkFlag;

   listLengs = hypre_TAlloc(int,  nlist , HYPRE_MEMORY_HOST);
   list  = hypre_TAlloc(int*,  nlist , HYPRE_MEMORY_HOST);
   list2 = hypre_TAlloc(int*,  nlist , HYPRE_MEMORY_HOST);
   for ( i = 0; i < nlist; i++ )
   {
      list[i] = hypre_TAlloc(int,  maxLeng , HYPRE_MEMORY_HOST);
      list2[i] = hypre_TAlloc(int,  maxLeng , HYPRE_MEMORY_HOST);
   }
   listLengs[0] = 5;
   list[0][0] = 4;
   list[0][1] = 5;
   list[0][2] = 6;
   list[0][3] = 8;
   list[0][4] = 9;
   listLengs[1] = 1;
   list[1][0] = 10;
   listLengs[2] = 4;
   list[2][0] = 5;
   list[2][1] = 6;
   list[2][2] = 7;
   list[2][3] = 9;
   listLengs[3] = 2;
   list[3][0] = 10;
   list[3][1] = 11;
   listLengs[4] = 3;
   list[4][0] = 6;
   list[4][1] = 7;
   list[4][2] = 8;
   listLengs[5] = 3;
   list[5][0] = 10;
   list[5][1] = 11;
   list[5][2] = 12;
   listLengs[6] = 3;
   list[6][0] = 7;
   list[6][1] = 8;
   list[6][2] = 9;
   checkN = 0;
   for ( i = 0; i < nlist; i++ )
      for ( j = 0; j < listLengs[i]; j++ ) list2[i][j] = checkN++;

   for ( i = 0; i < nlist; i++ )
      MLI_Utils_IntQSort2(list[i], NULL, 0, listLengs[i]-1);
   for ( i = 0; i < nlist; i++ )
      for ( j = 0; j < listLengs[i]; j++ )
         printf("original %5d %5d = %d\n", i, j, list[i][j]);
   printf("MergeSort begins...\n");
   MLI_Utils_IntMergeSort(nlist, listLengs, list, list2, &newNList, &newList);
   for ( i = 0; i < newNList; i++ )
       printf("after    %5d = %d\n", i, newList[i]);
   printf("MergeSort ends.\n");
/*
   for ( i = 0; i < newNList; i++ )
      printf("Merge List %5d = %d\n", i, newList[i]);
   checkList = hypre_TAlloc(int,  nlist * maxLeng , HYPRE_MEMORY_HOST);
   for ( i = 0; i < nlist; i++ )
      for ( j = 0; j < maxLeng; j++ ) checkList[i*maxLeng+j] = list[i][j];
   printf("QSort begins...\n");
   MLI_Utils_IntQSort2(checkList, NULL, 0, nlist*maxLeng-1);
   printf("QSort ends.\n");
   checkN = 1;
   for ( i = 1; i < nlist*maxLeng; i++ )
      if ( checkList[checkN-1] != checkList[i] )
         checkList[checkN++] = checkList[i];
   if ( checkN != newNList )
      printf("MergeSort and QSort lengths = %d %d\n", newNList, checkN);
   checkFlag = 0;
   for ( i = 0; i < newNList; i++ )
   {
      if ( checkList[i] != newList[i] )
      {
         printf("MergeSort and QSort discrepancy %5d = %5d %5d\n", i,
                newList[i], checkList[i]);
         checkFlag++;
      }
   }
   printf("MergeSort and QSort lengths = %d %d\n", newNList, checkN);
   if ( checkFlag == 0 )
      printf("MergeSort and QSort gives same result.\n");

   for ( i = 0; i < nlist; i++ )
   {
      hypre_TFree(list[i], HYPRE_MEMORY_HOST);
      hypre_TFree(list2[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(checkList , HYPRE_MEMORY_HOST);
*/
   hypre_TFree(listLengs, HYPRE_MEMORY_HOST);
   hypre_TFree(list, HYPRE_MEMORY_HOST);
   hypre_TFree(list2, HYPRE_MEMORY_HOST);
   hypre_TFree(newList, HYPRE_MEMORY_HOST);
}

