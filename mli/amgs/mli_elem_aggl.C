/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/* *********************************************************************
 * This file is customized to use HYPRE matrix format
 * ********************************************************************/

/* *********************************************************************
 * local includes
 * -------------------------------------------------------------------*/

#include <stdio.h>
#include "HYPRE.h"
#include "parcsr_mv/parcsr_mv.h"
#include "../matrix/mli_matrix.h"
 
/***********************************************************************
 * Function  : MLI_Coarsen_ElementAgglomeration
 * Purpose   : Form macroelements
 * Inputs    : element-element matrix 
 **********************************************************************/

void MLI_Coarsen_ElementAgglomerationLocal(MLI_Matrix *elemMatrix, 
                                           int **macro_labels_out)
{
   hypre_ParCSRMatrix  *hypre_EEMat;
   MPI_Comm            comm;
   int                 mypid, num_procs, *partition, start_elem, end_elem;
   int                 local_nElems, nmacros, *macro_labels, *macro_sizes;
   int                 *macro_list, ielem, jj, col_index, *dense_row;
   int                 max_weight, cur_weight, cur_index, row_leng, row_num;
   int                 elem_count, elem_index, macro_number, *cols;
   double              *vals;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   hypre_EEMat = (hypre_ParCSRMatrix *) elemMatrix->getMatrix();
   comm        = hypre_ParCSRMatrixComm(hypre_EEMat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&num_procs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypre_EEMat, 
                                        &partition);
   start_elem   = partition[mypid];
   end_elem     = partition[mypid+1] - 1;
   local_nElems = end_elem - start_elem + 1;

   /*-----------------------------------------------------------------
    * this array is used to determine which element has been agglomerated
    *-----------------------------------------------------------------*/

   macro_labels = (int *) malloc( local_nElems * sizeof(int) );
   for ( ielem = 0; ielem < local_nElems; ielem++ ) macro_labels[ielem] = -1;

   /*-----------------------------------------------------------------
    * this array is used to expand a sparse row into a full row 
    *-----------------------------------------------------------------*/

   dense_row = (int *) malloc( local_nElems * sizeof(int) );
   for ( ielem = 0; ielem < local_nElems; ielem++ ) dense_row[ielem] = 0;

   /*-----------------------------------------------------------------
    * allocate memory for the output data (assume no more than 
    * 100 elements in any macroelements 
    *-----------------------------------------------------------------*/

   nmacros = 0;
   macro_sizes = (int *) malloc( local_nElems/2 * sizeof(int) );
   macro_list  = (int *) malloc( 100 * sizeof(int) );

   /*-----------------------------------------------------------------
    * loop through all elements for agglomeration
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < local_nElems; ielem++ )
   {
      if ( macro_labels[ielem] < 0 ) /* element has not been agglomerated */
      {
         max_weight = 0;
         cur_weight = 0;
         cur_index  = -1;

         /* load row ielem into dense_row, keeping track of maximum weight */

         row_num = start_elem + ielem;
         hypre_ParCSRMatrixGetRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);
         for ( jj = 0; jj < row_leng; jj++ )
         {
            col_index = cols[jj] - start_elem;
            if ( col_index >= 0 && col_index < local_nElems )
            {
               if ( dense_row[col_index] >= 0 )
               {
                  dense_row[col_index] = (int) vals[jj];
                  if ( ((int) vals[jj]) > cur_weight )
                  {
                     cur_weight = (int) vals[jj];
                     cur_index  = col_index;
                  }
               }
            }    
         }    
         hypre_ParCSRMatrixRestoreRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);

         /* begin agglomeration using element ielem as root */

         elem_count = 0;
         macro_list[elem_count++] = ielem;
         dense_row[ielem] = -1;
         while ( cur_weight >= 4 && cur_weight > max_weight && elem_count < 100 )
         { 
            max_weight = cur_weight;
            macro_list[elem_count++] = cur_index;
            dense_row[cur_index] = -1;
            row_num = start_elem + cur_index;
            hypre_ParCSRMatrixGetRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);
            for ( jj = 0; jj < row_leng; jj++ )
            {
               col_index = cols[jj] - start_elem;
               if ( col_index >= 0 && col_index < local_nElems )
               {
                  if ( dense_row[col_index] >= 0 ) 
                  {
                     dense_row[col_index] += (int) vals[jj];
                     if ( ((int) dense_row[col_index]) > cur_weight )
                     {
                        cur_weight = dense_row[col_index];
                        cur_index  = col_index;
                     }
                  }
               }
            }
            hypre_ParCSRMatrixRestoreRow(hypre_EEMat,row_num,&row_leng,&cols,
                                         &vals);
         } 

         /* if macroelement has size > 1, register it and reset dense_row */

         if ( elem_count > 1 ) 
         {
            for ( jj = 0; jj < elem_count; jj++ )
            {
               elem_index = macro_list[jj];
               macro_labels[elem_index] = nmacros;
#if 1
               printf("Macroelement %4d has element %4d\n", nmacros, elem_index);
#endif
            }
            for ( jj = 0; jj < local_nElems; jj++ )
               if ( dense_row[jj] > 0 ) dense_row[jj] = 0;
            macro_sizes[nmacros++] = elem_count;
         } 
         else dense_row[ielem] = 0;
      }
   }

   /*-----------------------------------------------------------------
    * if there are still leftovers, put them into adjacent macroelement
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < local_nElems; ielem++ )
   {
      if ( macro_labels[ielem] < 0 ) /* not been agglomerated */
      {
         row_num = start_elem + ielem;
         hypre_ParCSRMatrixGetRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);
         cur_index = -1;
         max_weight = 3;
         for ( jj = 0; jj < row_leng; jj++ )
         {
            col_index   = cols[jj] - start_elem;
            if ( col_index >= 0 && col_index < local_nElems )
            {
               macro_number = macro_labels[col_index];
               if ( macro_number > 0 && vals[jj] > max_weight )
               {
                  max_weight = (int) vals[jj];
                  cur_index  = macro_number;
               }
            }
         } 
         hypre_ParCSRMatrixRestoreRow(hypre_EEMat,row_num,&row_leng,&cols,&vals);
         if ( cur_index >= 0 ) macro_labels[ielem] = cur_index;
      } 
   } 

   /*-----------------------------------------------------------------
    * finally lone zones will be all by themselves 
    *-----------------------------------------------------------------*/

   for ( ielem = 0; ielem < local_nElems; ielem++ )
   {
      if ( macro_labels[ielem] < 0 ) /* still has not been agglomerated */
      {
         macro_sizes[nmacros] = 1;
         macro_labels[ielem]  = nmacros++;
      }
   }

   /*-----------------------------------------------------------------
    * initialize the output arrays 
    *-----------------------------------------------------------------*/

   (*macro_labels_out) = macro_labels;
   free( macro_list );
   free( macro_sizes );
   free( dense_row );

}

void IncFlowElementAgglomeration(int nElem, int *ElemIA, int *ElemJA,
                                 int *ElemAA, int **macroLabels_out)
{
   int  ii, jj, nmacros, *macroLists, tempIndex[20];
   int  colIndex, *macroLabels, *denseRow, maxWeight, curWeight, curIndex;
   int  elemCount, elemIndex, macroNumber, curBegin, curEnd, loopflag;
   int  nextElem, connects, parent, neigh_cnt, min_neighs, secondchance;
   int  *macroIA, *macroJA, *macroAA, index, count, *denseRow2, *noroot;
   int  macro_nnz;

   /* ------------------------------------------------------------------- */
   /* this array is used to determine which element has been agglomerated */
   /* and which macroelement the current element belongs to               */
   /* ------------------------------------------------------------------- */

   macroLabels = (int *) malloc( nElem * sizeof(int) );
   for ( ii = 0; ii < nElem; ii++ ) macroLabels[ii] = -1;

   /* ------------------------------------------------------------------- */
   /* this array is used to indicate which element has been used as root  */
   /* for agglomeration so that no duplication will be done.              */
   /* ------------------------------------------------------------------- */

   noroot      = (int *) malloc( nElem * sizeof(int) );
   for ( ii = 0; ii < nElem; ii++ ) noroot[ii] = 0;

   /* ------------------------------------------------------------------- */
   /* These array are used to expand a sparse row into a full row         */
   /* (denseRow is used to register information for already agglomerated  */
   /* elements while denseRow2 is used to register information for        */
   /* possible macroelement).                                             */
   /* ------------------------------------------------------------------- */

   denseRow   = (int *) malloc( nElem * sizeof(int) );
   denseRow2  = (int *) malloc( nElem * sizeof(int) );
   for ( ii = 0; ii < nElem; ii++ ) denseRow[ii] = denseRow2[ii] = 0;

   /* ------------------------------------------------------------------- */
   /* These arrays are needed to find neighbor element for agglomeration  */
   /* that preserves nice geometric shapes                                */
   /* ------------------------------------------------------------------- */

   macroIA = (int *) malloc( (nElem/8+1) * sizeof(int) );
   macroJA = (int *) malloc( (nElem/8+1) * 216 * sizeof(int) );
   macroAA = (int *) malloc( (nElem/8+1) * 216 * sizeof(int) );

   /* ------------------------------------------------------------------- */
   /* allocate memory for the output data (assume no more than 60 elements*/
   /* in any macroelements                                                */
   /* ------------------------------------------------------------------- */

   nmacros = 0;
   macroLists = (int *) malloc( 60 * sizeof(int) );

   /* ------------------------------------------------------------------- */
   /* search for initial element (one with least number of neighbors)     */
   /* ------------------------------------------------------------------- */

   nextElem   = -1;
   min_neighs = 10000;
   for ( ii = 0; ii < nElem; ii++ )
   {
      neigh_cnt = ElemIA[ii+1] - ElemIA[ii];
      if ( neigh_cnt < min_neighs )
      {
         min_neighs = neigh_cnt;
         nextElem = ii;
      }
   }

   /* ------------------------------------------------------------------- */
   /* loop through all elements for agglomeration                         */
   /* ------------------------------------------------------------------- */

   if ( nextElem == -1 ) loopflag = 0; else loopflag = 1;
   parent     = -1;
   macroIA[0] = 0;
   macro_nnz  = 0;

   while ( loopflag )
   {
      if ( macroLabels[nextElem] < 0 )
      {
         /* ------------------------------------------------------------- */
         /* update the current macroelement connectivity row              */
         /* ------------------------------------------------------------- */

         for ( ii = 0; ii < nElem; ii++ ) denseRow2[ii] = denseRow[ii];

         /* ------------------------------------------------------------- */
         /* load row nextElem into denseRow, keeping track of max weight  */
         /* ------------------------------------------------------------- */

         curWeight = 0;
         curIndex  = -1;
         for ( ii = ElemIA[nextElem]; ii < ElemIA[nextElem+1]; ii++ )
         {
            colIndex = ElemJA[ii];
            if ( denseRow2[colIndex] >= 0 )
            {
               denseRow2[colIndex] = ElemAA[ii];
               if ( ElemAA[ii] > curWeight )
               {
                  curWeight = ElemAA[ii];
                  curIndex  = ElemJA[ii];
               }
            }
         }

         /* ------------------------------------------------------------- */
         /* if there is a parent macroelement to the root element, do the */
         /* following :                                                   */
         /* 1. find how many links between the selected neighbor element  */
         /*    and the parent element (there  may be none)                */
         /* 2. search for other neighbor elements to see if they have the */
         /*    same links to the root element but which is more connected */
         /*    to the parent element, and select it                       */
         /* ------------------------------------------------------------- */

         if ( parent >= 0 )
         {
            connects = 0;
            for ( jj = macroIA[parent]; jj < macroIA[parent+1]; jj++ )
               if ( macroJA[jj] == curIndex ) {connects = macroAA[jj]; break;}
            for ( ii = ElemIA[nextElem]; ii < ElemIA[nextElem+1]; ii++ )
            {
               colIndex = ElemJA[ii];
               if ( ElemAA[ii] == curWeight && colIndex != curIndex )
               {
                  for ( jj = macroIA[parent]; jj < macroIA[parent+1]; jj++ )
                  {
                     if ( macroJA[jj] == colIndex && macroAA[jj] > connects )
                     {
                        curWeight = ElemAA[ii];
                        curIndex  = ElemJA[ii];
                        break;
                     }
                  }
               }
            }
         }

         /* store the element on the macroelement list */

         elemCount = 0;
         maxWeight = 0;
         macroLists[elemCount++] = nextElem;
         denseRow2[nextElem] = -1;

         /* grab the neighboring elements */

         /*while ( elemCount < 8 || curWeight > maxWeight )*/
         secondchance = 0;
         while ( curWeight > maxWeight || secondchance == 0 )
         {
            /* if decent macroelement is unlikely to be formed, exit */
            if ( elemCount == 1 && curWeight <  4 ) break;
            if ( elemCount == 2 && curWeight <  6 ) break;
            if ( elemCount >  2 && curWeight <= 6 ) break;

            /* otherwise include this element in the list */

            if ( curWeight <= maxWeight ) secondchance = 1;
            maxWeight = curWeight;
            macroLists[elemCount++] = curIndex;
            denseRow2[curIndex] = - 1;

            /* update the macroelement connectivity */

            curBegin = ElemIA[curIndex];
            curEnd   = ElemIA[curIndex+1];
            for ( ii = curBegin; ii < curEnd; ii++ )
            {
               colIndex = ElemJA[ii];
               if (denseRow2[colIndex] >= 0)
               {
                  denseRow2[colIndex] += ElemAA[ii];
               }
            }

            /* search for next element to agglomerate (max connectivity) */

            curWeight = 0;
            curIndex  = -1;
            for ( ii = 0; ii < nElem; ii++ )
            {
               if (denseRow2[ii] > curWeight)
               {
                  curWeight = denseRow2[ii];
                  curIndex = ii;
               }
            }

            /* if more than one with same weight, use other criterion */

            if ( curIndex >= 0 && parent >= 0 )
            {
               for ( jj = macroIA[parent]; jj < macroIA[parent+1]; jj++ )
                  if ( macroJA[jj] == curIndex ) connects = macroAA[jj];
               for ( ii = 0; ii < nElem; ii++ )
               {
                  if (denseRow2[ii] == curWeight && ii != curIndex )
                  {
                     for ( jj = macroIA[parent]; jj < macroIA[parent+1]; jj++ )
                     {
                        if ( macroJA[jj] == ii && macroAA[jj] > connects )
                        {
                           curWeight = denseRow2[ii];
                           curIndex = ii;
                           break;
                        }
                     }
                  }
               }
            }
         }

         /* if decent macroelement has been found, validate it */

         if ( elemCount >= 8 )
         {
            for ( jj = 0; jj < elemCount; jj++ )
            {
               elemIndex = macroLists[jj];
               macroLabels[elemIndex] = nmacros;
               denseRow2[elemIndex] = -1;
               noroot[elemIndex] = 1;
            }
            for ( jj = 0; jj < nElem; jj++ ) denseRow[jj] = denseRow2[jj];
            for ( jj = 0; jj < nElem; jj++ )
            {
               if ( denseRow[jj] > 0 )
               {
                  macroJA[macro_nnz] = jj;
                  macroAA[macro_nnz++] = denseRow[jj];
               }
            }
            parent = nmacros++;
            macroIA[nmacros] = macro_nnz;
         }
         else
         {
            noroot[nextElem] = 1;
            denseRow[nextElem] = 0;
            if ( parent >= 0 )
            {
               for ( ii = macroIA[parent]; ii < macroIA[parent+1]; ii++ )
               {
                  jj = macroJA[ii];
                  if (noroot[jj] == 0) denseRow[jj] = macroAA[ii];
               }
            }
         }

         /* search for the root of the next macroelement */

         maxWeight = 0;
         nextElem = -1;
         for ( jj = 0; jj < nElem; jj++ )
         {
            if ( denseRow[jj] > 0 )
            {
               if ( denseRow[jj] > maxWeight )
               {
                  maxWeight = denseRow[jj];
                  nextElem = jj;
               }
               denseRow[jj] = 0;
            }
         }
         if ( nextElem == -1 )
         {
            parent = -1;
            for ( jj = 0; jj < nElem; jj++ )
               if (macroLabels[jj] < 0 && noroot[jj] == 0) { nextElem = jj; break;
 }
         }
         if ( nextElem == -1 ) loopflag = 0;
      }
   }

   /* if there are still leftovers, put them into adjacent macroelement
    * or form their own, if neighbor macroelement not found */

   loopflag = 1;
   while ( loopflag )
   {
      count = 0;
      for ( ii = 0; ii < nElem; ii++ )
      {
         if ( macroLabels[ii] < 0 )
         {
            for ( jj = ElemIA[ii]; jj < ElemIA[ii+1]; jj++ )
            {
               colIndex    = ElemJA[jj];
               macroNumber = macroLabels[colIndex];
               if ( ElemAA[jj] >= 4 && macroNumber >= 0 )
               {
                  macroLabels[ii] = - macroNumber - 10;
                  count++;
                  break;
               }
            }
         }
      }
      for ( ii = 0; ii < nElem; ii++ )
      {
         if ( macroLabels[ii] <= -10 ) macroLabels[ii] = - macroLabels[ii] - 10;
      }
      if ( count == 0 ) loopflag = 0;
   }

   /* finally lone zones will be all by themselves */

   for ( ii = 0; ii < nElem; ii++ )
   {
      if ( macroLabels[ii] < 0 ) /* element still has not been agglomerated */
      {
         macroLabels[ii] = nmacros++;
      }
   }

   /* initialize the output arrays */

   printf("number of macroelements = %d (%d) : %e\n", nmacros, nElem,
            (double) nElem/nmacros);
   (*macroLabels_out) = macroLabels;
   free( macroLists );
   free( macroIA );
   free( macroJA );
   free( macroAA );
   free( denseRow );
   free( denseRow2 );
   free( noroot );
}/*endfunction*/


