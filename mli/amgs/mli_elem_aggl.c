/***********************************************************************
 *
 **********************************************************************/

#include "parcsr_mv.h"
#include "mli_matrix.h"
 
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

   hypre_EEMat = (hypre_ParCSRMatrix *) elemMatrix->matrix;
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

