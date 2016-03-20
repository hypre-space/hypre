/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_parcsr_mv.h"

#include "_hypre_utilities.h"
#include "hypre_hopscotch_hash.h"
#include "_hypre_parcsr_mv.h"

/* RDF: The following prototype already exists in _hypre_parcsr_ls.h, so
 * something needs to be reorganized here.*/

#ifdef __cplusplus
extern "C" {
#endif

hypre_CSRMatrix *
hypre_ExchangeRAPData( hypre_CSRMatrix *RAP_int, hypre_ParCSRCommPkg *comm_pkg_RT);                                                                                                               
/* reference seems necessary to prevent a problem with the
   "headers" script... */

#ifdef __cplusplus
}
#endif

/* The following function was formerly part of hypre_ParMatmul
   but was removed so it can also be used for multiplication of
   Boolean matrices
*/

void hypre_ParMatmul_RowSizes(
   HYPRE_Int ** C_diag_i,
   HYPRE_Int ** C_offd_i,
   /*HYPRE_Int ** B_marker,*/
   HYPRE_Int * A_diag_i,
   HYPRE_Int * A_diag_j,
   HYPRE_Int * A_offd_i,
   HYPRE_Int * A_offd_j,
   HYPRE_Int * B_diag_i,
   HYPRE_Int * B_diag_j,
   HYPRE_Int * B_offd_i,
   HYPRE_Int * B_offd_j,
   HYPRE_Int * B_ext_diag_i,
   HYPRE_Int * B_ext_diag_j, 
   HYPRE_Int * B_ext_offd_i,
   HYPRE_Int * B_ext_offd_j,
   HYPRE_Int * map_B_to_C,
   HYPRE_Int *C_diag_size,
   HYPRE_Int *C_offd_size,
   HYPRE_Int num_rows_diag_A,
   HYPRE_Int num_cols_offd_A,
   HYPRE_Int allsquare,
   HYPRE_Int num_cols_diag_B,
   HYPRE_Int num_cols_offd_B,
   HYPRE_Int num_cols_offd_C
   )
{
   HYPRE_Int i1, i2, i3, jj2, jj3;
   HYPRE_Int jj_count_diag, jj_count_offd, jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int start_indexing = 0; /* start indexing for C_data at 0 */
   HYPRE_Int num_threads = hypre_NumThreads();
   HYPRE_Int *jj_count_diag_array;
   HYPRE_Int *jj_count_offd_array;
   HYPRE_Int ii, size, rest;
   /* First pass begins here.  Computes sizes of C rows.
      Arrays computed: C_diag_i, C_offd_i, B_marker
      Arrays needed: (11, all HYPRE_Int*)
      A_diag_i, A_diag_j, A_offd_i, A_offd_j,
      B_diag_i, B_diag_j, B_offd_i, B_offd_j,
      B_ext_i, B_ext_j, col_map_offd_B,
      col_map_offd_B, B_offd_i, B_offd_j, B_ext_i, B_ext_j,
      Scalars computed: C_diag_size, C_offd_size
      Scalars needed:
      num_rows_diag_A, num_rows_diag_A, num_cols_offd_A, allsquare,
      first_col_diag_B, n_cols_B, num_cols_offd_B, num_cols_diag_B
   */

   *C_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A+1);
   *C_offd_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A+1);
   jj_count_diag_array = hypre_CTAlloc(HYPRE_Int, num_threads);
   jj_count_offd_array = hypre_CTAlloc(HYPRE_Int, num_threads);
   /*-----------------------------------------------------------------------
    *  Loop over rows of A
    *-----------------------------------------------------------------------*/
   size = num_rows_diag_A/num_threads;
   rest = num_rows_diag_A - size*num_threads;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(ii, i1, jj_row_begin_diag, jj_row_begin_offd, jj_count_diag, jj_count_offd, jj2, i2, jj3, i3) 
#endif
   /*for (ii=0; ii < num_threads; ii++)*/
   {
    HYPRE_Int *B_marker = NULL;
    HYPRE_Int ns, ne;
    ii = hypre_GetThreadNum();
    if (ii < rest)
    {
       ns = ii*size+ii;
       ne = (ii+1)*size+ii+1;
    }
    else
    {
       ns = ii*size+rest;
       ne = (ii+1)*size+rest;
    }
    jj_count_diag = start_indexing;
    jj_count_offd = start_indexing;

    if (num_cols_diag_B || num_cols_offd_C)
    B_marker = hypre_CTAlloc(HYPRE_Int, num_cols_diag_B+num_cols_offd_C);
    for (i1 = 0; i1 < num_cols_diag_B+num_cols_offd_C; i1++)
      B_marker[i1] = -1;

    for (i1 = ns; i1 < ne; i1++)
    {
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, C_{i1,i1} (for square matrices). 
       *--------------------------------------------------------------------*/
 
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare ) {
         B_marker[i1] = jj_count_diag;
         jj_count_diag++;
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/
         
      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];
 
            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_ext.
             *-----------------------------------------------------------*/
 
            for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2+1]; jj3++)
            {
               i3 = num_cols_diag_B+B_ext_offd_j[jj3];
                  
               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, mark it and increment
                *  counter.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  jj_count_offd++;
               } 
            }
            for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2+1]; jj3++)
            {
               i3 = B_ext_diag_j[jj3];
                  
               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  jj_count_diag++;
               } 
            }
         }
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_diag.
       *-----------------------------------------------------------------*/
         
      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
      {
         i2 = A_diag_j[jj2];
 
         /*-----------------------------------------------------------
          *  Loop over entries in row i2 of B_diag.
          *-----------------------------------------------------------*/
 
         for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
         {
            i3 = B_diag_j[jj3];
                  
            /*--------------------------------------------------------
             *  Check B_marker to see that C_{i1,i3} has not already
             *  been accounted for. If it has not, mark it and increment
             *  counter.
             *--------------------------------------------------------*/
 
            if (B_marker[i3] < jj_row_begin_diag)
            {
               B_marker[i3] = jj_count_diag;
               jj_count_diag++;
            }
         }

         /*-----------------------------------------------------------
          *  Loop over entries in row i2 of B_offd.
          *-----------------------------------------------------------*/

         if (num_cols_offd_B)
         { 
            for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
            {
               i3 = num_cols_diag_B+map_B_to_C[B_offd_j[jj3]];
                  
               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, mark it and increment
                *  counter.
                *--------------------------------------------------------*/
 
               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  jj_count_offd++;
               }
            }
         }
      }
            
      /*--------------------------------------------------------------------
       * Set C_diag_i and C_offd_i for this row.
       *--------------------------------------------------------------------*/
 
      (*C_diag_i)[i1] = jj_row_begin_diag;
      (*C_offd_i)[i1] = jj_row_begin_offd;
      
    }
    jj_count_diag_array[ii] = jj_count_diag;
    jj_count_offd_array[ii] = jj_count_offd;

    hypre_TFree(B_marker);
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

    if (ii)
    {
       jj_count_diag = jj_count_diag_array[0];
       jj_count_offd = jj_count_offd_array[0];
       for (i1 = 1; i1 < ii; i1++)
       {
          jj_count_diag += jj_count_diag_array[i1];
          jj_count_offd += jj_count_offd_array[i1];
       }

       for (i1 = ns; i1 < ne; i1++)
       {
          (*C_diag_i)[i1] += jj_count_diag;
          (*C_offd_i)[i1] += jj_count_offd;
       }
    }
    else
    {
       (*C_diag_i)[num_rows_diag_A] = 0;
       (*C_offd_i)[num_rows_diag_A] = 0;
       for (i1 = 0; i1 < num_threads; i1++)
       {
          (*C_diag_i)[num_rows_diag_A] += jj_count_diag_array[i1];
          (*C_offd_i)[num_rows_diag_A] += jj_count_offd_array[i1];
       }
    }
   } /* end parallel loop */
 
   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   *C_diag_size = (*C_diag_i)[num_rows_diag_A];
   *C_offd_size = (*C_offd_i)[num_rows_diag_A];

   hypre_TFree(jj_count_diag_array);
   hypre_TFree(jj_count_offd_array);
 
   /* End of First Pass */
}

/*--------------------------------------------------------------------------
 * hypre_ParMatmul : multiplies two ParCSRMatrices A and B and returns
 * the product in ParCSRMatrix C
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *hypre_ParMatmul( hypre_ParCSRMatrix  *A,
                                     hypre_ParCSRMatrix  *B )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATMUL] -= hypre_MPI_Wtime();
#endif

   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   
   HYPRE_Complex   *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   
   HYPRE_Complex   *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int       *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int        num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   
   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
   
   HYPRE_Complex   *B_diag_data = hypre_CSRMatrixData(B_diag);
   HYPRE_Int       *B_diag_i = hypre_CSRMatrixI(B_diag);
   HYPRE_Int       *B_diag_j = hypre_CSRMatrixJ(B_diag);

   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int       *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);
   
   HYPRE_Complex   *B_offd_data = hypre_CSRMatrixData(B_offd);
   HYPRE_Int       *B_offd_i = hypre_CSRMatrixI(B_offd);
   HYPRE_Int       *B_offd_j = hypre_CSRMatrixJ(B_offd);

   HYPRE_Int        first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_Int        last_col_diag_B;
   HYPRE_Int       *col_starts_B = hypre_ParCSRMatrixColStarts(B);
   HYPRE_Int        num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int        num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int        num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_Int          *col_map_offd_C;
   HYPRE_Int          *map_B_to_C=NULL;

   hypre_CSRMatrix *C_diag;

   HYPRE_Complex   *C_diag_data;
   HYPRE_Int       *C_diag_i;
   HYPRE_Int       *C_diag_j;

   hypre_CSRMatrix *C_offd;

   HYPRE_Complex   *C_offd_data=NULL;
   HYPRE_Int       *C_offd_i=NULL;
   HYPRE_Int       *C_offd_j=NULL;

   HYPRE_Int        C_diag_size;
   HYPRE_Int        C_offd_size;
   HYPRE_Int        num_cols_offd_C = 0;
   
   hypre_CSRMatrix *Bs_ext;
   
   HYPRE_Complex   *Bs_ext_data;
   HYPRE_Int       *Bs_ext_i;
   HYPRE_Int       *Bs_ext_j;

   HYPRE_Complex   *B_ext_diag_data;
   HYPRE_Int       *B_ext_diag_i;
   HYPRE_Int       *B_ext_diag_j;
   HYPRE_Int        B_ext_diag_size;

   HYPRE_Complex   *B_ext_offd_data;
   HYPRE_Int       *B_ext_offd_i;
   HYPRE_Int       *B_ext_offd_j;
   HYPRE_Int        B_ext_offd_size;

   HYPRE_Int        n_rows_A, n_cols_A;
   HYPRE_Int        n_rows_B, n_cols_B;
   HYPRE_Int        allsquare = 0;
   HYPRE_Int        num_procs;
   HYPRE_Int       *my_diag_array;
   HYPRE_Int       *my_offd_array;
   HYPRE_Int        max_num_threads;

   HYPRE_Complex    zero = 0.0;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   max_num_threads = hypre_NumThreads();
   my_diag_array = hypre_CTAlloc(HYPRE_Int, max_num_threads);
   my_offd_array = hypre_CTAlloc(HYPRE_Int, max_num_threads);

   if (n_cols_A != n_rows_B || num_cols_diag_A != num_rows_diag_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC," Error! Incompatible matrix dimensions!\n");
      return NULL;
   }
   if ( num_rows_diag_A==num_cols_diag_B) allsquare = 1;

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product 
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within 
       * hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Bs_ext = hypre_ParCSRMatrixExtractBExt(B,A,1);
      Bs_ext_data = hypre_CSRMatrixData(Bs_ext);
      Bs_ext_i    = hypre_CSRMatrixI(Bs_ext);
      Bs_ext_j    = hypre_CSRMatrixJ(Bs_ext);
   }
   B_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A+1);
   B_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A+1);
   B_ext_diag_size = 0;
   B_ext_offd_size = 0;
   last_col_diag_B = first_col_diag_B + num_cols_diag_B -1;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   hypre_UnorderedIntSet set;

#pragma omp parallel 
   {
     HYPRE_Int size, rest, ii;
     HYPRE_Int ns, ne;
     HYPRE_Int i1, i, j;
     HYPRE_Int my_offd_size, my_diag_size;
     HYPRE_Int cnt_offd, cnt_diag;
     HYPRE_Int num_threads = hypre_NumActiveThreads();

     size = num_cols_offd_A/num_threads;
     rest = num_cols_offd_A - size*num_threads;
     ii = hypre_GetThreadNum();
     if (ii < rest)
     {
       ns = ii*size+ii;
       ne = (ii+1)*size+ii+1;
     }
     else
     {
       ns = ii*size+rest;
       ne = (ii+1)*size+rest;
     }

     my_diag_size = 0;
     my_offd_size = 0;
     for (i=ns; i < ne; i++)
     {
       B_ext_diag_i[i] = my_diag_size;
       B_ext_offd_i[i] = my_offd_size;
       for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            my_offd_size++;
         else
            my_diag_size++;
     }
     my_diag_array[ii] = my_diag_size;
     my_offd_array[ii] = my_offd_size;

#pragma omp barrier

     if (ii)
     {
       my_diag_size = my_diag_array[0];
       my_offd_size = my_offd_array[0];
       for (i1 = 1; i1 < ii; i1++)
       {
          my_diag_size += my_diag_array[i1];
          my_offd_size += my_offd_array[i1];
       }

       for (i1 = ns; i1 < ne; i1++)
       {
          B_ext_diag_i[i1] += my_diag_size;
          B_ext_offd_i[i1] += my_offd_size;
       }
     }
     else
     {
       B_ext_diag_size = 0;
       B_ext_offd_size = 0;
       for (i1 = 0; i1 < num_threads; i1++)
       {
          B_ext_diag_size += my_diag_array[i1];
          B_ext_offd_size += my_offd_array[i1];
       }
       B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
       B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

       if (B_ext_diag_size)
       {
          B_ext_diag_j = hypre_CTAlloc(HYPRE_Int, B_ext_diag_size);
          B_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, B_ext_diag_size);
       }
       if (B_ext_offd_size)
       {
          B_ext_offd_j = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size);
          B_ext_offd_data = hypre_CTAlloc(HYPRE_Complex, B_ext_offd_size);
       }
       hypre_UnorderedIntSetCreate(&set, B_ext_offd_size + num_cols_offd_B, 16*hypre_NumThreads());
     }

#pragma omp barrier

     cnt_offd = B_ext_offd_i[ns];
     cnt_diag = B_ext_diag_i[ns];
     for (i=ns; i < ne; i++)
     {
       for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
         {
            hypre_UnorderedIntSetPut(&set, Bs_ext_j[j]);
            B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
            B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
         }
         else
         {
            B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
            B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
         }
     }

     HYPRE_Int i_begin, i_end;
     hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_B);
     for (i = i_begin; i < i_end; i++)
     {
        hypre_UnorderedIntSetPut(&set, col_map_offd_B[i]);
     }
   } /* omp parallel */

   if (num_procs > 1)
    {
       hypre_CSRMatrixDestroy(Bs_ext);
       Bs_ext = NULL;
    }

    col_map_offd_C = hypre_UnorderedIntSetCopyToArray(&set, &num_cols_offd_C);
    hypre_UnorderedIntSetDestroy(&set);
    hypre_UnorderedIntMap col_map_offd_C_inverse;
    hypre_sort_and_create_inverse_map(col_map_offd_C, num_cols_offd_C, &col_map_offd_C, &col_map_offd_C_inverse);

    HYPRE_Int i, j;
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
    for (i = 0; i < num_cols_offd_A; i++)
       for (j=B_ext_offd_i[i]; j < B_ext_offd_i[i+1]; j++)
          B_ext_offd_j[j] = hypre_UnorderedIntMapGet(&col_map_offd_C_inverse, B_ext_offd_j[j]);

    if (num_cols_offd_C)
    {
       hypre_UnorderedIntMapDestroy(&col_map_offd_C_inverse);
    }

    hypre_TFree(my_diag_array);
    hypre_TFree(my_offd_array);

     if (num_cols_offd_B)
     {
         HYPRE_Int i;
         map_B_to_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_B);

#pragma omp parallel private(i)
         {
            HYPRE_Int i_begin, i_end;
            hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_C);

            HYPRE_Int cnt;
            if (i_end > i_begin)
            {
               cnt = hypre_LowerBound(col_map_offd_B, col_map_offd_B + num_cols_offd_B, col_map_offd_C[i_begin]) - col_map_offd_B;
            }

            for (i = i_begin; i < i_end && cnt < num_cols_offd_B; i++)
            {
               if (col_map_offd_C[i] == col_map_offd_B[cnt])
               {
                  map_B_to_C[cnt++] = i;
               }
            }
         }
      }
#else /* !HYPRE_CONCURRENT_HOPSCOTCH */

   HYPRE_Int *temp;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel 
#endif
   {
     HYPRE_Int size, rest, ii;
     HYPRE_Int ns, ne;
     HYPRE_Int i1, i, j;
     HYPRE_Int my_offd_size, my_diag_size;
     HYPRE_Int cnt_offd, cnt_diag;

     HYPRE_Int num_threads = hypre_NumActiveThreads();

     size = num_cols_offd_A/num_threads;
     rest = num_cols_offd_A - size*num_threads;
     ii = hypre_GetThreadNum();
     if (ii < rest)
     {
       ns = ii*size+ii;
       ne = (ii+1)*size+ii+1;
     }
     else
     {
       ns = ii*size+rest;
       ne = (ii+1)*size+rest;
     }

     my_diag_size = 0;
     my_offd_size = 0;
     for (i=ns; i < ne; i++)
     {
       B_ext_diag_i[i] = my_diag_size;
       B_ext_offd_i[i] = my_offd_size;
       for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            my_offd_size++;
         else
            my_diag_size++;
     }
     my_diag_array[ii] = my_diag_size;
     my_offd_array[ii] = my_offd_size;

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

     if (ii)
     {
       my_diag_size = my_diag_array[0];
       my_offd_size = my_offd_array[0];
       for (i1 = 1; i1 < ii; i1++)
       {
          my_diag_size += my_diag_array[i1];
          my_offd_size += my_offd_array[i1];
       }

       for (i1 = ns; i1 < ne; i1++)
       {
          B_ext_diag_i[i1] += my_diag_size;
          B_ext_offd_i[i1] += my_offd_size;
       }
     }
     else
     {
       B_ext_diag_size = 0;
       B_ext_offd_size = 0;
       for (i1 = 0; i1 < num_threads; i1++)
       {
          B_ext_diag_size += my_diag_array[i1];
          B_ext_offd_size += my_offd_array[i1];
       }
       B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
       B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

       if (B_ext_diag_size)
       {
          B_ext_diag_j = hypre_CTAlloc(HYPRE_Int, B_ext_diag_size);
          B_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, B_ext_diag_size);
       }
       if (B_ext_offd_size)
       {
          B_ext_offd_j = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size);
          B_ext_offd_data = hypre_CTAlloc(HYPRE_Complex, B_ext_offd_size);
       }
       if (B_ext_offd_size || num_cols_offd_B)
          temp = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size+num_cols_offd_B);
     }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

     cnt_offd = B_ext_offd_i[ns];
     cnt_diag = B_ext_diag_i[ns];
     for (i=ns; i < ne; i++)
     {
       for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
         {
            temp[cnt_offd] = Bs_ext_j[j];
            B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
            B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
         }
         else
         {
            B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
            B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
         }
     }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

     if (ii == 0)
     {
      HYPRE_Int        cnt;

      if (num_procs > 1)
      {
         hypre_CSRMatrixDestroy(Bs_ext);
         Bs_ext = NULL;
      }

      cnt = 0;
      if (B_ext_offd_size || num_cols_offd_B)
      {
         cnt = B_ext_offd_size;
         for (i=0; i < num_cols_offd_B; i++)
            temp[cnt++] = col_map_offd_B[i];
         if (cnt)
         {
            HYPRE_Int        value;
            hypre_qsort0(temp, 0, cnt-1);
            num_cols_offd_C = 1;
            value = temp[0];
            for (i=1; i < cnt; i++)
            {
               if (temp[i] > value)
               {
                  value = temp[i];
                  temp[num_cols_offd_C++] = value;
               }
            }
         }

         if (num_cols_offd_C)
            col_map_offd_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_C);

         for (i=0; i < num_cols_offd_C; i++)
            col_map_offd_C[i] = temp[i];

         hypre_TFree(temp);
      }
     }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

     for (i=ns; i < ne; i++)
        for (j=B_ext_offd_i[i]; j < B_ext_offd_i[i+1]; j++)
            B_ext_offd_j[j] = hypre_BinarySearch(col_map_offd_C, B_ext_offd_j[j],
                                           num_cols_offd_C);

    } /* end parallel region */

    hypre_TFree(my_diag_array);
    hypre_TFree(my_offd_array);

     if (num_cols_offd_B)
     {
         HYPRE_Int i, cnt;
         map_B_to_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_B);

         cnt = 0;
         for (i=0; i < num_cols_offd_C; i++)
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
               if (cnt == num_cols_offd_B) break;
            }
      }

#endif /* !HYPRE_CONCURRENT_HOPSCOTCH */

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif

   hypre_ParMatmul_RowSizes(
      /*&C_diag_i, &C_offd_i, &B_marker,*/
      &C_diag_i, &C_offd_i, 
      A_diag_i, A_diag_j, A_offd_i, A_offd_j,
      B_diag_i, B_diag_j, B_offd_i, B_offd_j,
      B_ext_diag_i, B_ext_diag_j, B_ext_offd_i, B_ext_offd_j,
      map_B_to_C,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_cols_offd_A, allsquare,
      num_cols_diag_B, num_cols_offd_B,
      num_cols_offd_C
      );

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;
   C_diag_data = hypre_CTAlloc(HYPRE_Complex, C_diag_size);
   C_diag_j    = hypre_CTAlloc(HYPRE_Int, C_diag_size);
   if (C_offd_size)
   { 
      C_offd_data = hypre_CTAlloc(HYPRE_Complex, C_offd_size);
      C_offd_j    = hypre_CTAlloc(HYPRE_Int, C_offd_size);
   } 

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_data and C_diag_j.
    *  Second Pass: Fill in C_offd_data and C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel 
#endif
   {
    HYPRE_Int *B_marker = NULL;
    HYPRE_Int ns, ne, size, rest, ii;
    HYPRE_Int i1, i2, i3, jj2, jj3;
    HYPRE_Int jj_row_begin_diag, jj_count_diag;
    HYPRE_Int jj_row_begin_offd, jj_count_offd;
    HYPRE_Int num_threads;
    HYPRE_Complex a_entry; /*, a_b_product;*/

    ii = hypre_GetThreadNum();
    num_threads = hypre_NumActiveThreads();
    size = num_rows_diag_A/num_threads;
    rest = num_rows_diag_A - size*num_threads;
    if (ii < rest)
    {
       ns = ii*size+ii;
       ne = (ii+1)*size+ii+1;
    }
    else
    {
       ns = ii*size+rest;
       ne = (ii+1)*size+rest;
    }
    jj_count_diag = C_diag_i[ns];
    jj_count_offd = C_offd_i[ns];
    if (num_cols_diag_B || num_cols_offd_C)
    B_marker = hypre_CTAlloc(HYPRE_Int, num_cols_diag_B+num_cols_offd_C);
    for (i1 = 0; i1 < num_cols_diag_B+num_cols_offd_C; i1++)
      B_marker[i1] = -1;

    /*-----------------------------------------------------------------------
     *  Loop over interior c-points.
     *-----------------------------------------------------------------------*/

    for (i1 = ns; i1 < ne; i1++)
    {

      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1} 
       *--------------------------------------------------------------------*/

      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare ) 
      {
         B_marker[i1] = jj_count_diag;
         C_diag_data[jj_count_diag] = zero;
         C_diag_j[jj_count_diag] = i1;
         jj_count_diag++;
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/
         
      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];
            a_entry = A_offd_data[jj2];
            
            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_ext.
             *-----------------------------------------------------------*/

            for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2+1]; jj3++)
            {
               i3 = num_cols_diag_B+B_ext_offd_j[jj3];
                  
               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  C_offd_data[jj_count_offd] = a_entry*B_ext_offd_data[jj3];
                  C_offd_j[jj_count_offd] = i3-num_cols_diag_B;
                  jj_count_offd++;
               }
               else
                  C_offd_data[B_marker[i3]] += a_entry*B_ext_offd_data[jj3];
            }
            for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2+1]; jj3++)
            {
               i3 = B_ext_diag_j[jj3];
               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  C_diag_data[jj_count_diag] = a_entry*B_ext_diag_data[jj3];
                  C_diag_j[jj_count_diag] = i3;
                  jj_count_diag++;
               }
               else
                  C_diag_data[B_marker[i3]] += a_entry*B_ext_diag_data[jj3];
            }
         }
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_diag.
       *-----------------------------------------------------------------*/

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
      {
         i2 = A_diag_j[jj2];
         a_entry = A_diag_data[jj2];
            
         /*-----------------------------------------------------------
          *  Loop over entries in row i2 of B_diag.
          *-----------------------------------------------------------*/

         for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
         {
            i3 = B_diag_j[jj3];
                  
            /*--------------------------------------------------------
             *  Check B_marker to see that C_{i1,i3} has not already
             *  been accounted for. If it has not, create a new entry.
             *  If it has, add new contribution.
             *--------------------------------------------------------*/

            if (B_marker[i3] < jj_row_begin_diag)
            {
               B_marker[i3] = jj_count_diag;
               C_diag_data[jj_count_diag] = a_entry*B_diag_data[jj3];
               C_diag_j[jj_count_diag] = i3;
               jj_count_diag++;
            }
            else
            {
               C_diag_data[B_marker[i3]] += a_entry*B_diag_data[jj3];
            }
         }
         if (num_cols_offd_B)
         {
            for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
            {
               i3 = num_cols_diag_B+map_B_to_C[B_offd_j[jj3]];
                  
               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  C_offd_data[jj_count_offd] = a_entry*B_offd_data[jj3];
                  C_offd_j[jj_count_offd] = i3-num_cols_diag_B;
                  jj_count_offd++;
               }
               else
               {
                  C_offd_data[B_marker[i3]] += a_entry*B_offd_data[jj3];
               }
            }
         }
      }
    }
    hypre_TFree(B_marker);
   } /*end parallel region */

   C = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
                                col_starts_B, num_cols_offd_C,
                                C_diag_size, C_offd_size);

   /* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRMatrixSetColStartsOwner(C,0);

   C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrixData(C_diag) = C_diag_data; 
   hypre_CSRMatrixI(C_diag) = C_diag_i; 
   hypre_CSRMatrixJ(C_diag) = C_diag_j; 

   C_offd = hypre_ParCSRMatrixOffd(C);
   hypre_CSRMatrixI(C_offd) = C_offd_i; 
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_CSRMatrixData(C_offd) = C_offd_data; 
      hypre_CSRMatrixJ(C_offd) = C_offd_j; 
      hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   }

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   hypre_TFree(B_ext_diag_i);
   if (B_ext_diag_size)
   {
      hypre_TFree(B_ext_diag_j);
      hypre_TFree(B_ext_diag_data);
   }
   hypre_TFree(B_ext_offd_i);
   if (B_ext_offd_size)
   {
      hypre_TFree(B_ext_offd_j);
      hypre_TFree(B_ext_offd_data);
   }
   if (num_cols_offd_B) hypre_TFree(map_B_to_C);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATMUL] += hypre_MPI_Wtime();
#endif

   return C;
}            

/* The following function was formerly part of hypre_ParCSRMatrixExtractBExt
   but the code was removed so it can be used for a corresponding function
   for Boolean matrices
 
   JSP: to allow communication overlapping, it returns comm_handle_idx and
   comm_handle_data. Before accessing B, they should be destroyed (including
   send_data contained in the comm_handle).
*/

void hypre_ParCSRMatrixExtractBExt_Arrays_Overlap(
   HYPRE_Int ** pB_ext_i,
   HYPRE_Int ** pB_ext_j,
   HYPRE_Complex ** pB_ext_data,
   HYPRE_Int ** pB_ext_row_map,
   HYPRE_Int * num_nonzeros,
   HYPRE_Int data,
   HYPRE_Int find_row_map,
   MPI_Comm comm,
   hypre_ParCSRCommPkg * comm_pkg,
   HYPRE_Int num_cols_B,
   HYPRE_Int num_recvs,
   HYPRE_Int num_sends,
   HYPRE_Int first_col_diag,
   HYPRE_Int * row_starts,
   HYPRE_Int * recv_vec_starts,
   HYPRE_Int * send_map_starts,
   HYPRE_Int * send_map_elmts,
   HYPRE_Int * diag_i,
   HYPRE_Int * diag_j,
   HYPRE_Int * offd_i,
   HYPRE_Int * offd_j,
   HYPRE_Int * col_map_offd,
   HYPRE_Real * diag_data,
   HYPRE_Real * offd_data,
   hypre_ParCSRCommHandle **comm_handle_idx,
   hypre_ParCSRCommHandle **comm_handle_data,
   HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd,
   HYPRE_Int skip_fine, /* 1 if only coarse points are needed */
   HYPRE_Int skip_same_sign /* 1 if only points that have the same sign are needed */
   // extended based long range interpolation: skip_fine = 1, skip_same_sign = 0 for S matrix, skip_fine = 1, skip_same_sign = 1 for A matrix
   // other interpolation: skip_fine = 0, skip_same_sign = 0
   )
{
   hypre_ParCSRCommHandle *comm_handle, *row_map_comm_handle = NULL;
   hypre_ParCSRCommPkg *tmp_comm_pkg;
   HYPRE_Int *B_int_i;
   HYPRE_Int *B_int_j;
   HYPRE_Int *B_ext_i;
   HYPRE_Int * B_ext_j;
   HYPRE_Complex * B_ext_data;
   HYPRE_Complex * B_int_data;
   HYPRE_Int * B_int_row_map;
   HYPRE_Int * B_ext_row_map;
   HYPRE_Int num_procs, my_id;
   HYPRE_Int *jdata_recv_vec_starts;
   HYPRE_Int *jdata_send_map_starts;
 
   HYPRE_Int i, j, k;
   HYPRE_Int start_index;
   /*HYPRE_Int jrow;*/
   HYPRE_Int num_rows_B_ext;
   HYPRE_Int *prefix_sum_workspace;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_Int first_row_index = row_starts[0];
#else
   HYPRE_Int first_row_index = row_starts[my_id];
   HYPRE_Int *send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
#endif

   num_rows_B_ext = recv_vec_starts[num_recvs];
   if ( num_rows_B_ext < 0 ) {  /* no B_ext, no communication */
      *pB_ext_i = NULL;
      *pB_ext_j = NULL;
      if ( data ) *pB_ext_data = NULL;
      if ( find_row_map ) *pB_ext_row_map = NULL;
      *num_nonzeros = 0;
      return;
   };
   B_int_i = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends]+1);
   B_ext_i = hypre_CTAlloc(HYPRE_Int, num_rows_B_ext+1);
   *pB_ext_i = B_ext_i;
   if ( find_row_map ) {
      B_int_row_map = hypre_CTAlloc( HYPRE_Int, send_map_starts[num_sends]+1 );
      B_ext_row_map = hypre_CTAlloc( HYPRE_Int, num_rows_B_ext+1 );
      *pB_ext_row_map = B_ext_row_map;
   };

   /*--------------------------------------------------------------------------
    * generate B_int_i through adding number of row-elements of offd and diag
    * for corresponding rows. B_int_i[j+1] contains the number of elements of
    * a row j (which is determined through send_map_elmts) 
    *--------------------------------------------------------------------------*/

   jdata_send_map_starts = hypre_CTAlloc(HYPRE_Int, num_sends+1);
   jdata_recv_vec_starts = hypre_CTAlloc(HYPRE_Int, num_recvs+1);
   jdata_send_map_starts[0] = B_int_i[0] = 0;

   /*HYPRE_Int prefix_sum_workspace[(hypre_NumThreads() + 1)*num_sends];*/
   prefix_sum_workspace = hypre_TAlloc(HYPRE_Int, (hypre_NumThreads() + 1)*num_sends);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,j,k)
#endif
   {
      /*HYPRE_Int counts[num_sends];*/
      HYPRE_Int *counts;
      counts = hypre_TAlloc(HYPRE_Int, num_sends);
      for (i=0; i < num_sends; i++)
      {
        HYPRE_Int j_begin, j_end;
        hypre_GetSimpleThreadPartition(&j_begin, &j_end, send_map_starts[i + 1] - send_map_starts[i]);
        j_begin += send_map_starts[i];
        j_end += send_map_starts[i];

        HYPRE_Int count = 0;
        if (skip_fine && skip_same_sign)
        {
#ifndef HYPRE_NO_GLOBAL_PARTITION
          HYPRE_Int send_proc = send_procs[i];
          HYPRE_Int send_proc_first_row = row_starts[send_proc];
          HYPRE_Int send_proc_last_row = row_starts[send_proc + 1];
#endif

          for (j = j_begin; j < j_end; j++)
          {
            HYPRE_Int jrow = send_map_elmts[j];
            HYPRE_Int len = 0;

            if (diag_data[diag_i[jrow]] >= 0)
            {
              for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
              {
                if (diag_data[k] < 0 && CF_marker[diag_j[k]] >= 0) len++;
              }
              for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
              {
#ifdef HYPRE_NO_GLOBAL_PARTITION
                if (offd_data[k] < 0) len++;
#else
                HYPRE_Int c = offd_j[k];
                HYPRE_Int c_global = col_map_offd[c];
                if (offd_data[k] < 0 && (CF_marker_offd[c] >= 0 || (c_global >= send_proc_first_row && c_global < send_proc_last_row))) len++;
#endif
              }
            }
            else
            {
              for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
              {
                if (diag_data[k] > 0 && CF_marker[diag_j[k]] >= 0) len++;
              }
              for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
              {
#ifdef HYPRE_NO_GLOBAL_PARTITION
                if (offd_data[k] > 0) len++;
#else
                HYPRE_Int c = offd_j[k];
                HYPRE_Int c_global = col_map_offd[c];
                if (offd_data[k] > 0 && (CF_marker_offd[c] >= 0 || (c_global >= send_proc_first_row && c_global < send_proc_last_row))) len++;
#endif
              }
            }

            B_int_i[j + 1] = len;
            count += len;
          }
        }
        else if (skip_fine)
        {
          for (j = j_begin; j < j_end; j++)
          {
            HYPRE_Int jrow = send_map_elmts[j];
            HYPRE_Int len = 0;

            for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
            {
              if (CF_marker[diag_j[k]] >= 0) len++;
            }
            for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
            {
              if (CF_marker_offd[offd_j[k]] >= 0) len++;
            }

            B_int_i[j + 1] = len;
            count += len;
          }
        }
        else
        {
          for (j = j_begin; j < j_end; j++)
          {
            HYPRE_Int jrow = send_map_elmts[j];
            HYPRE_Int len = diag_i[jrow + 1] - diag_i[jrow];
            len += offd_i[jrow + 1] - offd_i[jrow];
            B_int_i[j + 1] = len;
            count += len;
          }
        }

        if (find_row_map)
        {
          for (j = j_begin; j < j_end; j++)
          {
            HYPRE_Int jrow = send_map_elmts[j];
            B_int_row_map[j] = jrow + first_row_index;
          }
        }

        counts[i] = count;
      }

      hypre_prefix_sum_multiple(counts, jdata_send_map_starts + 1, num_sends, prefix_sum_workspace);

#ifdef HYPRE_USING_OPENMP
#pragma omp master
#endif
      {
        for (i = 1; i < num_sends; i++)
        {
           jdata_send_map_starts[i + 1] += jdata_send_map_starts[i];
        }

        /*--------------------------------------------------------------------------
         * initialize communication 
         *--------------------------------------------------------------------------*/

        comm_handle = hypre_ParCSRCommHandleCreate(11,comm_pkg,
                                                   &B_int_i[1],&(B_ext_i[1]) );
        if ( find_row_map )
        {
            /* scatter/gather B_int row numbers to form array of B_ext row numbers */
           row_map_comm_handle = hypre_ParCSRCommHandleCreate
              (11,comm_pkg, B_int_row_map, B_ext_row_map );
        }

        B_int_j = hypre_TAlloc(HYPRE_Int, jdata_send_map_starts[num_sends]);
        if (data) B_int_data = hypre_TAlloc(HYPRE_Complex, jdata_send_map_starts[num_sends]);
      }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
 
      for (i=0; i < num_sends; i++)
      {
        HYPRE_Int j_begin, j_end;
        hypre_GetSimpleThreadPartition(&j_begin, &j_end, send_map_starts[i + 1] - send_map_starts[i]);
        j_begin += send_map_starts[i];
        j_end += send_map_starts[i];

        HYPRE_Int count = counts[i] + jdata_send_map_starts[i];

        if (data)
        {
          if (skip_same_sign && skip_fine)
          {
#ifndef HYPRE_NO_GLOBAL_PARTITION
            HYPRE_Int send_proc = send_procs[i];
            HYPRE_Int send_proc_first_row = row_starts[send_proc];
            HYPRE_Int send_proc_last_row = row_starts[send_proc + 1];
#endif

            for (j = j_begin; j < j_end; j++)
            {
              HYPRE_Int jrow = send_map_elmts[j];
              /*HYPRE_Int count_begin = count;*/

              if (diag_data[diag_i[jrow]] >= 0)
              {
                for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                {
                  if (diag_data[k] < 0 && CF_marker[diag_j[k]] >= 0)
                  {
                    B_int_j[count] = diag_j[k]+first_col_diag;
                    B_int_data[count] = diag_data[k];
                    count++;
                  }
                }
                for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                {
                  HYPRE_Int c = offd_j[k];
                  HYPRE_Int c_global = col_map_offd[c];
#ifdef HYPRE_NO_GLOBAL_PARTITION
                  if (offd_data[k] < 0)
#else
                  if (offd_data[k] < 0 && (CF_marker_offd[c] >= 0 || (c_global >= send_proc_first_row && c_global < send_proc_last_row)))
#endif
                  {
                    B_int_j[count] = c_global;
                    B_int_data[count] = offd_data[k];
                    count++;
                  }
                }
              }
              else
              {
                for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                {
                  if (diag_data[k] > 0 && CF_marker[diag_j[k]] >= 0)
                  {
                    B_int_j[count] = diag_j[k]+first_col_diag;
                    B_int_data[count] = diag_data[k];
                    count++;
                  }
                }
                for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                {
                  HYPRE_Int c = offd_j[k];
                  HYPRE_Int c_global = col_map_offd[c];
#ifdef HYPRE_NO_GLOBAL_PARTITION
                  if (offd_data[k] > 0)
#else
                  if (offd_data[k] > 0 && (CF_marker_offd[c] >= 0 || (c_global >= send_proc_first_row && c_global < send_proc_last_row)))
#endif
                  {
                    B_int_j[count] = c_global;
                    B_int_data[count] = offd_data[k];
                    count++;
                  }
                }
              }
            }
          }
          else
          {
            for (j = j_begin; j < j_end; ++j) {
              HYPRE_Int jrow = send_map_elmts[j];
              for (k=diag_i[jrow]; k < diag_i[jrow+1]; k++)
              {
                B_int_j[count] = diag_j[k]+first_col_diag;
                B_int_data[count] = diag_data[k];
                count++;
              }
              for (k=offd_i[jrow]; k < offd_i[jrow+1]; k++)
              {
                B_int_j[count] = col_map_offd[offd_j[k]];
                B_int_data[count] = offd_data[k];
                count++;
              }
            }
          }
        } // data
        else
        {
          if (skip_fine)
          {
            for (j = j_begin; j < j_end; j++)
            {
              HYPRE_Int jrow = send_map_elmts[j];
              for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
              {
                if (CF_marker[diag_j[k]] >= 0)
                {
                  B_int_j[count] = diag_j[k] + first_col_diag;
                  count++;
                }
              }
              for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
              {
                if (CF_marker_offd[offd_j[k]] >= 0)
                {
                  B_int_j[count] = col_map_offd[offd_j[k]];
                  count++;
                }
              }
            }
          }
          else
          {
            for (j = j_begin; j < j_end; ++j) {
              HYPRE_Int jrow = send_map_elmts[j];
              for (k=diag_i[jrow]; k < diag_i[jrow+1]; k++)
              {
                B_int_j[count] = diag_j[k]+first_col_diag;
                count++;
              }
              for (k=offd_i[jrow]; k < offd_i[jrow+1]; k++)
              {
                B_int_j[count] = col_map_offd[offd_j[k]];
                count++;
              }
            }
          }
        } // !data
      } /* for each send target */
      hypre_TFree(counts);
   } /* omp parallel. JSP: this takes most of time in this function */
   hypre_TFree(prefix_sum_workspace);

   tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
   hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) =
      hypre_ParCSRCommPkgSendProcs(comm_pkg);
   hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) =
      hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_send_map_starts; 

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*--------------------------------------------------------------------------
    * after communication exchange B_ext_i[j+1] contains the number of elements
    * of a row j ! 
    * evaluate B_ext_i and compute *num_nonzeros for B_ext 
    *--------------------------------------------------------------------------*/

   for (i=0; i < num_recvs; i++)
      for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
         B_ext_i[j+1] += B_ext_i[j];

   *num_nonzeros = B_ext_i[num_rows_B_ext];

   *pB_ext_j = hypre_TAlloc(HYPRE_Int, *num_nonzeros);
   B_ext_j = *pB_ext_j;
   if (data) {
      *pB_ext_data = hypre_TAlloc(HYPRE_Complex, *num_nonzeros);
      B_ext_data = *pB_ext_data;
   };

   for (i=0; i < num_recvs; i++)
   {
      start_index = B_ext_i[recv_vec_starts[i]];
      *num_nonzeros = B_ext_i[recv_vec_starts[i+1]]-start_index;
      jdata_recv_vec_starts[i+1] = B_ext_i[recv_vec_starts[i+1]];
   }

   hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_recv_vec_starts;

   *comm_handle_idx = hypre_ParCSRCommHandleCreate(11,tmp_comm_pkg,B_int_j,B_ext_j);
   if (data)
   {
      *comm_handle_data = hypre_ParCSRCommHandleCreate(1,tmp_comm_pkg,B_int_data,
                                                 B_ext_data);
   }

   if (row_map_comm_handle)
   {
      hypre_ParCSRCommHandleDestroy(row_map_comm_handle);
      row_map_comm_handle = NULL;
   }

   hypre_TFree(jdata_send_map_starts);
   hypre_TFree(jdata_recv_vec_starts);
   hypre_TFree(tmp_comm_pkg);
   hypre_TFree(B_int_i);
   if ( find_row_map ) hypre_TFree(B_int_row_map);

   /* end generic part */
}

void hypre_ParCSRMatrixExtractBExt_Arrays(
   HYPRE_Int ** pB_ext_i,
   HYPRE_Int ** pB_ext_j,
   HYPRE_Complex ** pB_ext_data,
   HYPRE_Int ** pB_ext_row_map,
   HYPRE_Int * num_nonzeros,
   HYPRE_Int data,
   HYPRE_Int find_row_map,
   MPI_Comm comm,
   hypre_ParCSRCommPkg * comm_pkg,
   HYPRE_Int num_cols_B,
   HYPRE_Int num_recvs,
   HYPRE_Int num_sends,
   HYPRE_Int first_col_diag,
   HYPRE_Int * row_starts,
   HYPRE_Int * recv_vec_starts,
   HYPRE_Int * send_map_starts,
   HYPRE_Int * send_map_elmts,
   HYPRE_Int * diag_i,
   HYPRE_Int * diag_j,
   HYPRE_Int * offd_i,
   HYPRE_Int * offd_j,
   HYPRE_Int * col_map_offd,
   HYPRE_Real * diag_data,
   HYPRE_Real * offd_data
   )
{
   hypre_ParCSRCommHandle *comm_handle_idx, *comm_handle_data;

   hypre_ParCSRMatrixExtractBExt_Arrays_Overlap(
      pB_ext_i, pB_ext_j, pB_ext_data, pB_ext_row_map, num_nonzeros,
      data, find_row_map, comm, comm_pkg, num_cols_B, num_recvs, num_sends,
      first_col_diag, row_starts, recv_vec_starts, send_map_starts, send_map_elmts,
      diag_i, diag_j, offd_i, offd_j, col_map_offd, diag_data, offd_data,
      &comm_handle_idx, &comm_handle_data,
      NULL, NULL,
      0, 0);

   HYPRE_Int *send_idx = (HYPRE_Int *)comm_handle_idx->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_idx);
   hypre_TFree(send_idx);

   if (data)
   {
      HYPRE_Real *send_data = (HYPRE_Real *)comm_handle_data->send_data;
      hypre_ParCSRCommHandleDestroy(comm_handle_data);
      hypre_TFree(send_data);
   }
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBExt : extracts rows from B which are located on 
 * other processors and needed for multiplication with A locally. The rows
 * are returned as CSRMatrix.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix * 
hypre_ParCSRMatrixExtractBExt_Overlap( hypre_ParCSRMatrix *B,
                               hypre_ParCSRMatrix *A,
                               HYPRE_Int data,
                               hypre_ParCSRCommHandle **comm_handle_idx,
                               hypre_ParCSRCommHandle **comm_handle_data,
                               HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd,
                               HYPRE_Int skip_fine, HYPRE_Int skip_same_sign )
{
   MPI_Comm  comm = hypre_ParCSRMatrixComm(B);
   HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(B);
   /*HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(B);*/
   HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int num_recvs;
   HYPRE_Int *recv_vec_starts;
   HYPRE_Int num_sends;
   HYPRE_Int *send_map_starts;
   HYPRE_Int *send_map_elmts;
 
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(B);

   HYPRE_Int *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Real *diag_data = hypre_CSRMatrixData(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(B);

   HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Real *offd_data = hypre_CSRMatrixData(offd);

   HYPRE_Int num_cols_B, num_nonzeros;
   HYPRE_Int num_rows_B_ext;

   hypre_CSRMatrix *B_ext;

   HYPRE_Int *B_ext_i;
   HYPRE_Int *B_ext_j;
   HYPRE_Complex *B_ext_data;
   HYPRE_Int *idummy;

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings 
    *--------------------------------------------------------------------*/

   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }
    
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   hypre_ParCSRMatrixExtractBExt_Arrays_Overlap
      ( &B_ext_i, &B_ext_j, &B_ext_data, &idummy,
        &num_nonzeros,
        data, 0, comm, comm_pkg,
        num_cols_B, num_recvs, num_sends,
        first_col_diag, B->row_starts,
        recv_vec_starts, send_map_starts, send_map_elmts,
        diag_i, diag_j, offd_i, offd_j, col_map_offd,
        diag_data, offd_data,
        comm_handle_idx, comm_handle_data,
        CF_marker, CF_marker_offd,
        skip_fine, skip_same_sign
         );

   B_ext = hypre_CSRMatrixCreate(num_rows_B_ext,num_cols_B,num_nonzeros);
   hypre_CSRMatrixI(B_ext) = B_ext_i;
   hypre_CSRMatrixJ(B_ext) = B_ext_j;
   if (data) hypre_CSRMatrixData(B_ext) = B_ext_data;

   return B_ext;
}

hypre_CSRMatrix * 
hypre_ParCSRMatrixExtractBExt( hypre_ParCSRMatrix *B,
                               hypre_ParCSRMatrix *A,
                               HYPRE_Int data )
{
   hypre_ParCSRCommHandle *comm_handle_idx, *comm_handle_data;

   hypre_CSRMatrix *B_ext = hypre_ParCSRMatrixExtractBExt_Overlap(B, A, data, &comm_handle_idx, &comm_handle_data, NULL, NULL, 0, 0);

   HYPRE_Int *send_idx = (HYPRE_Int *)comm_handle_idx->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_idx);
   hypre_TFree(send_idx);

   if (data)
   {
      HYPRE_Real *send_data = (HYPRE_Real *)comm_handle_data->send_data;
      hypre_ParCSRCommHandleDestroy(comm_handle_data);
      hypre_TFree(send_data);
   }

   return B_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixTranspose
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixTranspose( hypre_ParCSRMatrix *A,
                             hypre_ParCSRMatrix **AT_ptr,
                             HYPRE_Int          data ) 
{
   hypre_ParCSRCommHandle *comm_handle;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix      *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int  num_cols = hypre_ParCSRMatrixNumCols(A);
   HYPRE_Int  first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int *row_starts = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int *col_starts = hypre_ParCSRMatrixColStarts(A);

   HYPRE_Int        num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int        ierr = 0;
   HYPRE_Int        num_sends, num_recvs, num_cols_offd_AT; 
   HYPRE_Int        i, j, k, index, counter, j_row;
   HYPRE_Int        value;

   hypre_ParCSRMatrix *AT;
   hypre_CSRMatrix *AT_diag;
   hypre_CSRMatrix *AT_offd;
   hypre_CSRMatrix *AT_tmp;

   HYPRE_Int first_row_index_AT, first_col_diag_AT;
   HYPRE_Int local_num_rows_AT, local_num_cols_AT;

   HYPRE_Int *AT_tmp_i;
   HYPRE_Int *AT_tmp_j;
   HYPRE_Complex *AT_tmp_data;

   HYPRE_Int *AT_buf_i;
   HYPRE_Int *AT_buf_j;
   HYPRE_Complex *AT_buf_data;

   HYPRE_Int *AT_offd_i;
   HYPRE_Int *AT_offd_j;
   HYPRE_Complex *AT_offd_data;
   HYPRE_Int *col_map_offd_AT;
   HYPRE_Int *row_starts_AT;
   HYPRE_Int *col_starts_AT;

   HYPRE_Int num_procs, my_id;

   HYPRE_Int *recv_procs;
   HYPRE_Int *send_procs;
   HYPRE_Int *recv_vec_starts;
   HYPRE_Int *send_map_starts;
   HYPRE_Int *send_map_elmts;
   HYPRE_Int *tmp_recv_vec_starts;
   HYPRE_Int *tmp_send_map_starts;
   hypre_ParCSRCommPkg *tmp_comm_pkg;

   hypre_MPI_Comm_size(comm,&num_procs);   
   hypre_MPI_Comm_rank(comm,&my_id);
  
   num_cols_offd_AT = 0;
   counter = 0;
   AT_offd_j = NULL;
   AT_offd_data = NULL;
   col_map_offd_AT = NULL;
 
   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   if (num_procs > 1)
   {
      hypre_CSRMatrixTranspose (A_offd, &AT_tmp, data);

      AT_tmp_i = hypre_CSRMatrixI(AT_tmp);
      AT_tmp_j = hypre_CSRMatrixJ(AT_tmp);
      if (data) AT_tmp_data = hypre_CSRMatrixData(AT_tmp);

      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

      AT_buf_i = hypre_CTAlloc(HYPRE_Int,send_map_starts[num_sends]); 

      for (i=0; i < AT_tmp_i[num_cols_offd]; i++)
         AT_tmp_j[i] += first_row_index;

      for (i=0; i < num_cols_offd; i++)
         AT_tmp_i[i] = AT_tmp_i[i+1]-AT_tmp_i[i];
        
      comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, AT_tmp_i, AT_buf_i);
   }

   hypre_CSRMatrixTranspose( A_diag, &AT_diag, data);

   AT_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols+1);

   if (num_procs > 1)
   {   
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      tmp_send_map_starts = hypre_CTAlloc(HYPRE_Int,num_sends+1);
      tmp_recv_vec_starts = hypre_CTAlloc(HYPRE_Int,num_recvs+1);

      tmp_send_map_starts[0] = send_map_starts[0];
      for (i=0; i < num_sends; i++)
      {
         tmp_send_map_starts[i+1] = tmp_send_map_starts[i];
         for (j=send_map_starts[i]; j < send_map_starts[i+1]; j++)
         {
            tmp_send_map_starts[i+1] += AT_buf_i[j];
            AT_offd_i[send_map_elmts[j]+1] += AT_buf_i[j];
         }
      }
      for (i=0; i < num_cols; i++)
         AT_offd_i[i+1] += AT_offd_i[i];

      tmp_recv_vec_starts[0] = recv_vec_starts[0];
      for (i=0; i < num_recvs; i++)
      {
         tmp_recv_vec_starts[i+1] = tmp_recv_vec_starts[i];
         for (j=recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
         {
            tmp_recv_vec_starts[i+1] +=  AT_tmp_i[j];
         }
      }

      tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
      hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
      hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
      hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
      hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = recv_procs;
      hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = send_procs;
      hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = tmp_recv_vec_starts;
      hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = tmp_send_map_starts;

      AT_buf_j = hypre_CTAlloc(HYPRE_Int,tmp_send_map_starts[num_sends]);
      comm_handle = hypre_ParCSRCommHandleCreate(12, tmp_comm_pkg, AT_tmp_j,
                                                 AT_buf_j);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      if (data)
      {
         AT_buf_data = hypre_CTAlloc(HYPRE_Complex,tmp_send_map_starts[num_sends]);
         comm_handle = hypre_ParCSRCommHandleCreate(2,tmp_comm_pkg,AT_tmp_data,
                                                    AT_buf_data);
         hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
      }

      hypre_TFree(tmp_recv_vec_starts);
      hypre_TFree(tmp_send_map_starts);
      hypre_TFree(tmp_comm_pkg);
      hypre_CSRMatrixDestroy(AT_tmp);

      if (AT_offd_i[num_cols])
      {
         AT_offd_j = hypre_CTAlloc(HYPRE_Int, AT_offd_i[num_cols]);
         if (data) AT_offd_data = hypre_CTAlloc(HYPRE_Complex, AT_offd_i[num_cols]);
      }
      else
      {
         AT_offd_j = NULL;
         AT_offd_data = NULL;
      }
         
      counter = 0;
      for (i=0; i < num_sends; i++)
      {
         for (j=send_map_starts[i]; j < send_map_starts[i+1]; j++)
         {
            j_row = send_map_elmts[j];
            index = AT_offd_i[j_row];
            for (k=0; k < AT_buf_i[j]; k++)
            {
               if (data) AT_offd_data[index] = AT_buf_data[counter];
               AT_offd_j[index++] = AT_buf_j[counter++];
            }
            AT_offd_i[j_row] = index;
         }
      }
      for (i=num_cols; i > 0; i--)
         AT_offd_i[i] = AT_offd_i[i-1];
      AT_offd_i[0] = 0;

      if (counter)
      {
         hypre_qsort0(AT_buf_j,0,counter-1);
         num_cols_offd_AT = 1;
         value = AT_buf_j[0];
         for (i=1; i < counter; i++)
         {
            if (value < AT_buf_j[i])
            {
               AT_buf_j[num_cols_offd_AT++] = AT_buf_j[i];
               value = AT_buf_j[i];
            }
         }
      }

      if (num_cols_offd_AT)
         col_map_offd_AT = hypre_CTAlloc(HYPRE_Int, num_cols_offd_AT);
      else
         col_map_offd_AT = NULL;

      for (i=0; i < num_cols_offd_AT; i++)
         col_map_offd_AT[i] = AT_buf_j[i];

      hypre_TFree(AT_buf_i);
      hypre_TFree(AT_buf_j);
      if (data) hypre_TFree(AT_buf_data);

      for (i=0; i < counter; i++)
         AT_offd_j[i] = hypre_BinarySearch(col_map_offd_AT,AT_offd_j[i],
                                           num_cols_offd_AT);
   }

   AT_offd = hypre_CSRMatrixCreate(num_cols,num_cols_offd_AT,counter);
   hypre_CSRMatrixI(AT_offd) = AT_offd_i;
   hypre_CSRMatrixJ(AT_offd) = AT_offd_j;
   hypre_CSRMatrixData(AT_offd) = AT_offd_data;

#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts_AT = hypre_CTAlloc(HYPRE_Int, 2);
   for (i=0; i < 2; i++)
      row_starts_AT[i] = col_starts[i];

   if (row_starts != col_starts)
   {
      col_starts_AT = hypre_CTAlloc(HYPRE_Int,2);
      for (i=0; i < 2; i++)
         col_starts_AT[i] = row_starts[i];
   }
   else
   {
      col_starts_AT = row_starts_AT;
   }

   first_row_index_AT =  row_starts_AT[0];
   first_col_diag_AT =  col_starts_AT[0];

   local_num_rows_AT = row_starts_AT[1]-first_row_index_AT ;
   local_num_cols_AT = col_starts_AT[1]-first_col_diag_AT;

#else
   row_starts_AT = hypre_CTAlloc(HYPRE_Int,num_procs+1);
   for (i=0; i < num_procs+1; i++)
      row_starts_AT[i] = col_starts[i];

   if (row_starts != col_starts)
   {
      col_starts_AT = hypre_CTAlloc(HYPRE_Int,num_procs+1);
      for (i=0; i < num_procs+1; i++)
         col_starts_AT[i] = row_starts[i];
   }
   else
   {
      col_starts_AT = row_starts_AT;
   }
   first_row_index_AT =  row_starts_AT[my_id];
   first_col_diag_AT =  col_starts_AT[my_id];

   local_num_rows_AT = row_starts_AT[my_id+1]-first_row_index_AT ;
   local_num_cols_AT = col_starts_AT[my_id+1]-first_col_diag_AT;

#endif

   AT = hypre_CTAlloc(hypre_ParCSRMatrix,1);
   hypre_ParCSRMatrixComm(AT) = comm;
   hypre_ParCSRMatrixDiag(AT) = AT_diag;
   hypre_ParCSRMatrixOffd(AT) = AT_offd;
   hypre_ParCSRMatrixGlobalNumRows(AT) = hypre_ParCSRMatrixGlobalNumCols(A);
   hypre_ParCSRMatrixGlobalNumCols(AT) = hypre_ParCSRMatrixGlobalNumRows(A);
   hypre_ParCSRMatrixRowStarts(AT) = row_starts_AT;
   hypre_ParCSRMatrixColStarts(AT) = col_starts_AT;
   hypre_ParCSRMatrixColMapOffd(AT) = col_map_offd_AT;
 
   hypre_ParCSRMatrixFirstRowIndex(AT) = first_row_index_AT;
   hypre_ParCSRMatrixFirstColDiag(AT) = first_col_diag_AT;

   hypre_ParCSRMatrixLastRowIndex(AT) = first_row_index_AT + local_num_rows_AT - 1;
   hypre_ParCSRMatrixLastColDiag(AT) = first_col_diag_AT + local_num_cols_AT - 1;

   hypre_ParCSRMatrixOwnsData(AT) = 1;
   hypre_ParCSRMatrixOwnsRowStarts(AT) = 1;
   hypre_ParCSRMatrixOwnsColStarts(AT) = 1;
   if (row_starts_AT == col_starts_AT)
      hypre_ParCSRMatrixOwnsColStarts(AT) = 0;

   hypre_ParCSRMatrixCommPkg(AT) = NULL;
   hypre_ParCSRMatrixCommPkgT(AT) = NULL;

   hypre_ParCSRMatrixRowindices(AT) = NULL;
   hypre_ParCSRMatrixRowvalues(AT) = NULL;
   hypre_ParCSRMatrixGetrowactive(AT) = 0;

   *AT_ptr = AT;
  
   return ierr;
}

/* -----------------------------------------------------------------------------
 * generate a parallel spanning tree (for Maxwell Equation)
 * G_csr is the node to edge connectivity matrix
 * ----------------------------------------------------------------------------- */

void hypre_ParCSRMatrixGenSpanningTree( hypre_ParCSRMatrix *G_csr,
                                        HYPRE_Int **indices,
                                        HYPRE_Int G_type )
{
   HYPRE_Int nrows_G, ncols_G, *G_diag_i, *G_diag_j, *GT_diag_mat, i, j, k, edge;
   HYPRE_Int *nodes_marked, *edges_marked, *queue, queue_tail, queue_head, node;
   HYPRE_Int mypid, nprocs, n_children, *children, nsends, *send_procs, *recv_cnts;
   HYPRE_Int nrecvs, *recv_procs, n_proc_array, *proc_array, *pgraph_i, *pgraph_j;
   HYPRE_Int parent, proc, proc2, node2, found, *t_indices, tree_size, *T_diag_i;
   HYPRE_Int *T_diag_j, *counts, offset;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg *comm_pkg;
   hypre_CSRMatrix     *G_diag;

   /* fetch G matrix (G_type = 0 ==> node to edge) */

   if (G_type == 0)
   {
      nrows_G = hypre_ParCSRMatrixGlobalNumRows(G_csr);
      ncols_G = hypre_ParCSRMatrixGlobalNumCols(G_csr);
      G_diag = hypre_ParCSRMatrixDiag(G_csr);
      G_diag_i = hypre_CSRMatrixI(G_diag);
      G_diag_j = hypre_CSRMatrixJ(G_diag);
   }
   else
   {
      nrows_G = hypre_ParCSRMatrixGlobalNumCols(G_csr);
      ncols_G = hypre_ParCSRMatrixGlobalNumRows(G_csr);
      G_diag = hypre_ParCSRMatrixDiag(G_csr);
      T_diag_i = hypre_CSRMatrixI(G_diag);
      T_diag_j = hypre_CSRMatrixJ(G_diag);
      counts = (HYPRE_Int *) malloc(nrows_G * sizeof(HYPRE_Int));
      for (i = 0; i < nrows_G; i++) counts[i] = 0;
      for (i = 0; i < T_diag_i[ncols_G]; i++) counts[T_diag_j[i]]++;
      G_diag_i = (HYPRE_Int *) malloc((nrows_G+1) * sizeof(HYPRE_Int));
      G_diag_j = (HYPRE_Int *) malloc(T_diag_i[ncols_G] * sizeof(HYPRE_Int));
      G_diag_i[0] = 0;
      for (i = 1; i <= nrows_G; i++) G_diag_i[i] = G_diag_i[i-1] + counts[i-1];
      for (i = 0; i < ncols_G; i++)
      {
         for (j = T_diag_i[i]; j < T_diag_i[i+1]; j++)
         {
            k = T_diag_j[j];
            offset = G_diag_i[k]++;
            G_diag_j[offset] = i;
         }
      }
      G_diag_i[0] = 0;
      for (i = 1; i <= nrows_G; i++) G_diag_i[i] = G_diag_i[i-1] + counts[i-1];
      free(counts);
   }

   /* form G transpose in special form (2 nodes per edge max) */

   GT_diag_mat = (HYPRE_Int *) malloc(2 * ncols_G * sizeof(HYPRE_Int));
   for (i = 0; i < 2 * ncols_G; i++) GT_diag_mat[i] = -1;
   for (i = 0; i < nrows_G; i++)
   {
      for (j = G_diag_i[i]; j < G_diag_i[i+1]; j++)
      {
         edge = G_diag_j[j];
         if (GT_diag_mat[edge*2] == -1) GT_diag_mat[edge*2] = i;
         else                           GT_diag_mat[edge*2+1] = i;
      }
   }

   /* BFS on the local matrix graph to find tree */

   nodes_marked = (HYPRE_Int *) malloc(nrows_G * sizeof(HYPRE_Int));
   edges_marked = (HYPRE_Int *) malloc(ncols_G * sizeof(HYPRE_Int));
   for (i = 0; i < nrows_G; i++) nodes_marked[i] = 0; 
   for (i = 0; i < ncols_G; i++) edges_marked[i] = 0; 
   queue = (HYPRE_Int *) malloc(nrows_G * sizeof(HYPRE_Int));
   queue_head = 0;
   queue_tail = 1;
   queue[0] = 0;
   nodes_marked[0] = 1;
   while ((queue_tail-queue_head) > 0)
   {
      node = queue[queue_tail-1];
      queue_tail--;
      for (i = G_diag_i[node]; i < G_diag_i[node+1]; i++)
      {
         edge = G_diag_j[i]; 
         if (edges_marked[edge] == 0)
         {
            if (GT_diag_mat[2*edge+1] != -1)
            {
               node2 = GT_diag_mat[2*edge];
               if (node2 == node) node2 = GT_diag_mat[2*edge+1];
               if (nodes_marked[node2] == 0)
               {
                  nodes_marked[node2] = 1;
                  edges_marked[edge] = 1;
                  queue[queue_tail] = node2;
                  queue_tail++;
               }
            }
         }
      }
   }
   free(nodes_marked);
   free(queue);
   free(GT_diag_mat);

   /* fetch the communication information from */

   comm = hypre_ParCSRMatrixComm(G_csr);
   hypre_MPI_Comm_rank(comm, &mypid);
   hypre_MPI_Comm_size(comm, &nprocs);
   comm_pkg = hypre_ParCSRMatrixCommPkg(G_csr);
   if (nprocs == 1 && comm_pkg == NULL)
   {

      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) G_csr);

      comm_pkg = hypre_ParCSRMatrixCommPkg(G_csr);
   }

   /* construct processor graph based on node-edge connection */
   /* (local edges connected to neighbor processor nodes)     */

   n_children = 0;
   nrecvs = nsends = 0;
   if (nprocs > 1)
   {
      nsends     = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      nrecvs     = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      proc_array = NULL;
      if ((nsends+nrecvs) > 0)
      {
         n_proc_array = 0;
         proc_array = (HYPRE_Int *) malloc((nsends+nrecvs) * sizeof(HYPRE_Int));
         for (i = 0; i < nsends; i++) proc_array[i] = send_procs[i];
         for (i = 0; i < nrecvs; i++) proc_array[nsends+i] = recv_procs[i];
         hypre_qsort0(proc_array, 0, nsends+nrecvs-1); 
         n_proc_array = 1;
         for (i = 1; i < nrecvs+nsends; i++) 
            if (proc_array[i] != proc_array[n_proc_array])
               proc_array[n_proc_array++] = proc_array[i];
      }
      pgraph_i = (HYPRE_Int *) malloc((nprocs+1) * sizeof(HYPRE_Int));
      recv_cnts = (HYPRE_Int *) malloc(nprocs * sizeof(HYPRE_Int));
      hypre_MPI_Allgather(&n_proc_array, 1, HYPRE_MPI_INT, recv_cnts, 1,
                          HYPRE_MPI_INT, comm);
      pgraph_i[0] = 0;
      for (i = 1; i <= nprocs; i++)
         pgraph_i[i] = pgraph_i[i-1] + recv_cnts[i-1];
      pgraph_j = (HYPRE_Int *) malloc(pgraph_i[nprocs] * sizeof(HYPRE_Int));
      hypre_MPI_Allgatherv(proc_array, n_proc_array, HYPRE_MPI_INT, pgraph_j,
                           recv_cnts, pgraph_i, HYPRE_MPI_INT, comm);
      free(recv_cnts);

      /* BFS on the processor graph to determine parent and children */

      nodes_marked = (HYPRE_Int *) malloc(nprocs * sizeof(HYPRE_Int));
      for (i = 0; i < nprocs; i++) nodes_marked[i] = -1; 
      queue = (HYPRE_Int *) malloc(nprocs * sizeof(HYPRE_Int));
      queue_head = 0;
      queue_tail = 1;
      node = 0;
      queue[0] = node;
      while ((queue_tail-queue_head) > 0)
      {
         proc = queue[queue_tail-1];
         queue_tail--;
         for (i = pgraph_i[proc]; i < pgraph_i[proc+1]; i++)
         {
            proc2 = pgraph_j[i]; 
            if (nodes_marked[proc2] < 0)
            {
               nodes_marked[proc2] = proc;
               queue[queue_tail] = proc2;
               queue_tail++;
            }
         }
      }
      parent = nodes_marked[mypid];
      n_children = 0;
      for (i = 0; i < nprocs; i++) if (nodes_marked[i] == mypid) n_children++;
      if (n_children == 0) {n_children = 0; children = NULL;}
      else
      {
         children = (HYPRE_Int *) malloc(n_children * sizeof(HYPRE_Int));
         n_children = 0;
         for (i = 0; i < nprocs; i++) 
            if (nodes_marked[i] == mypid) children[n_children++] = i;
      } 
      free(nodes_marked);
      free(queue);
      free(pgraph_i);
      free(pgraph_j);
   }

   /* first, connection with my parent : if the edge in my parent *
    * is incident to one of my nodes, then my parent will mark it */

   found = 0;
   for (i = 0; i < nrecvs; i++)
   {
      proc = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      if (proc == parent)
      {
         found = 1;
         break;
      }
   }

   /* but if all the edges connected to my parent are on my side, *
    * then I will just pick one of them as tree edge              */

   if (found == 0)
   {
      for (i = 0; i < nsends; i++)
      {
         proc = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
         if (proc == parent)
         {
            k = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
            edge = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,k);
            edges_marked[edge] = 1;
            break;
         }
      }
   }
   
   /* next, if my processor has an edge incident on one node in my *
    * child, put this edge on the tree. But if there is no such    *
    * edge, then I will assume my child will pick up an edge       */

   for (j = 0; j < n_children; j++)
   {
      proc = children[j];
      for (i = 0; i < nsends; i++)
      {
         proc2 = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
         if (proc == proc2)
         {
            k = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
            edge = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,k);
            edges_marked[edge] = 1;
            break;
         }
      }
   }
   if (n_children > 0) free(children);

   /* count the size of the tree */

   tree_size = 0;
   for (i = 0; i < ncols_G; i++)
      if (edges_marked[i] == 1) tree_size++;
   t_indices = (HYPRE_Int *) malloc((tree_size+1) * sizeof(HYPRE_Int));
   t_indices[0] = tree_size;
   tree_size = 1;
   for (i = 0; i < ncols_G; i++)
      if (edges_marked[i] == 1) t_indices[tree_size++] = i;
   (*indices) = t_indices;
   free(edges_marked);
   if (G_type != 0)
   {
      free(G_diag_i);
      free(G_diag_j);
   }
}

/* -----------------------------------------------------------------------------
 * extract submatrices based on given indices
 * ----------------------------------------------------------------------------- */

void hypre_ParCSRMatrixExtractSubmatrices( hypre_ParCSRMatrix *A_csr,
                                           HYPRE_Int *indices2,
                                           hypre_ParCSRMatrix ***submatrices )
{
   HYPRE_Int    nindices, *indices, nrows_A, *A_diag_i, *A_diag_j, mypid, nprocs;
   HYPRE_Int    i, j, k, *proc_offsets1, *proc_offsets2, *itmp_array, *exp_indices;
   HYPRE_Int    nnz11, nnz12, nnz21, nnz22, col, ncols_offd, nnz_offd, nnz_diag;
   HYPRE_Int    global_nrows, global_ncols, *row_starts, *col_starts, nrows, nnz;
   HYPRE_Int    *diag_i, *diag_j, row, *offd_i;
   HYPRE_Complex *A_diag_a, *diag_a;
   hypre_ParCSRMatrix *A11_csr, *A12_csr, *A21_csr, *A22_csr;
   hypre_CSRMatrix    *A_diag, *diag, *offd;
   MPI_Comm           comm;

   /* -----------------------------------------------------
    * first make sure the incoming indices are in order
    * ----------------------------------------------------- */

   nindices = indices2[0];
   indices  = &(indices2[1]);
   hypre_qsort0(indices, 0, nindices-1);

   /* -----------------------------------------------------
    * fetch matrix information
    * ----------------------------------------------------- */

   nrows_A = hypre_ParCSRMatrixGlobalNumRows(A_csr);
   A_diag = hypre_ParCSRMatrixDiag(A_csr);
   A_diag_i = hypre_CSRMatrixI(A_diag);
   A_diag_j = hypre_CSRMatrixJ(A_diag);
   A_diag_a = hypre_CSRMatrixData(A_diag);
   comm = hypre_ParCSRMatrixComm(A_csr);
   hypre_MPI_Comm_rank(comm, &mypid);
   hypre_MPI_Comm_size(comm, &nprocs);
   if (nprocs > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ExtractSubmatrices: cannot handle nprocs > 1 yet.\n");
      exit(1);
   }

   /* -----------------------------------------------------
    * compute new matrix dimensions
    * ----------------------------------------------------- */

   proc_offsets1 = (HYPRE_Int *) malloc((nprocs+1) * sizeof(HYPRE_Int));
   proc_offsets2 = (HYPRE_Int *) malloc((nprocs+1) * sizeof(HYPRE_Int));
   hypre_MPI_Allgather(&nindices, 1, HYPRE_MPI_INT, proc_offsets1, 1,
                       HYPRE_MPI_INT, comm);
   k = 0;
   for (i = 0; i < nprocs; i++) 
   {
      j = proc_offsets1[i];
      proc_offsets1[i] = k;
      k += j;
   } 
   proc_offsets1[nprocs] = k;
   itmp_array = hypre_ParCSRMatrixRowStarts(A_csr);
   for (i = 0; i <= nprocs; i++) 
      proc_offsets2[i] = itmp_array[i] - proc_offsets1[i];

   /* -----------------------------------------------------
    * assign id's to row and col for later processing
    * ----------------------------------------------------- */

   exp_indices = (HYPRE_Int *) malloc(nrows_A * sizeof(HYPRE_Int));
   for (i = 0; i < nrows_A; i++) exp_indices[i] = -1;
   for (i = 0; i < nindices; i++) 
   {
      if (exp_indices[indices[i]] == -1) exp_indices[indices[i]] = i;
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ExtractSubmatrices: wrong index %d %d\n");
         exit(1);
      }
   }
   k = 0;
   for (i = 0; i < nrows_A; i++) 
   {
      if (exp_indices[i] < 0)
      {
         exp_indices[i] = - k - 1;
         k++;
      }
   }

   /* -----------------------------------------------------
    * compute number of nonzeros for each block
    * ----------------------------------------------------- */

   nnz11 = nnz12 = nnz21 = nnz22 = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) nnz11++;
            else                       nnz12++;
         }
      }
      else
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) nnz21++;
            else                       nnz22++;
         }
      }
   }

   /* -----------------------------------------------------
    * create A11 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz11;
#ifdef HYPRE_NO_GLOBAL_PARTITION
   /* This case is not yet implemented! */
   global_nrows = 0;
   global_ncols = 0;
   row_starts = NULL;
   col_starts = NULL;
#else
   global_nrows = proc_offsets1[nprocs];
   global_ncols = proc_offsets1[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   col_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets1[i];
      col_starts[i] = proc_offsets1[i];
   }
#endif
   A11_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd); 
   nrows = nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   diag_j = hypre_CTAlloc(HYPRE_Int, nnz_diag);
   diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A11_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   for (i = 0; i <= nrows; i++) offd_i[i] = 0;
   offd = hypre_ParCSRMatrixOffd(A11_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * create A12 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz12;
   global_nrows = proc_offsets1[nprocs];
   global_ncols = proc_offsets2[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   col_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets1[i];
      col_starts[i] = proc_offsets2[i];
   }
   A12_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd); 
   nrows = nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   diag_j = hypre_CTAlloc(HYPRE_Int, nnz_diag);
   diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0)
            {
               diag_j[nnz] = - exp_indices[col] - 1;
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   if (nnz > nnz_diag) hypre_error(HYPRE_ERROR_GENERIC); 
		/*hypre_printf("WARNING WARNING WARNING\n");*/
   diag = hypre_ParCSRMatrixDiag(A12_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   for (i = 0; i <= nrows; i++) offd_i[i] = 0;
   offd = hypre_ParCSRMatrixOffd(A12_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * create A21 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz21;
   global_nrows = proc_offsets2[nprocs];
   global_ncols = proc_offsets1[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   col_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets2[i];
      col_starts[i] = proc_offsets1[i];
   }
   A21_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd); 
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   diag_j = hypre_CTAlloc(HYPRE_Int, nnz_diag);
   diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A21_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   for (i = 0; i <= nrows; i++) offd_i[i] = 0;
   offd = hypre_ParCSRMatrixOffd(A21_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * create A22 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz22;
   global_nrows = proc_offsets2[nprocs];
   global_ncols = proc_offsets2[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   col_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets2[i];
      col_starts[i] = proc_offsets2[i];
   }
   A22_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd); 
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   diag_j = hypre_CTAlloc(HYPRE_Int, nnz_diag);
   diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0)
            {
               diag_j[nnz] = - exp_indices[col] - 1;
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A22_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   for (i = 0; i <= nrows; i++) offd_i[i] = 0;
   offd = hypre_ParCSRMatrixOffd(A22_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * hand the matrices back to the caller and clean up 
    * ----------------------------------------------------- */

   (*submatrices)[0] = A11_csr;
   (*submatrices)[1] = A12_csr;
   (*submatrices)[2] = A21_csr;
   (*submatrices)[3] = A22_csr;
   free(proc_offsets1);
   free(proc_offsets2);
   free(exp_indices);
}

/* -----------------------------------------------------------------------------
 * extract submatrices of a rectangular matrix
 * ----------------------------------------------------------------------------- */

void hypre_ParCSRMatrixExtractRowSubmatrices( hypre_ParCSRMatrix *A_csr,
                                              HYPRE_Int *indices2,
                                              hypre_ParCSRMatrix ***submatrices )
{
   HYPRE_Int    nindices, *indices, nrows_A, *A_diag_i, *A_diag_j, mypid, nprocs;
   HYPRE_Int    i, j, k, *proc_offsets1, *proc_offsets2, *itmp_array, *exp_indices;
   HYPRE_Int    nnz11, nnz21, col, ncols_offd, nnz_offd, nnz_diag;
   HYPRE_Int    *A_offd_i, *A_offd_j;
   HYPRE_Int    global_nrows, global_ncols, *row_starts, *col_starts, nrows, nnz;
   HYPRE_Int    *diag_i, *diag_j, row, *offd_i, *offd_j, nnz11_offd, nnz21_offd;
   HYPRE_Complex *A_diag_a, *diag_a, *offd_a;
   hypre_ParCSRMatrix *A11_csr, *A21_csr;
   hypre_CSRMatrix    *A_diag, *diag, *A_offd, *offd;
   MPI_Comm           comm;

   /* -----------------------------------------------------
    * first make sure the incoming indices are in order
    * ----------------------------------------------------- */

   nindices = indices2[0];
   indices  = &(indices2[1]);
   hypre_qsort0(indices, 0, nindices-1);

   /* -----------------------------------------------------
    * fetch matrix information
    * ----------------------------------------------------- */

   nrows_A = hypre_ParCSRMatrixGlobalNumRows(A_csr);
   A_diag = hypre_ParCSRMatrixDiag(A_csr);
   A_diag_i = hypre_CSRMatrixI(A_diag);
   A_diag_j = hypre_CSRMatrixJ(A_diag);
   A_diag_a = hypre_CSRMatrixData(A_diag);
   A_offd = hypre_ParCSRMatrixOffd(A_csr);
   A_offd_i = hypre_CSRMatrixI(A_offd);
   A_offd_j = hypre_CSRMatrixJ(A_offd);
   comm = hypre_ParCSRMatrixComm(A_csr);
   hypre_MPI_Comm_rank(comm, &mypid);
   hypre_MPI_Comm_size(comm, &nprocs);

   /* -----------------------------------------------------
    * compute new matrix dimensions
    * ----------------------------------------------------- */

   proc_offsets1 = (HYPRE_Int *) malloc((nprocs+1) * sizeof(HYPRE_Int));
   proc_offsets2 = (HYPRE_Int *) malloc((nprocs+1) * sizeof(HYPRE_Int));
   hypre_MPI_Allgather(&nindices, 1, HYPRE_MPI_INT, proc_offsets1, 1,
                       HYPRE_MPI_INT, comm);
   k = 0;
   for (i = 0; i < nprocs; i++) 
   {
      j = proc_offsets1[i];
      proc_offsets1[i] = k;
      k += j;
   } 
   proc_offsets1[nprocs] = k;
   itmp_array = hypre_ParCSRMatrixRowStarts(A_csr);
   for (i = 0; i <= nprocs; i++) 
      proc_offsets2[i] = itmp_array[i] - proc_offsets1[i];

   /* -----------------------------------------------------
    * assign id's to row and col for later processing
    * ----------------------------------------------------- */

   exp_indices = (HYPRE_Int *) malloc(nrows_A * sizeof(HYPRE_Int));
   for (i = 0; i < nrows_A; i++) exp_indices[i] = -1;
   for (i = 0; i < nindices; i++) 
   {
      if (exp_indices[indices[i]] == -1) exp_indices[indices[i]] = i;
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ExtractRowSubmatrices: wrong index %d %d\n");
         exit(1);
      }
   }
   k = 0;
   for (i = 0; i < nrows_A; i++) 
   {
      if (exp_indices[i] < 0)
      {
         exp_indices[i] = - k - 1;
         k++;
      }
   }

   /* -----------------------------------------------------
    * compute number of nonzeros for each block
    * ----------------------------------------------------- */

   nnz11 = nnz21 = nnz11_offd = nnz21_offd = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) nnz11++;
         }
         nnz11_offd += A_offd_i[i+1] - A_offd_i[i];
      }
      else
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0) nnz21++;
         }
         nnz21_offd += A_offd_i[i+1] - A_offd_i[i];
      }
   }

   /* -----------------------------------------------------
    * create A11 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A_csr));
   nnz_diag   = nnz11;
   nnz_offd   = nnz11_offd; 

   global_nrows = proc_offsets1[nprocs];
   itmp_array   = hypre_ParCSRMatrixColStarts(A_csr);
   global_ncols = itmp_array[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   col_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets1[i];
      col_starts[i] = itmp_array[i];
   }
   A11_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd); 
   nrows = nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   diag_j = hypre_CTAlloc(HYPRE_Int, nnz_diag);
   diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A11_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   offd_j = hypre_CTAlloc(HYPRE_Int, nnz_offd);
   offd_a = hypre_CTAlloc(HYPRE_Complex, nnz_offd);
   nnz = 0;
   row = 0;
   offd_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
         {
            offd_j[nnz] = A_offd_j[j];
            offd_a[nnz++] = A_diag_a[j];
         }
         row++;
         offd_i[row] = nnz;
      }
   }
   offd = hypre_ParCSRMatrixOffd(A11_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = offd_j;
   hypre_CSRMatrixData(offd) = offd_a;

   /* -----------------------------------------------------
    * create A21 matrix
    * ----------------------------------------------------- */

   ncols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A_csr));
   nnz_offd   = nnz21_offd;
   nnz_diag   = nnz21;
   global_nrows = proc_offsets2[nprocs];
   itmp_array   = hypre_ParCSRMatrixColStarts(A_csr);
   global_ncols = itmp_array[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   col_starts = hypre_CTAlloc(HYPRE_Int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets2[i];
      col_starts[i] = itmp_array[i];
   }
   A21_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd); 
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   diag_j = hypre_CTAlloc(HYPRE_Int, nnz_diag);
   diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            diag_j[nnz] = A_diag_j[j];
            diag_a[nnz++] = A_diag_a[j];
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A21_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int, nrows+1);
   offd_j = hypre_CTAlloc(HYPRE_Int, nnz_offd);
   offd_a = hypre_CTAlloc(HYPRE_Complex, nnz_offd);
   nnz = 0;
   row = 0;
   offd_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
         {
            offd_j[nnz] = A_offd_j[j];
            offd_a[nnz++] = A_diag_a[j];
         }
         row++;
         offd_i[row] = nnz;
      }
   }
   offd = hypre_ParCSRMatrixOffd(A21_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = offd_j;
   hypre_CSRMatrixData(offd) = offd_a;

   /* -----------------------------------------------------
    * hand the matrices back to the caller and clean up 
    * ----------------------------------------------------- */

   (*submatrices)[0] = A11_csr;
   (*submatrices)[1] = A21_csr;
   free(proc_offsets1);
   free(proc_offsets2);
   free(exp_indices);
}

/* -----------------------------------------------------------------------------
 * return the sum of all local elements of the matrix
 * ----------------------------------------------------------------------------- */

HYPRE_Complex hypre_ParCSRMatrixLocalSumElts( hypre_ParCSRMatrix * A )
{
   hypre_CSRMatrix * A_diag = hypre_ParCSRMatrixDiag( A );
   hypre_CSRMatrix * A_offd = hypre_ParCSRMatrixOffd( A );

   return hypre_CSRMatrixSumElts(A_diag) + hypre_CSRMatrixSumElts(A_offd);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatAminvDB
 * computes C = (A - inv(D)B) where D is a diagonal matrix
 * Note: Data structure of A is expected to be a subset of data structure of B!
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixAminvDB( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B,
                 	   HYPRE_Complex *d, hypre_ParCSRMatrix **C_ptr)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(B);
   hypre_CSRMatrix      *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *A_offd   = hypre_ParCSRMatrixOffd(A);
   hypre_ParCSRMatrix  *C = NULL;
   HYPRE_Int	      num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_ParCSRCommPkg	*comm_pkg_B = hypre_ParCSRMatrixCommPkg(B);
   hypre_CSRMatrix      *B_diag   = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix      *B_offd   = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int	      num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);
   HYPRE_Int	      num_sends_B, num_recvs_B;
   HYPRE_Int	      i, j, cnt;

   HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);

   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int *B_diag_i = hypre_CSRMatrixI(B_diag);
   HYPRE_Int *B_diag_j = hypre_CSRMatrixJ(B_diag);
   HYPRE_Complex *B_diag_data = hypre_CSRMatrixData(B_diag);
   HYPRE_Int *B_offd_i = hypre_CSRMatrixI(B_offd);
   HYPRE_Int *B_offd_j = hypre_CSRMatrixJ(B_offd);
   HYPRE_Complex *B_offd_data = hypre_CSRMatrixData(B_offd);
   HYPRE_Int *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   hypre_CSRMatrix      *C_diag   = NULL;
   hypre_CSRMatrix      *C_offd   = NULL;
   HYPRE_Int *C_diag_i = NULL;
   HYPRE_Int *C_diag_j = NULL;
   HYPRE_Complex *C_diag_data = NULL;
   HYPRE_Int *C_offd_i = NULL;
   HYPRE_Int *C_offd_j = NULL;
   HYPRE_Complex *C_offd_data = NULL;

   HYPRE_Int num_procs, my_id;

   HYPRE_Int *recv_procs_B;
   HYPRE_Int *send_procs_B;
   HYPRE_Int *recv_vec_starts_B;
   HYPRE_Int *send_map_starts_B;
   HYPRE_Int *send_map_elmts_B;
   hypre_ParCSRCommPkg *comm_pkg_C;
   HYPRE_Int *recv_procs_C;
   HYPRE_Int *send_procs_C;
   HYPRE_Int *recv_vec_starts_C;
   HYPRE_Int *send_map_starts_C;
   HYPRE_Int *send_map_elmts_C;
   HYPRE_Int *map_to_B;

   /*HYPRE_Int *C_diag_array;
   HYPRE_Int *C_offd_array;*/
   HYPRE_Complex *D_tmp;
   HYPRE_Int size, rest, num_threads, ii;

   hypre_MPI_Comm_size(comm,&num_procs);   
   hypre_MPI_Comm_rank(comm,&my_id);
 
   num_threads = hypre_NumThreads();
   /*C_diag_array = hypre_CTAlloc(HYPRE_Int, num_threads);
   C_offd_array = hypre_CTAlloc(HYPRE_Int, num_threads);*/


   /*---------------------------------------------------------------------
    * If there exists no CommPkg for B, a CommPkg is generated 
    *--------------------------------------------------------------------*/

   if (!comm_pkg_B)
   {
      hypre_MatvecCommPkgCreate(B);
      comm_pkg_B = hypre_ParCSRMatrixCommPkg(B); 
   }

   C = hypre_ParCSRMatrixCompleteClone(B);
   /*hypre_ParCSRMatrixInitialize(C);*/

   C_diag = hypre_ParCSRMatrixDiag(C);
   C_diag_i = hypre_CSRMatrixI(C_diag);
   C_diag_j = hypre_CSRMatrixJ(C_diag);
   C_diag_data = hypre_CSRMatrixData(C_diag);
   C_offd = hypre_ParCSRMatrixOffd(C);
   C_offd_i = hypre_CSRMatrixI(C_offd);
   C_offd_j = hypre_CSRMatrixJ(C_offd);
   C_offd_data = hypre_CSRMatrixData(C_offd);

   size = num_rows/num_threads;
   rest = num_rows - size*num_threads;

   D_tmp = hypre_CTAlloc(HYPRE_Complex, num_rows);

   if (num_cols_offd_A)
   {
      map_to_B = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A);
      cnt = 0;
      for (i=0; i < num_cols_offd_A; i++)
      {
         while (col_map_offd_B[cnt] < col_map_offd_A[i])
         {
            cnt++;
         }
         map_to_B[i] = cnt;
         cnt++;
      }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(ii, i, j)
#endif
   for (ii=0; ii < num_threads; ii++)
   {
    HYPRE_Int *A_marker = NULL;
    HYPRE_Int ns, ne, A_col, num_cols, nmax;
    if (ii < rest)
    {
       ns = ii*size+ii;
       ne = (ii+1)*size+ii+1;
    }
    else
    {
       ns = ii*size+rest;
       ne = (ii+1)*size+rest;
    }
    nmax = hypre_max(num_rows, num_cols_offd_B);
    A_marker = hypre_CTAlloc(HYPRE_Int, nmax);

    for (i=0; i < num_rows; i++)
        A_marker[i] = -1;

    for (i=ns; i < ne; i++)
       D_tmp[i] = 1.0/d[i];

    num_cols = C_diag_i[ns];
    for (i=ns; i < ne; i++)
    {
      for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
      {
         A_col = A_diag_j[j];
         if (A_marker[A_col] < C_diag_i[i]) 
	 {
	    A_marker[A_col] = num_cols;
	    C_diag_j[num_cols] = A_col;
	    C_diag_data[num_cols] = A_diag_data[j];
	    num_cols++;
	 }
         else
	 {
	    C_diag_data[A_marker[A_col]] += A_diag_data[j];
	 }
      }
      for (j = B_diag_i[i]; j < B_diag_i[i+1]; j++)
      {
         A_col = B_diag_j[j];
         if (A_marker[A_col] < C_diag_i[i]) 
         {
            A_marker[A_col] = num_cols;
	    C_diag_j[num_cols] = A_col;
	    C_diag_data[num_cols] = -D_tmp[i]*B_diag_data[j];
	    num_cols++;
	 }
         else
	 {
	    C_diag_data[A_marker[A_col]] -= D_tmp[i]*B_diag_data[j];
	 }
      }
   }

   for (i=0; i < num_cols_offd_B; i++)
        A_marker[i] = -1;

   num_cols = C_offd_i[ns];
   for (i=ns; i < ne; i++)
   {
      for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
      {
         A_col = map_to_B[A_offd_j[j]];
         if (A_marker[A_col] < B_offd_i[i]) 
	 {
	    A_marker[A_col] = num_cols;
	    C_offd_j[num_cols] = A_col;
	    C_offd_data[num_cols] = A_offd_data[j];
	    num_cols++;
	 }
         else
	 {
	    C_offd_data[A_marker[A_col]] += A_offd_data[j];
	 }
      }
      for (j = B_offd_i[i]; j < B_offd_i[i+1]; j++)
      {
         A_col = B_offd_j[j];
         if (A_marker[A_col] < B_offd_i[i]) 
         {
            A_marker[A_col] = num_cols;
	    C_offd_j[num_cols] = A_col;
	    C_offd_data[num_cols] = -D_tmp[i]*B_offd_data[j];
	    num_cols++;
	 }
         else
	 {
	    C_offd_data[A_marker[A_col]] -= D_tmp[i]*B_offd_data[j];
	 }
      }
   }
   hypre_TFree(A_marker);

   } /* end parallel region */

   /*for (i=0; i < num_cols_offd_B; i++)
      col_map_offd_C[i] = col_map_offd_B[i]; */

   num_sends_B = hypre_ParCSRCommPkgNumSends(comm_pkg_B);
   num_recvs_B = hypre_ParCSRCommPkgNumRecvs(comm_pkg_B);
   recv_procs_B = hypre_ParCSRCommPkgRecvProcs(comm_pkg_B);
   recv_vec_starts_B = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_B);
   send_procs_B = hypre_ParCSRCommPkgSendProcs(comm_pkg_B);
   send_map_starts_B = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_B);
   send_map_elmts_B = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_B);

   recv_procs_C = hypre_CTAlloc(HYPRE_Int, num_recvs_B);
   recv_vec_starts_C = hypre_CTAlloc(HYPRE_Int, num_recvs_B+1);
   send_procs_C = hypre_CTAlloc(HYPRE_Int, num_sends_B);
   send_map_starts_C = hypre_CTAlloc(HYPRE_Int, num_sends_B+1);
   send_map_elmts_C = hypre_CTAlloc(HYPRE_Int, send_map_starts_B[num_sends_B]);

   for (i=0; i < num_recvs_B; i++)
      recv_procs_C[i] = recv_procs_B[i];
   for (i=0; i < num_recvs_B+1; i++)
      recv_vec_starts_C[i] = recv_vec_starts_B[i];
   for (i=0; i < num_sends_B; i++)
      send_procs_C[i] = send_procs_B[i];
   for (i=0; i < num_sends_B+1; i++)
      send_map_starts_C[i] = send_map_starts_B[i];
   for (i=0; i < send_map_starts_B[num_sends_B]; i++)
      send_map_elmts_C[i] = send_map_elmts_B[i];
   
   comm_pkg_C = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(comm_pkg_C) = comm;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg_C) = num_recvs_B;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg_C) = recv_procs_C;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_C) = recv_vec_starts_C;
   hypre_ParCSRCommPkgNumSends(comm_pkg_C) = num_sends_B;
   hypre_ParCSRCommPkgSendProcs(comm_pkg_C) = send_procs_C;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_C) = send_map_starts_C;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg_C) = send_map_elmts_C;
  
   hypre_ParCSRMatrixCommPkg(C) = comm_pkg_C; 

   hypre_TFree(D_tmp);
   if (num_cols_offd_A) hypre_TFree(map_to_B);

   *C_ptr = C;
   
   return (hypre_error_flag);
}

/*--------------------------------------------------------------------------
 * hypre_ParTMatmul : multiplies two ParCSRMatrices transpose(A) and B and returns
 * the product in ParCSRMatrix C
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *hypre_ParTMatmul( hypre_ParCSRMatrix  *A,
				     hypre_ParCSRMatrix  *B)
{
   MPI_Comm 	   comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *AT_diag = NULL;
   
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *AT_offd = NULL;
   
   HYPRE_Int	num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int	num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   
   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
   
   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int		   *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);
   
   HYPRE_Int	first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_Int *col_starts_A = hypre_ParCSRMatrixColStarts(A);
   HYPRE_Int *col_starts_B = hypre_ParCSRMatrixColStarts(B);
   HYPRE_Int	num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int	num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int	num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_Int		      *col_map_offd_C = NULL;
   HYPRE_Int		      *map_B_to_C;

   hypre_CSRMatrix *C_diag = NULL;
   hypre_CSRMatrix *C_tmp_diag = NULL;

   HYPRE_Complex          *C_diag_data = NULL;
   HYPRE_Int       *C_diag_i = NULL;
   HYPRE_Int       *C_diag_j = NULL;
   HYPRE_Int	first_col_diag_C;
   HYPRE_Int	last_col_diag_C;

   hypre_CSRMatrix *C_offd = NULL;
   hypre_CSRMatrix *C_tmp_offd = NULL;
   hypre_CSRMatrix *C_int = NULL;
   hypre_CSRMatrix *C_ext = NULL;
   HYPRE_Int   *C_ext_i;
   HYPRE_Int   *C_ext_j;
   HYPRE_Complex   *C_ext_data;
   HYPRE_Int   *C_ext_diag_i;
   HYPRE_Int   *C_ext_diag_j;
   HYPRE_Complex   *C_ext_diag_data;
   HYPRE_Int   *C_ext_offd_i;
   HYPRE_Int   *C_ext_offd_j;
   HYPRE_Complex   *C_ext_offd_data;
   HYPRE_Int	C_ext_size = 0;
   HYPRE_Int	C_ext_diag_size = 0;
   HYPRE_Int	C_ext_offd_size = 0;

   HYPRE_Int   *C_tmp_diag_i;
   HYPRE_Int   *C_tmp_diag_j;
   HYPRE_Complex   *C_tmp_diag_data;
   HYPRE_Int   *C_tmp_offd_i;
   HYPRE_Int   *C_tmp_offd_j;
   HYPRE_Complex   *C_tmp_offd_data;

   HYPRE_Complex          *C_offd_data=NULL;
   HYPRE_Int       *C_offd_i=NULL;
   HYPRE_Int       *C_offd_j=NULL;

   HYPRE_Int       *temp;
   HYPRE_Int       *send_map_starts_A;
   HYPRE_Int       *send_map_elmts_A;
   HYPRE_Int        num_sends_A;

   HYPRE_Int		    num_cols_offd_C = 0;
   
   HYPRE_Int		   *P_marker;

   HYPRE_Int              i, j;
   HYPRE_Int              i1, j_indx;
   
   HYPRE_Int		    n_rows_A, n_cols_A;
   HYPRE_Int		    n_rows_B, n_cols_B;
   /*HYPRE_Int              allsquare = 0;*/
   HYPRE_Int              cnt, cnt_offd, cnt_diag;
   HYPRE_Int 		    value;
   HYPRE_Int 		    num_procs, my_id;
   HYPRE_Int                max_num_threads;
   HYPRE_Int               *C_diag_array = NULL;
   HYPRE_Int               *C_offd_array = NULL;

   HYPRE_Int first_row_index, first_col_diag;
   HYPRE_Int local_num_rows, local_num_cols;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   max_num_threads = hypre_NumThreads();

   if (n_rows_A != n_rows_B || num_rows_diag_A != num_rows_diag_B)
   {
        hypre_error_w_msg(HYPRE_ERROR_GENERIC," Error! Incompatible matrix dimensions!\n");
	return NULL;
   }

   /*if (num_cols_diag_A == num_cols_diag_B) allsquare = 1;*/

   hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
   hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);

   C_tmp_diag = hypre_CSRMatrixMultiply(AT_diag, B_diag);
   C_ext_size = 0;
   if (num_procs > 1) 
   {
      hypre_CSRMatrix *C_int_diag;
      hypre_CSRMatrix *C_int_offd;
      C_tmp_offd = hypre_CSRMatrixMultiply(AT_diag, B_offd);
      C_int_diag = hypre_CSRMatrixMultiply(AT_offd, B_diag);
      C_int_offd = hypre_CSRMatrixMultiply(AT_offd, B_offd);
      hypre_ParCSRMatrixDiag(B) = C_int_diag;
      hypre_ParCSRMatrixOffd(B) = C_int_offd;
      C_int = hypre_MergeDiagAndOffd(B);
      hypre_ParCSRMatrixDiag(B) = B_diag;
      hypre_ParCSRMatrixOffd(B) = B_offd;
      C_ext = hypre_ExchangeRAPData(C_int, comm_pkg_A);
      C_ext_i = hypre_CSRMatrixI(C_ext);
      C_ext_j = hypre_CSRMatrixJ(C_ext);
      C_ext_data = hypre_CSRMatrixData(C_ext);
      C_ext_size = C_ext_i[hypre_CSRMatrixNumRows(C_ext)];
      hypre_CSRMatrixDestroy(C_int);
      hypre_CSRMatrixDestroy(C_int_diag);
      hypre_CSRMatrixDestroy(C_int_offd);
   }
   else
   {
     C_tmp_offd = hypre_CSRMatrixCreate(num_cols_diag_A, 0, 0);
     hypre_CSRMatrixInitialize(C_tmp_offd);
   }
   hypre_CSRMatrixDestroy(AT_diag);
   hypre_CSRMatrixDestroy(AT_offd);

   /*-----------------------------------------------------------------------
    *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
    *  to obtain C_diag and C_offd
    *-----------------------------------------------------------------------*/

   /* check for new nonzero columns in C_offd generated through C_ext */

   first_col_diag_C = first_col_diag_B;
   last_col_diag_C = first_col_diag_B + num_cols_diag_B - 1;

   C_tmp_diag_i = hypre_CSRMatrixI(C_tmp_diag);
   if (C_ext_size || num_cols_offd_B)
   {
      HYPRE_Int C_ext_num_rows;
      num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
      send_map_starts_A = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);
      send_map_elmts_A = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_A);
      C_ext_num_rows =  send_map_starts_A[num_sends_A];
     
      C_ext_diag_i = hypre_CTAlloc(HYPRE_Int, C_ext_num_rows+1);
      C_ext_offd_i = hypre_CTAlloc(HYPRE_Int, C_ext_num_rows+1);
      temp = hypre_CTAlloc(HYPRE_Int, C_ext_size+num_cols_offd_B);
      C_ext_diag_size = 0;
      C_ext_offd_size = 0;
      for (i=0; i < C_ext_num_rows; i++)
      {
         for (j=C_ext_i[i]; j < C_ext_i[i+1]; j++)
            if (C_ext_j[j] < first_col_diag_C || C_ext_j[j] > last_col_diag_C)
	       temp[C_ext_offd_size++] = C_ext_j[j];
            else
               C_ext_diag_size++;
         C_ext_diag_i[i+1] = C_ext_diag_size;
         C_ext_offd_i[i+1] = C_ext_offd_size;
      }
      cnt = C_ext_offd_size;
      for (i=0; i < num_cols_offd_B; i++)
         temp[cnt++] = col_map_offd_B[i];

      if (cnt)
      {
	  hypre_qsort0(temp,0,cnt-1);
          value = temp[0];
          num_cols_offd_C = 1;
          for (i=1; i < cnt; i++)
          {
 	     if (temp[i] > value)
	     {
		value = temp[i];
		temp[num_cols_offd_C++] = value;
	     }
  	  }
       }

       if (num_cols_offd_C)
	  col_map_offd_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_C);
       for (i=0; i < num_cols_offd_C; i++)
	  col_map_offd_C[i] = temp[i];

       hypre_TFree(temp);
   
      if (C_ext_diag_size)
      {
         C_ext_diag_j = hypre_CTAlloc(HYPRE_Int, C_ext_diag_size);
         C_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, C_ext_diag_size);
      }
      if (C_ext_offd_size)
      {
         C_ext_offd_j = hypre_CTAlloc(HYPRE_Int, C_ext_offd_size);
         C_ext_offd_data = hypre_CTAlloc(HYPRE_Complex, C_ext_offd_size);
      }

      C_tmp_diag_j = hypre_CSRMatrixJ(C_tmp_diag);
      C_tmp_diag_data = hypre_CSRMatrixData(C_tmp_diag);

      C_tmp_offd_i = hypre_CSRMatrixI(C_tmp_offd);
      C_tmp_offd_j = hypre_CSRMatrixJ(C_tmp_offd);
      C_tmp_offd_data = hypre_CSRMatrixData(C_tmp_offd);

      cnt_offd = 0;
      cnt_diag = 0;
      for (i=0; i < C_ext_num_rows; i++)
      {
         for (j=C_ext_i[i]; j < C_ext_i[i+1]; j++)
            if (C_ext_j[j] < first_col_diag_C || C_ext_j[j] > last_col_diag_C)
            {
               C_ext_offd_j[cnt_offd] = hypre_BinarySearch(col_map_offd_C,
                                           C_ext_j[j],
                                           num_cols_offd_C);
               C_ext_offd_data[cnt_offd++] = C_ext_data[j];
            }
            else
            {
               C_ext_diag_j[cnt_diag] = C_ext_j[j] - first_col_diag_C;
               C_ext_diag_data[cnt_diag++] = C_ext_data[j];
            }
      }
   }

   if (C_ext)
   {
      hypre_CSRMatrixDestroy(C_ext);
      C_ext = NULL;
   }

   if (num_cols_offd_B)
   {
      map_B_to_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_B);

      cnt = 0;
      for (i=0; i < num_cols_offd_C; i++)
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) break;
         }
      for (i=0; 
	i < hypre_CSRMatrixI(C_tmp_offd)[hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
      {
         j_indx = C_tmp_offd_j[i];
         C_tmp_offd_j[i] = map_B_to_C[j_indx];
      }
   }

   /*-----------------------------------------------------------------------
    *  Need to compute C_diag = C_tmp_diag + C_ext_diag
    *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
    *  First generate structure
    *-----------------------------------------------------------------------*/

   if (C_ext_size || num_cols_offd_B)
   {
     C_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_diag_A+1);
     C_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_diag_A+1);

     C_diag_array = hypre_CTAlloc(HYPRE_Int, max_num_threads);
     C_offd_array = hypre_CTAlloc(HYPRE_Int, max_num_threads);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel 
#endif
     {
        HYPRE_Int *B_marker = NULL;
        HYPRE_Int *B_marker_offd = NULL;
        HYPRE_Int ik, jk, j1, j2, jcol;
        HYPRE_Int ns, ne, ii, nnz_d, nnz_o;
        HYPRE_Int rest, size;
        HYPRE_Int num_threads = hypre_NumActiveThreads();

        size = num_cols_diag_A/num_threads;
        rest = num_cols_diag_A - size*num_threads;
        ii = hypre_GetThreadNum();
        if (ii < rest)
        {
           ns = ii*size+ii;
           ne = (ii+1)*size+ii+1;
        }
        else
        {
           ns = ii*size+rest;
           ne = (ii+1)*size+rest;
        }

        B_marker = hypre_CTAlloc(HYPRE_Int, num_cols_diag_B);
        B_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_C);

        for (ik = 0; ik < num_cols_diag_B; ik++)
           B_marker[ik] = -1;

        for (ik = 0; ik < num_cols_offd_C; ik++)
           B_marker_offd[ik] = -1;

        nnz_d = 0;
        nnz_o = 0;
        for (ik = ns; ik < ne; ik++)
        {
          for (jk = C_tmp_diag_i[ik]; jk < C_tmp_diag_i[ik+1]; jk++)
          {
             jcol = C_tmp_diag_j[jk];
             B_marker[jcol] = ik;
	     nnz_d++;
          }
          for (jk = C_tmp_offd_i[ik]; jk < C_tmp_offd_i[ik+1]; jk++)
          {
             jcol = C_tmp_offd_j[jk];
             B_marker_offd[jcol] = ik;
	     nnz_o++;
          }
          for (jk = 0; jk < num_sends_A; jk++)
            for (j1 = send_map_starts_A[jk]; j1 < send_map_starts_A[jk+1]; j1++)
             if (send_map_elmts_A[j1] == ik)
             {
                for (j2 = C_ext_diag_i[j1]; j2 < C_ext_diag_i[j1+1]; j2++)
                {
                    jcol = C_ext_diag_j[j2];
 	            if (B_marker[jcol] < ik)
                    {
                       B_marker[jcol] = ik;
	               nnz_d++;
                    }
                }
                for (j2 = C_ext_offd_i[j1]; j2 < C_ext_offd_i[j1+1]; j2++)
                {
                    jcol = C_ext_offd_j[j2];
 	            if (B_marker_offd[jcol] < ik)
                    {
                       B_marker_offd[jcol] = ik;
	               nnz_o++;
                    }
                }
                break;
             }
               C_diag_array[ii] = nnz_d;
               C_offd_array[ii] = nnz_o;
        }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
       if (ii == 0)
       {
          nnz_d = 0;
          nnz_o = 0;
          for (ik = 0; ik < num_threads-1; ik++)
          {
             C_diag_array[ik+1] += C_diag_array[ik];
             C_offd_array[ik+1] += C_offd_array[ik];
          }
          nnz_d = C_diag_array[num_threads-1];
          nnz_o = C_offd_array[num_threads-1];
          C_diag_i[num_cols_diag_A] = nnz_d;
          C_offd_i[num_cols_diag_A] = nnz_o;

          C_diag = hypre_CSRMatrixCreate(num_cols_diag_A, num_cols_diag_A, nnz_d);
          C_offd = hypre_CSRMatrixCreate(num_cols_diag_A, num_cols_offd_C, nnz_o);
          hypre_CSRMatrixI(C_diag) = C_diag_i;
          hypre_CSRMatrixInitialize(C_diag);
          C_diag_j = hypre_CSRMatrixJ(C_diag);
          C_diag_data = hypre_CSRMatrixData(C_diag);
          hypre_CSRMatrixI(C_offd) = C_offd_i;
          hypre_CSRMatrixInitialize(C_offd);
          C_offd_j = hypre_CSRMatrixJ(C_offd);
          C_offd_data = hypre_CSRMatrixData(C_offd); 
       }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   /*-----------------------------------------------------------------------
    *  Need to compute C_diag = C_tmp_diag + C_ext_diag
    *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
    *  Now fill in values
    *-----------------------------------------------------------------------*/
   
     for (ik = 0; ik < num_cols_diag_B; ik++)
        B_marker[ik] = -1;

     for (ik = 0; ik < num_cols_offd_C; ik++)
        B_marker_offd[ik] = -1;

   /*-----------------------------------------------------------------------
    *  Populate matrices
    *-----------------------------------------------------------------------*/

      nnz_d = 0;
      nnz_o = 0;
        nnz_o = 0;
        if (ii)
        {
           nnz_d = C_diag_array[ii-1];
           nnz_o = C_offd_array[ii-1];
        }
        for (ik = ns; ik < ne; ik++)
        {
           C_diag_i[ik] = nnz_d;
           C_offd_i[ik] = nnz_o;
           for (jk = C_tmp_diag_i[ik]; jk < C_tmp_diag_i[ik+1]; jk++)
           {
              jcol = C_tmp_diag_j[jk];
              C_diag_j[nnz_d] = jcol;
              C_diag_data[nnz_d] = C_tmp_diag_data[jk];
              B_marker[jcol] = nnz_d;
              nnz_d++;
           }
           for (jk = C_tmp_offd_i[ik]; jk < C_tmp_offd_i[ik+1]; jk++)
           {
              jcol = C_tmp_offd_j[jk];
              C_offd_j[nnz_o] = jcol;
              C_offd_data[nnz_o] = C_tmp_offd_data[jk];
              B_marker_offd[jcol] = nnz_o;
              nnz_o++;
           }
           for (jk = 0; jk < num_sends_A; jk++)
              for (j1 = send_map_starts_A[jk]; j1 < send_map_starts_A[jk+1]; j1++)
                 if (send_map_elmts_A[j1] == ik)
                 {
                    for (j2 = C_ext_diag_i[j1]; j2 < C_ext_diag_i[j1+1]; j2++)
                    {
                       jcol = C_ext_diag_j[j2];
                       if (B_marker[jcol] < C_diag_i[ik])
                       {
                          C_diag_j[nnz_d] = jcol;
                          C_diag_data[nnz_d] = C_ext_diag_data[j2];
                          B_marker[jcol] = nnz_d;
                          nnz_d++;
                       }
                       else
                          C_diag_data[B_marker[jcol]] += C_ext_diag_data[j2];
                    }
                    for (j2 = C_ext_offd_i[j1]; j2 < C_ext_offd_i[j1+1]; j2++)
                    {
                       jcol = C_ext_offd_j[j2];
                       if (B_marker_offd[jcol] < C_offd_i[ik])
                       {
                          C_offd_j[nnz_o] = jcol;
                          C_offd_data[nnz_o] = C_ext_offd_data[j2];
                          B_marker_offd[jcol] = nnz_o;
                          nnz_o++;
                       }
                       else
                          C_offd_data[B_marker_offd[jcol]] += C_ext_offd_data[j2];
                    }
                    break;
                 }
        }
        hypre_TFree(B_marker);
        hypre_TFree(B_marker_offd);
     } /*end parallel region */
     hypre_TFree(C_diag_array);
     hypre_TFree(C_offd_array);
   }

   /*C = hypre_ParCSRMatrixCreate(comm, n_cols_A, n_cols_B, col_starts_A,
	col_starts_B, num_cols_offd_C, nnz_diag, nnz_offd);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C)); */
#ifdef HYPRE_NO_GLOBAL_PARTITION
   /* row_starts[0] is start of local rows.  row_starts[1] is start of next 
      processor's rows */
   first_row_index = col_starts_A[0];
   local_num_rows = col_starts_A[1]-first_row_index ;
   first_col_diag = col_starts_B[0];
   local_num_cols = col_starts_B[1]-first_col_diag;
#else
   first_row_index = col_starts_A[my_id];
   local_num_rows = col_starts_A[my_id+1]-first_row_index;
   first_col_diag = col_starts_B[my_id];
   local_num_cols = col_starts_B[my_id+1]-first_col_diag;
#endif

   C = hypre_CTAlloc(hypre_ParCSRMatrix, 1);
   hypre_ParCSRMatrixComm(C) = comm;
   hypre_ParCSRMatrixGlobalNumRows(C) = n_cols_A;
   hypre_ParCSRMatrixGlobalNumCols(C) = n_cols_B;
   hypre_ParCSRMatrixFirstRowIndex(C) = first_row_index;
   hypre_ParCSRMatrixFirstColDiag(C) = first_col_diag;
   hypre_ParCSRMatrixLastRowIndex(C) = first_row_index + local_num_rows - 1;
   hypre_ParCSRMatrixLastColDiag(C) = first_col_diag + local_num_cols - 1;

   hypre_ParCSRMatrixColMapOffd(C) = NULL;

   hypre_ParCSRMatrixAssumedPartition(C) = NULL;

   hypre_ParCSRMatrixRowStarts(C) = col_starts_A;
   hypre_ParCSRMatrixColStarts(C) = col_starts_B;

   hypre_ParCSRMatrixCommPkg(C) = NULL;
   hypre_ParCSRMatrixCommPkgT(C) = NULL;

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(C) = 1;
   hypre_ParCSRMatrixRowindices(C) = NULL;
   hypre_ParCSRMatrixRowvalues(C) = NULL;
   hypre_ParCSRMatrixGetrowactive(C) = 0;

/* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRMatrixSetColStartsOwner(C,0);

   if (C_diag) hypre_ParCSRMatrixDiag(C) = C_diag;
   else hypre_ParCSRMatrixDiag(C) = C_tmp_diag;
   if (C_offd) hypre_ParCSRMatrixOffd(C) = C_offd;
   else hypre_ParCSRMatrixOffd(C) = C_tmp_offd;

   if (num_cols_offd_C)
   {
      HYPRE_Int jj_count_offd, nnz_offd;
      HYPRE_Int *new_col_map_offd_C = NULL;

      P_marker = hypre_CTAlloc(HYPRE_Int,num_cols_offd_C);
      for (i=0; i < num_cols_offd_C; i++)
         P_marker[i] = -1;

      jj_count_offd = 0;
      nnz_offd = C_offd_i[num_cols_diag_A];
      for (i=0; i < nnz_offd; i++)
      {
         i1 = C_offd_j[i];
         if (P_marker[i1])
         {
            P_marker[i1] = 0;
            jj_count_offd++;
         }
      }

      if (jj_count_offd < num_cols_offd_C)
      {
         new_col_map_offd_C = hypre_CTAlloc(HYPRE_Int,jj_count_offd);
         jj_count_offd = 0;
         for (i=0; i < num_cols_offd_C; i++)
            if (!P_marker[i])
            {
               P_marker[i] = jj_count_offd;
               new_col_map_offd_C[jj_count_offd++] = col_map_offd_C[i];
            }

         for (i=0; i < nnz_offd; i++)
         {
            i1 = C_offd_j[i];
            C_offd_j[i] = P_marker[i1];
         }

         num_cols_offd_C = jj_count_offd;
         hypre_TFree(col_map_offd_C);
         col_map_offd_C = new_col_map_offd_C;
         hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(C)) = num_cols_offd_C;
      }
      hypre_TFree(P_marker);
   }
   hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   if (C_ext_size || num_cols_offd_B)
   {
      hypre_TFree(C_ext_diag_i);
      hypre_TFree(C_ext_offd_i);
   }
   if (C_ext_diag_size)
   {
      hypre_TFree(C_ext_diag_j);
      hypre_TFree(C_ext_diag_data);
   }
   if (C_ext_offd_size)
   {
      hypre_TFree(C_ext_offd_j);
      hypre_TFree(C_ext_offd_data);
   }
   if (num_cols_offd_B) hypre_TFree(map_B_to_C);

   if (C_diag) hypre_CSRMatrixDestroy(C_tmp_diag);
   if (C_offd) hypre_CSRMatrixDestroy(C_tmp_offd);

   return C;
   
}            

