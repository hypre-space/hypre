/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "../HYPRE.h"
#include "_hypre_IJ_mv.h"

/* This file contains the routines for constructing non-Galerkin coarse grid
 * operators, based on the original Galerkin coarse grid
 */

/* Take all of the indices from indices[start, start+1, start+2, ..., end]
 * and take the corresponding entries in array and place them in-order in output.
 * Assumptions:
 *      output is of length end-start+1
 *      indices never contains an index that goes out of bounds in array
 * */
HYPRE_Int
hypre_GrabSubArray(HYPRE_Int * indices,
                   HYPRE_Int start,
                   HYPRE_Int end,
                   HYPRE_BigInt * array,
                   HYPRE_BigInt * output)
{
   HYPRE_Int i, length;
   length = end - start + 1;

   for (i = 0; i < length; i++)
   {
      output[i] = array[indices[start + i]];
   }

   return hypre_error_flag;
}

/* Compute the intersection of x and y, placing
 * the intersection in z.  Additionally, the array
 * x_data is associated with x, i.e., the entries
 * that we grab from x, we also grab from x_data.
 * If x[k] is placed in z[m], then x_data[k] goes to
 * output_x_data[m].
 *
 * Assumptions:
 *      z is of length min(x_length, y_length)
 *      x and y are sorted
 *      x_length and y_length are similar in size, otherwise,
 *          looping over the smaller array and doing binary search
 *          in the longer array is faster.
 * */
HYPRE_Int
hypre_IntersectTwoArrays(HYPRE_Int *x,
                         HYPRE_Real *x_data,
                         HYPRE_Int  x_length,
                         HYPRE_Int *y,
                         HYPRE_Int  y_length,
                         HYPRE_Int *z,
                         HYPRE_Real *output_x_data,
                         HYPRE_Int  *intersect_length)
{
   HYPRE_Int x_index = 0;
   HYPRE_Int y_index = 0;
   *intersect_length = 0;

   /* Compute Intersection, looping over each array */
   while ( (x_index < x_length) && (y_index < y_length) )
   {
      if (x[x_index] > y[y_index])
      {
         y_index = y_index + 1;
      }
      else if (x[x_index] < y[y_index])
      {
         x_index = x_index + 1;
      }
      else
      {
         z[*intersect_length] = x[x_index];
         output_x_data[*intersect_length] = x_data[x_index];
         x_index = x_index + 1;
         y_index = y_index + 1;
         *intersect_length = *intersect_length + 1;
      }
   }

   return 1;
}

HYPRE_Int
hypre_IntersectTwoBigArrays(HYPRE_BigInt *x,
                            HYPRE_Real *x_data,
                            HYPRE_Int  x_length,
                            HYPRE_BigInt *y,
                            HYPRE_Int  y_length,
                            HYPRE_BigInt *z,
                            HYPRE_Real *output_x_data,
                            HYPRE_Int  *intersect_length)
{
   HYPRE_Int x_index = 0;
   HYPRE_Int y_index = 0;
   *intersect_length = 0;

   /* Compute Intersection, looping over each array */
   while ( (x_index < x_length) && (y_index < y_length) )
   {
      if (x[x_index] > y[y_index])
      {
         y_index = y_index + 1;
      }
      else if (x[x_index] < y[y_index])
      {
         x_index = x_index + 1;
      }
      else
      {
         z[*intersect_length] = x[x_index];
         output_x_data[*intersect_length] = x_data[x_index];
         x_index = x_index + 1;
         y_index = y_index + 1;
         *intersect_length = *intersect_length + 1;
      }
   }

   return 1;
}

/* Copy CSR matrix A to CSR matrix B.  The column indices are
 * assumed to be sorted, and the sparsity pattern of B is a subset
 * of the sparsity pattern of A.
 *
 * Assumptions:
 *      Column indices of A and B are sorted
 *      Sparsity pattern of B is a subset of A's
 *      A and B are the same size and have same data layout
 **/
HYPRE_Int
hypre_SortedCopyParCSRData(hypre_ParCSRMatrix  *A,
                           hypre_ParCSRMatrix  *B)
{
   /* Grab off A and B's data structures */
   hypre_CSRMatrix     *A_diag               = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int           *A_diag_i             = hypre_CSRMatrixI(A_diag);
   HYPRE_Int           *A_diag_j             = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real          *A_diag_data          = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix     *A_offd               = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int           *A_offd_i             = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           *A_offd_j             = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real          *A_offd_data          = hypre_CSRMatrixData(A_offd);

   hypre_CSRMatrix     *B_diag               = hypre_ParCSRMatrixDiag(B);
   HYPRE_Int           *B_diag_i             = hypre_CSRMatrixI(B_diag);
   HYPRE_Int           *B_diag_j             = hypre_CSRMatrixJ(B_diag);
   HYPRE_Real          *B_diag_data          = hypre_CSRMatrixData(B_diag);

   hypre_CSRMatrix     *B_offd               = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int           *B_offd_i             = hypre_CSRMatrixI(B_offd);
   HYPRE_Int           *B_offd_j             = hypre_CSRMatrixJ(B_offd);
   HYPRE_Real          *B_offd_data          = hypre_CSRMatrixData(B_offd);

   HYPRE_Int            num_variables        = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            *temp_int_array      = NULL;
   HYPRE_Int            temp_int_array_length = 0;
   HYPRE_Int            i, length, offset_A, offset_B;

   for (i = 0; i < num_variables; i++)
   {

      /* Deal with the first row entries, which may be diagonal elements */
      if ( A_diag_j[A_diag_i[i]] == i)
      {   offset_A = 1; }
      else
      {   offset_A = 0; }
      if ( B_diag_j[B_diag_i[i]] == i)
      {   offset_B = 1; }
      else
      {   offset_B = 0; }
      if ( (offset_B == 1) && (offset_A == 1) )
      {   B_diag_data[B_diag_i[i]] = A_diag_data[A_diag_i[i]]; }

      /* This finds the intersection of the column indices, and
       * also copies the matching data in A to the data array in B
       **/
      if ( (A_diag_i[i + 1] - A_diag_i[i] - offset_A) > temp_int_array_length )
      {
         hypre_TFree(temp_int_array, HYPRE_MEMORY_HOST);
         temp_int_array_length = (A_diag_i[i + 1] - A_diag_i[i] - offset_A);
         temp_int_array = hypre_CTAlloc(HYPRE_Int,  temp_int_array_length, HYPRE_MEMORY_HOST);
      }
      hypre_IntersectTwoArrays(&(A_diag_j[A_diag_i[i] + offset_A]),
                               &(A_diag_data[A_diag_i[i] + offset_A]),
                               A_diag_i[i + 1] - A_diag_i[i] - offset_A,
                               &(B_diag_j[B_diag_i[i] + offset_B]),
                               B_diag_i[i + 1] - B_diag_i[i] - offset_B,
                               temp_int_array,
                               &(B_diag_data[B_diag_i[i] + offset_B]),
                               &length);

      if ( (A_offd_i[i + 1] - A_offd_i[i]) > temp_int_array_length )
      {
         hypre_TFree(temp_int_array, HYPRE_MEMORY_HOST);
         temp_int_array_length = (A_offd_i[i + 1] - A_offd_i[i]);
         temp_int_array = hypre_CTAlloc(HYPRE_Int,  temp_int_array_length, HYPRE_MEMORY_HOST);
      }
      hypre_IntersectTwoArrays(&(A_offd_j[A_offd_i[i]]),
                               &(A_offd_data[A_offd_i[i]]),
                               A_offd_i[i + 1] - A_offd_i[i],
                               &(B_offd_j[B_offd_i[i]]),
                               B_offd_i[i + 1] - B_offd_i[i],
                               temp_int_array,
                               &(B_offd_data[B_offd_i[i]]),
                               &length);
   }

   if (temp_int_array)
   {    hypre_TFree(temp_int_array, HYPRE_MEMORY_HOST); }
   return 1;
}

/*
 * Equivalent to hypre_BoomerAMGCreateS, except, the data array of S
 * is not Null and contains the data entries from A.
 */
HYPRE_Int
hypre_BoomerAMG_MyCreateS(hypre_ParCSRMatrix  *A,
                          HYPRE_Real           strength_threshold,
                          HYPRE_Real           max_row_sum,
                          HYPRE_Int            num_functions,
                          HYPRE_Int           *dof_func,
                          hypre_ParCSRMatrix  **S_ptr)
{
   MPI_Comm                 comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix         *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int               *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Real              *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix         *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int               *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Real              *A_offd_data     = NULL;
   HYPRE_Int               *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int               *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_BigInt            *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int                num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt             global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int                num_nonzeros_diag;
   HYPRE_Int                num_nonzeros_offd = 0;
   HYPRE_Int                num_cols_offd     = 0;

   hypre_ParCSRMatrix      *S;
   hypre_CSRMatrix         *S_diag;
   HYPRE_Int               *S_diag_i;
   HYPRE_Int               *S_diag_j;
   HYPRE_Real              *S_diag_data;
   hypre_CSRMatrix         *S_offd;
   HYPRE_Int               *S_offd_i = NULL;
   HYPRE_Int               *S_offd_j = NULL;
   HYPRE_Real              *S_offd_data = NULL;

   HYPRE_Real               diag, row_scale, row_sum;
   HYPRE_Int                i, jA, jS;

   HYPRE_Int                ierr = 0;

   HYPRE_Int               *dof_func_offd;
   HYPRE_Int                num_sends;
   HYPRE_Int               *int_buf_data;
   HYPRE_Int                index, start, j;

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = aij, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   A_offd_i = hypre_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   /* Initialize S */
   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
                                row_starts, row_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(HYPRE_Int,  num_variables + 1, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_diag, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixData(S_diag) = hypre_CTAlloc(HYPRE_Real,  num_nonzeros_diag, HYPRE_MEMORY_HOST);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(HYPRE_Int,  num_variables + 1, HYPRE_MEMORY_HOST);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   S_diag_data = hypre_CSRMatrixData(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   hypre_CSRMatrixMemoryLocation(S_diag) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixMemoryLocation(S_offd) = HYPRE_MEMORY_HOST;

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
      A_offd_data = hypre_CSRMatrixData(A_offd);
      hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_offd, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixData(S_offd) = hypre_CTAlloc(HYPRE_Real,  num_nonzeros_offd, HYPRE_MEMORY_HOST);
      S_offd_j = hypre_CSRMatrixJ(S_offd);
      S_offd_data = hypre_CSRMatrixData(S_offd);
      hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(HYPRE_BigInt,  num_cols_offd, HYPRE_MEMORY_HOST);
      if (num_functions > 1)
      {
         dof_func_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      }
   }


   /*-------------------------------------------------------------------
    * Get the dof_func data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   if (num_functions > 1)
   {
      int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                              num_sends), HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                                  dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   }

   /* give S same nonzero structure as A */
   hypre_ParCSRMatrixCopy(A, S, 1);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,diag,row_scale,row_sum,jA) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_variables; i++)
   {
      diag = A_diag_data[A_diag_i[i]];

      /* compute scaling factor and row sum */
      row_scale = 0.0;
      row_sum = diag;
      if (num_functions > 1)
      {
         if (diag < 0)
         {
            for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = hypre_max(row_scale, A_diag_data[jA]);
                  row_sum += A_diag_data[jA];
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
               {
                  row_scale = hypre_max(row_scale, A_offd_data[jA]);
                  row_sum += A_offd_data[jA];
               }
            }
         }
         else
         {
            for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = hypre_min(row_scale, A_diag_data[jA]);
                  row_sum += A_diag_data[jA];
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
               {
                  row_scale = hypre_min(row_scale, A_offd_data[jA]);
                  row_sum += A_offd_data[jA];
               }
            }
         }
      }
      else
      {
         if (diag < 0)
         {
            for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
            {
               row_scale = hypre_max(row_scale, A_diag_data[jA]);
               row_sum += A_diag_data[jA];
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
            {
               row_scale = hypre_max(row_scale, A_offd_data[jA]);
               row_sum += A_offd_data[jA];
            }
         }
         else
         {
            for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
            {
               row_scale = hypre_min(row_scale, A_diag_data[jA]);
               row_sum += A_diag_data[jA];
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
            {
               row_scale = hypre_min(row_scale, A_offd_data[jA]);
               row_sum += A_offd_data[jA];
            }
         }
      }

      /* compute row entries of S */
      S_diag_j[A_diag_i[i]] = -1;
      if ((hypre_abs(row_sum) > hypre_abs(diag)*max_row_sum) && (max_row_sum < 1.0))
      {
         /* make all dependencies weak */
         for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
         {
            S_diag_j[jA] = -1;
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
         {
            S_offd_j[jA] = -1;
         }
      }
      else
      {
         if (num_functions > 1)
         {
            if (diag < 0)
            {
               for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
               {
                  if (A_diag_data[jA] <= strength_threshold * row_scale
                      || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
               {
                  if (A_offd_data[jA] <= strength_threshold * row_scale
                      || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                  {
                     S_offd_j[jA] = -1;
                  }
               }
            }
            else
            {
               for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
               {
                  if (A_diag_data[jA] >= strength_threshold * row_scale
                      || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
               {
                  if (A_offd_data[jA] >= strength_threshold * row_scale
                      || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                  {
                     S_offd_j[jA] = -1;
                  }
               }
            }
         }
         else
         {
            if (diag < 0)
            {
               for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
               {
                  if (A_diag_data[jA] <= strength_threshold * row_scale)
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
               {
                  if (A_offd_data[jA] <= strength_threshold * row_scale)
                  {
                     S_offd_j[jA] = -1;
                  }
               }
            }
            else
            {
               for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
               {
                  if (A_diag_data[jA] >= strength_threshold * row_scale)
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
               {
                  if (A_offd_data[jA] >= strength_threshold * row_scale)
                  {
                     S_offd_j[jA] = -1;
                  }
               }
            }
         }
      }
   }

   /*--------------------------------------------------------------
    * "Compress" the strength matrix.
    *
    * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
    *
    * NOTE: This "compression" section of code may not be removed, the
    * non-Galerkin routine depends on it.
    *----------------------------------------------------------------*/

   /* RDF: not sure if able to thread this loop */
   jS = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_diag_i[i] = jS;
      for (jA = A_diag_i[i]; jA < A_diag_i[i + 1]; jA++)
      {
         if (S_diag_j[jA] > -1)
         {
            S_diag_j[jS]    = S_diag_j[jA];
            S_diag_data[jS] = S_diag_data[jA];
            jS++;
         }
      }
   }
   S_diag_i[num_variables] = jS;
   hypre_CSRMatrixNumNonzeros(S_diag) = jS;

   /* RDF: not sure if able to thread this loop */
   jS = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_offd_i[i] = jS;
      for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
      {
         if (S_offd_j[jA] > -1)
         {
            S_offd_j[jS]    = S_offd_j[jA];
            S_offd_data[jS] = S_offd_data[jA];
            jS++;
         }
      }
   }
   S_offd_i[num_variables] = jS;
   hypre_CSRMatrixNumNonzeros(S_offd) = jS;
   hypre_ParCSRMatrixCommPkg(S) = NULL;

   *S_ptr        = S;

   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);

   return (ierr);
}

/**
 * Initialize the IJBuffer counters
 **/
HYPRE_Int
hypre_NonGalerkinIJBufferInit( HYPRE_Int
                               *ijbuf_cnt,           /* See NonGalerkinIJBufferWrite for parameter descriptions */
                               HYPRE_Int     *ijbuf_rowcounter,
                               HYPRE_Int     *ijbuf_numcols )
{
   HYPRE_Int                ierr = 0;

   (*ijbuf_cnt)         = 0;
   (*ijbuf_rowcounter)  = 1; /*Always points to the next row*/
   ijbuf_numcols[0]     = 0;

   return ierr;
}


/**
 * Initialize the IJBuffer counters
 **/
HYPRE_Int
hypre_NonGalerkinIJBigBufferInit( HYPRE_Int
                                  *ijbuf_cnt,           /* See NonGalerkinIJBufferWrite for parameter descriptions */
                                  HYPRE_Int     *ijbuf_rowcounter,
                                  HYPRE_BigInt     *ijbuf_numcols )
{
   HYPRE_Int                ierr = 0;

   (*ijbuf_cnt)         = 0;
   (*ijbuf_rowcounter)  = 1; /*Always points to the next row*/
   ijbuf_numcols[0]     = 0;

   return ierr;
}



/**
 * Update the buffer counters
 **/
HYPRE_Int
hypre_NonGalerkinIJBufferNewRow(HYPRE_BigInt
                                *ijbuf_rownums, /* See NonGalerkinIJBufferWrite for parameter descriptions */
                                HYPRE_Int     *ijbuf_numcols,
                                HYPRE_Int     *ijbuf_rowcounter,
                                HYPRE_BigInt   new_row)
{
   HYPRE_Int                ierr = 0;

   /* First check to see if the previous row was empty, and if so, overwrite that row */
   if ( ijbuf_numcols[(*ijbuf_rowcounter) - 1] == 0 )
   {
      ijbuf_rownums[(*ijbuf_rowcounter) - 1] = new_row;
   }
   else
   {
      /* Move to the next row */
      ijbuf_rownums[(*ijbuf_rowcounter)] = new_row;
      ijbuf_numcols[(*ijbuf_rowcounter)] = 0;
      (*ijbuf_rowcounter)++;
   }

   return ierr;
}

/**
 * Compress the current row in an IJ Buffer by removing duplicate entries
 **/
HYPRE_Int
hypre_NonGalerkinIJBufferCompressRow( HYPRE_Int
                                      *ijbuf_cnt,      /* See NonGalerkinIJBufferWrite for parameter descriptions */
                                      HYPRE_Int      ijbuf_rowcounter,
                                      HYPRE_Real     *ijbuf_data,
                                      HYPRE_BigInt   *ijbuf_cols,
                                      HYPRE_BigInt   *ijbuf_rownums,
                                      HYPRE_Int      *ijbuf_numcols)
{
   HYPRE_UNUSED_VAR(ijbuf_rownums);

   HYPRE_Int                ierr = 0;
   HYPRE_Int                nentries, i, nduplicate;

   /* Compress the current row by removing any repeat entries,
    * making sure to decrement ijbuf_cnt by nduplicate */
   nentries = ijbuf_numcols[ ijbuf_rowcounter - 1 ];
   nduplicate = 0;
   hypre_BigQsort1(ijbuf_cols, ijbuf_data, (*ijbuf_cnt) - nentries, (*ijbuf_cnt) - 1 );

   for (i = (*ijbuf_cnt) - nentries + 1; i <= (*ijbuf_cnt) - 1; i++)
   {
      if ( ijbuf_cols[i] == ijbuf_cols[i - 1] )
      {
         /* Shift duplicate entry down */
         nduplicate++;
         ijbuf_data[i - nduplicate] += ijbuf_data[i];
      }
      else if (nduplicate > 0)
      {
         ijbuf_data[i - nduplicate] = ijbuf_data[i];
         ijbuf_cols[i - nduplicate] = ijbuf_cols[i];
      }
   }
   (*ijbuf_cnt) -= nduplicate;
   ijbuf_numcols[ ijbuf_rowcounter - 1 ] -= nduplicate;

   return ierr;
}



/**
 * Compress the entire buffer, removing duplicate rows
 **/
HYPRE_Int
hypre_NonGalerkinIJBufferCompress( HYPRE_MemoryLocation memory_location,
                                   HYPRE_Int            ijbuf_size,
                                   HYPRE_Int           *ijbuf_cnt,      /* See NonGalerkinIJBufferWrite for parameter descriptions */
                                   HYPRE_Int           *ijbuf_rowcounter,
                                   HYPRE_Real         **ijbuf_data,
                                   HYPRE_BigInt       **ijbuf_cols,
                                   HYPRE_BigInt       **ijbuf_rownums,
                                   HYPRE_Int          **ijbuf_numcols)
{
   HYPRE_Int                ierr       = 0;
   HYPRE_Int                *indys     = hypre_CTAlloc(HYPRE_Int,  (*ijbuf_rowcounter),
                                                       HYPRE_MEMORY_HOST);

   HYPRE_Int                i, duplicate, cnt_new, rowcounter_new;
   HYPRE_Int                row_loc;
   HYPRE_BigInt             row_start, row_stop, row, prev_row, j;

   HYPRE_Real               *data_new;
   HYPRE_BigInt             *cols_new;
   HYPRE_BigInt             *rownums_new;
   HYPRE_Int                *numcols_new;

   /* Do a sort on rownums, but store the original order in indys.
    * Then see if there are any duplicate rows */
   for (i = 0; i < (*ijbuf_rowcounter); i++)
   {
      indys[i] = i;
   }
   hypre_BigQsortbi((*ijbuf_rownums), indys, 0, (*ijbuf_rowcounter) - 1);
   duplicate = 0;
   for (i = 1; i < (*ijbuf_rowcounter); i++)
   {
      if (indys[i] != (indys[i - 1] + 1))
      {
         duplicate = 1;
         break;
      }
   }

   /* Compress duplicate rows */
   if (duplicate)
   {
      /* Accumulate numcols, so that it functions like a CSR row-pointer */
      for (i = 1; i < (*ijbuf_rowcounter); i++)
      {   (*ijbuf_numcols)[i] += (*ijbuf_numcols)[i - 1]; }

      /* Initialize new buffer */
      prev_row         = -1;
      rowcounter_new   = 0;
      cnt_new          = 0;
      data_new         = hypre_CTAlloc(HYPRE_Real,   ijbuf_size, memory_location);
      cols_new         = hypre_CTAlloc(HYPRE_BigInt, ijbuf_size, memory_location);
      rownums_new      = hypre_CTAlloc(HYPRE_BigInt, ijbuf_size, memory_location);
      numcols_new      = hypre_CTAlloc(HYPRE_Int,    ijbuf_size, memory_location);
      numcols_new[0]   = 0;

      /* Cycle through each row */
      for (i = 0; i < (*ijbuf_rowcounter); i++)
      {

         /* Find which row this is in local and global numberings, and where
          * this row's data starts and stops in the buffer*/
         row_loc = indys[i];
         row = (*ijbuf_rownums)[i];
         if (row_loc > 0)
         {
            row_start = (HYPRE_BigInt) (*ijbuf_numcols)[row_loc - 1];
            row_stop  = (HYPRE_BigInt) (*ijbuf_numcols)[row_loc];
         }
         else
         {
            row_start = 0;
            row_stop  = (HYPRE_BigInt) (*ijbuf_numcols)[row_loc];
         }

         /* Is this a new row?  If so, compress previous row, and add a new
          * one.  Noting that prev_row = -1 is a special value */
         if (row != prev_row)
         {
            if (prev_row != -1)
            {
               /* Compress previous row */
               hypre_NonGalerkinIJBufferCompressRow(&cnt_new, rowcounter_new, data_new,
                                                    cols_new, rownums_new, numcols_new);
            }
            prev_row = row;
            numcols_new[rowcounter_new] = 0;
            rownums_new[rowcounter_new] = row;
            rowcounter_new++;
         }

         /* Copy row into new buffer */
         for (j = row_start; j < row_stop; j++)
         {
            data_new[cnt_new] = (*ijbuf_data)[j];
            cols_new[cnt_new] = (*ijbuf_cols)[j];
            numcols_new[rowcounter_new - 1]++;
            cnt_new++;
         }
      }

      /* Compress the final row */
      if (i > 1)
      {
         hypre_NonGalerkinIJBufferCompressRow(&cnt_new, rowcounter_new, data_new,
                                              cols_new, rownums_new, numcols_new);
      }

      *ijbuf_cnt = cnt_new;
      *ijbuf_rowcounter = rowcounter_new;

      /* Point to the new buffer */
      hypre_TFree(*ijbuf_data,    memory_location);
      hypre_TFree(*ijbuf_cols,    memory_location);
      hypre_TFree(*ijbuf_rownums, memory_location);
      hypre_TFree(*ijbuf_numcols, memory_location);
      (*ijbuf_data)    = data_new;
      (*ijbuf_cols)    = cols_new;
      (*ijbuf_rownums) = rownums_new;
      (*ijbuf_numcols) = numcols_new;
   }

   hypre_TFree(indys, HYPRE_MEMORY_HOST);

   return ierr;
}


/**
 * Do a buffered write to an IJ matrix.
 * That is, write to the buffer, until the buffer is full. Then when the
 * buffer is full, write to the IJ matrix and reset the buffer counters
 * In effect, this buffers this operation
 *  A[row_to_write, col_to_write] += val_to_write
 **/
HYPRE_Int
hypre_NonGalerkinIJBufferWrite( HYPRE_IJMatrix
                                B,                 /* Unassembled matrix to add an entry to */
                                HYPRE_Int    *ijbuf_cnt,          /* current buffer size */
                                HYPRE_Int     ijbuf_size,         /* max buffer size */
                                HYPRE_Int    *ijbuf_rowcounter,   /* num of rows in rownums, (i.e., size of rownums) */
                                /* This counter will increase as you call this function for multiple rows */
                                HYPRE_Real   **ijbuf_data,         /* Array of values, of size ijbuf_size */
                                HYPRE_BigInt **ijbuf_cols,         /* Array of col indices, of size ijbuf_size */
                                HYPRE_BigInt
                                **ijbuf_rownums,      /* Holds row-indices that with numcols makes for a CSR-like data structure*/
                                HYPRE_Int
                                **ijbuf_numcols,      /* rownums[i] is the row num, and numcols holds the number of entries being added */
                                /* for that row. Note numcols is not cumulative like an actual CSR data structure*/
                                HYPRE_BigInt  row_to_write,       /* Entry to add to the buffer */
                                HYPRE_BigInt  col_to_write,       /*          Ditto             */
                                HYPRE_Real    val_to_write )      /*          Ditto             */
{
   HYPRE_Int                ierr = 0;

   HYPRE_MemoryLocation memory_location = hypre_IJMatrixMemoryLocation(B);

   if ( (*ijbuf_cnt) == 0 )
   {
      /* brand new buffer: increment buffer structures for the new row */
      hypre_NonGalerkinIJBufferNewRow((*ijbuf_rownums), (*ijbuf_numcols), ijbuf_rowcounter, row_to_write);

   }
   else if ((*ijbuf_rownums)[ (*ijbuf_rowcounter) - 1 ] != row_to_write)
   {
      /* If this is a new row, compress the previous row */
      hypre_NonGalerkinIJBufferCompressRow(ijbuf_cnt, (*ijbuf_rowcounter), (*ijbuf_data),
                                           (*ijbuf_cols), (*ijbuf_rownums), (*ijbuf_numcols));
      /* increment buffer structures for the new row */
      hypre_NonGalerkinIJBufferNewRow( (*ijbuf_rownums), (*ijbuf_numcols), ijbuf_rowcounter,
                                       row_to_write);
   }

   /* Add new entry to buffer */
   (*ijbuf_cols)[(*ijbuf_cnt)] = col_to_write;
   (*ijbuf_data)[(*ijbuf_cnt)] = val_to_write;
   (*ijbuf_numcols)[ (*ijbuf_rowcounter) - 1 ]++;
   (*ijbuf_cnt)++;

   /* Buffer is full, write to the matrix object */
   if ( (*ijbuf_cnt) == (ijbuf_size - 1) )
   {
      /* If the last row is empty, decrement rowcounter */
      if ( (*ijbuf_numcols)[ (*ijbuf_rowcounter) - 1 ] == 0)
      {    (*ijbuf_rowcounter)--; }

      /* Compress and Add Entries */
      hypre_NonGalerkinIJBufferCompressRow(ijbuf_cnt, (*ijbuf_rowcounter), (*ijbuf_data),
                                           (*ijbuf_cols), (*ijbuf_rownums), (*ijbuf_numcols));
      hypre_NonGalerkinIJBufferCompress(memory_location, ijbuf_size, ijbuf_cnt, ijbuf_rowcounter,
                                        ijbuf_data,
                                        ijbuf_cols, ijbuf_rownums, ijbuf_numcols);
      ierr += HYPRE_IJMatrixAddToValues(B, *ijbuf_rowcounter, (*ijbuf_numcols), (*ijbuf_rownums),
                                        (*ijbuf_cols), (*ijbuf_data));

      /* Reinitialize the buffer */
      hypre_NonGalerkinIJBufferInit( ijbuf_cnt, ijbuf_rowcounter, (*ijbuf_numcols));
      hypre_NonGalerkinIJBufferNewRow((*ijbuf_rownums), (*ijbuf_numcols), ijbuf_rowcounter, row_to_write);
   }

   return ierr;
}


/**
 * Empty the IJ Buffer with a final AddToValues.
 **/
HYPRE_Int
hypre_NonGalerkinIJBufferEmpty(HYPRE_IJMatrix
                               B, /* See NonGalerkinIJBufferWrite for parameter descriptions */
                               HYPRE_Int      ijbuf_size,
                               HYPRE_Int      *ijbuf_cnt,
                               HYPRE_Int      ijbuf_rowcounter,
                               HYPRE_Real     **ijbuf_data,
                               HYPRE_BigInt   **ijbuf_cols,
                               HYPRE_BigInt   **ijbuf_rownums,
                               HYPRE_Int      **ijbuf_numcols)
{
   HYPRE_Int                ierr = 0;
   HYPRE_MemoryLocation memory_location = hypre_IJMatrixMemoryLocation(B);

   if ( (*ijbuf_cnt) > 0)
   {
      /* Compress the last row and then write */
      hypre_NonGalerkinIJBufferCompressRow(ijbuf_cnt, ijbuf_rowcounter, (*ijbuf_data),
                                           (*ijbuf_cols), (*ijbuf_rownums), (*ijbuf_numcols));
      hypre_NonGalerkinIJBufferCompress(memory_location, ijbuf_size, ijbuf_cnt, &ijbuf_rowcounter,
                                        ijbuf_data,
                                        ijbuf_cols, ijbuf_rownums, ijbuf_numcols);
      ierr += HYPRE_IJMatrixAddToValues(B, ijbuf_rowcounter, (*ijbuf_numcols), (*ijbuf_rownums),
                                        (*ijbuf_cols), (*ijbuf_data));
   }
   (*ijbuf_cnt = 0);

   return ierr;
}


/*
 * Construct sparsity pattern based on R_I A P, plus entries required by drop tolerance
 */
hypre_ParCSRMatrix *
hypre_NonGalerkinSparsityPattern(hypre_ParCSRMatrix *R_IAP,
                                 hypre_ParCSRMatrix *RAP,
                                 HYPRE_Int * CF_marker,
                                 HYPRE_Real droptol,
                                 HYPRE_Int sym_collapse,
                                 HYPRE_Int collapse_beta )
{
   /* MPI Communicator */
   MPI_Comm            comm               = hypre_ParCSRMatrixComm(RAP);

   HYPRE_MemoryLocation memory_location_RAP = hypre_ParCSRMatrixMemoryLocation(RAP);

   /* Declare R_IAP */
   hypre_CSRMatrix    *R_IAP_diag         = hypre_ParCSRMatrixDiag(R_IAP);
   HYPRE_Int          *R_IAP_diag_i       = hypre_CSRMatrixI(R_IAP_diag);
   HYPRE_Int          *R_IAP_diag_j       = hypre_CSRMatrixJ(R_IAP_diag);

   hypre_CSRMatrix    *R_IAP_offd         = hypre_ParCSRMatrixOffd(R_IAP);
   HYPRE_Int          *R_IAP_offd_i       = hypre_CSRMatrixI(R_IAP_offd);
   HYPRE_Int          *R_IAP_offd_j       = hypre_CSRMatrixJ(R_IAP_offd);
   HYPRE_BigInt       *col_map_offd_R_IAP = hypre_ParCSRMatrixColMapOffd(R_IAP);

   /* Declare RAP */
   hypre_CSRMatrix    *RAP_diag           = hypre_ParCSRMatrixDiag(RAP);
   HYPRE_Int          *RAP_diag_i         = hypre_CSRMatrixI(RAP_diag);
   HYPRE_Real         *RAP_diag_data      = hypre_CSRMatrixData(RAP_diag);
   HYPRE_Int          *RAP_diag_j         = hypre_CSRMatrixJ(RAP_diag);
   HYPRE_BigInt        first_col_diag_RAP = hypre_ParCSRMatrixFirstColDiag(RAP);
   HYPRE_Int           num_cols_diag_RAP  = hypre_CSRMatrixNumCols(RAP_diag);
   HYPRE_BigInt        last_col_diag_RAP  = first_col_diag_RAP + (HYPRE_BigInt)num_cols_diag_RAP - 1;

   hypre_CSRMatrix    *RAP_offd           = hypre_ParCSRMatrixOffd(RAP);
   HYPRE_Int          *RAP_offd_i         = hypre_CSRMatrixI(RAP_offd);
   HYPRE_Real         *RAP_offd_data      = NULL;
   HYPRE_Int          *RAP_offd_j         = hypre_CSRMatrixJ(RAP_offd);
   HYPRE_BigInt       *col_map_offd_RAP   = hypre_ParCSRMatrixColMapOffd(RAP);
   HYPRE_Int           num_cols_RAP_offd  = hypre_CSRMatrixNumCols(RAP_offd);

   HYPRE_Int           num_variables      = hypre_CSRMatrixNumRows(RAP_diag);

   /* Declare A */
   HYPRE_Int           num_fine_variables = hypre_CSRMatrixNumRows(R_IAP_diag);

   /* Declare IJ matrices */
   HYPRE_IJMatrix      Pattern;
   hypre_ParCSRMatrix *Pattern_CSR        = NULL;

   /* Buffered IJAddToValues */
   HYPRE_Int           ijbuf_cnt, ijbuf_size, ijbuf_rowcounter;
   HYPRE_Real         *ijbuf_data;
   HYPRE_BigInt       *ijbuf_cols, *ijbuf_rownums;
   HYPRE_Int          *ijbuf_numcols;

   /* Buffered IJAddToValues for Symmetric Entries */
   HYPRE_Int           ijbuf_sym_cnt, ijbuf_sym_rowcounter;
   HYPRE_Real         *ijbuf_sym_data;
   HYPRE_BigInt       *ijbuf_sym_cols, *ijbuf_sym_rownums;
   HYPRE_Int          *ijbuf_sym_numcols;

   /* Other Declarations */
   HYPRE_Real          max_entry         = 0.0;
   HYPRE_Real          max_entry_offd    = 0.0;
   HYPRE_Int          *rownz             = NULL;
   HYPRE_Int           i, j, Cpt, row_start, row_end;
   HYPRE_BigInt        global_row, global_col;

   /* Other Setup */
   if (num_cols_RAP_offd)
   {
      RAP_offd_data = hypre_CSRMatrixData(RAP_offd);
   }

   /*
    * Initialize the IJ matrix, leveraging our rough knowledge of the
    * nonzero structure of Pattern based on RAP
    *
    *                         ilower,             iupper,            jlower,             jupper */
   HYPRE_IJMatrixCreate(comm, first_col_diag_RAP, last_col_diag_RAP, first_col_diag_RAP,
                        last_col_diag_RAP, &Pattern);
   HYPRE_IJMatrixSetObjectType(Pattern, HYPRE_PARCSR);
   rownz = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_variables; i++)
   {
      rownz[i] = (HYPRE_Int)(1.2 * (RAP_diag_i[i + 1] - RAP_diag_i[i]) +
                             1.2 * (RAP_offd_i[i + 1] - RAP_offd_i[i]));
   }
   HYPRE_IJMatrixSetRowSizes(Pattern, rownz);
   HYPRE_IJMatrixInitialize(Pattern);
   hypre_TFree(rownz, HYPRE_MEMORY_HOST);

   /*
    * For efficiency, we do a buffered IJAddToValues.
    * Here, we initialize the buffer and then initialize the buffer counters
    */
   ijbuf_size       = 1000;
   ijbuf_data       = hypre_CTAlloc(HYPRE_Real,   ijbuf_size, memory_location_RAP);
   ijbuf_cols       = hypre_CTAlloc(HYPRE_BigInt, ijbuf_size, memory_location_RAP);
   ijbuf_rownums    = hypre_CTAlloc(HYPRE_BigInt, ijbuf_size, memory_location_RAP);
   ijbuf_numcols    = hypre_CTAlloc(HYPRE_Int,    ijbuf_size, memory_location_RAP);
   hypre_NonGalerkinIJBigBufferInit(&ijbuf_cnt, &ijbuf_rowcounter, ijbuf_cols);
   if (sym_collapse)
   {
      ijbuf_sym_data    = hypre_CTAlloc(HYPRE_Real,    ijbuf_size, memory_location_RAP);
      ijbuf_sym_cols    = hypre_CTAlloc(HYPRE_BigInt,  ijbuf_size, memory_location_RAP);
      ijbuf_sym_rownums = hypre_CTAlloc(HYPRE_BigInt,  ijbuf_size, memory_location_RAP);
      ijbuf_sym_numcols = hypre_CTAlloc(HYPRE_Int,     ijbuf_size, memory_location_RAP);
      hypre_NonGalerkinIJBigBufferInit(&ijbuf_sym_cnt, &ijbuf_sym_rowcounter, ijbuf_sym_cols);
   }

   /*
    * Place entries in R_IAP into Pattern
    */
   Cpt = -1; /* Cpt contains the fine grid index of the i-th Cpt */
   for (i = 0; i < num_variables; i++)
   {
      global_row = i + first_col_diag_RAP;

      /* Find the next Coarse Point in CF_marker */
      for (j = Cpt + 1; j < num_fine_variables; j++)
      {
         if (CF_marker[j] == 1)  /* Found Next C-point */
         {
            Cpt = j;
            break;
         }
      }

      /* Diag Portion */
      row_start = R_IAP_diag_i[Cpt];
      row_end = R_IAP_diag_i[Cpt + 1];
      for (j = row_start; j < row_end; j++)
      {
         global_col = R_IAP_diag_j[j] + first_col_diag_RAP;
         /* This call adds a                       1 x 1 to  i            j           data */
         hypre_NonGalerkinIJBufferWrite(Pattern, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                        &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols,
                                        global_row, global_col, 1.0);
         if (sym_collapse)
         {
            hypre_NonGalerkinIJBufferWrite(Pattern, &ijbuf_sym_cnt,
                                           ijbuf_size, &ijbuf_sym_rowcounter, &ijbuf_sym_data,
                                           &ijbuf_sym_cols, &ijbuf_sym_rownums, &ijbuf_sym_numcols,
                                           global_col, global_row, 1.0);
         }
      }

      /* Offdiag Portion */
      row_start = R_IAP_offd_i[Cpt];
      row_end = R_IAP_offd_i[Cpt + 1];
      for (j = row_start; j < row_end; j++)
      {
         global_col = col_map_offd_R_IAP[R_IAP_offd_j[j]];
         /* This call adds a                       1 x 1 to  i            j           data */
         hypre_NonGalerkinIJBufferWrite(Pattern, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                        &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols,
                                        global_row, global_col, 1.0);

         if (sym_collapse)
         {
            hypre_NonGalerkinIJBufferWrite(Pattern, &ijbuf_sym_cnt,
                                           ijbuf_size, &ijbuf_sym_rowcounter, &ijbuf_sym_data,
                                           &ijbuf_sym_cols, &ijbuf_sym_rownums, &ijbuf_sym_numcols,
                                           global_col, global_row, 1.0);
         }
      }
   }

   /*
    * Use drop-tolerance to compute new entries for sparsity pattern
    */
   /*#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,max_entry,max_entry_offd,global_col,global_row) HYPRE_SMP_SCHEDULE
   #endif  */
   for (i = 0; i < num_variables; i++)
   {
      global_row = i + first_col_diag_RAP;

      /* Compute the drop tolerance for this row, which is just
       *  abs(max of row i)*droptol  */
      max_entry = -1.0;
      for (j = RAP_diag_i[i]; j < RAP_diag_i[i + 1]; j++)
      {
         if ( (RAP_diag_j[j] != i) && (max_entry < hypre_abs(RAP_diag_data[j]) ) )
         {   max_entry = hypre_abs(RAP_diag_data[j]); }
      }
      for (j = RAP_offd_i[i]; j < RAP_offd_i[i + 1]; j++)
      {
         {
            if ( max_entry < hypre_abs(RAP_offd_data[j]) )
            {   max_entry = hypre_abs(RAP_offd_data[j]); }
         }
      }
      max_entry *= droptol;
      max_entry_offd = max_entry * collapse_beta;

      /* Loop over diag portion, adding all entries that are "strong" */
      for (j = RAP_diag_i[i]; j < RAP_diag_i[i + 1]; j++)
      {
         if ( hypre_abs(RAP_diag_data[j]) > max_entry )
         {
            global_col = RAP_diag_j[j] + first_col_diag_RAP;
            /*#ifdef HYPRE_USING_OPENMP
            #pragma omp critical (IJAdd)
            #endif
            {*/
            /* For efficiency, we do a buffered IJAddToValues
             * A[global_row, global_col] += 1.0 */
            hypre_NonGalerkinIJBufferWrite(Pattern, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                           &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols,
                                           global_row, global_col, 1.0);
            if (sym_collapse)
            {
               hypre_NonGalerkinIJBufferWrite(Pattern, &ijbuf_sym_cnt,
                                              ijbuf_size, &ijbuf_sym_rowcounter, &ijbuf_sym_data,
                                              &ijbuf_sym_cols, &ijbuf_sym_rownums, &ijbuf_sym_numcols,
                                              global_col, global_row, 1.0);
            }
            /*}*/
         }
      }

      /* Loop over offd portion, adding all entries that are "strong" */
      for (j = RAP_offd_i[i]; j < RAP_offd_i[i + 1]; j++)
      {
         if ( hypre_abs(RAP_offd_data[j]) > max_entry_offd )
         {
            global_col = col_map_offd_RAP[ RAP_offd_j[j] ];
            /*#ifdef HYPRE_USING_OPENMP
            #pragma omp critical (IJAdd)
            #endif
            {*/
            /* For efficiency, we do a buffered IJAddToValues
             * A[global_row, global_col] += 1.0 */
            hypre_NonGalerkinIJBufferWrite(Pattern, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                           &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols,
                                           global_row, global_col, 1.0);
            if (sym_collapse)
            {
               hypre_NonGalerkinIJBufferWrite(Pattern, &ijbuf_sym_cnt,
                                              ijbuf_size, &ijbuf_sym_rowcounter, &ijbuf_sym_data,
                                              &ijbuf_sym_cols, &ijbuf_sym_rownums, &ijbuf_sym_numcols,
                                              global_col, global_row, 1.0);
            }
            /*}*/
         }
      }

   }

   /* For efficiency, we do a buffered IJAddToValues.
    * This empties the buffer of any remaining values */
   hypre_NonGalerkinIJBufferEmpty(Pattern, ijbuf_size, &ijbuf_cnt, ijbuf_rowcounter,
                                  &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols);
   if (sym_collapse)
   {
      hypre_NonGalerkinIJBufferEmpty(Pattern, ijbuf_size, &ijbuf_sym_cnt, ijbuf_sym_rowcounter,
                                     &ijbuf_sym_data, &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                     &ijbuf_sym_numcols);
   }

   /* Finalize Construction of Pattern */
   HYPRE_IJMatrixAssemble(Pattern);
   HYPRE_IJMatrixGetObject(Pattern, (void**) &Pattern_CSR);

   /* Deallocate */
   HYPRE_IJMatrixSetObjectType(Pattern, -1);
   HYPRE_IJMatrixDestroy(Pattern);
   hypre_TFree(ijbuf_data,    memory_location_RAP);
   hypre_TFree(ijbuf_cols,    memory_location_RAP);
   hypre_TFree(ijbuf_rownums, memory_location_RAP);
   hypre_TFree(ijbuf_numcols, memory_location_RAP);

   if (sym_collapse)
   {
      hypre_TFree(ijbuf_sym_data,    memory_location_RAP);
      hypre_TFree(ijbuf_sym_cols,    memory_location_RAP);
      hypre_TFree(ijbuf_sym_rownums, memory_location_RAP);
      hypre_TFree(ijbuf_sym_numcols, memory_location_RAP);
   }

   return Pattern_CSR;
}


HYPRE_Int
hypre_BoomerAMGBuildNonGalerkinCoarseOperator( hypre_ParCSRMatrix **RAP_ptr,
                                               hypre_ParCSRMatrix *AP,
                                               HYPRE_Real strong_threshold,
                                               HYPRE_Real max_row_sum,
                                               HYPRE_Int num_functions,
                                               HYPRE_Int * dof_func_value,
                                               HYPRE_Int * CF_marker,
                                               HYPRE_Real droptol, HYPRE_Int sym_collapse,
                                               HYPRE_Real lump_percent, HYPRE_Int collapse_beta )
{
   /* Initializations */
   MPI_Comm            comm                  = hypre_ParCSRMatrixComm(*RAP_ptr);
   hypre_ParCSRMatrix  *S                    = NULL;
   hypre_ParCSRMatrix  *RAP                  = *RAP_ptr;
   HYPRE_Int           i, j, k, row_start, row_end, num_cols_offd_Sext, num_procs;
   HYPRE_Int           S_ext_diag_size, S_ext_offd_size;
   HYPRE_BigInt        last_col_diag_RAP;
   HYPRE_Int           cnt_offd, cnt_diag, cnt;
   HYPRE_Int           col_indx_Pattern, current_Pattern_j, col_indx_RAP;
   HYPRE_BigInt        value;
   HYPRE_BigInt       *temp                = NULL;

   HYPRE_MemoryLocation memory_location_RAP = hypre_ParCSRMatrixMemoryLocation(RAP);

   /* Lumping related variables */
   HYPRE_IJMatrix      ijmatrix;
   HYPRE_BigInt        * Pattern_offd_indices          = NULL;
   HYPRE_BigInt        * S_offd_indices                = NULL;
   HYPRE_BigInt        * offd_intersection             = NULL;
   HYPRE_Real          * offd_intersection_data        = NULL;
   HYPRE_Int           * diag_intersection             = NULL;
   HYPRE_Real          * diag_intersection_data        = NULL;
   HYPRE_Int           Pattern_offd_indices_len        = 0;
   HYPRE_Int           Pattern_offd_indices_allocated_len = 0;
   HYPRE_Int           S_offd_indices_len              = 0;
   HYPRE_Int           S_offd_indices_allocated_len    = 0;
   HYPRE_Int           offd_intersection_len           = 0;
   HYPRE_Int           offd_intersection_allocated_len = 0;
   HYPRE_Int           diag_intersection_len           = 0;
   HYPRE_Int           diag_intersection_allocated_len = 0;
   HYPRE_Real          intersection_len                = 0;
   HYPRE_Int           * Pattern_indices_ptr           = NULL;
   HYPRE_Int           Pattern_diag_indices_len        = 0;
   HYPRE_Int           global_row                      = 0;
   HYPRE_Int           has_row_ended                   = 0;
   HYPRE_Real          lump_value                      = 0.;
   HYPRE_Real          diagonal_lump_value             = 0.;
   HYPRE_Real          neg_lump_value                  = 0.;
   HYPRE_Real          sum_strong_neigh                = 0.;
   HYPRE_Int           * rownz                         = NULL;

   /* offd and diag portions of RAP */
   hypre_CSRMatrix     *RAP_diag             = hypre_ParCSRMatrixDiag(RAP);
   HYPRE_Int           *RAP_diag_i           = hypre_CSRMatrixI(RAP_diag);
   HYPRE_Real          *RAP_diag_data        = hypre_CSRMatrixData(RAP_diag);
   HYPRE_Int           *RAP_diag_j           = hypre_CSRMatrixJ(RAP_diag);
   HYPRE_BigInt         first_col_diag_RAP   = hypre_ParCSRMatrixFirstColDiag(RAP);
   HYPRE_Int            num_cols_diag_RAP    = hypre_CSRMatrixNumCols(RAP_diag);

   hypre_CSRMatrix     *RAP_offd             = hypre_ParCSRMatrixOffd(RAP);
   HYPRE_Int           *RAP_offd_i           = hypre_CSRMatrixI(RAP_offd);
   HYPRE_Real          *RAP_offd_data        = NULL;
   HYPRE_Int           *RAP_offd_j           = hypre_CSRMatrixJ(RAP_offd);
   HYPRE_BigInt        *col_map_offd_RAP     = hypre_ParCSRMatrixColMapOffd(RAP);
   HYPRE_Int            num_cols_RAP_offd    = hypre_CSRMatrixNumCols(RAP_offd);
   HYPRE_Int            num_variables        = hypre_CSRMatrixNumRows(RAP_diag);

   /* offd and diag portions of S */
   hypre_CSRMatrix     *S_diag               = NULL;
   HYPRE_Int           *S_diag_i             = NULL;
   HYPRE_Real          *S_diag_data          = NULL;
   HYPRE_Int           *S_diag_j             = NULL;

   hypre_CSRMatrix     *S_offd               = NULL;
   HYPRE_Int           *S_offd_i             = NULL;
   HYPRE_Real          *S_offd_data          = NULL;
   HYPRE_Int           *S_offd_j             = NULL;
   HYPRE_BigInt        *col_map_offd_S       = NULL;

   HYPRE_Int            num_cols_offd_S;
   /* HYPRE_Int         num_nonzeros_S_diag; */

   /* off processor portions of S */
   hypre_CSRMatrix    *S_ext                 = NULL;
   HYPRE_Int          *S_ext_i               = NULL;
   HYPRE_Real         *S_ext_data            = NULL;
   HYPRE_BigInt       *S_ext_j               = NULL;

   HYPRE_Int          *S_ext_diag_i          = NULL;
   HYPRE_Real         *S_ext_diag_data       = NULL;
   HYPRE_Int          *S_ext_diag_j          = NULL;

   HYPRE_Int          *S_ext_offd_i          = NULL;
   HYPRE_Real         *S_ext_offd_data       = NULL;
   HYPRE_Int          *S_ext_offd_j          = NULL;
   HYPRE_BigInt       *col_map_offd_Sext     = NULL;
   /* HYPRE_Int            num_nonzeros_S_ext_diag;
      HYPRE_Int            num_nonzeros_S_ext_offd;
      HYPRE_Int            num_rows_Sext         = 0; */
   HYPRE_Int           row_indx_Sext         = 0;


   /* offd and diag portions of Pattern */
   hypre_ParCSRMatrix  *Pattern              = NULL;
   hypre_CSRMatrix     *Pattern_diag         = NULL;
   HYPRE_Int           *Pattern_diag_i       = NULL;
   HYPRE_Real          *Pattern_diag_data    = NULL;
   HYPRE_Int           *Pattern_diag_j       = NULL;

   hypre_CSRMatrix     *Pattern_offd         = NULL;
   HYPRE_Int           *Pattern_offd_i       = NULL;
   HYPRE_Real          *Pattern_offd_data    = NULL;
   HYPRE_Int           *Pattern_offd_j       = NULL;
   HYPRE_BigInt        *col_map_offd_Pattern = NULL;

   HYPRE_Int            num_cols_Pattern_offd;
   HYPRE_Int            my_id;

   /* Buffered IJAddToValues */
   HYPRE_Int           ijbuf_cnt, ijbuf_size, ijbuf_rowcounter;
   HYPRE_Real          *ijbuf_data;
   HYPRE_BigInt        *ijbuf_cols, *ijbuf_rownums;
   HYPRE_Int           *ijbuf_numcols;

   /* Buffered IJAddToValues for Symmetric Entries */
   HYPRE_Int           ijbuf_sym_cnt, ijbuf_sym_rowcounter;
   HYPRE_Real          *ijbuf_sym_data;
   HYPRE_BigInt        *ijbuf_sym_cols, *ijbuf_sym_rownums;
   HYPRE_Int           *ijbuf_sym_numcols;

   /* Further Initializations */
   if (num_cols_RAP_offd)
   {   RAP_offd_data = hypre_CSRMatrixData(RAP_offd); }
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Compute Sparsity Pattern  */
   Pattern                    = hypre_NonGalerkinSparsityPattern(AP, RAP, CF_marker, droptol,
                                                                 sym_collapse, collapse_beta);
   Pattern_diag               = hypre_ParCSRMatrixDiag(Pattern);
   Pattern_diag_i             = hypre_CSRMatrixI(Pattern_diag);
   Pattern_diag_data          = hypre_CSRMatrixData(Pattern_diag);
   Pattern_diag_j             = hypre_CSRMatrixJ(Pattern_diag);

   Pattern_offd               = hypre_ParCSRMatrixOffd(Pattern);
   Pattern_offd_i             = hypre_CSRMatrixI(Pattern_offd);
   Pattern_offd_j             = hypre_CSRMatrixJ(Pattern_offd);
   col_map_offd_Pattern       = hypre_ParCSRMatrixColMapOffd(Pattern);

   num_cols_Pattern_offd      = hypre_CSRMatrixNumCols(Pattern_offd);
   if (num_cols_Pattern_offd)
   {   Pattern_offd_data = hypre_CSRMatrixData(Pattern_offd); }

   /**
    * Fill in the entries of Pattern with entries from RAP
    **/

   /* First, sort column indices in RAP and Pattern */
   for (i = 0; i < num_variables; i++)
   {
      /* The diag matrices store the diagonal as first element in each row.
       * We maintain that for the case of Pattern and RAP, because the
       * strength of connection routine relies on it and we need to ignore
       * diagonal entries in Pattern later during set intersections.
       * */

      /* Sort diag portion of RAP */
      row_start = RAP_diag_i[i];
      if ( RAP_diag_j[row_start] == i)
      {   row_start = row_start + 1; }
      row_end = RAP_diag_i[i + 1];
      hypre_qsort1(RAP_diag_j, RAP_diag_data, row_start, row_end - 1 );

      /* Sort diag portion of Pattern */
      row_start = Pattern_diag_i[i];

      if ( Pattern_diag_j[row_start] == i)
      {   row_start = row_start + 1; }
      row_end = Pattern_diag_i[i + 1];
      hypre_qsort1(Pattern_diag_j, Pattern_diag_data, row_start, row_end - 1 );

      /* Sort offd portion of RAP */
      row_start = RAP_offd_i[i];
      row_end = RAP_offd_i[i + 1];
      hypre_qsort1(RAP_offd_j, RAP_offd_data, row_start, row_end - 1 );

      /* Sort offd portion of Pattern */
      /* Be careful to map coarse dof i with CF_marker into Pattern */
      row_start = Pattern_offd_i[i];
      row_end = Pattern_offd_i[i + 1];
      hypre_qsort1(Pattern_offd_j, Pattern_offd_data, row_start, row_end - 1 );

   }


   /* Create Strength matrix based on RAP or Pattern.  If Pattern is used,
    * then the SortedCopyParCSRData(...) function call must also be commented
    * back in */
   /* hypre_SortedCopyParCSRData(RAP, Pattern); */
   if (0)
   {
      /* hypre_BoomerAMG_MyCreateS(Pattern, strong_threshold, max_row_sum, */
      hypre_BoomerAMG_MyCreateS(RAP, strong_threshold, max_row_sum,
                                num_functions, dof_func_value, &S);
   }
   else
   {
      /* Passing in "1, NULL" because dof_array is not needed
       * because we assume that  the number of functions is 1 */
      /* hypre_BoomerAMG_MyCreateS(Pattern, strong_threshold, max_row_sum,*/
      hypre_BoomerAMG_MyCreateS(RAP, strong_threshold, max_row_sum,
                                1, NULL, &S);
   }
   /* Grab diag and offd parts of S */
   S_diag               = hypre_ParCSRMatrixDiag(S);
   S_diag_i             = hypre_CSRMatrixI(S_diag);
   S_diag_j             = hypre_CSRMatrixJ(S_diag);
   S_diag_data          = hypre_CSRMatrixData(S_diag);

   S_offd               = hypre_ParCSRMatrixOffd(S);
   S_offd_i             = hypre_CSRMatrixI(S_offd);
   S_offd_j             = hypre_CSRMatrixJ(S_offd);
   S_offd_data          = hypre_CSRMatrixData(S_offd);
   col_map_offd_S       = hypre_ParCSRMatrixColMapOffd(S);

   num_cols_offd_S      = hypre_CSRMatrixNumCols(S_offd);
   /* num_nonzeros_S_diag  = S_diag_i[num_variables]; */




   /* Grab part of S that is distance one away from the local rows
    * This is needed later for the stencil collapsing.  This section
    * of the code mimics par_rap.c when it extracts Ps_ext.
    * When moving from par_rap.c, the variable name changes were:
    * A      --> RAP
    * P      --> S
    * Ps_ext --> S_ext
    * P_ext_diag --> S_ext_diag
    * P_ext_offd --> S_ext_offd
    *
    * The data layout of S_ext as returned by ExtractBExt gives you only global
    * column indices, and must be converted to the local numbering.  This code
    * section constructs S_ext_diag and S_ext_offd, which are the distance 1
    * couplings in S based on the sparsity structure in RAP.
    * --> S_ext_diag corresponds to the same column slice that RAP_diag
    *     corresponds to.  Thus, the column indexing is the same as in
    *     RAP_diag such that S_ext_diag_j[k] just needs to be offset by
    *     the RAP_diag first global dof offset.
    * --> S_ext_offd column indexing is a little more complicated, and
    *     requires the computation below of col_map_S_ext_offd, which
    *     maps the local 0,1,2,... column indexing in S_ext_offd to global
    *     dof numbers.  Note, that the num_cols_RAP_offd is NOT equal to
    *     num_cols_offd_S_ext
    * --> The row indexing of S_ext_diag|offd is as follows.  Use
    *     col_map_offd_RAP, where the first index corresponds to the
    *     first global row index in S_ext_diag|offd.  Remember that ExtractBExt
    *     grabs the information from S required for locally computing
    *     (RAP*S)[proc_k row slice, :] */

   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S, RAP, 1);
      S_ext_data = hypre_CSRMatrixData(S_ext);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixBigJ(S_ext);
   }

   /* This uses the num_cols_RAP_offd to set S_ext_diag|offd_i, because S_ext
    * is the off-processor information needed to compute RAP*S.  That is,
    * num_cols_RAP_offd represents the number of rows needed from S_ext for
    * the multiplication */
   S_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_RAP_offd + 1, HYPRE_MEMORY_HOST);
   S_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_RAP_offd + 1, HYPRE_MEMORY_HOST);
   S_ext_diag_size = 0;
   S_ext_offd_size = 0;
   /* num_rows_Sext = num_cols_RAP_offd; */
   last_col_diag_RAP = first_col_diag_RAP + ((HYPRE_BigInt) (num_cols_diag_RAP - 1));

   /* construct the S_ext_diag and _offd row-pointer arrays by counting elements
    * This looks to create offd and diag blocks related to the local rows belonging
    * to this processor...we may not need to split up S_ext this way...or we could.
    * It would make for faster binary searching and set intersecting later...this will
    * be the bottle neck so LETS SPLIT THIS UP Between offd and diag*/
   for (i = 0; i < num_cols_RAP_offd; i++)
   {
      for (j = S_ext_i[i]; j < S_ext_i[i + 1]; j++)
      {
         if (S_ext_j[j] < first_col_diag_RAP || S_ext_j[j] > last_col_diag_RAP)
         {
            S_ext_offd_size++;
         }
         else
         {
            S_ext_diag_size++;
         }
      }
      S_ext_diag_i[i + 1] = S_ext_diag_size;
      S_ext_offd_i[i + 1] = S_ext_offd_size;
   }

   if (S_ext_diag_size)
   {
      S_ext_diag_j = hypre_CTAlloc(HYPRE_Int,  S_ext_diag_size, HYPRE_MEMORY_HOST);
      S_ext_diag_data = hypre_CTAlloc(HYPRE_Real,  S_ext_diag_size, HYPRE_MEMORY_HOST);
   }
   if (S_ext_offd_size)
   {
      S_ext_offd_j = hypre_CTAlloc(HYPRE_Int,  S_ext_offd_size, HYPRE_MEMORY_HOST);
      S_ext_offd_data = hypre_CTAlloc(HYPRE_Real,  S_ext_offd_size, HYPRE_MEMORY_HOST);
   }

   /* This copies over the column indices into the offd and diag parts.
    * The diag portion has it's local column indices shifted to start at 0.
    * The offd portion requires more work to construct the col_map_offd array
    * and a local column ordering. */
   cnt_offd = 0;
   cnt_diag = 0;
   cnt = 0;
   for (i = 0; i < num_cols_RAP_offd; i++)
   {
      for (j = S_ext_i[i]; j < S_ext_i[i + 1]; j++)
      {
         if (S_ext_j[j] < first_col_diag_RAP || S_ext_j[j] > last_col_diag_RAP)
         {
            S_ext_offd_data[cnt_offd] = S_ext_data[j];
            //S_ext_offd_j[cnt_offd++] = S_ext_j[j];
            S_ext_j[cnt_offd++] = S_ext_j[j];
         }
         else
         {
            S_ext_diag_data[cnt_diag] = S_ext_data[j];
            S_ext_diag_j[cnt_diag++] = (HYPRE_Int)(S_ext_j[j] - first_col_diag_RAP);
         }
      }
   }

   /* This creates col_map_offd_Sext */
   if (S_ext_offd_size || num_cols_offd_S)
   {
      temp = hypre_CTAlloc(HYPRE_BigInt,  S_ext_offd_size + num_cols_offd_S, HYPRE_MEMORY_HOST);
      for (i = 0; i < S_ext_offd_size; i++)
      {
         temp[i] = S_ext_j[i];
      }
      cnt = S_ext_offd_size;
      for (i = 0; i < num_cols_offd_S; i++)
      {
         temp[cnt++] = col_map_offd_S[i];
      }
   }
   if (cnt)
   {
      /* after this, the first so many entries of temp will hold the
       * unique column indices in S_ext_offd_j unioned with the indices
       * in col_map_offd_S */
      hypre_BigQsort0(temp, 0, cnt - 1);

      num_cols_offd_Sext = 1;
      value = temp[0];
      for (i = 1; i < cnt; i++)
      {
         if (temp[i] > value)
         {
            value = temp[i];
            temp[num_cols_offd_Sext++] = value;
         }
      }
   }
   else
   {
      num_cols_offd_Sext = 0;
   }

   /* num_nonzeros_S_ext_diag = cnt_diag;
    num_nonzeros_S_ext_offd = S_ext_offd_size; */

   col_map_offd_Sext = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_Sext, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_cols_offd_Sext; i++)
   {
      col_map_offd_Sext[i] = temp[i];
   }

   if (S_ext_offd_size || num_cols_offd_S)
   {
      hypre_TFree(temp, HYPRE_MEMORY_HOST);
   }

   /* look for S_ext_offd_j[i] in col_map_offd_Sext, and set S_ext_offd_j[i]
    * to the index of that column value in col_map_offd_Sext */
   for (i = 0 ; i < S_ext_offd_size; i++)
   {
      S_ext_offd_j[i] = hypre_BigBinarySearch(col_map_offd_Sext,
                                              S_ext_j[i],
                                              num_cols_offd_Sext);
   }

   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(S_ext);
      S_ext = NULL;
   }

   /* Need to sort column indices in S and S_ext */
   for (i = 0; i < num_variables; i++)
   {
      /* Re-Sort diag portion of Pattern, placing the diagonal entry in a
       * sorted position */
      row_start = Pattern_diag_i[i];
      row_end = Pattern_diag_i[i + 1];
      hypre_qsort1(Pattern_diag_j, Pattern_diag_data, row_start, row_end - 1 );

      /* Sort diag portion of S, noting that no diagonal entry */
      /* S has not "data" array...it's just NULL */
      row_start = S_diag_i[i];
      row_end = S_diag_i[i + 1];
      hypre_qsort1(S_diag_j, S_diag_data, row_start, row_end - 1 );

      /* Sort offd portion of S */
      /* S has no "data" array...it's just NULL */
      row_start = S_offd_i[i];
      row_end = S_offd_i[i + 1];
      hypre_qsort1(S_offd_j, S_offd_data, row_start, row_end - 1 );
   }

   /* Sort S_ext
    * num_cols_RAP_offd  equals  num_rows for S_ext*/
   for (i = 0; i < num_cols_RAP_offd; i++)
   {
      /* Sort diag portion of S_ext */
      row_start = S_ext_diag_i[i];
      row_end = S_ext_diag_i[i + 1];
      hypre_qsort1(S_ext_diag_j, S_ext_diag_data, row_start, row_end - 1 );

      /* Sort offd portion of S_ext */
      row_start = S_ext_offd_i[i];
      row_end = S_ext_offd_i[i + 1];
      hypre_qsort1(S_ext_offd_j, S_ext_offd_data, row_start, row_end - 1 );

   }

   /*
    * Now, for the fun stuff -- Computing the Non-Galerkin Operator
    */

   /* Initialize the ijmatrix, leveraging our knowledge of the nonzero
    * structure in Pattern */
   HYPRE_IJMatrixCreate(comm, first_col_diag_RAP, last_col_diag_RAP,
                        first_col_diag_RAP, last_col_diag_RAP, &ijmatrix);
   HYPRE_IJMatrixSetObjectType(ijmatrix, HYPRE_PARCSR);
   rownz = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_variables; i++)
   {
      rownz[i] = (HYPRE_Int)(1.2 * (Pattern_diag_i[i + 1] - Pattern_diag_i[i]) +
                             1.2 * (Pattern_offd_i[i + 1] - Pattern_offd_i[i]));
   }
   HYPRE_IJMatrixSetRowSizes(ijmatrix, rownz);
   HYPRE_IJMatrixInitialize(ijmatrix);
   hypre_TFree(rownz, HYPRE_MEMORY_HOST);

   /*
    *For efficiency, we do a buffered IJAddToValues.
    * Here, we initialize the buffer and then initialize the buffer counters
    */
   ijbuf_size       = 1000;
   ijbuf_data       = hypre_CTAlloc(HYPRE_Real,   ijbuf_size, memory_location_RAP);
   ijbuf_cols       = hypre_CTAlloc(HYPRE_BigInt, ijbuf_size, memory_location_RAP);
   ijbuf_rownums    = hypre_CTAlloc(HYPRE_BigInt, ijbuf_size, memory_location_RAP);
   ijbuf_numcols    = hypre_CTAlloc(HYPRE_Int,    ijbuf_size, memory_location_RAP);
   hypre_NonGalerkinIJBigBufferInit( &ijbuf_cnt, &ijbuf_rowcounter, ijbuf_cols );
   if (sym_collapse)
   {
      ijbuf_sym_data   = hypre_CTAlloc(HYPRE_Real,   ijbuf_size, memory_location_RAP);
      ijbuf_sym_cols   = hypre_CTAlloc(HYPRE_BigInt, ijbuf_size, memory_location_RAP);
      ijbuf_sym_rownums = hypre_CTAlloc(HYPRE_BigInt, ijbuf_size, memory_location_RAP);
      ijbuf_sym_numcols = hypre_CTAlloc(HYPRE_Int,    ijbuf_size, memory_location_RAP);
      hypre_NonGalerkinIJBigBufferInit( &ijbuf_sym_cnt, &ijbuf_sym_rowcounter, ijbuf_sym_cols );
   }

   /*
    * Eliminate Entries In RAP_diag
    * */
   for (i = 0; i < num_variables; i++)
   {
      global_row = (HYPRE_BigInt) i + first_col_diag_RAP;
      row_start = RAP_diag_i[i];
      row_end = RAP_diag_i[i + 1];
      has_row_ended = 0;

      /* Only do work if row has nonzeros */
      if ( row_start < row_end)
      {
         /* Grab pointer to current entry in Pattern_diag */
         current_Pattern_j = Pattern_diag_i[i];
         col_indx_Pattern = Pattern_diag_j[current_Pattern_j];

         /* Grab this row's indices out of Pattern offd and diag.  This will
          * be for computing index set intersections for lumping */
         /* Ensure adequate length */
         Pattern_offd_indices_len = Pattern_offd_i[i + 1] - Pattern_offd_i[i];
         if (Pattern_offd_indices_allocated_len < Pattern_offd_indices_len)
         {
            hypre_TFree(Pattern_offd_indices, HYPRE_MEMORY_HOST);
            Pattern_offd_indices = hypre_CTAlloc(HYPRE_BigInt,  Pattern_offd_indices_len, HYPRE_MEMORY_HOST);
            Pattern_offd_indices_allocated_len = Pattern_offd_indices_len;
         }
         /* Grab sub array from col_map, corresponding to the slice of Pattern_offd_j */
         hypre_GrabSubArray(Pattern_offd_j,
                            Pattern_offd_i[i], Pattern_offd_i[i + 1] - 1,
                            col_map_offd_Pattern, Pattern_offd_indices);
         /* No need to grab info out of Pattern_diag_j[...], here we just start from
          * Pattern_diag_i[i] and end at index Pattern_diag_i[i+1] - 1.  We do need to
          * ignore the diagonal entry in Pattern, because we don't lump entries there */
         if ( Pattern_diag_j[Pattern_diag_i[i]] == i )
         {
            Pattern_indices_ptr = &( Pattern_diag_j[Pattern_diag_i[i] + 1]);
            Pattern_diag_indices_len = Pattern_diag_i[i + 1] - Pattern_diag_i[i] - 1;
         }
         else
         {
            Pattern_indices_ptr = &( Pattern_diag_j[Pattern_diag_i[i]]);
            Pattern_diag_indices_len = Pattern_diag_i[i + 1] - Pattern_diag_i[i];
         }
      }

      for (j = row_start; j < row_end; j++)
      {
         col_indx_RAP = RAP_diag_j[j];

         /* Ignore zero entries in RAP */
         if ( RAP_diag_data[j] != 0.0)
         {
            /* Don't change the diagonal, just write it */
            if (col_indx_RAP == i)
            {
               /*#ifdef HY   PRE_USING_OPENMP
               #pragma omp    critical (IJAdd)
               #endif
               {*/
               /* For efficiency, we do a buffered IJAddToValues.
                * A[global_row, global_row] += RAP_diag_data[j] */
               hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                               &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row,
                                               global_row, RAP_diag_data[j] );
               /*}*/

            }
            /* The entry in RAP does not appear in Pattern, so LUMP it */
            else if ( (col_indx_RAP < col_indx_Pattern) || has_row_ended)
            {
               /* Lump entry (i, col_indx_RAP) in RAP */

               /* Grab the indices for row col_indx_RAP of S_offd and diag.  This will
                * be for computing lumping locations */
               S_offd_indices_len = S_offd_i[col_indx_RAP + 1] - S_offd_i[col_indx_RAP];
               if (S_offd_indices_allocated_len < S_offd_indices_len)
               {
                  hypre_TFree(S_offd_indices, HYPRE_MEMORY_HOST);
                  S_offd_indices = hypre_CTAlloc(HYPRE_BigInt,  S_offd_indices_len, HYPRE_MEMORY_HOST);
                  S_offd_indices_allocated_len = S_offd_indices_len;
               }
               /* Grab sub array from col_map, corresponding to the slice of S_offd_j */
               hypre_GrabSubArray(S_offd_j, S_offd_i[col_indx_RAP], S_offd_i[col_indx_RAP + 1] - 1,
                                  col_map_offd_S, S_offd_indices);
               /* No need to grab info out of S_diag_j[...], here we just start from
                * S_diag_i[col_indx_RAP] and end at index S_diag_i[col_indx_RAP+1] - 1 */

               /* Intersect the diag and offd pieces, remembering that the
                * diag array will need to have the offset +first_col_diag_RAP */
               cnt = hypre_max(S_offd_indices_len, Pattern_offd_indices_len);
               if (offd_intersection_allocated_len < cnt)
               {
                  hypre_TFree(offd_intersection, HYPRE_MEMORY_HOST);
                  hypre_TFree(offd_intersection_data, HYPRE_MEMORY_HOST);
                  offd_intersection = hypre_CTAlloc(HYPRE_BigInt,  cnt, HYPRE_MEMORY_HOST);
                  offd_intersection_data = hypre_CTAlloc(HYPRE_Real,  cnt, HYPRE_MEMORY_HOST);
                  offd_intersection_allocated_len = cnt;
               }
               /* This intersection also tracks S_offd_data and assumes that
                * S_offd_indices is the first argument here */
               hypre_IntersectTwoBigArrays(S_offd_indices,
                                           &(S_offd_data[ S_offd_i[col_indx_RAP] ]),
                                           S_offd_indices_len,
                                           Pattern_offd_indices,
                                           Pattern_offd_indices_len,
                                           offd_intersection,
                                           offd_intersection_data,
                                           &offd_intersection_len);


               /* Now, intersect the indices for the diag block.  Note that S_diag_j does
                * not have a diagonal entry, so no lumping occurs to the diagonal. */
               cnt = hypre_max(Pattern_diag_indices_len,
                               S_diag_i[col_indx_RAP + 1] - S_diag_i[col_indx_RAP] );
               if (diag_intersection_allocated_len < cnt)
               {
                  hypre_TFree(diag_intersection, HYPRE_MEMORY_HOST);
                  hypre_TFree(diag_intersection_data, HYPRE_MEMORY_HOST);
                  diag_intersection = hypre_CTAlloc(HYPRE_Int,  cnt, HYPRE_MEMORY_HOST);
                  diag_intersection_data = hypre_CTAlloc(HYPRE_Real,  cnt, HYPRE_MEMORY_HOST);
                  diag_intersection_allocated_len = cnt;
               }
               /* There is no diagonal entry in first position of S */
               hypre_IntersectTwoArrays( &(S_diag_j[S_diag_i[col_indx_RAP]]),
                                         &(S_diag_data[ S_diag_i[col_indx_RAP] ]),
                                         S_diag_i[col_indx_RAP + 1] - S_diag_i[col_indx_RAP],
                                         Pattern_indices_ptr,
                                         Pattern_diag_indices_len,
                                         diag_intersection,
                                         diag_intersection_data,
                                         &diag_intersection_len);

               /* Loop over these intersections, and lump a constant fraction of
                * RAP_diag_data[j] to each entry */
               intersection_len = diag_intersection_len + offd_intersection_len;
               if (intersection_len > 0)
               {
                  /* Sum the strength-of-connection values from row
                   * col_indx_RAP in S, corresponding to the indices we are
                   * collapsing to in row i This will give us our collapsing
                   * weights. */
                  sum_strong_neigh = 0.0;
                  for (k = 0; k < diag_intersection_len; k++)
                  {   sum_strong_neigh += hypre_abs(diag_intersection_data[k]); }
                  for (k = 0; k < offd_intersection_len; k++)
                  {   sum_strong_neigh += hypre_abs(offd_intersection_data[k]); }
                  sum_strong_neigh = RAP_diag_data[j] / sum_strong_neigh;

                  /* When lumping with the diag_intersection, must offset column index */
                  for (k = 0; k < diag_intersection_len; k++)
                  {
                     lump_value = lump_percent * hypre_abs(diag_intersection_data[k]) * sum_strong_neigh;
                     diagonal_lump_value = (1.0 - lump_percent) * hypre_abs(diag_intersection_data[k]) *
                                           sum_strong_neigh;
                     neg_lump_value = -1.0 * lump_value;
                     cnt = diag_intersection[k] + first_col_diag_RAP;

                     /*#ifdef HY   PRE_USING_OPENMP
                     #pragma omp    critical (IJAdd)
                     #endif
                     {*/
                     /* For efficiency, we do a buffered IJAddToValues.
                      * A[global_row, cnt] += RAP_diag_data[j] */
                     hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                     &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row,
                                                     cnt, lump_value );
                     if (lump_percent < 1.0)
                     {
                        /* Preserve row sum by updating diagonal */
                        hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                        &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row,
                                                        global_row, diagonal_lump_value );
                     }

                     /* Update mirror entries, if symmetric collapsing */
                     if (sym_collapse)
                     {
                        /* Update mirror entry */
                        hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                        &ijbuf_sym_cnt, ijbuf_size, &ijbuf_sym_rowcounter,
                                                        &ijbuf_sym_data, &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                        &ijbuf_sym_numcols, cnt, global_row, lump_value );
                        /* Update mirror entry diagonal */
                        hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                        &ijbuf_sym_cnt, ijbuf_size, &ijbuf_sym_rowcounter,
                                                        &ijbuf_sym_data, &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                        &ijbuf_sym_numcols, cnt, cnt, neg_lump_value );
                     }
                     /*}*/
                  }

                  /* The offd_intersection has global column indices, i.e., the
                   * col_map arrays contain global indices */
                  for (k = 0; k < offd_intersection_len; k++)
                  {
                     lump_value = lump_percent * hypre_abs(offd_intersection_data[k]) * sum_strong_neigh;
                     diagonal_lump_value = (1.0 - lump_percent) * hypre_abs(offd_intersection_data[k]) *
                                           sum_strong_neigh;
                     neg_lump_value = -1.0 * lump_value;

                     hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                     &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row,
                                                     offd_intersection[k], lump_value );

                     if (lump_percent < 1.0)
                     {
                        hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                        &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row,
                                                        global_row, diagonal_lump_value );
                     }

                     /* Update mirror entries, if symmetric collapsing */
                     if (sym_collapse)
                     {
                        hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                        &ijbuf_sym_cnt, ijbuf_size, &ijbuf_sym_rowcounter,
                                                        &ijbuf_sym_data, &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                        &ijbuf_sym_numcols, offd_intersection[k],
                                                        global_row, lump_value );
                        hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                        &ijbuf_sym_cnt, ijbuf_size, &ijbuf_sym_rowcounter,
                                                        &ijbuf_sym_data, &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                        &ijbuf_sym_numcols, offd_intersection[k],
                                                        offd_intersection[k], neg_lump_value );
                     }
                  }
               }
               /* If intersection is empty, do not eliminate entry */
               else
               {
                  /* Don't forget to update mirror entry if collapsing symmetrically */
                  if (sym_collapse)
                  {   lump_value = 0.5 * RAP_diag_data[j]; }
                  else
                  {   lump_value = RAP_diag_data[j]; }

                  cnt = col_indx_RAP + first_col_diag_RAP;
                  hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                  &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row,
                                                  cnt, lump_value );
                  if (sym_collapse)
                  {
                     hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                     &ijbuf_sym_cnt, ijbuf_size, &ijbuf_sym_rowcounter,
                                                     &ijbuf_sym_data, &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                     &ijbuf_sym_numcols, cnt, global_row, lump_value );
                  }
               }
            }
            /* The entry in RAP appears in Pattern, so keep it */
            else if (col_indx_RAP == col_indx_Pattern)
            {
               cnt = col_indx_RAP + first_col_diag_RAP;
               hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                               &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row,
                                               cnt, RAP_diag_data[j] );

               /* Only go to the next entry in Pattern, if this is not the end of a row */
               if ( current_Pattern_j < Pattern_diag_i[i + 1] - 1 )
               {
                  current_Pattern_j += 1;
                  col_indx_Pattern = Pattern_diag_j[current_Pattern_j];
               }
               else
               {   has_row_ended = 1;}
            }
            /* Increment col_indx_Pattern, and repeat this loop iter for current
             * col_ind_RAP value */
            else if (col_indx_RAP > col_indx_Pattern)
            {
               for (; current_Pattern_j < Pattern_diag_i[i + 1]; current_Pattern_j++)
               {
                  col_indx_Pattern = Pattern_diag_j[current_Pattern_j];
                  if (col_indx_RAP <= col_indx_Pattern)
                  {   break;}
               }

               /* If col_indx_RAP is still greater (i.e., we've reached a row end), then
                * we need to lump everything else in this row */
               if (col_indx_RAP > col_indx_Pattern)
               {   has_row_ended = 1; }

               /* Decrement j, in order to repeat this loop iteration for the current
                * col_indx_RAP value */
               j--;
            }
         }
      }

   }

   /*
    * Eliminate Entries In RAP_offd
    * Structure of this for-loop is very similar to the RAP_diag for-loop
    * But, not so similar that these loops should be combined into a single fuction.
    * */
   if (num_cols_RAP_offd)
   {
      for (i = 0; i < num_variables; i++)
      {
         global_row = i + first_col_diag_RAP;
         row_start = RAP_offd_i[i];
         row_end = RAP_offd_i[i + 1];
         has_row_ended = 0;

         /* Only do work if row has nonzeros */
         if ( row_start < row_end)
         {
            current_Pattern_j = Pattern_offd_i[i];
            Pattern_offd_indices_len = Pattern_offd_i[i + 1] - Pattern_offd_i[i];
            if ( (Pattern_offd_j != NULL) && (Pattern_offd_indices_len > 0) )
            {   col_indx_Pattern = col_map_offd_Pattern[ Pattern_offd_j[current_Pattern_j] ]; }
            else
            {
               /* if Pattern_offd_j is not allocated or this is a zero length row,
                then all entries need to be lumped.
                This is an analagous situation to has_row_ended=1. */
               col_indx_Pattern = -1;
               has_row_ended = 1;
            }

            /* Grab this row's indices out of Pattern offd and diag.  This will
             * be for computing index set intersections for lumping.  The above
             * loop over RAP_diag ensures adequate length of Pattern_offd_indices */
            /* Ensure adequate length */
            hypre_GrabSubArray(Pattern_offd_j,
                               Pattern_offd_i[i], Pattern_offd_i[i + 1] - 1,
                               col_map_offd_Pattern, Pattern_offd_indices);
            /* No need to grab info out of Pattern_diag_j[...], here we just start from
             * Pattern_diag_i[i] and end at index Pattern_diag_i[i+1] - 1.  We do need to
             * ignore the diagonal entry in Pattern, because we don't lump entries there */
            if ( Pattern_diag_j[Pattern_diag_i[i]] == i )
            {
               Pattern_indices_ptr = &( Pattern_diag_j[Pattern_diag_i[i] + 1]);
               Pattern_diag_indices_len = Pattern_diag_i[i + 1] - Pattern_diag_i[i] - 1;
            }
            else
            {
               Pattern_indices_ptr = &( Pattern_diag_j[Pattern_diag_i[i]]);
               Pattern_diag_indices_len = Pattern_diag_i[i + 1] - Pattern_diag_i[i];
            }

         }

         for (j = row_start; j < row_end; j++)
         {

            /* Ignore zero entries in RAP */
            if ( RAP_offd_data[j] != 0.0)
            {

               /* In general for all the offd_j arrays, we have to indirectly
                * index with the col_map_offd array to get a global index */
               col_indx_RAP = col_map_offd_RAP[ RAP_offd_j[j] ];

               /* The entry in RAP does not appear in Pattern, so LUMP it */
               if ( (col_indx_RAP < col_indx_Pattern) || has_row_ended)
               {
                  /* The row_indx_Sext would be found with:
                   row_indx_Sext     = hypre_BinarySearch(col_map_offd_RAP, col_indx_RAP, num_cols_RAP_offd);
                   But, we already know the answer to this with, */
                  row_indx_Sext        = RAP_offd_j[j];

                  /* Grab the indices for row row_indx_Sext from the offd and diag parts.  This will
                   * be for computing lumping locations */
                  S_offd_indices_len = S_ext_offd_i[row_indx_Sext + 1] - S_ext_offd_i[row_indx_Sext];
                  if (S_offd_indices_allocated_len < S_offd_indices_len)
                  {
                     hypre_TFree(S_offd_indices, HYPRE_MEMORY_HOST);
                     S_offd_indices = hypre_CTAlloc(HYPRE_BigInt,  S_offd_indices_len, HYPRE_MEMORY_HOST);
                     S_offd_indices_allocated_len = S_offd_indices_len;
                  }
                  /* Grab sub array from col_map, corresponding to the slice of S_ext_offd_j */
                  hypre_GrabSubArray(S_ext_offd_j, S_ext_offd_i[row_indx_Sext], S_ext_offd_i[row_indx_Sext + 1] - 1,
                                     col_map_offd_Sext, S_offd_indices);
                  /* No need to grab info out of S_ext_diag_j[...], here we just start from
                   * S_ext_diag_i[row_indx_Sext] and end at index S_ext_diag_i[row_indx_Sext+1] - 1 */

                  /* Intersect the diag and offd pieces, remembering that the
                   * diag array will need to have the offset +first_col_diag_RAP */
                  cnt = hypre_max(S_offd_indices_len, Pattern_offd_indices_len);
                  if (offd_intersection_allocated_len < cnt)
                  {
                     hypre_TFree(offd_intersection, HYPRE_MEMORY_HOST);
                     hypre_TFree(offd_intersection_data, HYPRE_MEMORY_HOST);
                     offd_intersection = hypre_CTAlloc(HYPRE_BigInt,  cnt, HYPRE_MEMORY_HOST);
                     offd_intersection_data = hypre_CTAlloc(HYPRE_Real,  cnt, HYPRE_MEMORY_HOST);
                     offd_intersection_allocated_len = cnt;
                  }
                  hypre_IntersectTwoBigArrays(S_offd_indices,
                                              &(S_ext_offd_data[ S_ext_offd_i[row_indx_Sext] ]),
                                              S_offd_indices_len,
                                              Pattern_offd_indices,
                                              Pattern_offd_indices_len,
                                              offd_intersection,
                                              offd_intersection_data,
                                              &offd_intersection_len);

                  /* Now, intersect the indices for the diag block. */
                  cnt = hypre_max(Pattern_diag_indices_len,
                                  S_ext_diag_i[row_indx_Sext + 1] - S_ext_diag_i[row_indx_Sext] );
                  if (diag_intersection_allocated_len < cnt)
                  {
                     hypre_TFree(diag_intersection, HYPRE_MEMORY_HOST);
                     hypre_TFree(diag_intersection_data, HYPRE_MEMORY_HOST);
                     diag_intersection = hypre_CTAlloc(HYPRE_Int,  cnt, HYPRE_MEMORY_HOST);
                     diag_intersection_data = hypre_CTAlloc(HYPRE_Real,  cnt, HYPRE_MEMORY_HOST);
                     diag_intersection_allocated_len = cnt;
                  }
                  hypre_IntersectTwoArrays( &(S_ext_diag_j[S_ext_diag_i[row_indx_Sext]]),
                                            &(S_ext_diag_data[ S_ext_diag_i[row_indx_Sext] ]),
                                            S_ext_diag_i[row_indx_Sext + 1] - S_ext_diag_i[row_indx_Sext],
                                            Pattern_indices_ptr,
                                            Pattern_diag_indices_len,
                                            diag_intersection,
                                            diag_intersection_data,
                                            &diag_intersection_len);

                  /* Loop over these intersections, and lump a constant fraction of
                   * RAP_offd_data[j] to each entry */
                  intersection_len = diag_intersection_len + offd_intersection_len;
                  if (intersection_len > 0)
                  {
                     /* Sum the strength-of-connection values from row
                      * row_indx_Sext in S, corresponding to the indices we are
                      * collapsing to in row i. This will give us our collapsing
                      * weights. */
                     sum_strong_neigh = 0.0;
                     for (k = 0; k < diag_intersection_len; k++)
                     {   sum_strong_neigh += hypre_abs(diag_intersection_data[k]); }
                     for (k = 0; k < offd_intersection_len; k++)
                     {   sum_strong_neigh += hypre_abs(offd_intersection_data[k]); }
                     sum_strong_neigh = RAP_offd_data[j] / sum_strong_neigh;

                     /* When lumping with the diag_intersection, must offset column index */
                     for (k = 0; k < diag_intersection_len; k++)
                     {
                        lump_value = lump_percent * hypre_abs(diag_intersection_data[k]) * sum_strong_neigh;
                        diagonal_lump_value = (1.0 - lump_percent) * hypre_abs(diag_intersection_data[k]) *
                                              sum_strong_neigh;
                        neg_lump_value = -1.0 * lump_value;
                        cnt = diag_intersection[k] + first_col_diag_RAP;

                        hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                        &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row, cnt, lump_value );
                        if (lump_percent < 1.0)
                        {
                           hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                           &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row, global_row,
                                                           diagonal_lump_value );
                        }

                        /* Update mirror entries, if symmetric collapsing */
                        if (sym_collapse)
                        {
                           hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                           &ijbuf_sym_cnt, ijbuf_size,
                                                           &ijbuf_sym_rowcounter, &ijbuf_sym_data,
                                                           &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                           &ijbuf_sym_numcols, cnt, global_row, lump_value);
                           hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                           &ijbuf_sym_cnt, ijbuf_size,
                                                           &ijbuf_sym_rowcounter, &ijbuf_sym_data,
                                                           &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                           &ijbuf_sym_numcols, cnt, cnt, neg_lump_value );
                        }
                     }

                     /* The offd_intersection has global column indices, i.e., the
                      * col_map arrays contain global indices */
                     for (k = 0; k < offd_intersection_len; k++)
                     {
                        lump_value = lump_percent * hypre_abs(offd_intersection_data[k]) * sum_strong_neigh;
                        diagonal_lump_value = (1.0 - lump_percent) * hypre_abs(offd_intersection_data[k]) *
                                              sum_strong_neigh;
                        neg_lump_value = -1.0 * lump_value;

                        hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                        &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row,
                                                        offd_intersection[k], lump_value );
                        if (lump_percent < 1.0)
                        {
                           hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                           &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row, global_row,
                                                           diagonal_lump_value );
                        }


                        /* Update mirror entries, if symmetric collapsing */
                        if (sym_collapse)
                        {
                           hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                           &ijbuf_sym_cnt, ijbuf_size,
                                                           &ijbuf_sym_rowcounter, &ijbuf_sym_data,
                                                           &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                           &ijbuf_sym_numcols, offd_intersection[k],
                                                           global_row, lump_value );
                           hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                           &ijbuf_sym_cnt, ijbuf_size,
                                                           &ijbuf_sym_rowcounter, &ijbuf_sym_data,
                                                           &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                           &ijbuf_sym_numcols, offd_intersection[k],
                                                           offd_intersection[k], neg_lump_value );
                        }
                     }
                  }
                  /* If intersection is empty, do not eliminate entry */
                  else
                  {
                     /* Don't forget to update mirror entry if collapsing symmetrically */
                     if (sym_collapse)
                     {   lump_value = 0.5 * RAP_offd_data[j]; }
                     else
                     {   lump_value = RAP_offd_data[j]; }

                     hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                     &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row, col_indx_RAP,
                                                     lump_value );
                     if (sym_collapse)
                     {
                        hypre_NonGalerkinIJBufferWrite( ijmatrix,
                                                        &ijbuf_sym_cnt, ijbuf_size, &ijbuf_sym_rowcounter,
                                                        &ijbuf_sym_data, &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                                        &ijbuf_sym_numcols, col_indx_RAP, global_row,
                                                        lump_value );
                     }
                  }
               }
               /* The entry in RAP appears in Pattern, so keep it */
               else if (col_indx_RAP == col_indx_Pattern)
               {
                  /* For the offd structure, col_indx_RAP is a global dof number */
                  hypre_NonGalerkinIJBufferWrite( ijmatrix, &ijbuf_cnt, ijbuf_size, &ijbuf_rowcounter,
                                                  &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols, global_row, col_indx_RAP,
                                                  RAP_offd_data[j]);

                  /* Only go to the next entry in Pattern, if this is not the end of a row */
                  if ( current_Pattern_j < Pattern_offd_i[i + 1] - 1 )
                  {
                     current_Pattern_j += 1;
                     col_indx_Pattern = col_map_offd_Pattern[ Pattern_offd_j[current_Pattern_j] ];
                  }
                  else
                  {   has_row_ended = 1;}
               }
               /* Increment col_indx_Pattern, and repeat this loop iter for current
                * col_ind_RAP value */
               else if (col_indx_RAP > col_indx_Pattern)
               {
                  for (; current_Pattern_j < Pattern_offd_i[i + 1]; current_Pattern_j++)
                  {
                     col_indx_Pattern = col_map_offd_Pattern[ Pattern_offd_j[current_Pattern_j] ];
                     if (col_indx_RAP <= col_indx_Pattern)
                     {   break;}
                  }

                  /* If col_indx_RAP is still greater (i.e., we've reached a row end), then
                   * we need to lump everything else in this row */
                  if (col_indx_RAP > col_indx_Pattern)
                  {   has_row_ended = 1; }

                  /* Decrement j, in order to repeat this loop iteration for the current
                   * col_indx_RAP value */
                  j--;
               }
            }
         }
      }
   }

   /* For efficiency, we do a buffered IJAddToValues.
    * This empties the buffer of any remaining values */
   hypre_NonGalerkinIJBufferEmpty(ijmatrix, ijbuf_size, &ijbuf_cnt, ijbuf_rowcounter,
                                  &ijbuf_data, &ijbuf_cols, &ijbuf_rownums, &ijbuf_numcols);
   if (sym_collapse)
   {
      hypre_NonGalerkinIJBufferEmpty(ijmatrix, ijbuf_size, &ijbuf_sym_cnt, ijbuf_sym_rowcounter,
                                     &ijbuf_sym_data, &ijbuf_sym_cols, &ijbuf_sym_rownums,
                                     &ijbuf_sym_numcols);
   }

   /* Assemble non-Galerkin Matrix, and overwrite current RAP*/
   HYPRE_IJMatrixAssemble(ijmatrix);
   HYPRE_IJMatrixGetObject(ijmatrix, (void**) RAP_ptr);

   /* Optional diagnostic matrix printing */
#if 0
   char  filename[256];

   hypre_sprintf(filename, "Pattern_%d.ij", global_num_vars);
   hypre_ParCSRMatrixPrintIJ(Pattern, 0, 0, filename);
   hypre_sprintf(filename, "Strength_%d.ij", global_num_vars);
   hypre_ParCSRMatrixPrintIJ(S, 0, 0, filename);
   hypre_sprintf(filename, "RAP_%d.ij", global_num_vars);
   hypre_ParCSRMatrixPrintIJ(RAP, 0, 0, filename);
   hypre_sprintf(filename, "RAPc_%d.ij", global_num_vars);
   hypre_ParCSRMatrixPrintIJ(*RAP_ptr, 0, 0, filename);
   hypre_sprintf(filename, "AP_%d.ij", global_num_vars);
   hypre_ParCSRMatrixPrintIJ(AP, 0, 0, filename);
#endif

   /* Free matrices and variables and arrays */
   hypre_TFree(ijbuf_data,    memory_location_RAP);
   hypre_TFree(ijbuf_cols,    memory_location_RAP);
   hypre_TFree(ijbuf_rownums, memory_location_RAP);
   hypre_TFree(ijbuf_numcols, memory_location_RAP);
   if (sym_collapse)
   {
      hypre_TFree(ijbuf_sym_data,    memory_location_RAP);
      hypre_TFree(ijbuf_sym_cols,    memory_location_RAP);
      hypre_TFree(ijbuf_sym_rownums, memory_location_RAP);
      hypre_TFree(ijbuf_sym_numcols, memory_location_RAP);
   }
   hypre_TFree(Pattern_offd_indices, HYPRE_MEMORY_HOST);
   hypre_TFree(S_ext_diag_i, HYPRE_MEMORY_HOST);
   hypre_TFree(S_ext_offd_i, HYPRE_MEMORY_HOST);
   hypre_TFree(S_offd_indices, HYPRE_MEMORY_HOST);
   hypre_TFree(offd_intersection, HYPRE_MEMORY_HOST);
   hypre_TFree(offd_intersection_data, HYPRE_MEMORY_HOST);
   hypre_TFree(diag_intersection, HYPRE_MEMORY_HOST);
   hypre_TFree(diag_intersection_data, HYPRE_MEMORY_HOST);
   hypre_TFree(S_ext_diag_j, HYPRE_MEMORY_HOST);
   hypre_TFree(S_ext_diag_data, HYPRE_MEMORY_HOST);
   hypre_TFree(S_ext_offd_j, HYPRE_MEMORY_HOST);
   hypre_TFree(S_ext_offd_data, HYPRE_MEMORY_HOST);
   hypre_TFree(col_map_offd_Sext, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrixDestroy(Pattern);
   hypre_ParCSRMatrixDestroy(RAP);
   hypre_ParCSRMatrixDestroy(S);
   HYPRE_IJMatrixSetObjectType(ijmatrix, -1);
   HYPRE_IJMatrixDestroy(ijmatrix);

   return hypre_error_flag;
}
