/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*==========================================================================*/
/*==========================================================================*/
/**
  Generates strength matrix

  Notes:
  \begin{itemize}
  \item The underlying matrix storage scheme is a hypre_ParCSR matrix.
  \item The routine returns the following:
  \begin{itemize}
  \item S - a ParCSR matrix representing the "strength matrix".  This is
  used in the coarsening and interpolation routines.
  \end{itemize}
  \item The graph of the "strength matrix" for A is a subgraph of the
  graph of A, but requires nonsymmetric storage even if A is
  symmetric.  This is because of the directional nature of the
  "strengh of dependence" notion (see below).  Since we are using
  nonsymmetric storage for A right now, this is not a problem.  If we
  ever add the ability to store A symmetrically, then we could store
  the strength graph as floats instead of doubles to save space.
  \item This routine currently "compresses" the strength matrix.  We
  should consider the possibility of defining this matrix to have the
  same "nonzero structure" as A.  To do this, we could use the same
  A\_i and A\_j arrays, and would need only define the S\_data array.
  There are several pros and cons to discuss.
  \end{itemize}

  Terminology:
  \begin{itemize}
  \item Ruge's terminology: A point is "strongly connected to" $j$, or
  "strongly depends on" $j$, if $-a_ij >= \theta max_{l != j} \{-a_il\}$.
  \item Here, we retain some of this terminology, but with a more
  generalized notion of "strength".  We also retain the "natural"
  graph notation for representing the directed graph of a matrix.
  That is, the nonzero entry $a_ij$ is represented as: i --> j.  In
  the strength matrix, S, the entry $s_ij$ is also graphically denoted
  as above, and means both of the following:
  \begin{itemize}
  \item $i$ "depends on" $j$ with "strength" $s_ij$
  \item $j$ "influences" $i$ with "strength" $s_ij$
  \end{itemize}
  \end{itemize}

  {\bf Input files:}
  _hypre_parcsr_ls.h

  @return Error code.

  @param A [IN]
  coefficient matrix
  @param strength_threshold [IN]
  threshold parameter used to define strength
  @param max_row_sum [IN]
  parameter used to modify definition of strength for diagonal dominant matrices
  @param S_ptr [OUT]
  strength matrix

  @see */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreateSHost(hypre_ParCSRMatrix    *A,
                           HYPRE_Real             strength_threshold,
                           HYPRE_Real             max_row_sum,
                           HYPRE_Int              num_functions,
                           HYPRE_Int             *dof_func,
                           hypre_ParCSRMatrix   **S_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] -= hypre_MPI_Wtime();
#endif

   MPI_Comm                 comm       = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg   = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int          *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Real         *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Real         *A_offd_data = NULL;
   HYPRE_Int          *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int          *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_BigInt       *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int           num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt        global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int           num_nonzeros_diag;
   HYPRE_Int           num_nonzeros_offd = 0;
   HYPRE_Int           num_cols_offd = 0;

   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   HYPRE_Int          *S_diag_i;
   HYPRE_Int          *S_diag_j;
   /* HYPRE_Real         *S_diag_data; */
   hypre_CSRMatrix    *S_offd;
   HYPRE_Int          *S_offd_i = NULL;
   HYPRE_Int          *S_offd_j = NULL;
   /* HYPRE_Real         *S_offd_data; */

   HYPRE_Real          diag, row_scale, row_sum;
   HYPRE_Int           i, jA, jS;

   HYPRE_Int           ierr = 0;

   HYPRE_Int          *dof_func_offd;
   HYPRE_Int           num_sends;
   HYPRE_Int          *int_buf_data;
   HYPRE_Int           index, start, j;

   HYPRE_Int *prefix_sum_workspace;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   A_offd_i = hypre_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
                                row_starts, row_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);

   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_location);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_diag, HYPRE_MEMORY_HOST);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_location);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int *S_temp_diag_j = hypre_CSRMatrixJ(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   S_diag_j = hypre_TAlloc(HYPRE_Int, num_nonzeros_diag, memory_location);
   HYPRE_Int *S_temp_offd_j = NULL;

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
      A_offd_data = hypre_CSRMatrixData(A_offd);
      hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_offd, HYPRE_MEMORY_HOST);
      S_temp_offd_j = hypre_CSRMatrixJ(S_offd);
      HYPRE_BigInt *col_map_offd_S = hypre_TAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;
      if (num_functions > 1)
      {
         dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
      }

      S_offd_j = hypre_TAlloc(HYPRE_Int, num_nonzeros_offd, memory_location);

      HYPRE_BigInt *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols_offd; i++)
      {
         col_map_offd_S[i] = col_map_offd_A[i];
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
      int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                   HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   }

   /*HYPRE_Int prefix_sum_workspace[2*(hypre_NumThreads() + 1)];*/
   prefix_sum_workspace = hypre_TAlloc(HYPRE_Int,  2 * (hypre_NumThreads() + 1), HYPRE_MEMORY_HOST);

   /* give S same nonzero structure as A */
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,diag,row_scale,row_sum,jA,jS)
#endif
   {
      HYPRE_Int start, stop;
      hypre_GetSimpleThreadPartition(&start, &stop, num_variables);
      HYPRE_Int jS_diag = 0, jS_offd = 0;

      for (i = start; i < stop; i++)
      {
         S_diag_i[i] = jS_diag;
         if (num_cols_offd)
         {
            S_offd_i[i] = jS_offd;
         }

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
            } /* diag >= 0 */
         } /* num_functions > 1 */
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
            } /* diag >= 0*/
         } /* num_functions <= 1 */

         jS_diag += A_diag_i[i + 1] - A_diag_i[i] - 1;
         jS_offd += A_offd_i[i + 1] - A_offd_i[i];

         /* compute row entries of S */
         S_temp_diag_j[A_diag_i[i]] = -1;
         if ((hypre_abs(row_sum) > hypre_abs(diag)*max_row_sum) && (max_row_sum < 1.0))
         {
            /* make all dependencies weak */
            for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
            {
               S_temp_diag_j[jA] = -1;
            }
            jS_diag -= A_diag_i[i + 1] - (A_diag_i[i] + 1);

            for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
            {
               S_temp_offd_j[jA] = -1;
            }
            jS_offd -= A_offd_i[i + 1] - A_offd_i[i];
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
                        S_temp_diag_j[jA] = -1;
                        --jS_diag;
                     }
                     else
                     {
                        S_temp_diag_j[jA] = A_diag_j[jA];
                     }
                  }
                  for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                  {
                     if (A_offd_data[jA] <= strength_threshold * row_scale
                         || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                     {
                        S_temp_offd_j[jA] = -1;
                        --jS_offd;
                     }
                     else
                     {
                        S_temp_offd_j[jA] = A_offd_j[jA];
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
                        S_temp_diag_j[jA] = -1;
                        --jS_diag;
                     }
                     else
                     {
                        S_temp_diag_j[jA] = A_diag_j[jA];
                     }
                  }
                  for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                  {
                     if (A_offd_data[jA] >= strength_threshold * row_scale
                         || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                     {
                        S_temp_offd_j[jA] = -1;
                        --jS_offd;
                     }
                     else
                     {
                        S_temp_offd_j[jA] = A_offd_j[jA];
                     }
                  }
               } /* diag >= 0 */
            } /* num_functions > 1 */
            else
            {
               if (diag < 0)
               {
                  for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
                  {
                     if (A_diag_data[jA] <= strength_threshold * row_scale)
                     {
                        S_temp_diag_j[jA] = -1;
                        --jS_diag;
                     }
                     else
                     {
                        S_temp_diag_j[jA] = A_diag_j[jA];
                     }
                  }
                  for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                  {
                     if (A_offd_data[jA] <= strength_threshold * row_scale)
                     {
                        S_temp_offd_j[jA] = -1;
                        --jS_offd;
                     }
                     else
                     {
                        S_temp_offd_j[jA] = A_offd_j[jA];
                     }
                  }
               }
               else
               {
                  for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
                  {
                     if (A_diag_data[jA] >= strength_threshold * row_scale)
                     {
                        S_temp_diag_j[jA] = -1;
                        --jS_diag;
                     }
                     else
                     {
                        S_temp_diag_j[jA] = A_diag_j[jA];
                     }
                  }
                  for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                  {
                     if (A_offd_data[jA] >= strength_threshold * row_scale)
                     {
                        S_temp_offd_j[jA] = -1;
                        --jS_offd;
                     }
                     else
                     {
                        S_temp_offd_j[jA] = A_offd_j[jA];
                     }
                  }
               } /* diag >= 0 */
            } /* num_functions <= 1 */
         } /* !((row_sum > max_row_sum) && (max_row_sum < 1.0)) */
      } /* for each variable */

      hypre_prefix_sum_pair(&jS_diag, S_diag_i + num_variables, &jS_offd, S_offd_i + num_variables,
                            prefix_sum_workspace);

      /*--------------------------------------------------------------
       * "Compress" the strength matrix.
       *
       * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
       *
       * NOTE: This "compression" section of code may be removed, and
       * coarsening will still be done correctly.  However, the routine
       * that builds interpolation would have to be modified first.
       *----------------------------------------------------------------*/

      for (i = start; i < stop; i++)
      {
         S_diag_i[i] += jS_diag;
         S_offd_i[i] += jS_offd;

         jS = S_diag_i[i];
         for (jA = A_diag_i[i]; jA < A_diag_i[i + 1]; jA++)
         {
            if (S_temp_diag_j[jA] > -1)
            {
               S_diag_j[jS]    = S_temp_diag_j[jA];
               jS++;
            }
         }

         jS = S_offd_i[i];
         for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
         {
            if (S_temp_offd_j[jA] > -1)
            {
               S_offd_j[jS]    = S_temp_offd_j[jA];
               jS++;
            }
         }
      } /* for each variable */

   } /* omp parallel */

   hypre_CSRMatrixNumNonzeros(S_diag) = S_diag_i[num_variables];
   hypre_CSRMatrixNumNonzeros(S_offd) = S_offd_i[num_variables];
   hypre_CSRMatrixJ(S_diag) = S_diag_j;
   hypre_CSRMatrixJ(S_offd) = S_offd_j;

   hypre_CSRMatrixMemoryLocation(S_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(S_offd) = memory_location;

   hypre_ParCSRMatrixCommPkg(S) = NULL;

   *S_ptr = S;

   hypre_TFree(prefix_sum_workspace, HYPRE_MEMORY_HOST);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(S_temp_diag_j, HYPRE_MEMORY_HOST);
   hypre_TFree(S_temp_offd_j, HYPRE_MEMORY_HOST);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] += hypre_MPI_Wtime();
#endif

   return (ierr);
}

/* ----------------------------------------------------------------------- */
HYPRE_Int
hypre_BoomerAMGCreateS(hypre_ParCSRMatrix    *A,
                       HYPRE_Real             strength_threshold,
                       HYPRE_Real             max_row_sum,
                       HYPRE_Int              num_functions,
                       HYPRE_Int             *dof_func,
                       hypre_ParCSRMatrix   **S_ptr)
{
   hypre_GpuProfilingPushRange("CreateS");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGCreateSDevice(A, 0, strength_threshold, max_row_sum, num_functions, dof_func,
                                          S_ptr);
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGCreateSHost(A, strength_threshold, max_row_sum, num_functions, dof_func,
                                        S_ptr);
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}


/* ----------------------------------------------------------------------- */
/*
   Create Strength matrix from CF marker array data. Provides a more
   general form to build S for specific nodes of the 'global' matrix
   (for example, F points or A_FF part), given the entire matrix.
   These nodes have the SMRK tag.

   Could possibly be merged with BoomerAMGCreateS() to yield a
   more general function.
 */
HYPRE_Int
hypre_BoomerAMGCreateSFromCFMarker(hypre_ParCSRMatrix   *A,
                                   HYPRE_Real            strength_threshold,
                                   HYPRE_Real            max_row_sum,
                                   HYPRE_Int            *CF_marker,
                                   HYPRE_Int             num_functions,
                                   HYPRE_Int            *dof_func,
                                   HYPRE_Int             SMRK,
                                   hypre_ParCSRMatrix  **S_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] -= hypre_MPI_Wtime();
#endif

   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int          *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Real         *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Real         *A_offd_data = NULL;
   HYPRE_Int          *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int          *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_BigInt       *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int           num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt        global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int           num_nonzeros_diag;
   HYPRE_Int           num_nonzeros_offd = 0;
   HYPRE_Int           num_cols_offd = 0;

   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   HYPRE_Int          *S_diag_i;
   HYPRE_Int          *S_diag_j;
   /* HYPRE_Real         *S_diag_data; */
   hypre_CSRMatrix    *S_offd;
   HYPRE_Int          *S_offd_i = NULL;
   HYPRE_Int          *S_offd_j = NULL;
   /* HYPRE_Real         *S_offd_data; */
   HYPRE_Int          *dof_func_offd = NULL;

   HYPRE_Real          diag, row_scale, row_sum;
   HYPRE_Int           i, jj, jA, jS;
   HYPRE_Int           num_sends, start, j, index;
   HYPRE_Int          *int_buf_data;

   HYPRE_Int           ierr = 0;
   HYPRE_Int          *CF_marker_offd = NULL;

   HYPRE_Int          *prefix_sum_workspace;
   HYPRE_Int           my_id;

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   hypre_MPI_Comm_rank(comm, &my_id);
   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   A_offd_i = hypre_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
                                row_starts, row_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);

   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(HYPRE_Int,  num_variables + 1, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_diag, HYPRE_MEMORY_HOST);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(HYPRE_Int,  num_variables + 1, HYPRE_MEMORY_HOST);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int *S_temp_diag_j = hypre_CSRMatrixJ(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   S_diag_j = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_diag, HYPRE_MEMORY_HOST);

   HYPRE_Int *S_temp_offd_j = NULL;

   if (num_cols_offd)
   {
      A_offd_data = hypre_CSRMatrixData(A_offd);
      hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_offd, HYPRE_MEMORY_HOST);
      S_temp_offd_j = hypre_CSRMatrixJ(S_offd);
      HYPRE_BigInt *col_map_offd_S = hypre_TAlloc(HYPRE_BigInt,  num_cols_offd, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;
      if (num_functions > 1)
      {
         dof_func_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      }

      S_offd_j = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_offd, HYPRE_MEMORY_HOST);

      HYPRE_BigInt *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols_offd; i++)
      {
         col_map_offd_S[i] = col_map_offd_A[i];
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

      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   }

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   if (num_cols_offd) { CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST); }
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                           num_sends), HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         int_buf_data[index++]
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);

   /*HYPRE_Int prefix_sum_workspace[2*(hypre_NumThreads() + 1)];*/
   prefix_sum_workspace = hypre_TAlloc(HYPRE_Int,  2 * (hypre_NumThreads() + 1), HYPRE_MEMORY_HOST);

   /* give S same nonzero structure as A */

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,diag,row_scale,row_sum,jA,jS)
#endif
   {
      HYPRE_Int start, stop;
      hypre_GetSimpleThreadPartition(&start, &stop, num_variables);
      HYPRE_Int jS_diag = 0, jS_offd = 0;

      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == SMRK)
         {
            S_diag_i[i] = jS_diag;
            if (num_cols_offd)
            {
               S_offd_i[i] = jS_offd;
            }

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
                     jj = A_diag_j[jA];
                     if ((CF_marker[jj] == SMRK) && (dof_func[i] == dof_func[jj]))
                     {
                        row_scale = hypre_max(row_scale, A_diag_data[jA]);
                        row_sum += A_diag_data[jA];
                     }
                  }
                  for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                  {
                     jj = A_offd_j[jA];
                     if ((CF_marker_offd[jj] == SMRK) && (dof_func[i] == dof_func_offd[jj]))
                     {
                        row_scale = hypre_max(row_scale, A_offd_data[jA]);
                        row_sum += A_offd_data[jA];
                     }
                  }
               } /* diag < 0 */
               else
               {
                  for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
                  {
                     jj = A_diag_j[jA];
                     if ((CF_marker[jj] == SMRK) && (dof_func[i] == dof_func[jj]))
                     {
                        row_scale = hypre_min(row_scale, A_diag_data[jA]);
                        row_sum += A_diag_data[jA];
                     }
                  }
                  for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                  {
                     jj = A_offd_j[jA];
                     if ((CF_marker_offd[jj] == SMRK) && (dof_func[i] == dof_func_offd[A_offd_j[jA]]))
                     {
                        row_scale = hypre_min(row_scale, A_offd_data[jA]);
                        row_sum += A_offd_data[jA];
                     }
                  }
               } /* diag >= 0 */
            } /* num_functions > 1 */
            else
            {
               if (diag < 0)
               {
                  for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
                  {
                     jj = A_diag_j[jA];
                     if (CF_marker[jj] == SMRK)
                     {
                        row_scale = hypre_max(row_scale, A_diag_data[jA]);
                        row_sum += A_diag_data[jA];
                     }
                  }
                  for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                  {
                     jj = A_offd_j[jA];
                     if (CF_marker_offd[jj] == SMRK)
                     {
                        row_scale = hypre_max(row_scale, A_offd_data[jA]);
                        row_sum += A_offd_data[jA];
                     }
                  }
               } /* diag < 0 */
               else
               {
                  for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
                  {
                     jj = A_diag_j[jA];
                     if (CF_marker[jj] == SMRK)
                     {
                        row_scale = hypre_min(row_scale, A_diag_data[jA]);
                        row_sum += A_diag_data[jA];
                     }
                  }
                  for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                  {
                     jj = A_offd_j[jA];
                     if (CF_marker_offd[jj] == SMRK)
                     {
                        row_scale = hypre_min(row_scale, A_offd_data[jA]);
                        row_sum += A_offd_data[jA];
                     }
                  }
               } /* diag >= 0*/
            } /* num_functions <=1 */


            /* compute row entries of S */
            S_temp_diag_j[A_diag_i[i]] = -1;
            if ((hypre_abs(row_sum) > hypre_abs(diag)*max_row_sum) && (max_row_sum < 1.0))
            {
               /* make all dependencies weak */
               for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
               {
                  S_temp_diag_j[jA] = -1;
               }

               for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
               {
                  S_temp_offd_j[jA] = -1;
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
                        jj = A_diag_j[jA];
                        if (CF_marker[jj] == SMRK)
                        {
                           if ((A_diag_data[jA] <= strength_threshold * row_scale)
                               || (dof_func[i] != dof_func[jj]))
                           {
                              S_temp_diag_j[jA] = -1;
                           }
                           else
                           {
                              S_temp_diag_j[jA] = jj;
                              ++jS_diag;
                           }
                        }
                        else
                        {
                           S_temp_diag_j[jA] = -1;
                        }
                     }
                     for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                     {
                        jj = A_offd_j[jA];
                        if (CF_marker_offd[jj] == SMRK)
                        {
                           if ((A_offd_data[jA] <= strength_threshold * row_scale)
                               || (dof_func[i] != dof_func_offd[jj]))
                           {
                              S_temp_offd_j[jA] = -1;
                           }
                           else
                           {
                              S_temp_offd_j[jA] = jj;
                              ++jS_offd;
                           }
                        }
                        else
                        {
                           S_temp_offd_j[jA] = -1;
                        }
                     }
                  } /* end diag < 0 */
                  else
                  {
                     for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
                     {
                        jj = A_diag_j[jA];
                        if (CF_marker[jj] == SMRK)
                        {
                           if ((A_diag_data[jA] >= strength_threshold * row_scale)
                               || (dof_func[i] != dof_func[jj]))
                           {
                              S_temp_diag_j[jA] = -1;
                           }
                           else
                           {
                              S_temp_diag_j[jA] = jj;
                              ++jS_diag;
                           }
                        }
                        else
                        {
                           S_temp_diag_j[jA] = -1;
                        }
                     }
                     for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                     {
                        jj = A_offd_j[jA];
                        if (CF_marker_offd[jj] == SMRK)
                        {
                           if ((A_offd_data[jA] >= strength_threshold * row_scale)
                               || (dof_func[i] != dof_func_offd[jj]))
                           {
                              S_temp_offd_j[jA] = -1;
                           }
                           else
                           {
                              S_temp_offd_j[jA] = jj;
                              ++jS_offd;
                           }
                        }
                        else
                        {
                           S_temp_offd_j[jA] = -1;
                        }
                     }
                  } /* diag >= 0 */
               } /* num_functions > 1 */
               else
               {
                  if (diag < 0)
                  {
                     for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
                     {
                        jj = A_diag_j[jA];
                        if (CF_marker[jj] == SMRK)
                        {
                           if (A_diag_data[jA] <= strength_threshold * row_scale)
                           {
                              S_temp_diag_j[jA] = -1;
                           }
                           else
                           {
                              S_temp_diag_j[jA] = jj;
                              ++jS_diag;
                           }
                        }
                        else
                        {
                           S_temp_diag_j[jA] = -1;
                        }
                     }
                     for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                     {
                        jj = A_offd_j[jA];
                        if (CF_marker_offd[jj] == SMRK)
                        {
                           if (A_offd_data[jA] <= strength_threshold * row_scale)
                           {
                              S_temp_offd_j[jA] = -1;
                           }
                           else
                           {
                              S_temp_offd_j[jA] = jj;
                              ++jS_offd;
                           }
                        }
                        else
                        {
                           S_temp_offd_j[jA] = -1;
                        }
                     }
                  } /* diag < 0 */
                  else
                  {
                     for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
                     {
                        jj = A_diag_j[jA];
                        if (CF_marker[jj] == SMRK)
                        {
                           if (A_diag_data[jA] >= strength_threshold * row_scale)
                           {
                              S_temp_diag_j[jA] = -1;
                           }
                           else
                           {
                              S_temp_diag_j[jA] = jj;
                              ++jS_diag;
                           }
                        }
                        else
                        {
                           S_temp_diag_j[jA] = -1;
                        }
                     }
                     for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
                     {
                        jj = A_offd_j[jA];
                        if (CF_marker_offd[jj] == SMRK)
                        {
                           if (A_offd_data[jA] >= strength_threshold * row_scale)
                           {
                              S_temp_offd_j[jA] = -1;
                           }
                           else
                           {
                              S_temp_offd_j[jA] = jj;
                              ++jS_offd;
                           }
                        }
                        else
                        {
                           S_temp_offd_j[jA] = -1;
                        }
                     }
                  } /* diag >= 0 */
               } /* num_functions <=1 */
            } /* !((row_sum > max_row_sum) && (max_row_sum < 1.0)) */
         } /* CF_marker == SMRK */
         else
         {
            S_diag_i[i] = jS_diag;
            if (num_cols_offd)
            {
               S_offd_i[i] = jS_offd;
            }

            for (jA = A_diag_i[i]; jA < A_diag_i[i + 1]; jA++)
            {
               S_temp_diag_j[jA] = -1;
            }

            for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
            {
               S_temp_offd_j[jA] = -1;
            }
         } /* CF_marker != SMRK */
      } /* for each variable */

      hypre_prefix_sum_pair(&jS_diag, S_diag_i + num_variables, &jS_offd, S_offd_i + num_variables,
                            prefix_sum_workspace);

      /*--------------------------------------------------------------
       * "Compress" the strength matrix.
       *
       * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
       *
       * NOTE: This "compression" section of code may be removed, and
       * coarsening will still be done correctly.  However, the routine
       * that builds interpolation would have to be modified first.
       *----------------------------------------------------------------*/

      for (i = start; i < stop; i++)
      {
         S_diag_i[i] += jS_diag;
         S_offd_i[i] += jS_offd;

         jS = S_diag_i[i];
         for (jA = A_diag_i[i]; jA < A_diag_i[i + 1]; jA++)
         {
            if (S_temp_diag_j[jA] > -1)
            {
               S_diag_j[jS]    = S_temp_diag_j[jA];
               jS++;
            }
         }

         jS = S_offd_i[i];
         for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
         {
            if (S_temp_offd_j[jA] > -1)
            {
               S_offd_j[jS]    = S_temp_offd_j[jA];
               jS++;
            }
         }
      } /* for each variable */

   } /* omp parallel */

   hypre_CSRMatrixNumNonzeros(S_diag) = S_diag_i[num_variables];
   hypre_CSRMatrixNumNonzeros(S_offd) = S_offd_i[num_variables];
   hypre_CSRMatrixJ(S_diag) = S_diag_j;
   hypre_CSRMatrixJ(S_offd) = S_offd_j;

   hypre_CSRMatrixMemoryLocation(S_diag) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixMemoryLocation(S_offd) = HYPRE_MEMORY_HOST;

   hypre_ParCSRMatrixCommPkg(S) = NULL;

   *S_ptr        = S;

   hypre_TFree(prefix_sum_workspace, HYPRE_MEMORY_HOST);
   hypre_TFree(S_temp_diag_j, HYPRE_MEMORY_HOST);
   hypre_TFree(S_temp_offd_j, HYPRE_MEMORY_HOST);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] += hypre_MPI_Wtime();
#endif

   return (ierr);
}

/*==========================================================================*/
/*==========================================================================*/
/**
  Generates strength matrix

  Notes:
  \begin{itemize}
  \item The underlying matrix storage scheme is a hypre_ParCSR matrix.
  \item The routine returns the following:
  \begin{itemize}
  \item S - a ParCSR matrix representing the "strength matrix".  This is
  used in the coarsening and interpolation routines.
  \end{itemize}
  \item The graph of the "strength matrix" for A is a subgraph of the
  graph of A, but requires nonsymmetric storage even if A is
  symmetric.  This is because of the directional nature of the
  "strengh of dependence" notion (see below).  Since we are using
  nonsymmetric storage for A right now, this is not a problem.  If we
  ever add the ability to store A symmetrically, then we could store
  the strength graph as floats instead of doubles to save space.
  \item This routine currently "compresses" the strength matrix.  We
  should consider the possibility of defining this matrix to have the
  same "nonzero structure" as A.  To do this, we could use the same
  A\_i and A\_j arrays, and would need only define the S\_data array.
  There are several pros and cons to discuss.
  \end{itemize}

  Terminology:
  \begin{itemize}
  \item Ruge's terminology: A point is "strongly connected to" $j$, or
  "strongly depends on" $j$, if $|a_ij| >= \theta max_{l != j} |a_il|}$.
  \item Here, we retain some of this terminology, but with a more
  generalized notion of "strength".  We also retain the "natural"
  graph notation for representing the directed graph of a matrix.
  That is, the nonzero entry $a_ij$ is represented as: i --> j.  In
  the strength matrix, S, the entry $s_ij$ is also graphically denoted
  as above, and means both of the following:
  \begin{itemize}
  \item $i$ "depends on" $j$ with "strength" $s_ij$
  \item $j$ "influences" $i$ with "strength" $s_ij$
  \end{itemize}
  \end{itemize}

  {\bf Input files:}
  _hypre_parcsr_ls.h

  @return Error code.

  @param A [IN]
  coefficient matrix
  @param strength_threshold [IN]
  threshold parameter used to define strength
  @param max_row_sum [IN]
  parameter used to modify definition of strength for diagonal dominant matrices
  @param S_ptr [OUT]
  strength matrix

  @see */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreateSabsHost(hypre_ParCSRMatrix    *A,
                              HYPRE_Real             strength_threshold,
                              HYPRE_Real             max_row_sum,
                              HYPRE_Int              num_functions,
                              HYPRE_Int             *dof_func,
                              hypre_ParCSRMatrix   **S_ptr)
{
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int          *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Real         *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Real         *A_offd_data = NULL;
   HYPRE_Int          *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int          *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_BigInt       *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int           num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt        global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int           num_nonzeros_diag;
   HYPRE_Int           num_nonzeros_offd = 0;
   HYPRE_Int           num_cols_offd = 0;

   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   HYPRE_Int          *S_diag_i;
   HYPRE_Int          *S_diag_j;
   /* HYPRE_Real         *S_diag_data; */
   hypre_CSRMatrix    *S_offd;
   HYPRE_Int          *S_offd_i = NULL;
   HYPRE_Int          *S_offd_j = NULL;
   /* HYPRE_Real         *S_offd_data; */

   HYPRE_Real          diag, row_scale, row_sum;
   HYPRE_Int           i, jA, jS;

   HYPRE_Int           ierr = 0;

   HYPRE_Int          *dof_func_offd;
   HYPRE_Int           num_sends;
   HYPRE_Int          *int_buf_data;
   HYPRE_Int           index, start, j;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * Absolute "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     abs(aij) > hypre_max (k != i) abs(aik)
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   A_offd_i = hypre_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
                                row_starts, row_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);

   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_location);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_diag, memory_location);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_location);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   hypre_CSRMatrixMemoryLocation(S_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(S_offd) = memory_location;

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
      A_offd_data = hypre_CSRMatrixData(A_offd);
      hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_offd, memory_location);
      S_offd_j = hypre_CSRMatrixJ(S_offd);
      hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);
      if (num_functions > 1)
      {
         dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
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
      int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                   HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   }

   /* give S same nonzero structure as A */
   hypre_ParCSRMatrixCopy(A, S, 0);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,diag,row_scale,row_sum,jA) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_variables; i++)
   {
      diag = A_diag_data[A_diag_i[i]];

      /* compute scaling factor and row sum */
      row_scale = 0.0;
      row_sum = hypre_abs(diag);
      if (num_functions > 1)
      {
         for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
         {
            if (dof_func[i] == dof_func[A_diag_j[jA]])
            {
               row_scale = hypre_max(row_scale, hypre_abs(A_diag_data[jA]));
               row_sum += hypre_abs(A_diag_data[jA]);
            }
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
         {
            if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
            {
               row_scale = hypre_max(row_scale, hypre_abs(A_offd_data[jA]));
               row_sum += hypre_abs(A_offd_data[jA]);
            }
         }
      }
      else
      {
         for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
         {
            row_scale = hypre_max(row_scale, hypre_abs(A_diag_data[jA]));
            row_sum += hypre_abs(A_diag_data[jA]);
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
         {
            row_scale = hypre_max(row_scale, hypre_abs(A_offd_data[jA]));
            row_sum += hypre_abs(A_offd_data[jA]);
         }
      }

      /* compute row entries of S */
      S_diag_j[A_diag_i[i]] = -1; /* reject diag entry */
      if ( hypre_abs(row_sum) < hypre_abs(diag) * (2.0 - max_row_sum) && max_row_sum < 1.0 )
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
            for (jA = A_diag_i[i] + 1; jA < A_diag_i[i + 1]; jA++)
            {
               if (hypre_abs(A_diag_data[jA]) <= strength_threshold * row_scale
                   || dof_func[i] != dof_func[A_diag_j[jA]])
               {
                  S_diag_j[jA] = -1;
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
            {
               if (hypre_abs(A_offd_data[jA]) <= strength_threshold * row_scale
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
               if (hypre_abs(A_diag_data[jA]) <= strength_threshold * row_scale)
               {
                  S_diag_j[jA] = -1;
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i + 1]; jA++)
            {
               if (hypre_abs(A_offd_data[jA]) <= strength_threshold * row_scale)
               {
                  S_offd_j[jA] = -1;
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
    * NOTE: This "compression" section of code may be removed, and
    * coarsening will still be done correctly.  However, the routine
    * that builds interpolation would have to be modified first.
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
            S_diag_j[jS] = S_diag_j[jA];
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
            jS++;
         }
      }
   }
   S_offd_i[num_variables] = jS;
   hypre_CSRMatrixNumNonzeros(S_offd) = jS;
   hypre_ParCSRMatrixCommPkg(S) = NULL;

   *S_ptr = S;

   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);

   return (ierr);
}

HYPRE_Int
hypre_BoomerAMGCreateSabs(hypre_ParCSRMatrix    *A,
                          HYPRE_Real             strength_threshold,
                          HYPRE_Real             max_row_sum,
                          HYPRE_Int              num_functions,
                          HYPRE_Int             *dof_func,
                          hypre_ParCSRMatrix   **S_ptr)
{
   hypre_GpuProfilingPushRange("CreateSabs");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGCreateSDevice(A, 1, strength_threshold, max_row_sum, num_functions, dof_func,
                                          S_ptr);
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGCreateSabsHost(A, strength_threshold, max_row_sum, num_functions, dof_func,
                                           S_ptr);
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}

/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreateSCommPkg(hypre_ParCSRMatrix *A,
                              hypre_ParCSRMatrix *S,
                              HYPRE_Int         **col_offd_S_to_A_ptr)
{
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   hypre_MPI_Status        *status;
   hypre_MPI_Request       *requests;
   hypre_ParCSRCommPkg     *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommPkg     *comm_pkg_S = NULL;
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix         *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_BigInt            *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);

   hypre_CSRMatrix    *S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix    *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int          *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int          *S_offd_j = hypre_CSRMatrixJ(S_offd);
   HYPRE_BigInt       *col_map_offd_S = hypre_ParCSRMatrixColMapOffd(S);

   HYPRE_Int          *recv_procs_A = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   HYPRE_Int          *recv_vec_starts_A = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   HYPRE_Int          *send_procs_A = hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   HYPRE_Int          *send_map_starts_A = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);
   HYPRE_Int          *recv_procs_S;
   HYPRE_Int          *recv_vec_starts_S;
   HYPRE_Int          *send_procs_S;
   HYPRE_Int          *send_map_starts_S;
   HYPRE_Int          *send_map_elmts_S = NULL;
   HYPRE_BigInt       *big_send_map_elmts_S = NULL;
   HYPRE_Int          *col_offd_S_to_A;

   HYPRE_Int          *S_marker;
   HYPRE_Int          *send_change;
   HYPRE_Int          *recv_change;

   HYPRE_Int           num_variables   = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int           num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int           num_cols_offd_S;
   HYPRE_Int           i, j, jcol;
   HYPRE_Int           proc, cnt, proc_cnt, total_nz;
   HYPRE_BigInt        first_row;

   HYPRE_Int           ierr = 0;

   HYPRE_Int           num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   HYPRE_Int           num_recvs_A = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   HYPRE_Int           num_sends_S;
   HYPRE_Int           num_recvs_S;
   HYPRE_Int           num_nonzeros;

   num_nonzeros = S_offd_i[num_variables];

   S_marker = NULL;
   if (num_cols_offd_A)
   {
      S_marker = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_cols_offd_A; i++)
   {
      S_marker[i] = -1;
   }

   for (i = 0; i < num_nonzeros; i++)
   {
      jcol = S_offd_j[i];
      S_marker[jcol] = 0;
   }

   proc = 0;
   proc_cnt = 0;
   cnt = 0;
   num_recvs_S = 0;
   for (i = 0; i < num_recvs_A; i++)
   {
      for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i + 1]; j++)
      {
         if (!S_marker[j])
         {
            S_marker[j] = cnt;
            cnt++;
            proc = 1;
         }
      }
      if (proc) {num_recvs_S++; proc = 0;}
   }


   num_cols_offd_S = cnt;
   recv_change = NULL;
   recv_procs_S = NULL;
   send_change = NULL;
   if (col_map_offd_S) { hypre_TFree(col_map_offd_S, HYPRE_MEMORY_HOST); }
   col_map_offd_S = NULL;
   col_offd_S_to_A = NULL;
   if (num_recvs_A) { recv_change = hypre_CTAlloc(HYPRE_Int,  num_recvs_A, HYPRE_MEMORY_HOST); }
   if (num_sends_A) { send_change = hypre_CTAlloc(HYPRE_Int,  num_sends_A, HYPRE_MEMORY_HOST); }
   if (num_recvs_S) { recv_procs_S = hypre_CTAlloc(HYPRE_Int,  num_recvs_S, HYPRE_MEMORY_HOST); }
   recv_vec_starts_S = hypre_CTAlloc(HYPRE_Int,  num_recvs_S + 1, HYPRE_MEMORY_HOST);
   if (num_cols_offd_S)
   {
      col_map_offd_S = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_S, HYPRE_MEMORY_HOST);
      col_offd_S_to_A = hypre_CTAlloc(HYPRE_Int, num_cols_offd_S, HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_S < num_cols_offd_A)
   {
      for (i = 0; i < num_nonzeros; i++)
      {
         jcol = S_offd_j[i];
         S_offd_j[i] = S_marker[jcol];
      }

      proc = 0;
      proc_cnt = 0;
      cnt = 0;
      recv_vec_starts_S[0] = 0;
      for (i = 0; i < num_recvs_A; i++)
      {
         for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i + 1]; j++)
         {
            if (S_marker[j] != -1)
            {
               col_map_offd_S[cnt] = col_map_offd_A[j];
               col_offd_S_to_A[cnt++] = j;
               proc = 1;
            }
         }
         recv_change[i] = j - cnt - recv_vec_starts_A[i] + recv_vec_starts_S[proc_cnt];
         if (proc)
         {
            recv_procs_S[proc_cnt++] = recv_procs_A[i];
            recv_vec_starts_S[proc_cnt] = cnt;
            proc = 0;
         }
      }
   }
   else
   {
      for (i = 0; i < num_recvs_A; i++)
      {
         for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i + 1]; j++)
         {
            col_map_offd_S[j] = col_map_offd_A[j];
            col_offd_S_to_A[j] = j;
         }
         recv_procs_S[i] = recv_procs_A[i];
         recv_vec_starts_S[i] = recv_vec_starts_A[i];
      }
      recv_vec_starts_S[num_recvs_A] = recv_vec_starts_A[num_recvs_A];
   }

   requests = hypre_CTAlloc(hypre_MPI_Request, num_sends_A + num_recvs_A, HYPRE_MEMORY_HOST);
   j = 0;
   for (i = 0; i < num_sends_A; i++)
   {
      hypre_MPI_Irecv(&send_change[i], 1, HYPRE_MPI_INT, send_procs_A[i], 0, comm, &requests[j++]);
   }

   for (i = 0; i < num_recvs_A; i++)
   {
      hypre_MPI_Isend(&recv_change[i], 1, HYPRE_MPI_INT, recv_procs_A[i], 0, comm, &requests[j++]);
   }

   status = hypre_CTAlloc(hypre_MPI_Status, j, HYPRE_MEMORY_HOST);
   hypre_MPI_Waitall(j, requests, status);
   hypre_TFree(status, HYPRE_MEMORY_HOST);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);

   num_sends_S = 0;
   total_nz = send_map_starts_A[num_sends_A];
   for (i = 0; i < num_sends_A; i++)
   {
      if (send_change[i])
      {
         if ((send_map_starts_A[i + 1] - send_map_starts_A[i]) > send_change[i])
         {
            num_sends_S++;
         }
      }
      else
      {
         num_sends_S++;
      }
      total_nz -= send_change[i];
   }

   send_procs_S = NULL;
   if (num_sends_S)
   {
      send_procs_S = hypre_CTAlloc(HYPRE_Int, num_sends_S, HYPRE_MEMORY_HOST);
   }
   send_map_starts_S = hypre_CTAlloc(HYPRE_Int, num_sends_S + 1, HYPRE_MEMORY_HOST);
   send_map_elmts_S = NULL;
   if (total_nz)
   {
      send_map_elmts_S = hypre_CTAlloc(HYPRE_Int, total_nz, HYPRE_MEMORY_HOST);
      big_send_map_elmts_S = hypre_CTAlloc(HYPRE_BigInt, total_nz, HYPRE_MEMORY_HOST);
   }

   proc = 0;
   proc_cnt = 0;
   for (i = 0; i < num_sends_A; i++)
   {
      cnt = send_map_starts_A[i + 1] - send_map_starts_A[i] - send_change[i];
      if (cnt)
      {
         send_procs_S[proc_cnt++] = send_procs_A[i];
         send_map_starts_S[proc_cnt] = send_map_starts_S[proc_cnt - 1] + cnt;
      }
   }

   /* Create communication package for S */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_S, recv_procs_S, recv_vec_starts_S,
                                    num_sends_S, send_procs_S, send_map_starts_S,
                                    send_map_elmts_S,
                                    &comm_pkg_S);

   comm_handle = hypre_ParCSRCommHandleCreate(22, comm_pkg_S, col_map_offd_S,
                                              big_send_map_elmts_S);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   first_row = hypre_ParCSRMatrixFirstRowIndex(A);
   if (first_row)
   {
      for (i = 0; i < send_map_starts_S[num_sends_S]; i++)
      {
         send_map_elmts_S[i] = (HYPRE_Int)(big_send_map_elmts_S[i] - first_row);
      }
   }

   hypre_ParCSRMatrixCommPkg(S) = comm_pkg_S;
   hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;
   hypre_CSRMatrixNumCols(S_offd) = num_cols_offd_S;

   hypre_TFree(S_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(send_change, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_change, HYPRE_MEMORY_HOST);

   *col_offd_S_to_A_ptr = col_offd_S_to_A;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCreate2ndS : creates strength matrix on coarse points
 * for second coarsening pass in aggressive coarsening (S*S+2S)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreate2ndSHost( hypre_ParCSRMatrix  *S,
                               HYPRE_Int           *CF_marker,
                               HYPRE_Int            num_paths,
                               HYPRE_BigInt        *coarse_row_starts,
                               hypre_ParCSRMatrix **C_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATE_2NDS] -= hypre_MPI_Wtime();
#endif

   MPI_Comm             comm = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommPkg *tmp_comm_pkg = NULL;
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);

   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);

   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   HYPRE_Int    num_cols_diag_S = hypre_CSRMatrixNumCols(S_diag);
   HYPRE_Int    num_cols_offd_S = hypre_CSRMatrixNumCols(S_offd);

   hypre_ParCSRMatrix *S2;
   HYPRE_BigInt       *col_map_offd_C = NULL;

   hypre_CSRMatrix *C_diag;

   /*HYPRE_Int          *C_diag_data = NULL;*/
   HYPRE_Int             *C_diag_i;
   HYPRE_Int             *C_diag_j = NULL;

   hypre_CSRMatrix *C_offd;

   /*HYPRE_Int          *C_offd_data=NULL;*/
   HYPRE_Int             *C_offd_i;
   HYPRE_Int             *C_offd_j = NULL;

   HYPRE_Int              num_cols_offd_C = 0;

   HYPRE_Int             *S_ext_diag_i = NULL;
   HYPRE_Int             *S_ext_diag_j = NULL;
   HYPRE_Int              S_ext_diag_size = 0;

   HYPRE_Int             *S_ext_offd_i = NULL;
   HYPRE_Int             *S_ext_offd_j = NULL;
   HYPRE_Int              S_ext_offd_size = 0;

   HYPRE_Int             *CF_marker_offd = NULL;

   HYPRE_Int             *S_marker = NULL;
   HYPRE_Int             *S_marker_offd = NULL;
   //HYPRE_Int           *temp = NULL;

   HYPRE_Int             *fine_to_coarse = NULL;
   HYPRE_BigInt          *fine_to_coarse_offd = NULL;
   HYPRE_Int             *map_S_to_C = NULL;

   HYPRE_Int             num_sends = 0;
   HYPRE_Int             num_recvs = 0;
   HYPRE_Int            *send_map_starts;
   HYPRE_Int            *tmp_send_map_starts = NULL;
   HYPRE_Int            *send_map_elmts;
   HYPRE_Int            *recv_vec_starts;
   HYPRE_Int            *tmp_recv_vec_starts = NULL;
   HYPRE_Int            *int_buf_data = NULL;
   HYPRE_BigInt         *big_int_buf_data = NULL;
   HYPRE_BigInt         *temp = NULL;

   HYPRE_Int              i, j, k;
   HYPRE_Int              i1, i2, i3;
   HYPRE_BigInt           big_i1;
   HYPRE_Int              jj1, jj2, jrow, j_cnt;

   /*HYPRE_Int            cnt, cnt_offd, cnt_diag;*/
   HYPRE_Int              num_procs, my_id;
   HYPRE_Int              index;
   /*HYPRE_Int            value;*/
   HYPRE_Int              num_coarse;
   HYPRE_Int              num_nonzeros;
   HYPRE_BigInt           global_num_coarse;
   HYPRE_BigInt           my_first_cpt, my_last_cpt;

   HYPRE_Int *S_int_i = NULL;
   HYPRE_BigInt *S_int_j = NULL;
   HYPRE_Int *S_ext_i = NULL;
   HYPRE_BigInt *S_ext_j = NULL;

   /*HYPRE_Int prefix_sum_workspace[2*(hypre_NumThreads() + 1)];*/
   HYPRE_Int *prefix_sum_workspace;
   HYPRE_Int *num_coarse_prefix_sum;
   prefix_sum_workspace = hypre_TAlloc(HYPRE_Int,  2 * (hypre_NumThreads() + 1), HYPRE_MEMORY_HOST);
   num_coarse_prefix_sum = hypre_TAlloc(HYPRE_Int,  hypre_NumThreads() + 1, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Extract S_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = coarse_row_starts[0];
   my_last_cpt = coarse_row_starts[1] - 1;
   if (my_id == (num_procs - 1)) { global_num_coarse = coarse_row_starts[1]; }
   hypre_MPI_Bcast(&global_num_coarse, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (num_cols_offd_S)
   {
      CF_marker_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd_S, HYPRE_MEMORY_HOST);
      fine_to_coarse_offd = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_S, HYPRE_MEMORY_HOST);
   }

   HYPRE_Int *coarse_to_fine = NULL;
   if (num_cols_diag_S)
   {
      fine_to_coarse = hypre_TAlloc(HYPRE_Int,  num_cols_diag_S, HYPRE_MEMORY_HOST);
      coarse_to_fine = hypre_TAlloc(HYPRE_Int,  num_cols_diag_S, HYPRE_MEMORY_HOST);
   }

   /*HYPRE_Int num_coarse_prefix_sum[hypre_NumThreads() + 1];*/

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i)
#endif
   {
      HYPRE_Int num_coarse_private = 0;

      HYPRE_Int i_begin, i_end;
      hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_diag_S);

      for (i = i_begin; i < i_end; i++)
      {
         if (CF_marker[i] > 0) { num_coarse_private++; }
      }

      hypre_prefix_sum(&num_coarse_private, &num_coarse, num_coarse_prefix_sum);

      for (i = i_begin; i < i_end; i++)
      {
         if (CF_marker[i] > 0)
         {
            fine_to_coarse[i] = num_coarse_private;
            coarse_to_fine[num_coarse_private] = i;
            num_coarse_private++;
         }
         else
         {
            fine_to_coarse[i] = -1;
         }
      }
   } /* omp parallel */

   if (num_procs > 1)
   {
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(S);

         comm_pkg = hypre_ParCSRMatrixCommPkg(S);
      }
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

      HYPRE_Int begin = send_map_starts[0];
      HYPRE_Int end = send_map_starts[num_sends];
      big_int_buf_data = hypre_TAlloc(HYPRE_BigInt,  end, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (index = begin; index < end; index++)
      {
         big_int_buf_data[index - begin] = (HYPRE_BigInt)fine_to_coarse[send_map_elmts[index]] +
                                           my_first_cpt;
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_int_buf_data,
                                                  fine_to_coarse_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);

      int_buf_data = hypre_TAlloc(HYPRE_Int,  end, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (index = begin; index < end; index++)
      {
         int_buf_data[index - begin] = CF_marker[send_map_elmts[index]];
      }

      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 CF_marker_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
      hypre_TFree(big_int_buf_data, HYPRE_MEMORY_HOST);

      S_int_i = hypre_TAlloc(HYPRE_Int,  end + 1, HYPRE_MEMORY_HOST);
      S_ext_i = hypre_CTAlloc(HYPRE_Int,  recv_vec_starts[num_recvs] + 1, HYPRE_MEMORY_HOST);

      /*--------------------------------------------------------------------------
       * generate S_int_i through adding number of coarse row-elements of offd and diag
       * for corresponding rows. S_int_i[j+1] contains the number of coarse elements of
       * a row j (which is determined through send_map_elmts)
       *--------------------------------------------------------------------------*/
      S_int_i[0] = 0;
      num_nonzeros = 0;
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j,k) reduction(+:num_nonzeros) HYPRE_SMP_SCHEDULE
#endif
      for (j = begin; j < end; j++)
      {
         HYPRE_Int jrow = send_map_elmts[j];
         HYPRE_Int index = 0;
         for (k = S_diag_i[jrow]; k < S_diag_i[jrow + 1]; k++)
         {
            if (CF_marker[S_diag_j[k]] > 0) { index++; }
         }
         for (k = S_offd_i[jrow]; k < S_offd_i[jrow + 1]; k++)
         {
            if (CF_marker_offd[S_offd_j[k]] > 0) { index++; }
         }
         S_int_i[j - begin + 1] = index;
         num_nonzeros += S_int_i[j - begin + 1];
      }

      /*--------------------------------------------------------------------------
       * initialize communication
       *--------------------------------------------------------------------------*/
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, &S_int_i[1], &S_ext_i[1]);
      }

      if (num_nonzeros) { S_int_j = hypre_TAlloc(HYPRE_BigInt,  num_nonzeros, HYPRE_MEMORY_HOST); }

      tmp_send_map_starts = hypre_CTAlloc(HYPRE_Int,  num_sends + 1, HYPRE_MEMORY_HOST);
      tmp_recv_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_recvs + 1, HYPRE_MEMORY_HOST);

      tmp_send_map_starts[0] = 0;
      j_cnt = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            jrow = send_map_elmts[j];
            for (k = S_diag_i[jrow]; k < S_diag_i[jrow + 1]; k++)
            {
               if (CF_marker[S_diag_j[k]] > 0)
               {
                  S_int_j[j_cnt++] = (HYPRE_BigInt)fine_to_coarse[S_diag_j[k]] + my_first_cpt;
               }
            }
            for (k = S_offd_i[jrow]; k < S_offd_i[jrow + 1]; k++)
            {
               if (CF_marker_offd[S_offd_j[k]] > 0)
               {
                  S_int_j[j_cnt++] = fine_to_coarse_offd[S_offd_j[k]];
               }
            }
         }
         tmp_send_map_starts[i + 1] = j_cnt;
      }

      /* Create temporary communication package */
      hypre_ParCSRCommPkgCreateAndFill(comm,
                                       num_recvs,
                                       hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                       tmp_recv_vec_starts,
                                       num_sends,
                                       hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                       tmp_send_map_starts,
                                       NULL,
                                       &tmp_comm_pkg);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      /*--------------------------------------------------------------------------
       * after communication exchange S_ext_i[j+1] contains the number of coarse elements
       * of a row j !
       * evaluate S_ext_i and compute num_nonzeros for S_ext
       *--------------------------------------------------------------------------*/

      for (i = 0; i < recv_vec_starts[num_recvs]; i++)
      {
         S_ext_i[i + 1] += S_ext_i[i];
      }

      num_nonzeros = S_ext_i[recv_vec_starts[num_recvs]];

      if (num_nonzeros)
      {
         S_ext_j = hypre_TAlloc(HYPRE_BigInt,  num_nonzeros, HYPRE_MEMORY_HOST);
      }

      for (i = 0; i < num_recvs; i++)
      {
         tmp_recv_vec_starts[i + 1] = S_ext_i[recv_vec_starts[i + 1]];
      }

      comm_handle = hypre_ParCSRCommHandleCreate(21, tmp_comm_pkg, S_int_j, S_ext_j);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      hypre_TFree(tmp_send_map_starts, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_recv_vec_starts, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_comm_pkg, HYPRE_MEMORY_HOST);

      hypre_TFree(S_int_i, HYPRE_MEMORY_HOST);
      hypre_TFree(S_int_j, HYPRE_MEMORY_HOST);

#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
      HYPRE_BigInt          *S_big_offd_j = NULL;
      S_ext_diag_i = hypre_TAlloc(HYPRE_Int,  num_cols_offd_S + 1, HYPRE_MEMORY_HOST);
      S_ext_diag_i[0] = 0;
      S_ext_offd_i = hypre_TAlloc(HYPRE_Int,  num_cols_offd_S + 1, HYPRE_MEMORY_HOST);
      S_ext_offd_i[0] = 0;

      hypre_UnorderedBigIntSet found_set;
      hypre_UnorderedBigIntSetCreate(&found_set, S_ext_i[num_cols_offd_S] + num_cols_offd_S,
                                     16 * hypre_NumThreads());

      #pragma omp parallel private(i,j, big_i1)
      {
         HYPRE_Int S_ext_offd_size_private = 0;
         HYPRE_Int S_ext_diag_size_private = 0;

         HYPRE_Int i_begin, i_end;
         hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_S);

         for (i = i_begin; i < i_end; i++)
         {
            if (CF_marker_offd[i] > 0)
            {
               hypre_UnorderedBigIntSetPut(&found_set, fine_to_coarse_offd[i]);
            }
            for (j = S_ext_i[i]; j < S_ext_i[i + 1]; j++)
            {
               big_i1 = S_ext_j[j];
               if (big_i1 < my_first_cpt || big_i1 > my_last_cpt)
               {
                  S_ext_offd_size_private++;
                  hypre_UnorderedBigIntSetPut(&found_set, big_i1);
               }
               else
               {
                  S_ext_diag_size_private++;
               }
            }
         }

         hypre_prefix_sum_pair(
            &S_ext_diag_size_private, &S_ext_diag_size,
            &S_ext_offd_size_private, &S_ext_offd_size,
            prefix_sum_workspace);

         #pragma omp master
         {
            if (S_ext_diag_size)
            {
               S_ext_diag_j = hypre_TAlloc(HYPRE_Int,  S_ext_diag_size, HYPRE_MEMORY_HOST);
            }
            if (S_ext_offd_size)
            {
               S_ext_offd_j = hypre_TAlloc(HYPRE_Int,  S_ext_offd_size, HYPRE_MEMORY_HOST);
               S_big_offd_j = hypre_TAlloc(HYPRE_BigInt,  S_ext_offd_size, HYPRE_MEMORY_HOST);
            }
         }

         #pragma omp barrier

         for (i = i_begin; i < i_end; i++)
         {
            for (j = S_ext_i[i]; j < S_ext_i[i + 1]; j++)
            {
               big_i1 = S_ext_j[j];
               if (big_i1 < my_first_cpt || big_i1 > my_last_cpt)
               {
                  S_big_offd_j[S_ext_offd_size_private++] = big_i1;
               }
               //S_ext_offd_j[S_ext_offd_size_private++] = big_i1;
               else
               {
                  S_ext_diag_j[S_ext_diag_size_private++] = (HYPRE_Int)(big_i1 - my_first_cpt);
               }
            }
            S_ext_diag_i[i + 1] = S_ext_diag_size_private;
            S_ext_offd_i[i + 1] = S_ext_offd_size_private;
         }
      } // omp parallel

      temp = hypre_UnorderedBigIntSetCopyToArray(&found_set, &num_cols_offd_C);

      hypre_UnorderedBigIntSetDestroy(&found_set);
      hypre_TFree(S_ext_i, HYPRE_MEMORY_HOST);

      hypre_UnorderedBigIntMap col_map_offd_C_inverse;
      hypre_big_sort_and_create_inverse_map(temp, num_cols_offd_C, &col_map_offd_C,
                                            &col_map_offd_C_inverse);

      #pragma omp parallel for HYPRE_SMP_SCHEDULE
      for (i = 0 ; i < S_ext_offd_size; i++)
      {
         S_ext_offd_j[i] = hypre_UnorderedBigIntMapGet(&col_map_offd_C_inverse, S_big_offd_j[i]);
      }
      //S_ext_offd_j[i] = hypre_UnorderedIntMapGet(&col_map_offd_C_inverse, S_ext_offd_j[i]);

      hypre_TFree(S_ext_j, HYPRE_MEMORY_HOST);
      hypre_TFree(S_big_offd_j, HYPRE_MEMORY_HOST);
      if (num_cols_offd_C) { hypre_UnorderedBigIntMapDestroy(&col_map_offd_C_inverse); }
#else /* !HYPRE_CONCURRENT_HOPSCOTCH */
      HYPRE_Int cnt_offd, cnt_diag, cnt, value;
      S_ext_diag_size = 0;
      S_ext_offd_size = 0;

      for (i = 0; i < num_cols_offd_S; i++)
      {
         for (j = S_ext_i[i]; j < S_ext_i[i + 1]; j++)
         {
            if (S_ext_j[j] < my_first_cpt || S_ext_j[j] > my_last_cpt)
            {
               S_ext_offd_size++;
            }
            else
            {
               S_ext_diag_size++;
            }
         }
      }
      S_ext_diag_i = hypre_CTAlloc(HYPRE_Int,  num_cols_offd_S + 1, HYPRE_MEMORY_HOST);
      S_ext_offd_i = hypre_CTAlloc(HYPRE_Int,  num_cols_offd_S + 1, HYPRE_MEMORY_HOST);

      if (S_ext_diag_size)
      {
         S_ext_diag_j = hypre_CTAlloc(HYPRE_Int,  S_ext_diag_size, HYPRE_MEMORY_HOST);
      }
      if (S_ext_offd_size)
      {
         S_ext_offd_j = hypre_CTAlloc(HYPRE_Int,  S_ext_offd_size, HYPRE_MEMORY_HOST);
      }

      cnt_offd = 0;
      cnt_diag = 0;
      cnt = 0;
      HYPRE_Int num_coarse_offd = 0;
      for (i = 0; i < num_cols_offd_S; i++)
      {
         if (CF_marker_offd[i] > 0) { num_coarse_offd++; }

         for (j = S_ext_i[i]; j < S_ext_i[i + 1]; j++)
         {
            big_i1 = S_ext_j[j];
            if (big_i1 < my_first_cpt || big_i1 > my_last_cpt)
            {
               S_ext_j[cnt_offd++] = big_i1;
            }
            else
            {
               S_ext_diag_j[cnt_diag++] = (HYPRE_Int)(big_i1 - my_first_cpt);
            }
         }
         S_ext_diag_i[++cnt] = cnt_diag;
         S_ext_offd_i[cnt] = cnt_offd;
      }

      hypre_TFree(S_ext_i, HYPRE_MEMORY_HOST);

      cnt = 0;
      if (S_ext_offd_size || num_coarse_offd)
      {
         temp = hypre_CTAlloc(HYPRE_BigInt,  S_ext_offd_size + num_coarse_offd, HYPRE_MEMORY_HOST);
         for (i = 0; i < S_ext_offd_size; i++)
         {
            temp[i] = S_ext_j[i];
         }
         cnt = S_ext_offd_size;
         for (i = 0; i < num_cols_offd_S; i++)
            if (CF_marker_offd[i] > 0) { temp[cnt++] = fine_to_coarse_offd[i]; }
      }
      if (cnt)
      {
         hypre_BigQsort0(temp, 0, cnt - 1);

         num_cols_offd_C = 1;
         value = temp[0];
         for (i = 1; i < cnt; i++)
         {
            if (temp[i] > value)
            {
               value = temp[i];
               temp[num_cols_offd_C++] = value;
            }
         }
      }

      if (num_cols_offd_C)
      {
         col_map_offd_C = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
      }

      for (i = 0; i < num_cols_offd_C; i++)
      {
         col_map_offd_C[i] = temp[i];
      }

      if (S_ext_offd_size || num_coarse_offd)
      {
         hypre_TFree(temp, HYPRE_MEMORY_HOST);
      }

      for (i = 0 ; i < S_ext_offd_size; i++)
         S_ext_offd_j[i] = hypre_BigBinarySearch(col_map_offd_C,
                                                 S_ext_j[i],
                                                 num_cols_offd_C);
      hypre_TFree(S_ext_j, HYPRE_MEMORY_HOST);

#endif /* !HYPRE_CONCURRENT_HOPSCOTCH */

      if (num_cols_offd_S)
      {
         map_S_to_C = hypre_TAlloc(HYPRE_Int, num_cols_offd_S, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel private(i)
#endif
         {
            HYPRE_Int i_begin, i_end;
            hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_S);

            HYPRE_BigInt cnt = 0;
            for (i = i_begin; i < i_end; i++)
            {
               if (CF_marker_offd[i] > 0)
               {
                  cnt = hypre_BigLowerBound(col_map_offd_C + cnt, col_map_offd_C + num_cols_offd_C,
                                            fine_to_coarse_offd[i]) - col_map_offd_C;
                  map_S_to_C[i] = cnt++;
               }
               else { map_S_to_C[i] = -1; }
            }
         } /* omp parallel */
      }

#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif
   } /* num_procs > 1 */

   /*-----------------------------------------------------------------------
    *  Allocate and initialize some stuff.
    *-----------------------------------------------------------------------*/

   HYPRE_Int *S_marker_array = NULL, *S_marker_offd_array = NULL;
   if (num_coarse) { S_marker_array = hypre_TAlloc(HYPRE_Int,  num_coarse * hypre_NumThreads(), HYPRE_MEMORY_HOST); }
   if (num_cols_offd_C) { S_marker_offd_array = hypre_TAlloc(HYPRE_Int,  num_cols_offd_C * hypre_NumThreads(), HYPRE_MEMORY_HOST); }

   HYPRE_Int *C_temp_offd_j_array = NULL;
   HYPRE_Int *C_temp_diag_j_array = NULL;
   HYPRE_Int *C_temp_offd_data_array = NULL;
   HYPRE_Int *C_temp_diag_data_array = NULL;

   if (num_paths > 1)
   {
      C_temp_diag_j_array = hypre_TAlloc(HYPRE_Int,  num_coarse * hypre_NumThreads(), HYPRE_MEMORY_HOST);
      C_temp_offd_j_array = hypre_TAlloc(HYPRE_Int,  num_cols_offd_C * hypre_NumThreads(),
                                         HYPRE_MEMORY_HOST);

      C_temp_diag_data_array = hypre_TAlloc(HYPRE_Int,  num_coarse * hypre_NumThreads(),
                                            HYPRE_MEMORY_HOST);
      C_temp_offd_data_array = hypre_TAlloc(HYPRE_Int,  num_cols_offd_C * hypre_NumThreads(),
                                            HYPRE_MEMORY_HOST);
   }

   C_diag_i = hypre_CTAlloc(HYPRE_Int,  num_coarse + 1, HYPRE_MEMORY_HOST);
   C_offd_i = hypre_CTAlloc(HYPRE_Int,  num_coarse + 1, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Loop over rows of S
    *-----------------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i1,i2,i3,jj1,jj2,index)
#endif
   {
      HYPRE_Int my_thread_num = hypre_GetThreadNum();

      HYPRE_Int i1_begin, i1_end;
      hypre_GetSimpleThreadPartition(&i1_begin, &i1_end, num_cols_diag_S);

      HYPRE_Int *C_temp_diag_j = NULL, *C_temp_offd_j = NULL;
      HYPRE_Int *C_temp_diag_data = NULL, *C_temp_offd_data = NULL;

      if (num_paths > 1)
      {
         C_temp_diag_j = C_temp_diag_j_array + num_coarse * my_thread_num;
         C_temp_offd_j = C_temp_offd_j_array + num_cols_offd_C * my_thread_num;

         C_temp_diag_data = C_temp_diag_data_array + num_coarse * my_thread_num;
         C_temp_offd_data = C_temp_offd_data_array + num_cols_offd_C * my_thread_num;
      }

      HYPRE_Int *S_marker = NULL, *S_marker_offd = NULL;
      if (num_coarse) { S_marker = S_marker_array + num_coarse * my_thread_num; }
      if (num_cols_offd_C) { S_marker_offd = S_marker_offd_array + num_cols_offd_C * my_thread_num; }
      for (i1 = 0; i1 < num_coarse; i1++)
      {
         S_marker[i1] = -1;
      }
      for (i1 = 0; i1 < num_cols_offd_C; i1++)
      {
         S_marker_offd[i1] = -1;
      }

      // These two counters are for before filtering by num_paths
      HYPRE_Int jj_count_diag = 0;
      HYPRE_Int jj_count_offd = 0;

      // These two counters are for after filtering by num_paths
      HYPRE_Int num_nonzeros_diag = 0;
      HYPRE_Int num_nonzeros_offd = 0;

      HYPRE_Int ic_begin = num_coarse_prefix_sum[my_thread_num];
      HYPRE_Int ic_end = num_coarse_prefix_sum[my_thread_num + 1];
      HYPRE_Int ic;

      if (num_paths == 1)
      {
         for (ic = ic_begin; ic < ic_end; ic++)
         {
            /*--------------------------------------------------------------------
             *  Set marker for diagonal entry, C_{i1,i1} (for square matrices).
             *--------------------------------------------------------------------*/

            i1 = coarse_to_fine[ic];

            HYPRE_Int jj_row_begin_diag = num_nonzeros_diag;
            HYPRE_Int jj_row_begin_offd = num_nonzeros_offd;

            C_diag_i[ic] = num_nonzeros_diag;
            if (num_cols_offd_C)
            {
               C_offd_i[ic] = num_nonzeros_offd;
            }

            for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1 + 1]; jj1++)
            {
               i2 = S_diag_j[jj1];
               if (CF_marker[i2] > 0)
               {
                  index = fine_to_coarse[i2];
                  if (S_marker[index] < jj_row_begin_diag)
                  {
                     S_marker[index] = num_nonzeros_diag;
                     num_nonzeros_diag++;
                  }
               }
               for (jj2 = S_diag_i[i2]; jj2 < S_diag_i[i2 + 1]; jj2++)
               {
                  i3 = S_diag_j[jj2];
                  if (CF_marker[i3] > 0)
                  {
                     index = fine_to_coarse[i3];
                     if (index != ic && S_marker[index] < jj_row_begin_diag)
                     {
                        S_marker[index] = num_nonzeros_diag;
                        num_nonzeros_diag++;
                     }
                  }
               }
               for (jj2 = S_offd_i[i2]; jj2 < S_offd_i[i2 + 1]; jj2++)
               {
                  i3 = S_offd_j[jj2];
                  if (CF_marker_offd[i3] > 0)
                  {
                     index = map_S_to_C[i3];
                     if (S_marker_offd[index] < jj_row_begin_offd)
                     {
                        S_marker_offd[index] = num_nonzeros_offd;
                        num_nonzeros_offd++;
                     }
                  }
               }
            }
            for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1 + 1]; jj1++)
            {
               i2 = S_offd_j[jj1];
               if (CF_marker_offd[i2] > 0)
               {
                  index = map_S_to_C[i2];
                  if (S_marker_offd[index] < jj_row_begin_offd)
                  {
                     S_marker_offd[index] = num_nonzeros_offd;
                     num_nonzeros_offd++;
                  }
               }
               for (jj2 = S_ext_diag_i[i2]; jj2 < S_ext_diag_i[i2 + 1]; jj2++)
               {
                  i3 = S_ext_diag_j[jj2];
                  if (i3 != ic && S_marker[i3] < jj_row_begin_diag)
                  {
                     S_marker[i3] = num_nonzeros_diag;
                     num_nonzeros_diag++;
                  }
               }
               for (jj2 = S_ext_offd_i[i2]; jj2 < S_ext_offd_i[i2 + 1]; jj2++)
               {
                  i3 = S_ext_offd_j[jj2];
                  if (S_marker_offd[i3] < jj_row_begin_offd)
                  {
                     S_marker_offd[i3] = num_nonzeros_offd;
                     num_nonzeros_offd++;
                  }
               }
            }
         } /* for each row */

      } /* num_paths == 1 */
      else
      {
         for (ic = ic_begin; ic < ic_end; ic++)
         {
            /*--------------------------------------------------------------------
             *  Set marker for diagonal entry, C_{i1,i1} (for square matrices).
             *--------------------------------------------------------------------*/

            i1 = coarse_to_fine[ic];

            HYPRE_Int jj_row_begin_diag = jj_count_diag;
            HYPRE_Int jj_row_begin_offd = jj_count_offd;

            C_diag_i[ic] = num_nonzeros_diag;
            if (num_cols_offd_C)
            {
               C_offd_i[ic] = num_nonzeros_offd;
            }

            for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1 + 1]; jj1++)
            {
               i2 = S_diag_j[jj1];
               if (CF_marker[i2] > 0)
               {
                  index = fine_to_coarse[i2];
                  if (S_marker[index] < jj_row_begin_diag)
                  {
                     S_marker[index] = jj_count_diag;
                     C_temp_diag_data[jj_count_diag - jj_row_begin_diag] = 2;
                     jj_count_diag++;
                  }
                  else
                  {
                     C_temp_diag_data[S_marker[index] - jj_row_begin_diag] += 2;
                  }
               }
               for (jj2 = S_diag_i[i2]; jj2 < S_diag_i[i2 + 1]; jj2++)
               {
                  i3 = S_diag_j[jj2];
                  if (CF_marker[i3] > 0 && fine_to_coarse[i3] != ic)
                  {
                     index = fine_to_coarse[i3];
                     if (S_marker[index] < jj_row_begin_diag)
                     {
                        S_marker[index] = jj_count_diag;
                        C_temp_diag_data[jj_count_diag - jj_row_begin_diag] = 1;
                        jj_count_diag++;
                     }
                     else
                     {
                        C_temp_diag_data[S_marker[index] - jj_row_begin_diag]++;
                     }
                  }
               }
               for (jj2 = S_offd_i[i2]; jj2 < S_offd_i[i2 + 1]; jj2++)
               {
                  i3 = S_offd_j[jj2];
                  if (CF_marker_offd[i3] > 0)
                  {
                     index = map_S_to_C[i3];
                     if (S_marker_offd[index] < jj_row_begin_offd)
                     {
                        S_marker_offd[index] = jj_count_offd;
                        C_temp_offd_data[jj_count_offd - jj_row_begin_offd] = 1;
                        jj_count_offd++;
                     }
                     else
                     {
                        C_temp_offd_data[S_marker_offd[index] - jj_row_begin_offd]++;
                     }
                  }
               }
            }
            for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1 + 1]; jj1++)
            {
               i2 = S_offd_j[jj1];
               if (CF_marker_offd[i2] > 0)
               {
                  index = map_S_to_C[i2];
                  if (S_marker_offd[index] < jj_row_begin_offd)
                  {
                     S_marker_offd[index] = jj_count_offd;
                     C_temp_offd_data[jj_count_offd - jj_row_begin_offd] = 2;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_temp_offd_data[S_marker_offd[index] - jj_row_begin_offd] += 2;
                  }
               }
               for (jj2 = S_ext_diag_i[i2]; jj2 < S_ext_diag_i[i2 + 1]; jj2++)
               {
                  i3 = S_ext_diag_j[jj2];
                  if (i3 != ic)
                  {
                     if (S_marker[i3] < jj_row_begin_diag)
                     {
                        S_marker[i3] = jj_count_diag;
                        C_temp_diag_data[jj_count_diag - jj_row_begin_diag] = 1;
                        jj_count_diag++;
                     }
                     else
                     {
                        C_temp_diag_data[S_marker[i3] - jj_row_begin_diag]++;
                     }
                  }
               }
               for (jj2 = S_ext_offd_i[i2]; jj2 < S_ext_offd_i[i2 + 1]; jj2++)
               {
                  i3 = S_ext_offd_j[jj2];
                  if (S_marker_offd[i3] < jj_row_begin_offd)
                  {
                     S_marker_offd[i3] = jj_count_offd;
                     C_temp_offd_data[jj_count_offd - jj_row_begin_offd] = 1;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_temp_offd_data[S_marker_offd[i3] - jj_row_begin_offd]++;
                  }
               }
            }

            for (jj1 = jj_row_begin_diag; jj1 < jj_count_diag; jj1++)
            {
               if (C_temp_diag_data[jj1 - jj_row_begin_diag] >= num_paths)
               {
                  ++num_nonzeros_diag;
               }
               C_temp_diag_data[jj1 - jj_row_begin_diag] = 0;
            }
            for (jj1 = jj_row_begin_offd; jj1 < jj_count_offd; jj1++)
            {
               if (C_temp_offd_data[jj1 - jj_row_begin_offd] >= num_paths)
               {
                  ++num_nonzeros_offd;
               }
               C_temp_offd_data[jj1 - jj_row_begin_offd] = 0;
            }
         } /* for each row */
      } /* num_paths > 1 */

      hypre_prefix_sum_pair(
         &num_nonzeros_diag, &C_diag_i[num_coarse],
         &num_nonzeros_offd, &C_offd_i[num_coarse],
         prefix_sum_workspace);

      for (i1 = 0; i1 < num_coarse; i1++)
      {
         S_marker[i1] = -1;
      }
      for (i1 = 0; i1 < num_cols_offd_C; i1++)
      {
         S_marker_offd[i1] = -1;
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
      #pragma omp master
#endif
      {
         if (C_diag_i[num_coarse])
         {
            C_diag_j = hypre_TAlloc(HYPRE_Int,  C_diag_i[num_coarse], HYPRE_MEMORY_HOST);
         }
         if (C_offd_i[num_coarse])
         {
            C_offd_j = hypre_TAlloc(HYPRE_Int,  C_offd_i[num_coarse], HYPRE_MEMORY_HOST);
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (ic = ic_begin; ic < ic_end - 1; ic++)
      {
         if (C_diag_i[ic + 1] == C_diag_i[ic] && C_offd_i[ic + 1] == C_offd_i[ic])
         {
            CF_marker[coarse_to_fine[ic]] = 2;
         }

         C_diag_i[ic] += num_nonzeros_diag;
         C_offd_i[ic] += num_nonzeros_offd;
      }
      if (ic_begin < ic_end)
      {
         C_diag_i[ic] += num_nonzeros_diag;
         C_offd_i[ic] += num_nonzeros_offd;

         HYPRE_Int next_C_diag_i = prefix_sum_workspace[2 * (my_thread_num + 1)];
         HYPRE_Int next_C_offd_i = prefix_sum_workspace[2 * (my_thread_num + 1) + 1];

         if (next_C_diag_i == C_diag_i[ic] && next_C_offd_i == C_offd_i[ic])
         {
            CF_marker[coarse_to_fine[ic]] = 2;
         }
      }

      if (num_paths == 1)
      {
         for (ic = ic_begin; ic < ic_end; ic++)
         {
            /*--------------------------------------------------------------------
             *  Set marker for diagonal entry, C_{i1,i1} (for square matrices).
             *--------------------------------------------------------------------*/

            i1 = coarse_to_fine[ic];

            HYPRE_Int jj_row_begin_diag = num_nonzeros_diag;
            HYPRE_Int jj_row_begin_offd = num_nonzeros_offd;

            for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1 + 1]; jj1++)
            {
               i2 = S_diag_j[jj1];
               if (CF_marker[i2] > 0)
               {
                  index = fine_to_coarse[i2];
                  if (S_marker[index] < jj_row_begin_diag)
                  {
                     S_marker[index] = num_nonzeros_diag;
                     C_diag_j[num_nonzeros_diag] = index;
                     num_nonzeros_diag++;
                  }
               }
               for (jj2 = S_diag_i[i2]; jj2 < S_diag_i[i2 + 1]; jj2++)
               {
                  i3 = S_diag_j[jj2];
                  if (CF_marker[i3] > 0)
                  {
                     index = fine_to_coarse[i3];
                     if (index != ic && S_marker[index] < jj_row_begin_diag)
                     {
                        S_marker[index] = num_nonzeros_diag;
                        C_diag_j[num_nonzeros_diag] = index;
                        num_nonzeros_diag++;
                     }
                  }
               }
               for (jj2 = S_offd_i[i2]; jj2 < S_offd_i[i2 + 1]; jj2++)
               {
                  i3 = S_offd_j[jj2];
                  if (CF_marker_offd[i3] > 0)
                  {
                     index = map_S_to_C[i3];
                     if (S_marker_offd[index] < jj_row_begin_offd)
                     {
                        S_marker_offd[index] = num_nonzeros_offd;
                        C_offd_j[num_nonzeros_offd] = index;
                        num_nonzeros_offd++;
                     }
                  }
               }
            }
            for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1 + 1]; jj1++)
            {
               i2 = S_offd_j[jj1];
               if (CF_marker_offd[i2] > 0)
               {
                  index = map_S_to_C[i2];
                  if (S_marker_offd[index] < jj_row_begin_offd)
                  {
                     S_marker_offd[index] = num_nonzeros_offd;
                     C_offd_j[num_nonzeros_offd] = index;
                     num_nonzeros_offd++;
                  }
               }
               for (jj2 = S_ext_diag_i[i2]; jj2 < S_ext_diag_i[i2 + 1]; jj2++)
               {
                  i3 = S_ext_diag_j[jj2];
                  if (i3 != ic && S_marker[i3] < jj_row_begin_diag)
                  {
                     S_marker[i3] = num_nonzeros_diag;
                     C_diag_j[num_nonzeros_diag] = i3;
                     num_nonzeros_diag++;
                  }
               }
               for (jj2 = S_ext_offd_i[i2]; jj2 < S_ext_offd_i[i2 + 1]; jj2++)
               {
                  i3 = S_ext_offd_j[jj2];
                  if (S_marker_offd[i3] < jj_row_begin_offd)
                  {
                     S_marker_offd[i3] = num_nonzeros_offd;
                     C_offd_j[num_nonzeros_offd] = i3;
                     num_nonzeros_offd++;
                  }
               }
            }
         } /* for each row */

      } /* num_paths == 1 */
      else
      {
         jj_count_diag = num_nonzeros_diag;
         jj_count_offd = num_nonzeros_offd;

         for (ic = ic_begin; ic < ic_end; ic++)
         {
            /*--------------------------------------------------------------------
             *  Set marker for diagonal entry, C_{i1,i1} (for square matrices).
             *--------------------------------------------------------------------*/

            i1 = coarse_to_fine[ic];

            HYPRE_Int jj_row_begin_diag = jj_count_diag;
            HYPRE_Int jj_row_begin_offd = jj_count_offd;

            for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1 + 1]; jj1++)
            {
               i2 = S_diag_j[jj1];
               if (CF_marker[i2] > 0)
               {
                  index = fine_to_coarse[i2];
                  if (S_marker[index] < jj_row_begin_diag)
                  {
                     S_marker[index] = jj_count_diag;
                     C_temp_diag_j[jj_count_diag - jj_row_begin_diag] = index;
                     C_temp_diag_data[jj_count_diag - jj_row_begin_diag] = 2;
                     jj_count_diag++;
                  }
                  else
                  {
                     C_temp_diag_data[S_marker[index] - jj_row_begin_diag] += 2;
                  }
               }
               for (jj2 = S_diag_i[i2]; jj2 < S_diag_i[i2 + 1]; jj2++)
               {
                  i3 = S_diag_j[jj2];
                  if (CF_marker[i3] > 0 && fine_to_coarse[i3] != ic)
                  {
                     index = fine_to_coarse[i3];
                     if (S_marker[index] < jj_row_begin_diag)
                     {
                        S_marker[index] = jj_count_diag;
                        C_temp_diag_j[jj_count_diag - jj_row_begin_diag] = index;
                        C_temp_diag_data[jj_count_diag - jj_row_begin_diag] = 1;
                        jj_count_diag++;
                     }
                     else
                     {
                        C_temp_diag_data[S_marker[index] - jj_row_begin_diag]++;
                     }
                  }
               }
               for (jj2 = S_offd_i[i2]; jj2 < S_offd_i[i2 + 1]; jj2++)
               {
                  i3 = S_offd_j[jj2];
                  if (CF_marker_offd[i3] > 0)
                  {
                     index = map_S_to_C[i3];
                     if (S_marker_offd[index] < jj_row_begin_offd)
                     {
                        S_marker_offd[index] = jj_count_offd;
                        C_temp_offd_j[jj_count_offd - jj_row_begin_offd] = index;
                        C_temp_offd_data[jj_count_offd - jj_row_begin_offd] = 1;
                        jj_count_offd++;
                     }
                     else
                     {
                        C_temp_offd_data[S_marker_offd[index] - jj_row_begin_offd]++;
                     }
                  }
               }
            }
            for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1 + 1]; jj1++)
            {
               i2 = S_offd_j[jj1];
               if (CF_marker_offd[i2] > 0)
               {
                  index = map_S_to_C[i2];
                  if (S_marker_offd[index] < jj_row_begin_offd)
                  {
                     S_marker_offd[index] = jj_count_offd;
                     C_temp_offd_j[jj_count_offd - jj_row_begin_offd] = index;
                     C_temp_offd_data[jj_count_offd - jj_row_begin_offd] = 2;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_temp_offd_data[S_marker_offd[index] - jj_row_begin_offd] += 2;
                  }
               }
               for (jj2 = S_ext_diag_i[i2]; jj2 < S_ext_diag_i[i2 + 1]; jj2++)
               {
                  i3 = S_ext_diag_j[jj2];
                  if (i3 != ic)
                  {
                     if (S_marker[i3] < jj_row_begin_diag)
                     {
                        S_marker[i3] = jj_count_diag;
                        C_temp_diag_j[jj_count_diag - jj_row_begin_diag] = i3;
                        C_temp_diag_data[jj_count_diag - jj_row_begin_diag] = 1;
                        jj_count_diag++;
                     }
                     else
                     {
                        C_temp_diag_data[S_marker[i3] - jj_row_begin_diag]++;
                     }
                  }
               }
               for (jj2 = S_ext_offd_i[i2]; jj2 < S_ext_offd_i[i2 + 1]; jj2++)
               {
                  i3 = S_ext_offd_j[jj2];
                  if (S_marker_offd[i3] < jj_row_begin_offd)
                  {
                     S_marker_offd[i3] = jj_count_offd;
                     C_temp_offd_j[jj_count_offd - jj_row_begin_offd] = i3;
                     C_temp_offd_data[jj_count_offd - jj_row_begin_offd] = 1;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_temp_offd_data[S_marker_offd[i3] - jj_row_begin_offd]++;
                  }
               }
            }

            for (jj1 = jj_row_begin_diag; jj1 < jj_count_diag; jj1++)
            {
               if (C_temp_diag_data[jj1 - jj_row_begin_diag] >= num_paths)
               {
                  C_diag_j[num_nonzeros_diag++] = C_temp_diag_j[jj1 - jj_row_begin_diag];
               }
               C_temp_diag_data[jj1 - jj_row_begin_diag] = 0;
            }
            for (jj1 = jj_row_begin_offd; jj1 < jj_count_offd; jj1++)
            {
               if (C_temp_offd_data[jj1 - jj_row_begin_offd] >= num_paths)
               {
                  C_offd_j[num_nonzeros_offd++] = C_temp_offd_j[jj1 - jj_row_begin_offd];
               }
               C_temp_offd_data[jj1 - jj_row_begin_offd] = 0;
            }
         } /* for each row */
      } /* num_paths > 1 */
   } /* omp parallel */

   S2 = hypre_ParCSRMatrixCreate(comm, global_num_coarse,
                                 global_num_coarse, coarse_row_starts,
                                 coarse_row_starts, num_cols_offd_C, C_diag_i[num_coarse], C_offd_i[num_coarse]);

   C_diag = hypre_ParCSRMatrixDiag(S2);
   hypre_CSRMatrixI(C_diag) = C_diag_i;
   if (C_diag_i[num_coarse]) { hypre_CSRMatrixJ(C_diag) = C_diag_j; }

   C_offd = hypre_ParCSRMatrixOffd(S2);
   hypre_CSRMatrixI(C_offd) = C_offd_i;
   hypre_ParCSRMatrixOffd(S2) = C_offd;

   if (num_cols_offd_C)
   {
      if (C_offd_i[num_coarse]) { hypre_CSRMatrixJ(C_offd) = C_offd_j; }
      hypre_ParCSRMatrixColMapOffd(S2) = col_map_offd_C;
   }

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/
   hypre_TFree(C_temp_diag_j_array, HYPRE_MEMORY_HOST);
   hypre_TFree(C_temp_diag_data_array, HYPRE_MEMORY_HOST);

   hypre_TFree(C_temp_offd_j_array, HYPRE_MEMORY_HOST);
   hypre_TFree(C_temp_offd_data_array, HYPRE_MEMORY_HOST);

   hypre_TFree(S_marker_array, HYPRE_MEMORY_HOST);
   hypre_TFree(S_marker_offd_array, HYPRE_MEMORY_HOST);

   hypre_TFree(S_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(S_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(S_ext_diag_i, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_to_fine, HYPRE_MEMORY_HOST);
   if (S_ext_diag_size)
   {
      hypre_TFree(S_ext_diag_j, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(S_ext_offd_i, HYPRE_MEMORY_HOST);
   if (S_ext_offd_size)
   {
      hypre_TFree(S_ext_offd_j, HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_S)
   {
      hypre_TFree(map_S_to_C, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   }

   hypre_CSRMatrixMemoryLocation(C_diag) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixMemoryLocation(C_offd) = HYPRE_MEMORY_HOST;

   *C_ptr = S2;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATE_2NDS] += hypre_MPI_Wtime();
#endif

   hypre_TFree(prefix_sum_workspace, HYPRE_MEMORY_HOST);
   hypre_TFree(num_coarse_prefix_sum, HYPRE_MEMORY_HOST);

   return 0;
}

//-----------------------------------------------------------------------
HYPRE_Int
hypre_BoomerAMGCreate2ndS( hypre_ParCSRMatrix  *S,
                           HYPRE_Int           *CF_marker,
                           HYPRE_Int            num_paths,
                           HYPRE_BigInt        *coarse_row_starts,
                           hypre_ParCSRMatrix **C_ptr)
{
   hypre_GpuProfilingPushRange("Create2ndS");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(S) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGCreate2ndSDevice( S, CF_marker, num_paths, coarse_row_starts, C_ptr );
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGCreate2ndSHost( S, CF_marker, num_paths, coarse_row_starts, C_ptr );
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCorrectCFMarker : corrects CF_marker after aggr. coarsening
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCorrectCFMarkerHost(hypre_IntArray *CF_marker, hypre_IntArray *new_CF_marker)
{
   HYPRE_Int i, cnt;

   cnt = 0;
   for (i = 0; i < hypre_IntArraySize(CF_marker); i++)
   {
      if (hypre_IntArrayData(CF_marker)[i] > 0 )
      {
         if (hypre_IntArrayData(CF_marker)[i] == 1)
         {
            hypre_IntArrayData(CF_marker)[i] = hypre_IntArrayData(new_CF_marker)[cnt++];
         }
         else
         {
            hypre_IntArrayData(CF_marker)[i] = 1;
            cnt++;
         }
      }
   }

   return 0;
}
/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCorrectCFMarker2 : corrects CF_marker after aggr. coarsening,
 * but marks new F-points (previous C-points) as -2
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCorrectCFMarker2Host(hypre_IntArray *CF_marker, hypre_IntArray *new_CF_marker)
{
   HYPRE_Int i, cnt;

   cnt = 0;
   for (i = 0; i < hypre_IntArraySize(CF_marker); i++)
   {
      if (hypre_IntArrayData(CF_marker)[i] > 0 )
      {
         if (hypre_IntArrayData(new_CF_marker)[cnt] == -1)
         {
            hypre_IntArrayData(CF_marker)[i] = -2;
         }
         else
         {
            hypre_IntArrayData(CF_marker)[i] = 1;
         }
         cnt++;
      }
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCorrectCFMarker : corrects CF_marker after aggr. coarsening
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCorrectCFMarker(hypre_IntArray *CF_marker, hypre_IntArray *new_CF_marker)
{

   hypre_GpuProfilingPushRange("CorrectCFMarker");

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_IntArrayMemoryLocation(CF_marker),
                                                      hypre_IntArrayMemoryLocation(new_CF_marker));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_BoomerAMGCorrectCFMarkerDevice(CF_marker, new_CF_marker);
   }
   else
#endif
   {
      hypre_BoomerAMGCorrectCFMarkerHost(CF_marker, new_CF_marker);
   }

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCorrectCFMarker2 : corrects CF_marker after aggr. coarsening,
 * but marks new F-points (previous C-points) as -2
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCorrectCFMarker2(hypre_IntArray *CF_marker, hypre_IntArray *new_CF_marker)
{

   hypre_GpuProfilingPushRange("CorrectCFMarker2");

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_IntArrayMemoryLocation(CF_marker),
                                                      hypre_IntArrayMemoryLocation(new_CF_marker));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_BoomerAMGCorrectCFMarker2Device(CF_marker, new_CF_marker);
   }
   else
#endif
   {
      hypre_BoomerAMGCorrectCFMarker2Host(CF_marker, new_CF_marker);
   }

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}
