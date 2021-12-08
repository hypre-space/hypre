/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "_hypre_parcsr_ls.h"

/*==========================================================================*/
/*==========================================================================*/
/**
  Selects a coarse "grid" based on the graph of a matrix.

  Notes:
  \begin{itemize}
  \item The underlying matrix storage scheme is a hypre_ParCSR matrix.
  \item The routine returns the following:
  \begin{itemize}
  \item S - a ParCSR matrix representing the "strength matrix".  This is
  used in the "build interpolation" routine.
  \item CF\_marker - an array indicating both C-pts (value = 1) and
  F-pts (value = -1)
  \end{itemize}
  \item We define the following temporary storage:
  \begin{itemize}
  \item measure\_array - an array containing the "measures" for each
  of the fine-grid points
  \item graph\_array - an array containing the list of points in the
  "current subgraph" being considered in the coarsening process.
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
  @param S_ptr [OUT]
  strength matrix
  @param CF_marker_ptr [IN/OUT]
  array indicating C/F points

  @see */
/*--------------------------------------------------------------------------*/

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

HYPRE_Int
hypre_BoomerAMGCoarsen( hypre_ParCSRMatrix    *S,
                        hypre_ParCSRMatrix    *A,
                        HYPRE_Int              CF_init,
                        HYPRE_Int              debug_flag,
                        hypre_IntArray       **CF_marker_ptr)
{
   MPI_Comm                comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg    *comm_pkg        = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int          *S_diag_i        = hypre_CSRMatrixI(S_diag);
   HYPRE_Int          *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int          *S_offd_i        = hypre_CSRMatrixI(S_offd);
   HYPRE_Int          *S_offd_j = NULL;

   HYPRE_BigInt       *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);
   HYPRE_Int           num_variables   = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_BigInt        col_1 = hypre_ParCSRMatrixFirstColDiag(S);
   HYPRE_BigInt        col_n = col_1 + (HYPRE_BigInt)hypre_CSRMatrixNumCols(S_diag);
   HYPRE_Int           num_cols_offd = 0;

   hypre_CSRMatrix    *S_ext;
   HYPRE_Int          *S_ext_i = NULL;
   HYPRE_BigInt       *S_ext_j = NULL;

   HYPRE_Int           num_sends = 0;
   HYPRE_Int          *int_buf_data;
   HYPRE_Real         *buf_data;

   HYPRE_Int          *CF_marker;
   HYPRE_Int          *CF_marker_offd;

   HYPRE_Real         *measure_array;
   HYPRE_Int          *graph_array;
   HYPRE_Int          *graph_array_offd;
   HYPRE_Int           graph_size;
   HYPRE_BigInt        big_graph_size;
   HYPRE_Int           graph_offd_size;
   HYPRE_BigInt        global_graph_size;

   HYPRE_Int           i, j, k, kc, jS, kS, ig, elmt;
   HYPRE_Int           index, start, my_id, num_procs, jrow, cnt, nnzrow;

   HYPRE_Int           use_commpkg_A = 0;
   HYPRE_Int           break_var = 1;

   HYPRE_Real       wall_time;
   HYPRE_Int        iter = 0;
   HYPRE_BigInt     big_k;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

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

   S_ext = NULL;
   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      use_commpkg_A = 1;
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                            num_sends), HYPRE_MEMORY_HOST);
   buf_data = hypre_CTAlloc(HYPRE_Real,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                         num_sends), HYPRE_MEMORY_HOST);

   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }
   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *
    * The measures are augmented by a random number
    * between 0 and 1.
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(HYPRE_Real,  num_variables + num_cols_offd, HYPRE_MEMORY_HOST);

   for (i = 0; i < S_offd_i[num_variables]; i++)
   {
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }
   if (num_procs > 1)
      comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
                                                 &measure_array[num_variables], buf_data);

   for (i = 0; i < S_diag_i[num_variables]; i++)
   {
      measure_array[S_diag_j[i]] += 1.0;
   }

   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)]
         += buf_data[index++];
   }

   for (i = num_variables; i < num_variables + num_cols_offd; i++)
   {
      measure_array[i] = 0;
   }

   /* this augments the measures */
   if (CF_init == 2)
   {
      hypre_BoomerAMGIndepSetInit(S, measure_array, 1);
   }
   else
   {
      hypre_BoomerAMGIndepSetInit(S, measure_array, 0);
   }

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);
   if (num_cols_offd)
   {
      graph_array_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
   }
   else
   {
      graph_array_offd = NULL;
   }

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_cols_offd; ig++)
   {
      graph_array_offd[ig] = ig;
   }

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ...
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   graph_offd_size = num_cols_offd;

   /* Allocate CF_marker if not done before */
   if (*CF_marker_ptr == NULL)
   {
      *CF_marker_ptr = hypre_IntArrayCreate(num_variables);
      hypre_IntArrayInitialize(*CF_marker_ptr);
   }
   CF_marker = hypre_IntArrayData(*CF_marker_ptr);

   if (CF_init == 1)
   {
      cnt = 0;
      for (i = 0; i < num_variables; i++)
      {
         if ( CF_marker[i] != SF_PT )
         {
            if ( (S_offd_i[i + 1] - S_offd_i[i]) > 0 ||
                 (CF_marker[i] == F_PT) )
            {
               CF_marker[i] = 0;
            }
            if ( CF_marker[i] == Z_PT)
            {
               if ( (S_diag_i[i + 1] - S_diag_i[i]) > 0 ||
                    (measure_array[i] >= 1.0) )
               {
                  CF_marker[i] = 0;
                  graph_array[cnt++] = i;
               }
               else
               {
                  CF_marker[i] = F_PT;
               }
            }
            else
            {
               graph_array[cnt++] = i;
            }
         }
         else
         {
            measure_array[i] = 0;
         }
      }
   }
   else
   {
      cnt = 0;
      for (i = 0; i < num_variables; i++)
      {
         if ( CF_marker[i] != SF_PT )
         {
            CF_marker[i] = 0;
            nnzrow = (S_diag_i[i + 1] - S_diag_i[i]) + (S_offd_i[i + 1] - S_offd_i[i]);
            if (nnzrow == 0)
            {
               CF_marker[i] = SF_PT;
               measure_array[i] = 0;
            }
            else
            {
               graph_array[cnt++] = i;
            }
         }
         else
         {
            measure_array[i] = 0;
         }
      }
   }
   graph_size = cnt;
   if (num_cols_offd)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
   }
   else
   {
      CF_marker_offd = NULL;
   }
   for (i = 0; i < num_cols_offd; i++)
   {
      CF_marker_offd[i] = 0;
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (num_procs > 1)
   {
      if (use_commpkg_A)
      {
         S_ext      = hypre_ParCSRMatrixExtractBExt(S, A, 0);
      }
      else
      {
         S_ext      = hypre_ParCSRMatrixExtractBExt(S, S, 0);
      }
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixBigJ(S_ext);
   }

   /*  compress S_ext  and convert column numbers*/

   index = 0;
   for (i = 0; i < num_cols_offd; i++)
   {
      for (j = S_ext_i[i]; j < S_ext_i[i + 1]; j++)
      {
         big_k = S_ext_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            S_ext_j[index++] = big_k - col_1;
         }
         else
         {
            kc = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_offd);
            if (kc > -1) { S_ext_j[index++] = (HYPRE_BigInt)(-kc - 1); }
         }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i--)
   {
      S_ext_i[i] = S_ext_i[i - 1];
   }
   if (num_procs > 1) { S_ext_i[0] = 0; }

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Initialize CLJP phase = %f\n",
                   my_id, wall_time);
   }

   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

      if (num_procs > 1)
         comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
                                                    &measure_array[num_variables], buf_data);

      if (num_procs > 1)
      {
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)]
            += buf_data[index++];
      }

      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/

      if (iter || (CF_init != 1))
      {
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];

            if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) )
            {
               /* set to be an F-pt */
               CF_marker[i] = F_PT;

               /* make sure all dependencies have been accounted for */
               for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
            }
            if (CF_marker[i])
            {
               measure_array[i] = 0;

               /* take point out of the subgraph */
               graph_size--;
               graph_array[ig] = graph_array[graph_size];
               graph_array[graph_size] = i;
               ig--;
            }
         }
      }

      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures
       *------------------------------------------------*/

      if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
            buf_data[index++] = measure_array[jrow];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data,
                                                    &measure_array[num_variables]);

         hypre_ParCSRCommHandleDestroy(comm_handle);

      }
      /*------------------------------------------------
       * Debugging:
       *
       * Uncomment the sections of code labeled
       * "debugging" to generate several files that
       * can be visualized using the `coarsen.m'
       * matlab routine.
       *------------------------------------------------*/

#if 0 /* debugging */
      /* print out measures */
      hypre_sprintf(filename, "coarsen.out.measures.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         hypre_fprintf(fp, "%f\n", measure_array[i]);
      }
      fclose(fp);

      /* print out strength matrix */
      hypre_sprintf(filename, "coarsen.out.strength.%04d", iter);
      hypre_CSRMatrixPrint(S, filename);

      /* print out C/F marker */
      hypre_sprintf(filename, "coarsen.out.CF.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         hypre_fprintf(fp, "%d\n", CF_marker[i]);
      }
      fclose(fp);

      iter++;
#endif

      /*------------------------------------------------
       * Test for convergence
       *------------------------------------------------*/

      big_graph_size = (HYPRE_BigInt) graph_size;
      hypre_MPI_Allreduce(&big_graph_size, &global_graph_size, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      if (global_graph_size == 0)
      {
         break;
      }

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/
      if (iter || (CF_init != 1))
      {
         hypre_BoomerAMGIndepSet(S, measure_array, graph_array,
                                 graph_size,
                                 graph_array_offd, graph_offd_size,
                                 CF_marker, CF_marker_offd);
         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                                                       CF_marker_offd, int_buf_data);

            hypre_ParCSRCommHandleDestroy(comm_handle);
         }

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
               if (!int_buf_data[index++] && CF_marker[elmt] > 0)
               {
                  CF_marker[elmt] = 0;
               }
            }
         }
      }

      iter++;
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/


      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
            int_buf_data[index++] = CF_marker[elmt];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    CF_marker_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      for (ig = 0; ig < graph_offd_size; ig++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] < 0)
         {
            /* take point out of the subgraph */
            graph_offd_size--;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig--;
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         hypre_printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",
                      my_id, iter, wall_time);
      }
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

      for (i = num_variables; i < num_variables + num_cols_offd; i++)
      {
         measure_array[i] = 0;
      }

      if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them.
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = C_PT;

            for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
            {
               j = S_diag_j[jS];
               if (j > -1)
               {

                  /* "remove" edge from S */
                  S_diag_j[jS] = -S_diag_j[jS] - 1;

                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j]--;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
            {
               j = S_offd_j[jS];
               if (j > -1)
               {

                  /* "remove" edge from S */
                  S_offd_j[jS] = -S_offd_j[jS] - 1;

                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker_offd[j])
                  {
                     measure_array[j + num_variables]--;
                  }
               }
            }
         }
         else
         {
            /* marked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
            {
               j = S_diag_j[jS];
               if (j < 0) { j = -j - 1; }

               if (CF_marker[j] > 0)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS] - 1;
                  }

                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker[j] = COMMON_C_PT;
               }
               else if (CF_marker[j] == SF_PT)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS] - 1;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
            {
               j = S_offd_j[jS];
               if (j < 0) { j = -j - 1; }

               if (CF_marker_offd[j] > 0)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS] - 1;
                  }

                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker_offd[j] = COMMON_C_PT;
               }
               else if (CF_marker_offd[j] == SF_PT)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS] - 1;
                  }
               }
            }

            /* unmarked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
            {
               if (S_diag_j[jS] > -1)
               {
                  j = S_diag_j[jS];
                  break_var = 1;
                  /* check for common C-pt */
                  for (kS = S_diag_i[j]; kS < S_diag_i[j + 1]; kS++)
                  {
                     k = S_diag_j[kS];
                     if (k < 0) { k = -k - 1; }

                     /* IMPORTANT: consider all dependencies */
                     if (CF_marker[k] == COMMON_C_PT)
                     {
                        /* "remove" edge from S and update measure*/
                        S_diag_j[jS] = -S_diag_j[jS] - 1;
                        measure_array[j]--;
                        break_var = 0;
                        break;
                     }
                  }
                  if (break_var)
                  {
                     for (kS = S_offd_i[j]; kS < S_offd_i[j + 1]; kS++)
                     {
                        k = S_offd_j[kS];
                        if (k < 0) { k = -k - 1; }

                        /* IMPORTANT: consider all dependencies */
                        if ( CF_marker_offd[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = -S_diag_j[jS] - 1;
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
            {
               if (S_offd_j[jS] > -1)
               {
                  j = S_offd_j[jS];

                  /* check for common C-pt */
                  for (kS = S_ext_i[j]; kS < S_ext_i[j + 1]; kS++)
                  {
                     k = (HYPRE_Int)S_ext_j[kS];
                     if (k >= 0)
                     {
                        /* IMPORTANT: consider all dependencies */
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS] - 1;
                           measure_array[j + num_variables]--;
                           break;
                        }
                     }
                     else
                     {
                        kc = -k - 1;
                        if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS] - 1;
                           measure_array[j + num_variables]--;
                           break;
                        }
                     }
                  }
               }
            }
         }

         /* reset CF_marker */
         for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
         {
            j = S_diag_j[jS];
            if (j < 0) { j = -j - 1; }

            if (CF_marker[j] == COMMON_C_PT)
            {
               CF_marker[j] = C_PT;
            }
         }
         for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
         {
            j = S_offd_j[jS];
            if (j < 0) { j = -j - 1; }

            if (CF_marker_offd[j] == COMMON_C_PT)
            {
               CF_marker_offd[j] = C_PT;
            }
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         hypre_printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                      my_id, wall_time, graph_size, num_cols_offd);
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /* Reset S_matrix */
   for (i = 0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
      {
         S_diag_j[i] = -S_diag_j[i] - 1;
      }
   }
   for (i = 0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
      {
         S_offd_j[i] = -S_offd_j[i] - 1;
      }
   }
   /*for (i=0; i < num_variables; i++)
     if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT;*/

   hypre_TFree(measure_array, HYPRE_MEMORY_HOST);
   hypre_TFree(graph_array, HYPRE_MEMORY_HOST);
   if (num_cols_offd) { hypre_TFree(graph_array_offd, HYPRE_MEMORY_HOST); }
   hypre_TFree(buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   if (num_procs > 1) { hypre_CSRMatrixDestroy(S_ext); }

   return hypre_error_flag;
}

/*==========================================================================
 * Ruge's coarsening algorithm
 *==========================================================================*/

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define SC_PT 3  /* special coarse points */
#define UNDECIDED 0


/**************************************************************
 *
 *      Ruge Coarsening routine
 *
 **************************************************************/
HYPRE_Int
hypre_BoomerAMGCoarsenRuge( hypre_ParCSRMatrix    *S,
                            hypre_ParCSRMatrix    *A,
                            HYPRE_Int              measure_type,
                            HYPRE_Int              coarsen_type,
                            HYPRE_Int              cut_factor,
                            HYPRE_Int              debug_flag,
                            hypre_IntArray       **CF_marker_ptr)
{
   MPI_Comm                comm          = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg    *comm_pkg      = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle *comm_handle;
   hypre_CSRMatrix *A_diag        = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *S_diag        = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix *A_offd        = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *S_offd        = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *A_i           = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *S_i           = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_j           = hypre_CSRMatrixJ(S_diag);
   HYPRE_Int       *A_offd_i      = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *S_offd_i      = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j      = NULL;
   HYPRE_Int        num_variables = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int        num_cols_offd = hypre_CSRMatrixNumCols(S_offd);
   HYPRE_BigInt    *col_map_offd  = hypre_ParCSRMatrixColMapOffd(S);

   HYPRE_BigInt     num_nonzeros    = hypre_ParCSRMatrixNumNonzeros(A);
   HYPRE_BigInt     global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int        avg_nnzrow;

   hypre_CSRMatrix *S_ext = NULL;
   HYPRE_Int       *S_ext_i = NULL;
   HYPRE_BigInt    *S_ext_j = NULL;

   hypre_CSRMatrix *ST;
   HYPRE_Int       *ST_i;
   HYPRE_Int       *ST_j;

   HYPRE_Int       *CF_marker;
   HYPRE_Int       *CF_marker_offd = NULL;
   HYPRE_Int        ci_tilde = -1;
   HYPRE_Int        ci_tilde_mark = -1;
   HYPRE_Int        ci_tilde_offd = -1;
   HYPRE_Int        ci_tilde_offd_mark = -1;

   HYPRE_Int       *measure_array;
   HYPRE_Int       *graph_array;
   HYPRE_Int       *int_buf_data = NULL;
   HYPRE_Int       *ci_array = NULL;

   HYPRE_BigInt     big_k;
   HYPRE_Int        i, j, k, jS;
   HYPRE_Int        ji, jj, jk, jm, index;
   HYPRE_Int        set_empty = 1;
   HYPRE_Int        C_i_nonempty = 0;
   HYPRE_Int        cut, nnzrow;
   HYPRE_Int        num_procs, my_id;
   HYPRE_Int        num_sends = 0;
   HYPRE_BigInt     first_col;
   HYPRE_Int        start;
   HYPRE_BigInt     col_0, col_n;

   hypre_LinkList   LoL_head;
   hypre_LinkList   LoL_tail;

   HYPRE_Int       *lists, *where;
   HYPRE_Int        measure, new_meas;
   HYPRE_Int        meas_type = 0;
   HYPRE_Int        agg_2 = 0;
   HYPRE_Int        num_left, elmt;
   HYPRE_Int        nabor, nabor_two;

   HYPRE_Int        use_commpkg_A = 0;
   HYPRE_Int        break_var = 0;
   HYPRE_Int        f_pnt = F_PT;
   HYPRE_Real       wall_time;

   if (coarsen_type < 0)
   {
      coarsen_type = -coarsen_type;
   }
   if (measure_type == 1 || measure_type == 4)
   {
      meas_type = 1;
   }
   if (measure_type == 4 || measure_type == 3)
   {
      agg_2 = 1;
   }

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);
   where = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
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

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   first_col = hypre_ParCSRMatrixFirstColDiag(S);
   col_0 = first_col - 1;
   col_n = col_0 + (HYPRE_BigInt)num_variables;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      use_commpkg_A = 1;
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }

   jS = S_i[num_variables];

   ST = hypre_CSRMatrixCreate(num_variables, num_variables, jS);
   hypre_CSRMatrixMemoryLocation(ST) = HYPRE_MEMORY_HOST;
   ST_i = hypre_CTAlloc(HYPRE_Int, num_variables + 1, HYPRE_MEMORY_HOST);
   ST_j = hypre_CTAlloc(HYPRE_Int, jS, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixI(ST) = ST_i;
   hypre_CSRMatrixJ(ST) = ST_j;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i = 0; i <= num_variables; i++)
   {
      ST_i[i] = 0;
   }
   for (i = 0; i < jS; i++)
   {
      ST_i[S_j[i] + 1]++;
   }
   for (i = 0; i < num_variables; i++)
   {
      ST_i[i + 1] += ST_i[i];
   }
   for (i = 0; i < num_variables; i++)
   {
      for (j = S_i[i]; j < S_i[i + 1]; j++)
      {
         index = S_j[j];
         ST_j[ST_i[index]] = i;
         ST_i[index]++;
      }
   }
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i - 1];
   }
   ST_i[0] = 0;

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are given by the row sums of ST.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    * correct actual measures through adding influences from
    * neighbor processors
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = ST_i[i + 1] - ST_i[i];
   }

   /* special case for Falgout coarsening */
   if (coarsen_type == 6)
   {
      f_pnt = Z_PT;
      coarsen_type = 1;
   }
   if (coarsen_type == 10)
   {
      f_pnt = Z_PT;
      coarsen_type = 11;
   }

   if ((meas_type || (coarsen_type != 1 && coarsen_type != 11)) && num_procs > 1)
   {
      if (use_commpkg_A)
      {
         S_ext      = hypre_ParCSRMatrixExtractBExt(S, A, 0);
      }
      else
      {
         S_ext      = hypre_ParCSRMatrixExtractBExt(S, S, 0);
      }
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixBigJ(S_ext);
      HYPRE_Int num_nonzeros = S_ext_i[num_cols_offd];
      /*first_col = hypre_ParCSRMatrixFirstColDiag(S);
        col_0 = first_col-1;
        col_n = col_0+num_variables; */
      if (meas_type)
      {
         for (i = 0; i < num_nonzeros; i++)
         {
            index = (HYPRE_Int)(S_ext_j[i] - first_col);
            if (index > -1 && index < num_variables)
            {
               measure_array[index]++;
            }
         }
      }
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   /* first coarsening phase */

   /*************************************************************
    *
    *   Initialize the lists
    *
    *************************************************************/

   /* Allocate CF_marker if not done before */
   if (*CF_marker_ptr == NULL)
   {
      *CF_marker_ptr = hypre_IntArrayCreate(num_variables);
      hypre_IntArrayInitialize(*CF_marker_ptr);
   }
   CF_marker = hypre_IntArrayData(*CF_marker_ptr);

   num_left = 0;
   for (j = 0; j < num_variables; j++)
   {
      if (CF_marker[j] == 0)
      {
         nnzrow = (S_i[j + 1] - S_i[j]) + (S_offd_i[j + 1] - S_offd_i[j]);
         if (nnzrow == 0)
         {
            CF_marker[j] = SF_PT;
            if (agg_2)
            {
               CF_marker[j] = SC_PT;
            }
            measure_array[j] = 0;
         }
         else
         {
            CF_marker[j] = UNDECIDED;
            num_left++;
         }
      }
      else
      {
         measure_array[j] = 0;
      }
   }

   /* Set dense rows as SF_PT */
   if ((cut_factor > 0) && (global_num_rows > 0))
   {
      avg_nnzrow = num_nonzeros / global_num_rows;
      cut = cut_factor * avg_nnzrow;
      for (j = 0; j < num_variables; j++)
      {
         nnzrow = (A_i[j + 1] - A_i[j]) + (A_offd_i[j + 1] - A_offd_i[j]);
         if (nnzrow > cut)
         {
            if (CF_marker[j] == UNDECIDED)
            {
               num_left--;
            }
            CF_marker[j] = SF_PT;
         }
      }
   }

   for (j = 0; j < num_variables; j++)
   {
      measure = measure_array[j];
      if (CF_marker[j] != SF_PT && CF_marker[j] != SC_PT)
      {
         if (measure > 0)
         {
            hypre_enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
         }
         else
         {
            if (measure < 0)
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "negative measure!\n");
            }

            CF_marker[j] = f_pnt;
            for (k = S_i[j]; k < S_i[j + 1]; k++)
            {
               nabor = S_j[k];
               if (CF_marker[nabor] != SF_PT && CF_marker[nabor] != SC_PT)
               {
                  if (nabor < j)
                  {
                     new_meas = measure_array[nabor];
                     if (new_meas > 0)
                     {
                        hypre_remove_point(&LoL_head, &LoL_tail, new_meas,
                                           nabor, lists, where);
                     }

                     new_meas = ++(measure_array[nabor]);
                     hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                          nabor, lists, where);
                  }
                  else
                  {
                     new_meas = ++(measure_array[nabor]);
                  }
               }
            }
            --num_left;
         }
      }
   }

   /****************************************************************
    *
    *  Main loop of Ruge-Stueben first coloring pass.
    *
    *  WHILE there are still points to classify DO:
    *        1) find first point, i,  on list with max_measure
    *           make i a C-point, remove it from the lists
    *        2) For each point, j,  in S_i^T,
    *           a) Set j to be an F-point
    *           b) For each point, k, in S_j
    *                  move k to the list in LoL with measure one
    *                  greater than it occupies (creating new LoL
    *                  entry if necessary)
    *        3) For each point, j,  in S_i,
    *                  move j to the list in LoL with measure one
    *                  smaller than it occupies (creating new LoL
    *                  entry if necessary)
    *
    ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;

      CF_marker[index] = C_PT;
      measure = measure_array[index];
      measure_array[index] = 0;
      --num_left;

      hypre_remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);

      for (j = ST_i[index]; j < ST_i[index + 1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = F_PT;
            measure = measure_array[nabor];

            hypre_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;

            for (k = S_i[nabor]; k < S_i[nabor + 1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  hypre_remove_point(&LoL_head, &LoL_tail, measure,
                                     nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);

                  hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                       nabor_two, lists, where);
               }
            }
         }
      }
      for (j = S_i[index]; j < S_i[index + 1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            hypre_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;

            if (measure > 0)
            {
               hypre_enter_on_lists(&LoL_head, &LoL_tail, measure, nabor,
                                    lists, where);
            }
            else
            {
               CF_marker[nabor] = F_PT;
               --num_left;

               for (k = S_i[nabor]; k < S_i[nabor + 1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     hypre_remove_point(&LoL_head, &LoL_tail, new_meas,
                                        nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);

                     hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                          nabor_two, lists, where);
                  }
               }
            }
         }
      }
   }

   hypre_TFree(measure_array, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Coarsen 1st pass = %f\n",
                   my_id, wall_time);
   }

   hypre_TFree(lists, HYPRE_MEMORY_HOST);
   hypre_TFree(where, HYPRE_MEMORY_HOST);
   hypre_TFree(LoL_head, HYPRE_MEMORY_HOST);
   hypre_TFree(LoL_tail, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_variables; i++)
   {
      if (CF_marker[i] == SC_PT)
      {
         CF_marker[i] = C_PT;
      }
   }

   if (coarsen_type == 11)
   {
      if (meas_type && num_procs > 1)
      {
         hypre_CSRMatrixDestroy(S_ext);
      }
      return 0;
   }

   /* second pass, check fine points for coarse neighbors
      for coarsen_type = 2, the second pass includes
      off-processore boundary points */

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   if (coarsen_type == 2)
   {
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                               num_sends), HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    CF_marker_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      ci_array = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_cols_offd; i++)
      {
         ci_array[i] = -1;
      }

      for (i = 0; i < num_variables; i++)
      {
         if (ci_tilde_mark != i) { ci_tilde = -1; }
         if (ci_tilde_offd_mark != i) { ci_tilde_offd = -1; }
         if (CF_marker[i] == -1)
         {
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i + 1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
               {
                  graph_array[j] = i;
               }
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i + 1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
               {
                  ci_array[j] = i;
               }
            }
            for (ji = S_i[i]; ji < S_i[i + 1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_i[j]; jj < S_i[j + 1]; jj++)
                  {
                     index = S_j[jj];
                     if (graph_array[index] == i)
                     {
                        set_empty = 0;
                        break;
                     }
                  }
                  if (set_empty)
                  {
                     for (jj = S_offd_i[j]; jj < S_offd_i[j + 1]; jj++)
                     {
                        index = S_offd_j[jj];
                        if (ci_array[index] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                  }
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = 1;
                        if (ci_tilde > -1)
                        {
                           CF_marker[ci_tilde] = -1;
                           ci_tilde = -1;
                        }
                        if (ci_tilde_offd > -1)
                        {
                           CF_marker_offd[ci_tilde_offd] = -1;
                           ci_tilde_offd = -1;
                        }
                        C_i_nonempty = 0;
                        break_var = 0;
                        break;
                     }
                     else
                     {
                        ci_tilde = j;
                        ci_tilde_mark = i;
                        CF_marker[j] = 1;
                        C_i_nonempty = 1;
                        i--;
                        break_var = 0;
                        break;
                     }
                  }
               }
            }
            if (break_var)
            {
               for (ji = S_offd_i[i]; ji < S_offd_i[i + 1]; ji++)
               {
                  j = S_offd_j[ji];
                  if (CF_marker_offd[j] == -1)
                  {
                     set_empty = 1;
                     for (jj = S_ext_i[j]; jj < S_ext_i[j + 1]; jj++)
                     {
                        big_k = S_ext_j[jj];
                        if (big_k > col_0 && big_k < col_n) /* index interior */
                        {
                           if (graph_array[(HYPRE_Int)(big_k - first_col)] == i)
                           {
                              set_empty = 0;
                              break;
                           }
                        }
                        else
                        {
                           jk = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_offd);
                           if (jk != -1)
                           {
                              if (ci_array[jk] == i)
                              {
                                 set_empty = 0;
                                 break;
                              }
                           }
                        }
                     }
                     if (set_empty)
                     {
                        if (C_i_nonempty)
                        {
                           CF_marker[i] = 1;
                           if (ci_tilde > -1)
                           {
                              CF_marker[ci_tilde] = -1;
                              ci_tilde = -1;
                           }
                           if (ci_tilde_offd > -1)
                           {
                              CF_marker_offd[ci_tilde_offd] = -1;
                              ci_tilde_offd = -1;
                           }
                           C_i_nonempty = 0;
                           break;
                        }
                        else
                        {
                           ci_tilde_offd = j;
                           ci_tilde_offd_mark = i;
                           CF_marker_offd[j] = 1;
                           C_i_nonempty = 1;
                           i--;
                           break;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   else
   {
      for (i = 0; i < num_variables; i++)
      {
         if (ci_tilde_mark != i) { ci_tilde = -1; }
         if (CF_marker[i] == -1)
         {
            for (ji = S_i[i]; ji < S_i[i + 1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
               {
                  graph_array[j] = i;
               }
            }
            for (ji = S_i[i]; ji < S_i[i + 1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_i[j]; jj < S_i[j + 1]; jj++)
                  {
                     index = S_j[jj];
                     if (graph_array[index] == i)
                     {
                        set_empty = 0;
                        break;
                     }
                  }
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = 1;
                        if (ci_tilde > -1)
                        {
                           CF_marker[ci_tilde] = -1;
                           ci_tilde = -1;
                        }
                        C_i_nonempty = 0;
                        break;
                     }
                     else
                     {
                        ci_tilde = j;
                        ci_tilde_mark = i;
                        CF_marker[j] = 1;
                        C_i_nonempty = 1;
                        i--;
                        break;
                     }
                  }
               }
            }
         }
      }
   }

   if (debug_flag == 3 && coarsen_type != 2)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Coarsen 2nd pass = %f\n",
                   my_id, wall_time);
   }

   /* third pass, check boundary fine points for coarse neighbors */

   if (coarsen_type == 3 || coarsen_type == 4)
   {
      if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

      CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                   HYPRE_MEMORY_HOST);

      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    CF_marker_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      ci_array = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_cols_offd; i++)
      {
         ci_array[i] = -1;
      }
   }

   if (coarsen_type > 1 && coarsen_type < 5)
   {
      for (i = 0; i < num_variables; i++)
      {
         graph_array[i] = -1;
      }
      for (i = 0; i < num_cols_offd; i++)
      {
         if (ci_tilde_mark != i) { ci_tilde = -1; }
         if (ci_tilde_offd_mark != i) { ci_tilde_offd = -1; }
         if (CF_marker_offd[i] == -1)
         {
            for (ji = S_ext_i[i]; ji < S_ext_i[i + 1]; ji++)
            {
               big_k = S_ext_j[ji];
               if (big_k > col_0 && big_k < col_n)
               {
                  j = (HYPRE_Int)(big_k - first_col);
                  if (CF_marker[j] > 0)
                  {
                     graph_array[j] = i;
                  }
               }
               else
               {
                  jj = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_offd);
                  if (jj != -1 && CF_marker_offd[jj] > 0)
                  {
                     ci_array[jj] = i;
                  }
               }
            }
            for (ji = S_ext_i[i]; ji < S_ext_i[i + 1]; ji++)
            {
               big_k = S_ext_j[ji];
               if (big_k > col_0 && big_k < col_n)
               {
                  j = (HYPRE_Int)(big_k - first_col);
                  if ( CF_marker[j] == -1)
                  {
                     set_empty = 1;
                     for (jj = S_i[j]; jj < S_i[j + 1]; jj++)
                     {
                        index = S_j[jj];
                        if (graph_array[index] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                     for (jj = S_offd_i[j]; jj < S_offd_i[j + 1]; jj++)
                     {
                        index = S_offd_j[jj];
                        if (ci_array[index] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                     if (set_empty)
                     {
                        if (C_i_nonempty)
                        {
                           CF_marker_offd[i] = 1;
                           if (ci_tilde > -1)
                           {
                              CF_marker[ci_tilde] = -1;
                              ci_tilde = -1;
                           }
                           if (ci_tilde_offd > -1)
                           {
                              CF_marker_offd[ci_tilde_offd] = -1;
                              ci_tilde_offd = -1;
                           }
                           C_i_nonempty = 0;
                           break;
                        }
                        else
                        {
                           ci_tilde = j;
                           ci_tilde_mark = i;
                           CF_marker[j] = 1;
                           C_i_nonempty = 1;
                           i--;
                           break;
                        }
                     }
                  }
               }
               else
               {
                  jm = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_offd);
                  if (jm != -1 && CF_marker_offd[jm] == -1)
                  {
                     set_empty = 1;
                     for (jj = S_ext_i[jm]; jj < S_ext_i[jm + 1]; jj++)
                     {
                        big_k = S_ext_j[jj];
                        if (big_k > col_0 && big_k < col_n)
                        {
                           if (graph_array[(HYPRE_Int)(big_k - first_col)] == i)
                           {
                              set_empty = 0;
                              break;
                           }
                        }
                        else
                        {
                           jk = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_offd);
                           if (jk != -1)
                           {
                              if (ci_array[jk] == i)
                              {
                                 set_empty = 0;
                                 break;
                              }
                           }
                        }
                     }
                     if (set_empty)
                     {
                        if (C_i_nonempty)
                        {
                           CF_marker_offd[i] = 1;
                           if (ci_tilde > -1)
                           {
                              CF_marker[ci_tilde] = -1;
                              ci_tilde = -1;
                           }
                           if (ci_tilde_offd > -1)
                           {
                              CF_marker_offd[ci_tilde_offd] = -1;
                              ci_tilde_offd = -1;
                           }
                           C_i_nonempty = 0;
                           break;
                        }
                        else
                        {
                           ci_tilde_offd = jm;
                           ci_tilde_offd_mark = i;
                           CF_marker_offd[jm] = 1;
                           C_i_nonempty = 1;
                           i--;
                           break;
                        }
                     }
                  }
               }
            }
         }
      }
      /*------------------------------------------------
       * Send boundary data for CF_marker back
       *------------------------------------------------*/
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, CF_marker_offd,
                                                    int_buf_data);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      /* only CF_marker entries from larger procs are accepted
         if coarsen_type = 4 coarse points are not overwritten  */

      index = 0;
      if (coarsen_type != 4)
      {
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            if (hypre_ParCSRCommPkgSendProc(comm_pkg, i) > my_id)
            {
               for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
                  CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)] =
                     int_buf_data[index++];
            }
            else
            {
               index += hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - start;
            }
         }
      }
      else
      {
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            if (hypre_ParCSRCommPkgSendProc(comm_pkg, i) > my_id)
            {
               for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
               {
                  elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
                  if (CF_marker[elmt] != 1)
                  {
                     CF_marker[elmt] = int_buf_data[index];
                  }
                  index++;
               }
            }
            else
            {
               index += hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - start;
            }
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         if (coarsen_type == 4)
            hypre_printf("Proc = %d    Coarsen 3rd pass = %f\n",
                         my_id, wall_time);
         if (coarsen_type == 3)
            hypre_printf("Proc = %d    Coarsen 3rd pass = %f\n",
                         my_id, wall_time);
         if (coarsen_type == 2)
            hypre_printf("Proc = %d    Coarsen 2nd pass = %f\n",
                         my_id, wall_time);
      }
   }
   if (coarsen_type == 5)
   {
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

      CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                               num_sends), HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    CF_marker_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      ci_array = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_cols_offd; i++)
      {
         ci_array[i] = -1;
      }
      for (i = 0; i < num_variables; i++)
      {
         graph_array[i] = -1;
      }

      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == -1 && (S_offd_i[i + 1] - S_offd_i[i]) > 0)
         {
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i + 1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
               {
                  graph_array[j] = i;
               }
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i + 1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
               {
                  ci_array[j] = i;
               }
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i + 1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_ext_i[j]; jj < S_ext_i[j + 1]; jj++)
                  {
                     big_k = S_ext_j[jj];
                     if (big_k > col_0 && big_k < col_n) /* index interior */
                     {
                        if (graph_array[(HYPRE_Int)(big_k - first_col)] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                     else
                     {
                        jk = hypre_BigBinarySearch(col_map_offd, big_k, num_cols_offd);
                        if (jk != -1)
                        {
                           if (ci_array[jk] == i)
                           {
                              set_empty = 0;
                              break;
                           }
                        }
                     }
                  }
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = -2;
                        C_i_nonempty = 0;
                        break;
                     }
                     else
                     {
                        C_i_nonempty = 1;
                        i--;
                        break;
                     }
                  }
               }
            }
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         hypre_printf("Proc = %d    Coarsen special points = %f\n",
                      my_id, wall_time);
      }

   }
   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /*if (coarsen_type != 1)
     { */
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(ci_array, HYPRE_MEMORY_HOST);
   /*} */
   hypre_TFree(graph_array, HYPRE_MEMORY_HOST);
   if ((meas_type || (coarsen_type != 1 && coarsen_type != 11)) && num_procs > 1)
   {
      hypre_CSRMatrixDestroy(S_ext);
   }

   return hypre_error_flag;
}


HYPRE_Int
hypre_BoomerAMGCoarsenFalgout( hypre_ParCSRMatrix  *S,
                               hypre_ParCSRMatrix  *A,
                               HYPRE_Int            measure_type,
                               HYPRE_Int            cut_factor,
                               HYPRE_Int            debug_flag,
                               hypre_IntArray     **CF_marker_ptr)
{
   HYPRE_Int              ierr = 0;

   /*-------------------------------------------------------
    * Perform Ruge coarsening followed by CLJP coarsening
    *-------------------------------------------------------*/

   ierr += hypre_BoomerAMGCoarsenRuge (S, A, measure_type, 6, cut_factor,
                                       debug_flag, CF_marker_ptr);

   ierr += hypre_BoomerAMGCoarsen (S, A, 1, debug_flag, CF_marker_ptr);

   return (ierr);
}


/*--------------------------------------------------------------------------*/

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

/* begin HANS added */
/**************************************************************
 *
 *      Modified Independent Set Coarsening routine
 *          (don't worry about strong F-F connections
 *           without a common C point)
 *
 **************************************************************/
HYPRE_Int
hypre_BoomerAMGCoarsenPMISHost( hypre_ParCSRMatrix    *S,
                                hypre_ParCSRMatrix    *A,
                                HYPRE_Int              CF_init,
                                HYPRE_Int              debug_flag,
                                hypre_IntArray       **CF_marker_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PMIS] -= hypre_MPI_Wtime();
#endif

   MPI_Comm                  comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix          *S_diag          = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   HYPRE_Int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix          *S_offd          = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   HYPRE_Int                *S_offd_j;

   HYPRE_Int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int                 num_cols_offd = 0;

   /* hypre_CSRMatrix       *S_ext;
      HYPRE_Int                *S_ext_i;
      HYPRE_Int                *S_ext_j; */

   HYPRE_Int                 num_sends = 0;
   HYPRE_Int                *int_buf_data;
   HYPRE_Real               *buf_data;

   HYPRE_Int                *CF_marker;
   HYPRE_Int                *CF_marker_offd;

   HYPRE_Real               *measure_array;
   HYPRE_Int                *graph_array;
   HYPRE_Int                *graph_array_offd;
   HYPRE_Int                 graph_size;
   HYPRE_BigInt              big_graph_size;
   HYPRE_Int                 graph_offd_size;
   HYPRE_BigInt              global_graph_size;

   HYPRE_Int                 i, j, jj, jS, ig;
   HYPRE_Int                 index, start, my_id, num_procs, jrow, cnt, elmt;
   HYPRE_Int                 nnzrow;

   HYPRE_Int                 ierr = 0;

   HYPRE_Real                wall_time;
   HYPRE_Int                 iter = 0;

   HYPRE_Int                *prefix_sum_workspace;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   /*******************************************************************************
     BEFORE THE INDEPENDENT SET COARSENING LOOP:
   measure_array: calculate the measures, and communicate them
   (this array contains measures for both local and external nodes)
   CF_marker, CF_marker_offd: initialize CF_marker
   (separate arrays for local and external; 0=unassigned, negative=F point, positive=C point)
    ******************************************************************************/

   /*--------------------------------------------------------------
    * Use the ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: S_data is not used; in stead, only strong columns are retained
    *       in S_j, which can then be used like S_data
    *----------------------------------------------------------------*/

   /*S_ext = NULL; */
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds();
   }

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);
   buf_data     = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *
    * The measures are augmented by a random number
    * between 0 and 1.
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(HYPRE_Real, num_variables + num_cols_offd, HYPRE_MEMORY_HOST);

   /* first calculate the local part of the sums for the external nodes */
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int *measure_array_temp = hypre_CTAlloc(HYPRE_Int,  num_variables + num_cols_offd,
                                                 HYPRE_MEMORY_HOST);

   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   for (i = 0; i < S_offd_i[num_variables]; i++)
   {
      #pragma omp atomic
      measure_array_temp[num_variables + S_offd_j[i]]++;
   }

   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   for (i = 0; i < num_cols_offd; i++)
   {
      measure_array[i + num_variables] = measure_array_temp[i + num_variables];
   }
#else
   for (i = 0; i < S_offd_i[num_variables]; i++)
   {
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }
#endif // HYPRE_USING_OPENMP

   /* now send those locally calculated values for the external nodes to the neighboring processors */
   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, &measure_array[num_variables], buf_data);
   }

   /* calculate the local part for the local nodes */
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   for (i = 0; i < S_diag_i[num_variables]; i++)
   {
      #pragma omp atomic
      measure_array_temp[S_diag_j[i]]++;
   }

   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = measure_array_temp[i];
   }

   hypre_TFree(measure_array_temp, HYPRE_MEMORY_HOST);
#else
   for (i = 0; i < S_diag_i[num_variables]; i++)
   {
      measure_array[S_diag_j[i]] += 1.0;
   }
#endif // HYPRE_USING_OPENMP

   /* finish the communication */
   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /* now add the externally calculated part of the local nodes to the local nodes */
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)] += buf_data[index++];
      }
   }

   /* set the measures of the external nodes to zero */
   for (i = num_variables; i < num_variables + num_cols_offd; i++)
   {
      measure_array[i] = 0;
   }

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   /* this augments the measures */
   if (CF_init == 2 || CF_init == 4)
   {
      hypre_BoomerAMGIndepSetInit(S, measure_array, 1);
   }
   else
   {
      hypre_BoomerAMGIndepSetInit(S, measure_array, 0);
   }

   /*---------------------------------------------------
    * Initialize the graph arrays, and CF_marker arrays
    *---------------------------------------------------*/

   /* first the off-diagonal part of the graph array */
   if (num_cols_offd)
   {
      graph_array_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
   }
   else
   {
      graph_array_offd = NULL;
   }

   for (ig = 0; ig < num_cols_offd; ig++)
   {
      graph_array_offd[ig] = ig;
   }

   graph_offd_size = num_cols_offd;

   /* now the local part of the graph array, and the local CF_marker array */
   graph_array = hypre_CTAlloc(HYPRE_Int, num_variables, HYPRE_MEMORY_HOST);

   /* Allocate CF_marker if not done before */
   if (*CF_marker_ptr == NULL)
   {
      *CF_marker_ptr = hypre_IntArrayCreate(num_variables);
      hypre_IntArrayInitialize(*CF_marker_ptr);
   }
   CF_marker = hypre_IntArrayData(*CF_marker_ptr);

   if (CF_init == 1)
   {
      cnt = 0;
      for (i = 0; i < num_variables; i++)
      {
         if ( CF_marker[i] != SF_PT )
         {
            if ( S_offd_i[i + 1] - S_offd_i[i] > 0 || CF_marker[i] == -1 )
            {
               CF_marker[i] = 0;
            }
            if ( CF_marker[i] == Z_PT)
            {
               if ( measure_array[i] >= 1.0 || S_diag_i[i + 1] - S_diag_i[i] > 0 )
               {
                  CF_marker[i] = 0;
                  graph_array[cnt++] = i;
               }
               else
               {
                  CF_marker[i] = F_PT;
               }
            }
            else
            {
               graph_array[cnt++] = i;
            }
         }
         else
         {
            measure_array[i] = 0;
         }
      }
   }
   else
   {
      cnt = 0;
      for (i = 0; i < num_variables; i++)
      {
         CF_marker[i] = 0;
         nnzrow = (S_diag_i[i + 1] - S_diag_i[i]) + (S_offd_i[i + 1] - S_offd_i[i]);
         if (nnzrow == 0)
         {
            CF_marker[i] = SF_PT; /* an isolated fine grid */
            if (CF_init == 3 || CF_init == 4)
            {
               CF_marker[i] = C_PT;
            }
            measure_array[i] = 0;
         }
         else
         {
            graph_array[cnt++] = i;
         }
      }
   }

   graph_size = cnt;

   /* now the off-diagonal part of CF_marker */
   if (num_cols_offd)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
   }
   else
   {
      CF_marker_offd = NULL;
   }

   for (i = 0; i < num_cols_offd; i++)
   {
      CF_marker_offd[i] = 0;
   }

   /*------------------------------------------------
    * Communicate the local measures, which are complete,
    to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         buf_data[index++] = measure_array[jrow];
      }
   }

   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, &measure_array[num_variables]);
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Initialize CLJP phase = %f\n", my_id, wall_time);
   }

   /* graph_array2 */
   HYPRE_Int *graph_array2 = hypre_CTAlloc(HYPRE_Int, num_variables, HYPRE_MEMORY_HOST);
   HYPRE_Int *graph_array_offd2 = NULL;
   if (num_cols_offd)
   {
      graph_array_offd2 = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
   }

   /*******************************************************************************
     THE INDEPENDENT SET COARSENING LOOP:
    ******************************************************************************/

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/
   while (1)
   {
      big_graph_size = (HYPRE_BigInt) graph_size;

      /* stop the coarsening if nothing left to be coarsened */
      hypre_MPI_Allreduce(&big_graph_size, &global_graph_size, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      /* if (my_id == 0) { hypre_printf("graph size %b\n", global_graph_size); } */

      if (global_graph_size == 0)
      {
         break;
      }

      /*
         hypre_printf("\n");
         hypre_printf("*** MIS iteration %d\n",iter);
         hypre_printf("graph_size remaining %d\n",graph_size);
         */

      /*-----------------------------------------------------------------------------------------
       * Pick an independent set of points with maximal measure
       * At the end, CF_marker is complete, but still needs to be communicated to CF_marker_offd
       * for CF_init == 1, as in HMIS, the first IS was fed from prior R-S coarsening
       *----------------------------------------------------------------------------------------*/
      if (!CF_init || iter)
      {
         /*
            hypre_BoomerAMGIndepSet(S, measure_array, graph_array, graph_size,
            graph_array_offd, graph_offd_size, CF_marker, CF_marker_offd);
            */

#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(ig, i) HYPRE_SMP_SCHEDULE
#endif
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];
            if (measure_array[i] > 1)
            {
               CF_marker[i] = 1;
            }
         }

#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(ig, i) HYPRE_SMP_SCHEDULE
#endif
         for (ig = 0; ig < graph_offd_size; ig++)
         {
            i = graph_array_offd[ig];
            if (measure_array[i + num_variables] > 1)
            {
               CF_marker_offd[i] = 1;
            }
         }

         /*-------------------------------------------------------
          * Remove nodes from the initial independent set
          *-------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(ig, i, jS, j, jj) HYPRE_SMP_SCHEDULE
#endif
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];

            if (measure_array[i] > 1)
            {
               /* for each local neighbor j of i */
               for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
               {
                  j = S_diag_j[jS];
                  if (measure_array[j] > 1)
                  {
                     if (measure_array[i] > measure_array[j])
                     {
                        CF_marker[j] = 0;
                     }
                     else if (measure_array[j] > measure_array[i])
                     {
                        CF_marker[i] = 0;
                     }
                  }
               }

               /* for each offd neighbor j of i */
               for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
               {
                  jj = S_offd_j[jS];
                  j = num_variables + jj;
                  if (measure_array[j] > 1)
                  {
                     if (measure_array[i] > measure_array[j])
                     {
                        CF_marker_offd[jj] = 0;
                     }
                     else if (measure_array[j] > measure_array[i])
                     {
                        CF_marker[i] = 0;
                     }
                  }
               }
            } /* for each node with measure > 1 */
         } /* for each node i */

         /*------------------------------------------------------------------------------
          * Exchange boundary data for CF_marker: send external CF to internal CF
          *------------------------------------------------------------------------------*/
         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, CF_marker_offd, int_buf_data);
            hypre_ParCSRCommHandleDestroy(comm_handle);
         }

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
               if (!int_buf_data[index] && CF_marker[elmt] > 0)
               {
                  CF_marker[elmt] = 0;
                  index++;
               }
               else
               {
                  int_buf_data[index++] = CF_marker[elmt];
               }
            }
         }

         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
            hypre_ParCSRCommHandleDestroy(comm_handle);
         }
      } /* if (!CF_init || iter) */

      iter++;

      /*------------------------------------------------
       * Set C-pts and F-pts.
       *------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(ig, i, jS, j) HYPRE_SMP_SCHEDULE
#endif
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * If the measure of i is smaller than 1, then
          * make i and F point (because it does not influence
          * any other point)
          *---------------------------------------------*/

         if (measure_array[i] < 1)
         {
            CF_marker[i] = F_PT;
         }

         /*---------------------------------------------
          * First treat the case where point i is in the
          * independent set: make i a C point,
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            CF_marker[i] = C_PT;
         }
         /*---------------------------------------------
          * Now treat the case where point i is not in the
          * independent set: loop over
          * all the points j that influence equation i; if
          * j is a C point, then make i an F point.
          *---------------------------------------------*/
         else
         {
            /* first the local part */
            for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
            {
               /* j is the column number, or the local number of the point influencing i */
               j = S_diag_j[jS];
               if (CF_marker[j] > 0) /* j is a C-point */
               {
                  CF_marker[i] = F_PT;
               }
            }
            /* now the external part */
            for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
            {
               j = S_offd_j[jS];
               if (CF_marker_offd[j] > 0) /* j is a C-point */
               {
                  CF_marker[i] = F_PT;
               }
            }
         } /* end else */
      } /* end first loop over graph */

      /* now communicate CF_marker to CF_marker_offd, to make
         sure that new external F points are known on this processor */

      /*------------------------------------------------------------------------------
       * Exchange boundary data for CF_marker: send internal points to external points
       *------------------------------------------------------------------------------*/
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      /*------------------------------------------------
       * Update subgraph
       *------------------------------------------------*/

      /*HYPRE_Int prefix_sum_workspace[2*(hypre_NumThreads() + 1)];*/
      prefix_sum_workspace = hypre_TAlloc(HYPRE_Int, 2 * (hypre_NumThreads() + 1), HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel private(ig,i)
#endif
      {
         HYPRE_Int private_graph_size_cnt = 0;
         HYPRE_Int private_graph_offd_size_cnt = 0;

         HYPRE_Int ig_begin, ig_end;
         hypre_GetSimpleThreadPartition(&ig_begin, &ig_end, graph_size);

         HYPRE_Int ig_offd_begin, ig_offd_end;
         hypre_GetSimpleThreadPartition(&ig_offd_begin, &ig_offd_end, graph_offd_size);

         for (ig = ig_begin; ig < ig_end; ig++)
         {
            i = graph_array[ig];

            if (CF_marker[i] != 0) /* C or F point */
            {
               /* the independent set subroutine needs measure 0 for removed nodes */
               measure_array[i] = 0;
            }
            else
            {
               private_graph_size_cnt++;
            }
         }

         for (ig = ig_offd_begin; ig < ig_offd_end; ig++)
         {
            i = graph_array_offd[ig];

            if (CF_marker_offd[i] != 0) /* C of F point */
            {
               /* the independent set subroutine needs measure 0 for removed nodes */
               measure_array[i + num_variables] = 0;
            }
            else
            {
               private_graph_offd_size_cnt++;
            }
         }

         hypre_prefix_sum_pair(&private_graph_size_cnt, &graph_size, &private_graph_offd_size_cnt,
                               &graph_offd_size, prefix_sum_workspace);

         for (ig = ig_begin; ig < ig_end; ig++)
         {
            i = graph_array[ig];
            if (CF_marker[i] == 0)
            {
               graph_array2[private_graph_size_cnt++] = i;
            }
         }

         for (ig = ig_offd_begin; ig < ig_offd_end; ig++)
         {
            i = graph_array_offd[ig];
            if (CF_marker_offd[i] == 0)
            {
               graph_array_offd2[private_graph_offd_size_cnt++] = i;
            }
         }
      } /* omp parallel */

      HYPRE_Int *temp = graph_array;
      graph_array = graph_array2;
      graph_array2 = temp;

      temp = graph_array_offd;
      graph_array_offd = graph_array_offd2;
      graph_array_offd2 = temp;

      hypre_TFree(prefix_sum_workspace, HYPRE_MEMORY_HOST);

   } /* end while */

   /*
      hypre_printf("*** MIS iteration %d\n",iter);
      hypre_printf("graph_size remaining %d\n",graph_size);

      hypre_printf("num_cols_offd %d\n",num_cols_offd);
      for (i=0;i<num_variables;i++)
      {
      if(CF_marker[i] == 1)
      {
      hypre_printf("node %d CF %d\n",i,CF_marker[i]);
      }
      }
      */


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/
   hypre_TFree(measure_array, HYPRE_MEMORY_HOST);
   hypre_TFree(graph_array, HYPRE_MEMORY_HOST);
   hypre_TFree(graph_array2, HYPRE_MEMORY_HOST);
   hypre_TFree(graph_array_offd2, HYPRE_MEMORY_HOST);
   if (num_cols_offd)
   {
      hypre_TFree(graph_array_offd, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   /*if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);*/

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PMIS] += hypre_MPI_Wtime();
#endif

   return (ierr);
}

HYPRE_Int
hypre_BoomerAMGCoarsenPMIS( hypre_ParCSRMatrix    *S,
                            hypre_ParCSRMatrix    *A,
                            HYPRE_Int              CF_init,
                            HYPRE_Int              debug_flag,
                            hypre_IntArray       **CF_marker_ptr)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_GpuProfilingPushRange("PMIS");
#endif

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGCoarsenPMISDevice( S, A, CF_init, debug_flag, CF_marker_ptr );
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGCoarsenPMISHost( S, A, CF_init, debug_flag, CF_marker_ptr );
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_GpuProfilingPopRange();
#endif

   return ierr;
}

HYPRE_Int
hypre_BoomerAMGCoarsenHMIS( hypre_ParCSRMatrix    *S,
                            hypre_ParCSRMatrix    *A,
                            HYPRE_Int              measure_type,
                            HYPRE_Int              cut_factor,
                            HYPRE_Int              debug_flag,
                            hypre_IntArray       **CF_marker_ptr)
{
   HYPRE_Int              ierr = 0;

   /*-------------------------------------------------------
    * Perform Ruge coarsening followed by CLJP coarsening
    *-------------------------------------------------------*/

   ierr += hypre_BoomerAMGCoarsenRuge (S, A, measure_type, 10, cut_factor,
                                       debug_flag, CF_marker_ptr);

   ierr += hypre_BoomerAMGCoarsenPMISHost (S, A, 1, debug_flag, CF_marker_ptr);

   return (ierr);
}
