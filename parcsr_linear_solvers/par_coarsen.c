/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "headers.h"



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
  headers.h

  @return Error code.
  
  @param A [IN]
  coefficient matrix
  @param strength_threshold [IN]
  threshold parameter used to define strength
  @param S_ptr [OUT]
  strength matrix
  @param CF_marker_ptr [OUT]
  array indicating C/F points
  @param coarse_size_ptr [OUT]
  size of the coarse grid
  
  @see */
/*--------------------------------------------------------------------------*/

#define C_PT  1
#define F_PT -1
#define COMMON_C_PT  2

int
hypre_ParAMGCoarsen( hypre_ParCSRMatrix    *A,
                     double                 strength_threshold,
                     double                 max_row_sum,
                     int                    debug_flag,
                     hypre_ParCSRMatrix   **S_ptr,
                     int                  **CF_marker_ptr,
                     int                   *coarse_size_ptr     )
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommPkg      *comm_pkg_mS;
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   double             *A_offd_data;

   int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
   int 		      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   int                 global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   int		       col_1 = hypre_ParCSRMatrixFirstColDiag(A);
   int		       col_n = col_1 + hypre_CSRMatrixNumCols(A_diag);
   int 		       num_nonzeros_diag;
   int 		       num_nonzeros_offd = 0;
   int 		       num_cols_offd = 0;
                  
   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   int                *S_diag_i;
   int                *S_diag_j;
   /* double             *S_diag_data; */
   hypre_CSRMatrix    *S_offd;
   int                *S_offd_i;
   int                *S_offd_j;
   /* double             *S_offd_data; */
                 
   hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j;
   /* double             *S_ext_data; */

   int		       num_sends = 0;
   int		       num_recvs = 0;
   int  	      *int_buf_data;
   int  	      *S_recv_vec_starts;
   int  	      *S_send_map_starts;
   double	      *buf_data;
   int		      *S_buf_j;

   int                *CF_marker;
   int                *CF_marker_offd;
   int                 coarse_size;
                      
   /* hypre_ParVector    *ones_vector;
   hypre_ParVector    *measure_vector; */
   double             *measure_array;
   int                *graph_array;
   int                 graph_size;
   int                 global_graph_size;
                      
   double              diag, row_scale, row_sum;
   int                 i, j, k, ic, jc, kc, jj, kk, jA, jS, kS, ig;
   int		       index, index_S, start, my_id, num_procs, jrow;
                      
   int                 ierr = 0;
   int                 break_var = 1;

   int		       num_data, start_index;
   int		       *recv_vec_starts;
   double	    wall_time;
   double	    wall_time_ip = 0;
   double	    wall_time_bp = 0;
   double	    wall_time_rs = 0;
   double	    sum_time_ip = 0;
   double	    sum_time_bp = 0;
   double	    sum_time_rs = 0;
   int   iter = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
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
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

   comm_pkg_mS = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(comm_pkg_mS) = comm;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg_mS) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg_mS) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgNumSends(comm_pkg_mS) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg_mS) = hypre_ParCSRCommPkgSendProcs(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   A_offd_i = hypre_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
			row_starts, row_starts,
			num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
/* row_starts is owned by A, col_starts = row_starts */
   hypre_ParCSRMatrixSetRowStartsOwner(S,0);
   /* hypre_ParCSRMatrixInitialize(S); */
   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(int, num_variables+1);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(int, num_nonzeros_diag);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(int, num_variables+1);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   /* S_diag_data = hypre_CSRMatrixData(S_diag); */
   S_offd_i = hypre_CSRMatrixI(S_offd);

   if (num_cols_offd)
   {
        A_offd_data = hypre_CSRMatrixData(A_offd);
        hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(int, num_nonzeros_offd);
        S_offd_j = hypre_CSRMatrixJ(S_offd);
        /* S_offd_data = hypre_CSRMatrixData(S_offd); */
        hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(int, num_cols_offd);
   }

   /* give S same nonzero structure as A */
   hypre_ParCSRMatrixCopy(A,S,0);

   for (i = 0; i < num_variables; i++)
   {
      diag = A_diag_data[A_diag_i[i]];

      /* compute scaling factor and row sum */
      row_scale = 0.0;
      row_sum = diag;
      if (diag < 0)
      {
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            row_scale = hypre_max(row_scale, A_diag_data[jA]);
            row_sum += A_diag_data[jA];
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = hypre_max(row_scale, A_offd_data[jA]);
            row_sum += A_offd_data[jA];
         }
      }
      else
      {
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            row_scale = hypre_min(row_scale, A_diag_data[jA]);
            row_sum += A_diag_data[jA];
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = hypre_min(row_scale, A_offd_data[jA]);
            row_sum += A_offd_data[jA];
         }
      }
      row_sum = fabs( row_sum / diag );

      /* compute row entries of S */
      S_diag_j[A_diag_i[i]] = -1;
      if ((row_sum > max_row_sum) && (max_row_sum < 1.0))
      {
         /* make all dependencies weak */
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            S_diag_j[jA] = -1;
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            S_offd_j[jA] = -1;
         }
      }
      else
      {
         if (diag < 0) 
         { 
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               if (A_diag_data[jA] <= strength_threshold * row_scale)
               {
                  S_diag_j[jA] = -1;
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               if (A_offd_data[jA] <= strength_threshold * row_scale)
               {
                  S_offd_j[jA] = -1;
               }
            }
         }
         else
         {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               if (A_diag_data[jA] >= strength_threshold * row_scale)
               {
                  S_diag_j[jA] = -1;
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               if (A_offd_data[jA] >= strength_threshold * row_scale)
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

   jS = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_diag_i[i] = jS;
      for (jA = A_diag_i[i]; jA < A_diag_i[i+1]; jA++)
      {
         if (S_diag_j[jA] > -1)
         {
            S_diag_j[jS]    = S_diag_j[jA];
            jS++;
         }
      }
   }
   S_diag_i[num_variables] = jS;
   hypre_CSRMatrixNumNonzeros(S_diag) = jS;

   jS = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_offd_i[i] = jS;
      for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
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
/*
   ones_vector = hypre_ParVectorCreate(comm,global_num_vars,row_starts);
   hypre_ParVectorSetPartitioningOwner(ones_vector,0);
   hypre_ParVectorInitialize(ones_vector);
   hypre_ParVectorSetConstantValues(ones_vector,-1.0);

   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);
   measure_vector = hypre_ParVectorCreate(comm,global_num_vars,row_starts);
   hypre_ParVectorSetPartitioningOwner(measure_vector,0);
   hypre_VectorData(hypre_ParVectorLocalVector(measure_vector))=measure_array;

   hypre_ParCSRMatrixCommPkg(S) = hypre_ParCSRMatrixCommPkg(A);

   hypre_ParCSRMatrixMatvecT(1.0, S, ones_vector, 0.0, measure_vector);
*/
   hypre_ParCSRMatrixCommPkg(S) = NULL;

   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);

   for (i=0; i < S_offd_i[num_variables]; i++)
   { 
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }
   if (num_procs > 1)
   comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, 
                        &measure_array[num_variables], buf_data);

   for (i=0; i < S_diag_i[num_variables]; i++)
   { 
      measure_array[S_diag_j[i]] += 1.0;
   }

   if (num_procs > 1)
   hypre_ParCSRCommHandleDestroy(comm_handle);
      
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
   }

   /* this augments the measures */
   hypre_InitParAMGIndepSet(S, measure_array);

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   graph_array   = hypre_CTAlloc(int, num_variables+num_cols_offd);

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_variables+num_cols_offd; ig++)
      graph_array[ig] = ig;

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ... 
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(int, num_variables);
   if (num_cols_offd)
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      CF_marker_offd = NULL;
   for (i=0; i < num_variables; i++)
	CF_marker[i] = 0;
   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   coarse_size = 0;
   graph_size = num_variables+num_cols_offd;
   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
      /* S_ext_data = hypre_CSRMatrixData(S_ext); */
   }

   S_send_map_starts = hypre_CTAlloc(int, num_sends+1);

   num_data = 0;
   S_send_map_starts[0] = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         num_data += S_diag_i[jrow+1] - S_diag_i[jrow]
         		+ S_offd_i[jrow+1] - S_offd_i[jrow];
      }
      S_send_map_starts[i+1] = num_data;
   }

   S_buf_j = hypre_CTAlloc(int, num_data);

   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_mS) = S_send_map_starts;   

   S_recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);
   S_recv_vec_starts[0] = 0;
   for (i=0; i < num_recvs; i++)
   {
      S_recv_vec_starts[i+1] = S_ext_i[recv_vec_starts[i+1]];
   }
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_mS) = S_recv_vec_starts;   
 
   
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time); 
   }
   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      index = 0;
      index_S = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            buf_data[index++] = measure_array[jrow];
	    for (k = S_diag_i[jrow]; k < S_diag_i[jrow+1]; k++)
	    {
               if (S_diag_j[k] > -1)
                  S_buf_j[index_S++] = S_diag_j[k]+col_1;
               else
                  S_buf_j[index_S++] = S_diag_j[k]-col_1;
            }
	    for (k = S_offd_i[jrow]; k < S_offd_i[jrow+1]; k++)
	    {
               if (S_offd_j[k] > -1)
                  S_buf_j[index_S++] = col_map_offd[S_offd_j[k]];
               else
                  S_buf_j[index_S++] = -col_map_offd[-S_offd_j[k]-1]-1;
            }
         }
      }

      if (num_procs > 1)
      { 
      comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, 
        &measure_array[num_variables]);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
 
 
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg_mS, S_buf_j,
			S_ext_j);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      } 
      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/
 
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

	 if (i < num_variables)
	 { 
            if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) )
            {
               /* set to be an F-pt */
               CF_marker[i] = F_PT;
 
	       /* make sure all dependencies have been accounted for */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
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
	 else
	 {
            ic = i - num_variables;
            if ( (CF_marker_offd[ic] != C_PT) && (measure_array[i] < 1) )
            {
               /* set to be an F-pt */
               CF_marker_offd[ic] = F_PT;
 
               for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
               {
                  if (S_ext_j[jS] > -1)
                  {
                     CF_marker_offd[ic] = 0;
                  }
               }
            }
 
            if (CF_marker_offd[ic])
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
       * Debugging:
       *
       * Uncomment the sections of code labeled
       * "debugging" to generate several files that
       * can be visualized using the `coarsen.m'
       * matlab routine.
       *------------------------------------------------*/

#if 0 /* debugging */
      /* print out measures */
      sprintf(filename, "coarsen.out.measures.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%f\n", measure_array[i]);
      }
      fclose(fp);

      /* print out strength matrix */
      sprintf(filename, "coarsen.out.strength.%04d", iter);
      hypre_CSRMatrixPrint(S, filename);

      /* print out C/F marker */
      sprintf(filename, "coarsen.out.CF.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%d\n", CF_marker[i]);
      }
      fclose(fp);

      iter++;
#endif

      /*------------------------------------------------
       * Test for convergence
       *------------------------------------------------*/

      MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

      if (global_graph_size == 0)
         break;

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/

      hypre_ParAMGIndepSet(S, S_ext, measure_array, graph_array, graph_size, 
				CF_marker, CF_marker_offd);

      iter++;
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++] 
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      }
 
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",
                     my_id, iter, wall_time); 
      }
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them.
          *---------------------------------------------*/

         if (i < num_variables)  /*interior points */
	 {
            if (CF_marker[i] > 0)
            {  
               /* set to be a C-pt */
               CF_marker[i] = C_PT;
	       coarse_size++;

               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  j = S_diag_j[jS];
                  if (j > -1)
                  {
               
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
               
                     /* decrement measures of unmarked neighbors */
                     if (!CF_marker[j])
                     {
                        measure_array[j]--;
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  j = S_offd_j[jS];
                  if (j > -1)
                  {
               
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
               
                     /* decrement measures of unmarked neighbors */
                     if (!CF_marker_offd[j])
                     {
                        measure_array[j+num_variables]--;
                     }
                  }
               }
            }
	    else
    	    {
               /* marked dependencies */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  j = S_diag_j[jS];
		  if (j < 0) j = -j-1;
   
                  if (CF_marker[j] > 0)
                  {
                     if (S_diag_j[jS] > -1)
                     {
                        /* "remove" edge from S */
                        S_diag_j[jS] = -S_diag_j[jS]-1;
                     }
   
                     /* IMPORTANT: consider all dependencies */
                     /* if (S_diag_data[jS]) */
                     {
                        /* temporarily modify CF_marker */
                        CF_marker[j] = COMMON_C_PT;
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  j = S_offd_j[jS];
		  if (j < 0) j = -j-1;
   
                  if (CF_marker_offd[j] > 0)
                  {
                     if (S_offd_j[jS] > -1)
                     {
                        /* "remove" edge from S */
                        S_offd_j[jS] = -S_offd_j[jS]-1;
                     }
   
                     /* IMPORTANT: consider all dependencies */
                     /* if (S_offd_data[jS]) */
                     {
                        /* temporarily modify CF_marker */
                        CF_marker_offd[j] = COMMON_C_PT;
                     }
                  }
               }
   
               /* unmarked dependencies */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     j = S_diag_j[jS];
   		     break_var = 1;
                     /* check for common C-pt */
                     for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++)
                     {
                        k = S_diag_j[kS];
			if (k < 0) k = -k-1;
   
                        /* IMPORTANT: consider all dependencies */
                        /* if (S_diag_data[kS] && CF_marker[k] == COMMON_C_PT)*/
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = -S_diag_j[jS]-1;
                           measure_array[j]--;
                           break_var = 0;
                           break;
                        }
                     }
   		     if (break_var)
                     {
                        for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS++)
                        {
                           k = S_offd_j[kS];
			   if (k < 0) k = -k-1;
   
                           /* IMPORTANT: consider all dependencies */
                           /* if (S_offd_data[kS] &&
   				CF_marker_offd[k] == COMMON_C_PT) */
                           if ( CF_marker_offd[k] == COMMON_C_PT)
                           {
                              /* "remove" edge from S and update measure*/
                              S_diag_j[jS] = -S_diag_j[jS]-1;
                              measure_array[j]--;
                              break;
                           }
                        }
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     j = S_offd_j[jS];
   
                     /* check for common C-pt */
                     for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++)
                     {
                        k = S_ext_j[kS];
			if (k < 0) k = -k-1;
   		        if (k >= col_1 && k < col_n)
   		        {
   			   kc = k - col_1;
   
                           /* IMPORTANT: consider all dependencies */
                        /*if (S_ext_data[kS] && CF_marker[kc] == COMMON_C_PT)*/
                           if (CF_marker[kc] == COMMON_C_PT)
                           {
                              /* "remove" edge from S and update measure*/
                              S_offd_j[jS] = -S_offd_j[jS]-1;
                              measure_array[j+num_variables]--;
                              break;
                           }
                        }
   		        else
   		        {
   		           kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
   		           /* if (kc > -1 && S_ext_data[kS] && 
   				CF_marker_offd[kc] == COMMON_C_PT) */
   		           if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
   		           {
                              /* "remove" edge from S and update measure*/
                              S_offd_j[jS] = -S_offd_j[jS]-1;
                              measure_array[j+num_variables]--;
                              break;
   		           }
   		        }
                     }
                  }
               }
            }
    	 }

	 else /* boundary points */
	 {
	    ic = i - num_variables;
            if (CF_marker_offd[ic] > 0)
            {  
               /* set to be a C-pt */
               CF_marker_offd[ic] = C_PT;

               for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
               {
                  j = S_ext_j[jS];
                  if (j > -1)
                  {
	       	     if (j >= col_1 && j < col_n)
		     {
		        jj = j-col_1;
		     }
		     else
		     {
   		        jj = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
			if (jj != -1) jj += num_variables;
		     }
               
                     /* "remove" edge from S */
                     S_ext_j[jS] = -S_ext_j[jS]-1;
               
                     /* decrement measures of unmarked neighbors */
                     if (jj < num_variables)
		     {
			if (jj != -1 && !CF_marker[jj])
                        {
                           measure_array[jj]--;
                        }
                     }
		     else
		     {
			if (!CF_marker_offd[jj-num_variables])
                        {
                           measure_array[jj]--;
                        }
                     }
                  }
               }
            }
	    else

            /*---------------------------------------------
             * Heuristic: points that interpolate from a
             * common C-pt are less dependent on each other.
             *
             * NOTE: CF_marker is used to help check for
             * common C-pt's in the heuristic.
             *---------------------------------------------*/

 	    {
	       ic = i - num_variables;
               for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
               {
                  j = S_ext_j[jS];
		  if (j < 0) j = -j-1;
	          if (j >= col_1 && j < col_n)
	          {
	             jc = j - col_1;
                     if (CF_marker[jc] > 0)
                     {
                        if (S_ext_j[jS] > -1)
                        {
                           /* "remove" edge from S */
                           S_ext_j[jS] = -S_ext_j[jS]-1;
                        }

                        /* IMPORTANT: consider all dependencies */
                        /* if (S_ext_data[jS]) */
                        {
                           /* temporarily modify CF_marker */
                           CF_marker[jc] = COMMON_C_PT;
                        }
                     }
	          }
	          else
	          {
   		     jj = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
		     if (jj != -1)
		     {
                        if (CF_marker_offd[jj] > 0)
                  	{
                     	   if (S_ext_j[jS] > -1)
                     	   {
                              /* "remove" edge from S */
                       	      S_ext_j[jS] = -S_ext_j[jS]-1;
                     	   }

                     	   /* IMPORTANT: consider all dependencies */
                     	   /* if (S_ext_data[jS]) */
                     	   {
                              /* temporarily modify CF_marker */
                              CF_marker_offd[jj] = COMMON_C_PT;
                  	   }
                        }
                     }
	          }
               }

               /* unmarked dependencies */
               for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
               {
                  j = S_ext_j[jS];
                  if (j > -1)
                  {

                     /* check for common C-pt */
		     if (j >= col_1 && j < col_n)
		     {
		        jc = j - col_1;
		        break_var = 1;
                        for (kS = S_diag_i[jc]; kS < S_diag_i[jc+1]; kS++)
                        {
                           k = S_diag_j[kS];
                           if (k < 0) k = -k-1;

                           /* IMPORTANT: consider all dependencies */
                           /* if (S_diag_data[kS]) */
                           {
                              if (CF_marker[k] == COMMON_C_PT)
                              {
                                 /* "remove" edge from S and update measure*/
                                 S_ext_j[jS] = -S_ext_j[jS]-1;
                                 measure_array[jc]--;
                                 break_var = 0;
                                 break;
                              }
                           }
                        }
		        if (break_var)
                        {
                           for (kS = S_offd_i[jc]; kS < S_offd_i[jc+1]; kS++)
                           {
                              k = S_offd_j[kS];
			      if (k < 0) k = -k-1;

                              /* IMPORTANT: consider all dependencies */
                              /* if (S_offd_data[kS]) */
                              {
                                 if (CF_marker_offd[k] == COMMON_C_PT)
                                 {
                                    /* "remove" edge from S and update measure*/
                                    S_ext_j[jS] = -S_ext_j[jS]-1;
                                    measure_array[jc]--;
                                    break;
                                 }
                              }
                           }
                        }
                     }
		     else
		     {
   		        jc = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
		        if (jc > -1 )
                        {
		           for (kS = S_ext_i[jc]; kS < S_ext_i[jc+1]; kS++)
                           {
                      	      k = S_ext_j[kS];
			      if (k < 0) k = -k-1;

                      	      /* IMPORTANT: consider all dependencies */
                      	    /* if (k >= col_1 && k < col_n && S_ext_data[kS])*/
                      	      if (k >= col_1 && k < col_n)
                      	      {
                                 if (CF_marker[k-col_1] == COMMON_C_PT)
                                 {
                                    /* "remove" edge from S and update measure*/
                                    S_ext_j[jS] = -S_ext_j[jS]-1;
                                    measure_array[jc+num_variables]--;
                                    break;
                                 }
                              }
			      else
			      {
   		                 kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
			         if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
			         {
                                    /* "remove" edge from S and update measure*/
                                    S_ext_j[jS] = -S_ext_j[jS]-1;
                                    measure_array[jc+num_variables]--;
			            break;
			         }
                              }
                           }
                        }
                     }
                  }
               }
 	    }
         }

         /* reset CF_marker */
         if (i < num_variables)
	 {
	    for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
	    {
               j = S_diag_j[jS];
	       if (j < 0) j = -j-1;

               if (CF_marker[j] == COMMON_C_PT)
               {
                  CF_marker[j] = C_PT;
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
	       if (j < 0) j = -j-1;

               if (CF_marker_offd[j] == COMMON_C_PT)
               {
                  CF_marker_offd[j] = C_PT;
               }
            }
         }
	 else
	 {
	    ic = i - num_variables;
	    for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
	    {
               j = S_ext_j[jS];
	       if (j < 0) j = -j-1;
	       if (j >= col_1 && j < col_n &&
			CF_marker[j - col_1] == COMMON_C_PT)
               {
                  CF_marker[j - col_1] = C_PT;
               }
	       else
	       {
   		  jj = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
	          if (jj != -1 && CF_marker_offd[jj] == COMMON_C_PT)
                           CF_marker_offd[jj] = C_PT;
               }
            }
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                     my_id, wall_time, graph_size, num_cols_offd); 
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }

   /* hypre_ParVectorDestroy(ones_vector); 
   hypre_ParVectorDestroy(measure_vector); */
   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   hypre_TFree(buf_data);
   hypre_TFree(S_buf_j);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);
   hypre_TFree(S_recv_vec_starts);
   hypre_TFree(S_send_map_starts);

   hypre_TFree(comm_pkg_mS);
   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);


   *S_ptr        = S;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/*==========================================================================
 * Ruge's coarsening algorithm                        
 *==========================================================================*/

#define CPOINT 1
#define FPOINT -1
#define UNDECIDED 0 


/**************************************************************
 *
 *      Ruge Coarsening routine
 *
 **************************************************************/
int
hypre_ParAMGCoarsenRuge( hypre_ParCSRMatrix    *A,
                         double                 strength_threshold,
                         double                 max_row_sum,
                         int                    measure_type,
                         int                    coarsen_type,
                         int                    debug_flag,
                         hypre_ParCSRMatrix   **S_ptr,
                         int                  **CF_marker_ptr,
                         int                   *coarse_size_ptr     )
{
   MPI_Comm         comm          = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg   *comm_pkg      = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;
   int		   *row_starts    = hypre_ParCSRMatrixRowStarts(A);
   hypre_CSRMatrix *A_diag        = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd        = hypre_ParCSRMatrixOffd(A);
   int             *A_i           = hypre_CSRMatrixI(A_diag);
   int             *A_j           = hypre_CSRMatrixJ(A_diag);
   double          *A_data        = hypre_CSRMatrixData(A_diag);
   int             *A_offd_i      = hypre_CSRMatrixI(A_offd);
   int             *A_offd_j      = hypre_CSRMatrixJ(A_offd);
   double          *A_offd_data   = hypre_CSRMatrixData(A_offd);
   int              num_variables = hypre_CSRMatrixNumRows(A_diag);
   int              num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   int		    global_num_vars = hypre_ParCSRMatrixGlobalNumCols(A);
   int 	           *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
                  
   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix *S_diag;
   hypre_CSRMatrix *S_offd;
   int             *S_i;
   int             *S_j;
   int             *S_offd_i;
   int             *S_offd_j;
   int		   *col_map_offd_S;

   hypre_CSRMatrix *S_ext;
   int             *S_ext_i;
   int             *S_ext_j;
                 
   hypre_CSRMatrix *ST;
   int             *ST_i;
   int             *ST_j;
                 
   int             *CF_marker;
   int             *CF_marker_offd;
   int              coarse_size;
   int              ci_tilde = -1;
   int              ci_tilde_offd = -1;

   int             *measure_array;
   int             *graph_array;
   int              graph_size;
   int 	           *int_buf_data;
   int 	           *ci_array;

   double           diag, row_scale, row_sum;
   int              measure, max_measure;
   int              i, j, k, jA, jS, jS_offd, kS, ig;
   int		    ic, ji, jj, jk, jl, jm, index;
   int		    set_empty = 1;
   int		    C_i_nonempty = 0;
   int		    num_strong;
   int		    num_nonzeros;
   int		    num_procs, my_id;
   int		    num_sends = 0;
   int		    first_col, start;
   int		    col_0, col_n;

   hypre_LinkList   LoL_head;
   hypre_LinkList   LoL_tail;

   int             *lists, *where;
   int              points_left, new_C;
   int              new_meas, bumps, top_max;
   int              num_left, elmt;
   int              nabor, nabor_two;

   int              ierr = 0;
   int              break_var = 0;
   double	    wall_time;

   if (coarsen_type < 0) coarsen_type = -coarsen_type;

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(int, num_variables);
   where = hypre_CTAlloc(int, num_variables);

   CF_marker = hypre_CTAlloc(int, num_variables);
   for (j = 0; j < num_variables; j++)
   {
      CF_marker[j] = UNDECIDED;
   } 

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
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

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   num_strong = A_i[num_variables] - num_variables;

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
			row_starts, row_starts,
			num_cols_offd, 0, 0);
/* row_starts is owned by A, col_starts = row_starts */
   hypre_ParCSRMatrixSetRowStartsOwner(S,0);
   col_map_offd_S = hypre_CTAlloc(int,num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
	col_map_offd_S[i] = col_map_offd[i];

   hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;

   S_diag = hypre_ParCSRMatrixDiag(S);
   S_offd = hypre_ParCSRMatrixOffd(S);

   ST = hypre_CSRMatrixCreate(num_variables, num_variables, num_strong);

   S_i = hypre_CTAlloc(int,num_variables+1);
   hypre_CSRMatrixI(S_diag) = S_i;

   S_offd_i = hypre_CTAlloc(int,num_variables+1);
   hypre_CSRMatrixI(S_offd) = S_offd_i;

   S_offd_j = hypre_CTAlloc(int,A_offd_i[num_variables]);
   hypre_CSRMatrixJ(S_offd) = S_offd_j;

   ST_i = hypre_CTAlloc(int,num_variables+1);
   hypre_CSRMatrixI(ST) = ST_i;

   ST_j = hypre_CTAlloc(int,A_i[num_variables]);
   hypre_CSRMatrixJ(ST) = ST_j;

   /* give S same nonzero structure as A, store in ST*/
   for (i = 0; i < num_variables; i++)
   {
      ST_i[i] = A_i[i];
      for (jA = A_i[i]; jA < A_i[i+1]; jA++)
      {
         ST_j[jA] = A_j[jA];
      }
      S_offd_i[i] = A_offd_i[i];
      for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
      {
         S_offd_j[jA] = A_offd_j[jA];
      }
   }
   ST_i[num_variables] = A_i[num_variables];
   S_offd_i[num_variables] = A_offd_i[num_variables];

   for (i = 0; i < num_variables; i++)
   {
      diag = A_data[A_i[i]];

      /* compute scaling factor */
      row_scale = 0.0;
      row_sum = diag;
      if (diag < 0)
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            row_scale = hypre_max(row_scale, A_data[jA]);
            row_sum += A_data[jA];
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = hypre_max(row_scale, A_offd_data[jA]);
            row_sum += A_offd_data[jA];
         }
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            row_scale = hypre_min(row_scale, A_data[jA]);
            row_sum += A_data[jA];
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = hypre_min(row_scale, A_offd_data[jA]);
            row_sum += A_offd_data[jA];
         }
      }
      row_sum = fabs( row_sum / diag );

      /* compute row entries of S */
      if ((row_sum > max_row_sum) && (max_row_sum < 1.0))
      {
         /* make all dependencies weak */
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            ST_j[jA] = -1;
            num_strong--;
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            S_offd_j[jA] = -1;
         }
      }
      else
      {
         if (diag < 0) 
         {
            for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
            {
               if (A_data[jA] <= strength_threshold * row_scale)
               {
                  ST_j[jA] = -1;
                  num_strong--;
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               if (A_offd_data[jA] <= strength_threshold * row_scale)
               {
                  S_offd_j[jA] = -1;
               }
            }
         }
         else
         {
            for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
            {
               if (A_data[jA] >= strength_threshold * row_scale)
               {
                  ST_j[jA] = -1;
                  num_strong--;
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               if (A_offd_data[jA] >= strength_threshold * row_scale)
               {
                  S_offd_j[jA] = -1;
               }
            }
         }
      }
   }

   S_j = hypre_CTAlloc(int,num_strong);
   hypre_CSRMatrixJ(S_diag) = S_j;

   /*--------------------------------------------------------------
    * "Compress" the strength matrix.
    *
    * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
    *
    * NOTE: This "compression" section of code may be removed, and
    * coarsening will still be done correctly.  However, the routine
    * that builds interpolation would have to be modified first.
    *----------------------------------------------------------------*/

   jS = 0;
   jS_offd = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_i[i] = jS;
      for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
      {
         if (ST_j[jA] != -1)
         {
            S_j[jS]    = ST_j[jA];
            jS++;
         }
      }
      S_offd_i[i] = jS_offd;
      for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
      {
         if (S_offd_j[jA] != -1)
         {
            S_offd_j[jS_offd] = S_offd_j[jA];
            jS_offd++;
         }
      }
   }
   S_i[num_variables] = jS;
   S_offd_i[num_variables] = jS_offd;
   hypre_CSRMatrixNumNonzeros(S_diag) = jS;
   hypre_CSRMatrixNumNonzeros(ST) = jS;
   hypre_CSRMatrixNumNonzeros(S_offd) = jS_offd;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i=0; i <= num_variables; i++)
      ST_i[i] = 0;
 
   for (i=0; i < jS; i++)
   {
	 ST_i[S_j[i]+1]++;
   }
   for (i=0; i < num_variables; i++)
   {
      ST_i[i+1] += ST_i[i];
   }
   for (i=0; i < num_variables; i++)
   {
      for (j=S_i[i]; j < S_i[i+1]; j++)
      {
	 index = S_j[j];
       	 ST_j[ST_i[index]] = i;
       	 ST_i[index]++;
      }
   }      
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i-1];
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

   measure_array = hypre_CTAlloc(int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = ST_i[i+1]-ST_i[i];
   }

   if ((measure_type || coarsen_type != 1) && num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
      num_nonzeros = S_ext_i[num_cols_offd];
      first_col = hypre_ParCSRMatrixFirstColDiag(A);
      col_0 = first_col-1;
      col_n = col_0+num_variables;
      if (measure_type)
      {
	 for (i=0; i < num_nonzeros; i++)
         {
	    index = S_ext_j[i] - first_col;
	    if (index > -1 && index < num_variables)
		measure_array[index]++;
         } 
      } 
   }



   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   graph_size = num_variables;

   /* first coarsening phase */

  /*************************************************************
   *
   *   Initialize the lists
   *
   *************************************************************/

   num_left = num_variables;
   coarse_size = 0;
 
   for (j = 0; j < num_variables; j++) 
   {    
      measure = measure_array[j];
      if (measure > 0) 
      {
         enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
      }
      else
      {
         if (measure < 0) printf("negative measure!\n");
         CF_marker[j] = CPOINT;
         ++coarse_size;
         --num_left;
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

      CF_marker[index] = CPOINT;
      ++coarse_size;
      measure = measure_array[index];
      measure_array[index] = 0;
      --num_left;
      
      remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
  
      for (j = ST_i[index]; j < ST_i[index+1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = FPOINT;
            measure = measure_array[nabor];

            remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;

            for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  remove_point(&LoL_head, &LoL_tail, measure, 
                               nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);
                 
                  enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor_two, lists, where);
               }
            }
         }
      }
      for (j = S_i[index]; j < S_i[index+1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;

            enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, lists, where);
         }
      }

   }

   hypre_TFree(measure_array);
   hypre_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Coarsen 1st pass = %f\n",
                     my_id, wall_time); 
   }

   hypre_TFree(lists);
   hypre_TFree(where);
   hypre_TFree(LoL_head);
   hypre_TFree(LoL_tail);

   /* second pass, check fine points for coarse neighbors 
      for coarsen_type = 2, the second pass includes
      off-processore boundary points */

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   if (coarsen_type == 2)
   {
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/
    
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                   num_sends));
    
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
    
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        CF_marker_offd);
    
      hypre_ParCSRCommHandleDestroy(comm_handle);
      }
    
      ci_array = hypre_CTAlloc(int,num_cols_offd);
      for (i=0; i < num_cols_offd; i++)
	 ci_array[i] = -1;
	
      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] == -1)
         {
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i+1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
                  graph_array[j] = i;
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
                  ci_array[j] = i;
            }
            for (ji = S_i[i]; ji < S_i[i+1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_i[j]; jj < S_i[j+1]; jj++)
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
                     for (jj = S_offd_i[j]; jj < S_offd_i[j+1]; jj++)
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
                        coarse_size++;
                        if (ci_tilde > -1)
                        {
                           CF_marker[ci_tilde] = -1;
                           coarse_size--;
                           ci_tilde = -1;
                        }
                        C_i_nonempty = 0;
                        break_var = 0;
                        break;
                     }
                     else
                     {
                        ci_tilde = j;
                        CF_marker[j] = 1;
                        coarse_size++;
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
               for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
               {
                  j = S_offd_j[ji];
                  if (CF_marker_offd[j] == -1)
                  {
                     set_empty = 1;
                     for (jj = S_ext_i[j]; jj < S_ext_i[j+1]; jj++)
                     {
                        index = S_ext_j[jj];
                        if (index > col_0 && index < col_n) /* index interior */
                        {
                           if (graph_array[index-first_col] == i)
                           {
                              set_empty = 0;
                              break;
                           }
                        }
                        else
                        {
   		           jk = hypre_BinarySearch(col_map_offd,index,num_cols_offd);
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
                           coarse_size++;
                           if (ci_tilde > -1)
                           {
                              CF_marker[ci_tilde] = -1;
                              coarse_size--;
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
      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] == -1)
         {
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] > 0)
   	          graph_array[j] = i;
    	    }
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] == -1)
   	       {
   	          set_empty = 1;
   	          for (jj = S_i[j]; jj < S_i[j+1]; jj++)
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
   		        coarse_size++;
   		        if (ci_tilde > -1)
   		        {
   			   CF_marker[ci_tilde] = -1;
   		           coarse_size--;
   		           ci_tilde = -1;
   		        }
   	    		C_i_nonempty = 0;
   		        break;
   		     }
   		     else
   		     {
   		        ci_tilde = j;
   		        CF_marker[j] = 1;
   		        coarse_size++;
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
      printf("Proc = %d    Coarsen 2nd pass = %f\n",
                       my_id, wall_time); 
   }

   /* third pass, check boundary fine points for coarse neighbors */

   if (coarsen_type == 3 || coarsen_type == 4)
   {
      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));

      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
             int_buf_data[index++] 
              = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
     		CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      }

      ci_array = hypre_CTAlloc(int,num_cols_offd);
      for (i=0; i < num_cols_offd; i++)
	 ci_array[i] = -1;
   }

   if (coarsen_type > 1 && coarsen_type < 5)
   { 
      for (i=0; i < num_variables; i++)
	 graph_array[i] = -1;
      for (i=0; i < num_cols_offd; i++)
      {
         if (CF_marker_offd[i] == -1)
         {
   	    for (ji = S_ext_i[i]; ji < S_ext_i[i+1]; ji++)
   	    {
   	       j = S_ext_j[ji];
   	       if (j > col_0 && j < col_n)
   	       {
   	          j = j - first_col;
   	          if (CF_marker[j] > 0)
   	             graph_array[j] = i;
   	       }
   	       else
   	       {
   		  jj = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
   		  if (jj != -1 && CF_marker_offd[jj] > 0)
   	                ci_array[jj] = i;
    	       }	
    	    }
   	    for (ji = S_ext_i[i]; ji < S_ext_i[i+1]; ji++)
   	    {
   	       j = S_ext_j[ji];
   	       if (j > col_0 && j < col_n)
   	       {
   	          j = j - first_col;
   	          if ( CF_marker[j] == -1)
   	          {
   	             set_empty = 1;
   	             for (jj = S_i[j]; jj < S_i[j+1]; jj++)
   	             {
   		        index = S_j[jj];
   		        if (graph_array[index] == i)
   		        {
   		           set_empty = 0;
   		           break;
   		        }
   	             }
   	             for (jj = S_offd_i[j]; jj < S_offd_i[j+1]; jj++)
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
   		  jm = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
   		  if (jm != -1 && CF_marker_offd[jm] == -1)
   	          {
   	             set_empty = 1;
   	             for (jj = S_ext_i[jm]; jj < S_ext_i[jm+1]; jj++)
   	             {
   		        index = S_ext_j[jj];
   		        if (index > col_0 && index < col_n) 
   		  	{
   		           if (graph_array[index-first_col] == i)
   		           {
   		              set_empty = 0;
   		              break;
   		           }
   	                }
   			else
   			{
   		           jk = hypre_BinarySearch(col_map_offd,index,num_cols_offd);
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
            if (hypre_ParCSRCommPkgSendProc(comm_pkg,i) > my_id)
	    {
              for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                   CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)] =
                   int_buf_data[index++]; 
            }
	    else
	    {
	       index += hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - start;
	    }
         }
      }
      else
      {
         for (i = 0; i < num_sends; i++)
         {
	    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            if (hypre_ParCSRCommPkgSendProc(comm_pkg,i) > my_id)
	    {
              for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
              {
                 elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
                 if (CF_marker[elmt] != 1)
                   CF_marker[elmt] = int_buf_data[index];
		 index++; 
              }
            }
	    else
	    {
	       index += hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - start;
	    }
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         if (coarsen_type == 4)
		printf("Proc = %d    Coarsen 3rd pass = %f\n",
                my_id, wall_time); 
         if (coarsen_type == 3)
		printf("Proc = %d    Coarsen 3rd pass = %f\n",
                my_id, wall_time); 
         if (coarsen_type == 2)
		printf("Proc = %d    Coarsen 2nd pass = %f\n",
                my_id, wall_time); 
      }
   }
   if (coarsen_type == 5)
   {
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
    
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                   num_sends));
    
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
    
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        CF_marker_offd);
    
      hypre_ParCSRCommHandleDestroy(comm_handle);
      }
    
      ci_array = hypre_CTAlloc(int,num_cols_offd);
      for (i=0; i < num_cols_offd; i++)
   	 ci_array[i] = -1;
      for (i=0; i < num_variables; i++)
   	 graph_array[i] = -1;

      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] == -1 && (S_offd_i[i+1]-S_offd_i[i]) > 0)
         {
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i+1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
                  graph_array[j] = i;
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
                  ci_array[j] = i;
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_ext_i[j]; jj < S_ext_i[j+1]; jj++)
                  {
                     index = S_ext_j[jj];
                     if (index > col_0 && index < col_n) /* index interior */
                     {
                        if (graph_array[index-first_col] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                     else
                     {
   		        jk = hypre_BinarySearch(col_map_offd,index,num_cols_offd);
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
         printf("Proc = %d    Coarsen special points = %f\n",
                       my_id, wall_time); 
      }

   }
   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   if (coarsen_type != 1)
   {   
      hypre_TFree(CF_marker_offd);
      hypre_TFree(int_buf_data);
      hypre_TFree(ci_array);
   }   
   hypre_TFree(graph_array);
   if ((measure_type || coarsen_type != 1) && num_procs > 1)
   	hypre_CSRMatrixDestroy(S_ext); 
   
   *S_ptr           = S;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;
   
   return (ierr);
}

#define C_PT  1
#define F_PT -1
#define COMMON_C_PT  2
#define CPOINT 1
#define FPOINT -1
#define UNDECIDED 0 


int
hypre_ParAMGCoarsenFalgout( hypre_ParCSRMatrix    *A,
                            double                 strength_threshold,
                            double                 max_row_sum,
                            int                    debug_flag,
                            hypre_ParCSRMatrix   **S_ptr,
                            int                  **CF_marker_ptr,
                            int                   *coarse_size_ptr     )
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommPkg      *comm_pkg_mS;
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   double             *A_offd_data;

   int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
   int 		      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   int                 global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   int		       col_1 = hypre_ParCSRMatrixFirstColDiag(A);
   int		       col_n = col_1 + hypre_CSRMatrixNumCols(A_diag);
   int 		       num_nonzeros_diag;
   int 		       num_nonzeros_offd = 0;
   int 		       num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
                  
   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   int                *S_diag_i;
   int                *S_diag_j;
   /* double             *S_diag_data; */
   hypre_CSRMatrix    *S_offd;
   int                *S_offd_i;
   int                *S_offd_j;
   /* double             *S_offd_data; */
                 
   hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j;
   /* double             *S_ext_data; */

   hypre_CSRMatrix *ST;
   int             *ST_i;
   int             *ST_j;
                 
   int		       num_sends = 0;
   int		       num_recvs = 0;
   int  	      *int_buf_data;
   int  	      *S_recv_vec_starts;
   int  	      *S_send_map_starts;
   double	      *buf_data;
   int		      *S_buf_j; 

   int                *CF_marker;
   int                *CF_marker_offd;
   int                 coarse_size;
                      
   /* hypre_ParVector    *ones_vector; 
   hypre_ParVector    *measure_vector; */
   double             *measure_array;
   int                *i_measure_array;
   int                *graph_array;
   int                 graph_size;
   int                 global_graph_size;

                      
   double              diag, row_scale, row_sum;
   int                 i, j, k, ic, jc, kc, jj, kk, jA, jS, kS, ig;
   int		       index, index_S, jrow;
   int              measure, max_measure;
   int              jS_offd;
   int		    ji, jk, jl, jm;
   int		    set_empty = 1;
   int		    C_i_nonempty = 0;
   int		    ci_tilde = -1;
   int		    num_strong;
   int		    num_nonzeros;
   int		    num_procs, my_id;
   int		    first_col, start;
   int		    col_0, iter;
                      
   int		       num_data, start_index;
   int		       *recv_vec_starts;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif
                  

   hypre_LinkList   LoL_head;
   hypre_LinkList   LoL_tail;

   int             *lists, *where;
   int              points_left, new_C;
   int              new_meas, bumps, top_max;
   int              num_left;
   int              nabor, nabor_two;

   int              ierr = 0;
   int              break_var = 0;
   double	    wall_time;
   double	    wall_time_ip = 0;
   double	    wall_time_bp = 0;
   double	    wall_time_rs = 0;
   double	    sum_time_ip = 0;
   double	    sum_time_bp = 0;
   double	    sum_time_rs = 0;



   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(int, num_variables);
   where = hypre_CTAlloc(int, num_variables);

   CF_marker = hypre_CTAlloc(int, num_variables);
   for (j = 0; j < num_variables; j++)
   {
      CF_marker[j] = UNDECIDED;
   } 

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > max (k != i) aik,    aii < 0
    * or
    *     aij < min (k != i) aik,    aii >= 0
    * Then S_diag_ij = 1, else S_diag_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   num_strong = A_diag_i[num_variables] - num_variables;

   num_nonzeros_offd = A_offd_i[num_variables];
   num_nonzeros_diag = A_diag_i[num_variables];
   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
			row_starts, row_starts,
			num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
/* row_starts is owned by A, col_starts = row_starts */
   hypre_ParCSRMatrixSetRowStartsOwner(S,0);
   /* hypre_ParCSRMatrixInitialize(S); */
   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(int, num_variables+1);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(int, num_nonzeros_diag);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(int, num_variables+1);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   /* S_diag_data = hypre_CSRMatrixData(S_diag); */
   S_offd_i = hypre_CSRMatrixI(S_offd);

   if (num_cols_offd)
   {
   	A_offd_data = hypre_CSRMatrixData(A_offd);
        hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(int, num_nonzeros_offd);
   	S_offd_j = hypre_CSRMatrixJ(S_offd);
   	/* S_offd_data = hypre_CSRMatrixData(S_offd); */
 	hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(int, num_cols_offd);
   }

   /* give S same nonzero structure as A */
   hypre_ParCSRMatrixCopy(A,S,0);

   for (i = 0; i < num_variables; i++)
   {
      diag = A_diag_data[A_diag_i[i]];

      /* compute scaling factor */
      row_scale = 0.0;
      row_sum = diag;
      if (diag < 0)
      {
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            row_scale = hypre_max(row_scale, A_diag_data[jA]);
            row_sum += A_diag_data[jA];
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = hypre_max(row_scale, A_offd_data[jA]);
            row_sum += A_offd_data[jA];
         }
      }
      else
      {
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            row_scale = hypre_min(row_scale, A_diag_data[jA]);
            row_sum += A_diag_data[jA];
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = hypre_min(row_scale, A_offd_data[jA]);
            row_sum += A_offd_data[jA];
         }
      }
      row_sum = fabs( row_sum / diag );

      /* compute row entries of S */
      S_diag_j[A_diag_i[i]] = -1;
      if ((row_sum > max_row_sum) && (max_row_sum < 1.0))
      {
         /* make all dependencies weak */
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            S_diag_j[jA] = -1;
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            S_offd_j[jA] = -1;
         }
      }
      else
      {
         if (diag < 0) 
         { 
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               if (A_diag_data[jA] <= strength_threshold * row_scale)
               {
                  S_diag_j[jA] = -1;
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               if (A_offd_data[jA] <= strength_threshold * row_scale)
               {
                  S_offd_j[jA] = -1;
               }
            }
         }
         else
         {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               if (A_diag_data[jA] >= strength_threshold * row_scale)
               {
                  S_diag_j[jA] = -1;
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               if (A_offd_data[jA] >= strength_threshold * row_scale)
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

   jS = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_diag_i[i] = jS;
      for (jA = A_diag_i[i]; jA < A_diag_i[i+1]; jA++)
      {
         if (S_diag_j[jA] > -1)
         {
            S_diag_j[jS]    = S_diag_j[jA];
            jS++;
         }
      }
   }
   S_diag_i[num_variables] = jS;
   hypre_CSRMatrixNumNonzeros(S_diag) = jS;

   jS_offd = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_offd_i[i] = jS_offd;
      for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
      {
         if (S_offd_j[jA] > -1)
         {
            S_offd_j[jS_offd]    = S_offd_j[jA];
            jS_offd++;
         }
      }
   }
   S_offd_i[num_variables] = jS_offd;
   hypre_CSRMatrixNumNonzeros(S_offd) = jS_offd;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   ST = hypre_CSRMatrixCreate(num_variables, num_variables, jS);

   ST_i = hypre_CTAlloc(int,num_variables+1);
   hypre_CSRMatrixI(ST) = ST_i;

   ST_j = hypre_CTAlloc(int,jS);
   hypre_CSRMatrixJ(ST) = ST_j;

   for (i=0; i <= num_variables; i++)
      ST_i[i] = 0;
 
   for (i=0; i < jS; i++)
   {
	 ST_i[S_diag_j[i]+1]++;
   }
   for (i=0; i < num_variables; i++)
   {
      ST_i[i+1] += ST_i[i];
   }
   for (i=0; i < num_variables; i++)
   {
      for (j=S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
	 index = S_diag_j[j];
       	 ST_j[ST_i[index]] = i;
       	 ST_i[index]++;
      }
   }      
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i-1];
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

   i_measure_array = hypre_CTAlloc(int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      i_measure_array[i] = ST_i[i+1]-ST_i[i];
   }

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize S, measures = %f\n",
                     my_id, wall_time); 
   }
   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   graph_size = num_variables;

   /* first coarsening phase */

  /*************************************************************
   *
   *   Initialize the lists
   *
   *************************************************************/

   num_left = num_variables;
   coarse_size = 0;
 
   for (j = 0; j < num_variables; j++) 
   {    
      measure = i_measure_array[j];
      if (measure > 0) 
      {
         enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
      }
      else
      {
         if (measure < 0) printf("negative measure!\n");
         CF_marker[j] = FPOINT;
         /* CF_marker[j] = CPOINT;
         ++coarse_size; */
         --num_left;
      }
   }

   /****************************************************************
    *
    *  Main loop of Ruge-Stueben first coloring pass.
    *
    *  WHILE there are still points to classify DO:
    *        1) find first point, i,  on list with max_measure
    *           make i a C-point, remove it from the lists
    *        2) For each point, j,  in S_diag_i^T,
    *           a) Set j to be an F-point
    *           b) For each point, k, in S_diag_j
    *                  move k to the list in LoL with measure one
    *                  greater than it occupies (creating new LoL
    *                  entry if necessary)
    *        3) For each point, j,  in S_diag_i,
    *                  move j to the list in LoL with measure one
    *                  smaller than it occupies (creating new LoL
    *                  entry if necessary)
    *
    ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;

      CF_marker[index] = CPOINT;
      ++coarse_size;
      measure = i_measure_array[index];
      i_measure_array[index] = 0;
      --num_left;
      
      remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
  
      for (j = ST_i[index]; j < ST_i[index+1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = FPOINT;
            measure = i_measure_array[nabor];

            remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;

            for (k = S_diag_i[nabor]; k < S_diag_i[nabor+1]; k++)
            {
               nabor_two = S_diag_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = i_measure_array[nabor_two];
                  remove_point(&LoL_head, &LoL_tail, measure, 
                               nabor_two, lists, where);

                  new_meas = ++(i_measure_array[nabor_two]);
                 
                  enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor_two, lists, where);
               }
            }
         }
      }
      for (j = S_diag_i[index]; j < S_diag_i[index+1]; j++)
      {
         nabor = S_diag_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = i_measure_array[nabor];

            remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            i_measure_array[nabor] = --measure;

            enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, lists, where);
         }
      }
   }

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Coarsen 1st pass = %f\n",
                     my_id, wall_time); 
   }

   hypre_TFree(lists);
   hypre_TFree(where);
   hypre_TFree(LoL_head);
   hypre_TFree(LoL_tail);
   hypre_TFree(i_measure_array);

   /* second pass, check fine points for coarse neighbors  */

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables+num_cols_offd);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   for (i=0; i < num_variables; i++)
   {
      if (CF_marker[i] == -1)
      {
   	 for (ji = S_diag_i[i]; ji < S_diag_i[i+1]; ji++)
   	 {
   	    j = S_diag_j[ji];
   	    if (CF_marker[j] > 0)
   	       graph_array[j] = i;
    	 }
   	 for (ji = S_diag_i[i]; ji < S_diag_i[i+1]; ji++)
   	 {
   	    j = S_diag_j[ji];
   	    if (CF_marker[j] == -1)
   	    {
   	       set_empty = 1;
   	       for (jj = S_diag_i[j]; jj < S_diag_i[j+1]; jj++)
   	       {
   		  index = S_diag_j[jj];
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
   		     coarse_size++;
   		     if (ci_tilde > -1)
   		     {
   		        CF_marker[ci_tilde] = -1;
   		        coarse_size--;
   		        ci_tilde = -1;
   		     } 
                     C_i_nonempty = 0;
   		     break;
   		  }
   		  else
   		  {
   		     ci_tilde = j; 
   		     CF_marker[j] = 1;
   		     coarse_size++;
   		     C_i_nonempty = 1;
		     i--;
		     break;
		  }
	       }
	    }
	 }
      }
   }

   if (debug_flag == 3 )
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Coarsen 2nd pass = %f\n",
                       my_id, wall_time); 
   }

   
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

   comm_pkg_mS = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(comm_pkg_mS) = comm;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg_mS) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg_mS) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgNumSends(comm_pkg_mS) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg_mS) = hypre_ParCSRCommPkgSendProcs(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   A_offd_i = hypre_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

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

   hypre_ParCSRMatrixCommPkg(S) = NULL;

   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);

   for (i=0; i < S_offd_i[num_variables]; i++)
   { 
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }
   if (num_procs > 1)
   comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, 
			&measure_array[num_variables], buf_data);

   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = (double) (ST_i[i+1]-ST_i[i]);
   }

   hypre_CSRMatrixDestroy(ST);
  
   if (num_procs > 1)
   hypre_ParCSRCommHandleDestroy(comm_handle);
      
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
   }


   /* this augments the measures */
   hypre_InitParAMGIndepSet(S, measure_array);

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_variables+num_cols_offd; ig++)
      graph_array[ig] = ig;

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ... 
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
   for (i=0; i < num_variables; i++)
   {
	if ( (CF_marker[i] == -1)||
	     (CF_marker[i] == 1 && (S_offd_i[i+1]-S_offd_i[i]) > 0)) 
	   CF_marker[i] = 0;
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   coarse_size = 0;
   graph_size = num_variables+num_cols_offd;
   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
   }

   num_data = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         num_data += S_diag_i[jrow+1] - S_diag_i[jrow]
         		+ S_offd_i[jrow+1] - S_offd_i[jrow];
      }
   }

   S_buf_j = hypre_CTAlloc(int, num_data); 
   S_send_map_starts = hypre_CTAlloc(int, num_sends+1);

   index = 0;
   S_send_map_starts[0] = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         index += S_diag_i[jrow+1] - S_diag_i[jrow]
	 		+ S_offd_i[jrow+1] - S_offd_i[jrow];
      }
      S_send_map_starts[i+1] = index;
   }
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_mS) = S_send_map_starts;   

   S_recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);
   S_recv_vec_starts[0] = 0;

   for (i=0; i < num_recvs; i++)
   {
      S_recv_vec_starts[i+1] = S_ext_i[recv_vec_starts[i+1]];
   }
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_mS) = S_recv_vec_starts;   
 
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time); 
   }
   iter = 0; 
   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
      index = 0;
      index_S = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
         {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            buf_data[index++] = measure_array[jrow];
	    for (k = S_diag_i[jrow]; k < S_diag_i[jrow+1]; k++)
	    {
	       if (S_diag_j[k] > -1)
	          S_buf_j[index_S++] = S_diag_j[k]+col_1;
	       else
	          S_buf_j[index_S++] = S_diag_j[k]-col_1;
            }
	    for (k = S_offd_i[jrow]; k < S_offd_i[jrow+1]; k++)
	    {
	       if (S_offd_j[k] > -1)
	          S_buf_j[index_S++] = col_map_offd[S_offd_j[k]];
	       else
	          S_buf_j[index_S++] = -col_map_offd[-S_offd_j[k]-1]-1;
            }
         }
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, buf_data, 
			&measure_array[num_variables]); 
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
 
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg_mS, S_buf_j, 
			S_ext_j);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      }
 
      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/
 
      if (iter)
      {
     	 for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];

	    if (i < num_variables)
	    { 
               if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) )
               {
                  /* set to be an F-pt */
                  CF_marker[i] = F_PT;
 
	          /* make sure all dependencies have been accounted for */
                  for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
                  {
                     if (S_diag_j[jS] > -1)
                     {
                        CF_marker[i] = 0;
                     }
                  }
                  for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
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
	    else
	    {
               ic = i - num_variables;
               if ( (CF_marker_offd[ic] != C_PT) && (measure_array[i] < 1) )
               {
                  /* set to be an F-pt */
                  CF_marker_offd[ic] = F_PT;
 
                  for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
                  {
                     if (S_ext_j[jS] > -1)
                     {
                        CF_marker_offd[ic] = 0;
                     }
                  }
               }
 
               if (CF_marker_offd[ic])
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
      sprintf(filename, "coarsen.out.measures.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%f\n", measure_array[i]);
      }
      fclose(fp);

      /* print out strength matrix */
      sprintf(filename, "coarsen.out.strength.%04d", iter);
      hypre_CSRMatrixPrint(S, filename);

      /* print out C/F marker */
      sprintf(filename, "coarsen.out.CF.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%d\n", CF_marker[i]);
      }
      fclose(fp);

      iter++;
#endif

      /*------------------------------------------------
       * Test for convergence
       *------------------------------------------------*/

      MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

      if (global_graph_size == 0)
         break;

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/
      if (iter)
         hypre_ParAMGIndepSet(S, S_ext, measure_array, graph_array, 
				graph_size, CF_marker, CF_marker_offd);
      iter++;

      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++] 
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }

      if (num_procs > 1)
      { 
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      } 
 
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",
                     my_id, iter, wall_time); 
   }
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them.
          *---------------------------------------------*/

         if (i < num_variables)  /*interior points */
	 {
   if (debug_flag == 3) wall_time_ip = time_getWallclockSeconds();
            if (CF_marker[i] > 0)
            {  
               /* set to be a C-pt */
               CF_marker[i] = C_PT;
	       coarse_size++;

               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  j = S_diag_j[jS];
                  if (j > -1)
                  {
               
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
               
                     /* decrement measures of unmarked neighbors */
                     if (!CF_marker[j])
                     {
                        measure_array[j]--;
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  j = S_offd_j[jS];
                  if (j > -1)
                  {
               
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
               
                     /* decrement measures of unmarked neighbors */
                     if (!CF_marker_offd[j])
                     {
                        measure_array[j+num_variables]--;
                     }
                  }
               }
            }
	    else
    	    {
               /* marked dependencies */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  j = S_diag_j[jS];
  		  if (j < 0) j = -j-1;
 
                  if (CF_marker[j] > 0)
                  {
                     if (S_diag_j[jS] > -1)
                     {
                        /* "remove" edge from S */
                        S_diag_j[jS] = -S_diag_j[jS]-1;
                     }
   
                     /* IMPORTANT: consider all dependencies */
                     /* if (S_diag_data[jS]) */
                     {
                        /* temporarily modify CF_marker */
                        CF_marker[j] = COMMON_C_PT;
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  j = S_offd_j[jS];
  		  if (j < 0) j = -j-1;
   
                  if (CF_marker_offd[j] > 0)
                  {
                     if (S_offd_j[jS] > -1)
                     {
                        /* "remove" edge from S */
                        S_offd_j[jS] = -S_offd_j[jS]-1;
                     }
   
                     /* IMPORTANT: consider all dependencies */
                     /* if (S_offd_data[jS]) */
                     {
                        /* temporarily modify CF_marker */
                        CF_marker_offd[j] = COMMON_C_PT;
                     }
                  }
               }
   
               /* unmarked dependencies */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     j = S_diag_j[jS];
  		     if (j < 0) j = -j-1;
   		     break_var = 1;
                     /* check for common C-pt */
                     for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++)
                     {
                        k = S_diag_j[kS];
  		        if (k < 0) k = -k-1;
   
                        /* IMPORTANT: consider all dependencies */
                        /* if (S_diag_data[kS] && CF_marker[k] == COMMON_C_PT)*/
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = -S_diag_j[jS]-1;
                           measure_array[j]--;
                           break_var = 0;
                           break;
                        }
                     }
   		     if (break_var)
                     {
                        for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS++)
                        {
                           k = S_offd_j[kS];
  		           if (k < 0) k = -k-1;
   
                           /* IMPORTANT: consider all dependencies */
                           /*if (S_offd_data[kS] &&
   				CF_marker_offd[k] == COMMON_C_PT)*/
                           if ( CF_marker_offd[k] == COMMON_C_PT)
                           {
                              /* "remove" edge from S and update measure*/
                              S_diag_j[jS] = -S_diag_j[jS]-1;
                              measure_array[j]--;
                              break;
                           }
                        }
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  j = S_offd_j[jS];
                  if (j > -1)
                  {
   
                     /* check for common C-pt */
                     for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++)
                     {
                        k = S_ext_j[kS];
                        if (k < 0) k = -k-1;
   		        if (k >= col_1 && k < col_n)
   		        {
   			   kc = k - col_1;
   
                           /* IMPORTANT: consider all dependencies */
                        /*if (S_ext_data[kS] && CF_marker[kc] == COMMON_C_PT)*/
                           if (CF_marker[kc] == COMMON_C_PT)
                           {
                              /* "remove" edge from S and update measure*/
                              S_offd_j[jS] = -S_offd_j[jS]-1;
                              measure_array[j+num_variables]--;
                              break;
                           }
                        }
   		        else
   		        {
   		           kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
   		           /* kc = -1;
   		           for (kk = 0; kk < num_cols_offd; kk++)
   		           {
   			      if (col_map_offd[kk] == k)
   			      {
   			         kc = kk;
   			         break;
   			      }
   		           } */
   		           /* if (kc > -1 && S_ext_data[kS] && 
   				CF_marker_offd[kc] == COMMON_C_PT) */
   		           if (kc > -1 && 
   				CF_marker_offd[kc] == COMMON_C_PT)
   		           {
                              /* "remove" edge from S and update measure*/
                              S_offd_j[jS] = -S_offd_j[jS]-1;
                              measure_array[j+num_variables]--;
                              break;
   		           }
   		        }
                     }
                  }
               }
            }
   if (debug_flag == 3) sum_time_ip += time_getWallclockSeconds()-wall_time_ip;
    	 }

	 else /* boundary points */
	 {
   if (debug_flag == 3) wall_time_bp = time_getWallclockSeconds();
	    ic = i - num_variables;
            if (CF_marker_offd[ic] > 0)
            {  
               /* set to be a C-pt */
               CF_marker_offd[ic] = C_PT;

               for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
               {
                  j = S_ext_j[jS];
                  if (j > -1)
                  {
	       	     if (j >= col_1 && j < col_n)
		     {
		        jj = j-col_1;
		     }
		     else
		     {
   		        jj = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
			if (jj != -1) jj += num_variables;
		     }
               
                     /* "remove" edge from S */
                     S_ext_j[jS] = -S_ext_j[jS]-1;
               
                     /* decrement measures of unmarked neighbors */
                     if (jj < num_variables)
		     {
			if (jj != -1 && !CF_marker[jj])
                        {
                           measure_array[jj]--;
                        }
                     }
		     else
		     {
			if (!CF_marker_offd[jj-num_variables])
                        {
                           measure_array[jj]--;
                        }
                     }
                  }
               }
            }
	    else

            /*---------------------------------------------
             * Heuristic: points that interpolate from a
             * common C-pt are less dependent on each other.
             *
             * NOTE: CF_marker is used to help check for
             * common C-pt's in the heuristic.
             *---------------------------------------------*/

 	    {
	       ic = i - num_variables;
               for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
               {
                  j = S_ext_j[jS];
		  if (j < 0) j = -j-1;
	          if (j >= col_1 && j < col_n)
	          {
	             jc = j - col_1;
                     if (CF_marker[jc] > 0)
                     {
                        if (S_ext_j[jS] > -1)
                        {
                           /* "remove" edge from S */
                           S_ext_j[jS] = -S_ext_j[jS]-1;
                        }

                        /* IMPORTANT: consider all dependencies */
                        /* if (S_ext_data[jS]) */
                        {
                           /* temporarily modify CF_marker */
                           CF_marker[jc] = COMMON_C_PT;
                        }
                     }
	          }
	          else
	          {
   		     jj = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
		     if (jj != -1)
		     {
                  	if (CF_marker_offd[jj] > 0)
                  	{
                     	   if (S_ext_j[jS] > -1)
                     	   {
                              /* "remove" edge from S */
                              S_ext_j[jS] = -S_ext_j[jS]-1;
                     	   }

                     	   /* IMPORTANT: consider all dependencies */
                     	   /* if (S_ext_data[jS]) */
                     	   {
                              /* temporarily modify CF_marker */
                              CF_marker_offd[jj] = COMMON_C_PT;
                     	   }
                        }
                     }
	          }
               }

               /* unmarked dependencies */
               for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
               {
                  j = S_ext_j[jS];
                  if (j > -1)
                  {
                     /* check for common C-pt */
		     if (j >= col_1 && j < col_n)
		     {
		        jc = j - col_1;
		        break_var = 1;
                        for (kS = S_diag_i[jc]; kS < S_diag_i[jc+1]; kS++)
                        {
                           k = S_diag_j[kS];
			   if (k < 0) k = -k-1;
                           /* IMPORTANT: consider all dependencies */
                           /* if (S_diag_data[kS]) */
                           {
                              if (CF_marker[k] == COMMON_C_PT)
                              {
                                 /* "remove" edge from S and update measure*/
                                 S_ext_j[jS] = -S_ext_j[jS]-1;
                                 measure_array[jc]--;
                                 break_var = 0;
                                 break;
                              }
                           }
                        }
		        if (break_var)
                        {
                           for (kS = S_offd_i[jc]; kS < S_offd_i[jc+1]; kS++)
                           {
                              k = S_offd_j[kS];
			      if (k < 0) k = -k-1;

                              /* IMPORTANT: consider all dependencies */
                              /* if (S_offd_data[kS]) */
                              {
                                 if (CF_marker_offd[k] == COMMON_C_PT)
                                 {
                                    /* "remove" edge from S and update measure*/
                                    S_ext_j[jS] = -S_ext_j[jS]-1;
                                    measure_array[jc]--;
                                    break;
                                 }
                              }
                           }
                        }
                     }
		     else
		     {
   		        jc = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
		        if (jc > -1 )
                        {
		           for (kS = S_ext_i[jc]; kS < S_ext_i[jc+1]; kS++)
                           {
                      	      k = S_ext_j[kS];
			      if (k < 0) k = -k-1;

                      	      /* IMPORTANT: consider all dependencies */
                      	   /* if (k >= col_1 && k < col_n && S_ext_data[kS]) */
                      	      if (k >= col_1 && k < col_n )
                      	      {
                                 if (CF_marker[k-col_1] == COMMON_C_PT)
                                 {
                                    /* "remove" edge from S and update measure*/
                                    S_ext_j[jS] = -S_ext_j[jS]-1;
                                    measure_array[jc+num_variables]--;
                                    break;
                                 }
                              }
			      else
			      {
   		                 kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
			         /* kc = -1;
			         for (kk = 0; kk < num_cols_offd; kk++)
			         {
			            if (col_map_offd[kk] == k)
			            {
			               kc = kk;
			               break;
			            }
			         } */
			         if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
			         {
                                    /* "remove" edge from S and update measure*/
                                    S_ext_j[jS] = -S_ext_j[jS]-1;
                                    measure_array[jc+num_variables]--;
			            break;
			         }
                              }
                           }
                        }
                     }
                  }
               }
 	    }
   if (debug_flag == 3) sum_time_bp += time_getWallclockSeconds()-wall_time_bp;
         }

   if (debug_flag == 3) wall_time_rs = time_getWallclockSeconds();
         /* reset CF_marker */
         if (i < num_variables)
	 {
	    for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
	    {
               j = S_diag_j[jS];
	       if (j < 0) j = -j-1;

               if (CF_marker[j] == COMMON_C_PT)
               {
                  CF_marker[j] = C_PT;
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
	       if (j < 0) j = -j-1;

               if (CF_marker_offd[j] == COMMON_C_PT)
               {
                  CF_marker_offd[j] = C_PT;
               }
            }
         }
	 else
	 {
	    ic = i - num_variables;
	    for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
	    {
               j = S_ext_j[jS];
	       if (j < 0) j = -j-1;
	       if (j >= col_1 && j < col_n &&
			CF_marker[j - col_1] == COMMON_C_PT)
               {
                  CF_marker[j - col_1] = C_PT;
               }
	       else
	       {
   		  jj = hypre_BinarySearch(col_map_offd,j,num_cols_offd);
		  if (jj != -1 && CF_marker_offd[jj] == COMMON_C_PT)
                      CF_marker_offd[jj] = C_PT;
               }
            }
         }
   if (debug_flag == 3) sum_time_rs += time_getWallclockSeconds()-wall_time_rs;
      }
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                     my_id, wall_time, graph_size, num_cols_offd); 
      printf("Proc = %d    ip = %f bp = %f rs = %f\n",
                     my_id, sum_time_ip, sum_time_bp, sum_time_rs);
	sum_time_ip = 0;
	sum_time_bp = 0;
	sum_time_rs = 0;
   }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
	 S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
	 S_offd_j[i] = -S_offd_j[i]-1;
   }
   /* hypre_ParVectorDestroy(ones_vector); 
   hypre_ParVectorDestroy(measure_vector); */
   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   hypre_TFree(buf_data);
   hypre_TFree(S_buf_j);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);

   hypre_TFree(S_send_map_starts);
   hypre_TFree(S_recv_vec_starts);
   hypre_TFree(comm_pkg_mS);
   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);


   *S_ptr        = S;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}
