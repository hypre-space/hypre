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
                     hypre_ParCSRMatrix   **S_ptr,
                     int                  **CF_marker_ptr,
                     int                   *coarse_size_ptr     )
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_CommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_CommHandle   *comm_handle;
   hypre_CommHandle   *comm_handle2;

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
   double             *S_diag_data;
   hypre_CSRMatrix    *S_offd;
   int                *S_offd_i;
   int                *S_offd_j;
   double             *S_offd_data;
                 
   hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j;
   double             *S_ext_data;

   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker;
   int                 coarse_size;
                      
   hypre_ParVector    *ones_vector;
   hypre_ParVector    *measure_vector;
   double             *measure_array;
   int                *graph_array;
   int                 graph_size;
                      
   double              diag, row_scale;
   int                 i, j, k, ic, jc, kc, jj, kk, jA, jS, kS, ig;
   int		       index, start, num_procs;
                      
   int                 ierr = 0;
   int                 break_var = 1;

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
    *     aij > max (k != i) aik,    aii < 0
    * or
    *     aij < min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);

   if (!comm_pkg)
   {
        hypre_GenerateMatvecCommunicationInfo(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }
 
   num_sends = hypre_CommPkgNumSends(comm_pkg);
   buf_data = hypre_CTAlloc(double, hypre_CommPkgSendMapStart(comm_pkg,
                                                num_sends));
   int_buf_data = hypre_CTAlloc(int, hypre_CommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   A_offd_i = hypre_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   S = hypre_CreateParCSRMatrix(comm, global_num_vars, global_num_vars,
			row_starts, row_starts,
			num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
/* row_starts is owned by A, col_starts = row_starts */
   hypre_SetParCSRMatrixRowStartsOwner(S,0);
   hypre_InitializeParCSRMatrix(S);

   /* give S same nonzero structure as A */
   hypre_CopyParCSRMatrix(A,S,0);

   S_diag = hypre_ParCSRMatrixDiag(S);
   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   S_diag_data = hypre_CSRMatrixData(S_diag);
   S_offd = hypre_ParCSRMatrixOffd(S);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   if (num_cols_offd)
   {
   	A_offd_data = hypre_CSRMatrixData(A_offd);
   	S_offd_j = hypre_CSRMatrixJ(S_offd);
   	S_offd_data = hypre_CSRMatrixData(S_offd);
   }

   for (i = 0; i < num_variables; i++)
   {
      diag = A_diag_data[A_diag_i[i]];

      /* compute scaling factor */
      row_scale = 0.0;
      if (diag < 0)
      {
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            row_scale = max(row_scale, A_diag_data[jA]);
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = max(row_scale, A_offd_data[jA]);
         }
      }
      else
      {
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            row_scale = min(row_scale, A_diag_data[jA]);
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = min(row_scale, A_offd_data[jA]);
         }
      }

      /* compute row entries of S */
      S_diag_data[A_diag_i[i]] = 0;
      if (diag < 0) 
      { 
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            S_diag_data[jA] = 0;
            if (A_diag_data[jA] > strength_threshold * row_scale)
            {
               S_diag_data[jA] = -1;
            }
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            S_offd_data[jA] = 0;
            if (A_offd_data[jA] > strength_threshold * row_scale)
            {
               S_offd_data[jA] = -1;
            }
         }
      }
      else
      {
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            S_diag_data[jA] = 0;
            if (A_diag_data[jA] < strength_threshold * row_scale)
            {
               S_diag_data[jA] = -1;
            }
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            S_offd_data[jA] = 0;
            if (A_offd_data[jA] < strength_threshold * row_scale)
            {
               S_offd_data[jA] = -1;
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
         if (S_diag_data[jA])
         {
            S_diag_j[jS]    = S_diag_j[jA];
            S_diag_data[jS] = S_diag_data[jA];
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
         if (S_offd_data[jA])
         {
            S_offd_j[jS]    = S_offd_j[jA];
            S_offd_data[jS] = S_offd_data[jA];
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

   ones_vector = hypre_CreateParVector(comm,global_num_vars,row_starts);
   hypre_SetParVectorPartitioningOwner(ones_vector,0);
   hypre_InitializeParVector(ones_vector);
   hypre_SetParVectorConstantValues(ones_vector,-1.0);

   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);
   measure_vector = hypre_CreateParVector(comm,global_num_vars,row_starts);
   hypre_SetParVectorPartitioningOwner(measure_vector,0);
   hypre_VectorData(hypre_ParVectorLocalVector(measure_vector))=measure_array;

   hypre_ParCSRMatrixCommPkg(S) = hypre_ParCSRMatrixCommPkg(A);

   hypre_ParMatvecT(1.0, S, ones_vector, 0.0, measure_vector);

   hypre_ParCSRMatrixCommPkg(S) = NULL;

   /* this augments the measures */
   hypre_InitParAMGIndepSet(S, measure_array);

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   graph_array   = hypre_CTAlloc(int, num_variables+num_cols_offd);

   /* intialize measure array and graph array */
   for (i = 0; i < num_variables+num_cols_offd; i++)
   {
      graph_array[i] = i;
   }

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ... 
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(int, num_variables+num_cols_offd);

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   coarse_size = 0;
   graph_size = num_variables+num_cols_offd;
   while (1)
   {
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
      hypre_PrintCSRMatrix(S, filename);

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

      if (graph_size == 0)
         break;

      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_CommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg, i+1); j++)
                buf_data[index++] 
                 = measure_array[hypre_CommPkgSendMapElmt(comm_pkg,j)];
      }
 
      comm_handle = hypre_InitializeCommunication( 1, comm_pkg, buf_data, 
        &measure_array[num_variables]);
 
      hypre_FinalizeCommunication(comm_handle);   
 
      if (num_procs > 1)
      {
         S_ext      = hypre_ExtractBExt(S,A);
         S_ext_i    = hypre_CSRMatrixI(S_ext);
         S_ext_j    = hypre_CSRMatrixJ(S_ext);
         S_ext_data = hypre_CSRMatrixData(S_ext);
      }
   
      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/

      hypre_ParAMGIndepSet(S, S_ext, measure_array,
                           graph_array, graph_size, CF_marker);

      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_CommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++] 
                 = CF_marker[hypre_CommPkgSendMapElmt(comm_pkg,j)];
      }
 
      comm_handle2 = hypre_InitializeCommunication(11, comm_pkg, int_buf_data, 
        &CF_marker[num_variables]);
 
      hypre_FinalizeCommunication(comm_handle2);   
 
      /*------------------------------------------------
       * Apply heuristics.
       *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Set to be a C-pt
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            CF_marker[i] = C_PT;
            if (i < num_variables) coarse_size++;
         }

         /*---------------------------------------------
          * Set to be an F-pt
          *---------------------------------------------*/

         else if (measure_array[i] < 1)
         {
            CF_marker[i] = F_PT;
         }

         /*---------------------------------------------
          * Heuristic: points that interpolate from a
          * common C-pt are less dependent on each other.
          *
          * NOTE: CF_marker is used to help check for
          * common C-pt's in the heuristic.
          *---------------------------------------------*/

         else if (i < num_variables) /* local points */
 	 {
            /* marked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];

               if (CF_marker[j] > 0)
               {
                  if (S_diag_data[jS] < 0)
                  {
                     /* "remove" edge from S */
                     S_diag_data[jS] = -S_diag_data[jS];
                  }

                  /* IMPORTANT: consider all dependencies */
                  if (S_diag_data[jS])
                  {
                     /* temporarily modify CF_marker */
                     CF_marker[j] = COMMON_C_PT;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = num_variables+S_offd_j[jS];

               if (CF_marker[j] > 0)
               {
                  if (S_offd_data[jS] < 0)
                  {
                     /* "remove" edge from S */
                     S_offd_data[jS] = -S_offd_data[jS];
                  }

                  /* IMPORTANT: consider all dependencies */
                  if (S_offd_data[jS])
                  {
                     /* temporarily modify CF_marker */
                     CF_marker[j] = COMMON_C_PT;
                  }
               }
            }

            /* unmarked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               if (S_diag_data[jS] < 0)
               {
                  j = S_diag_j[jS];
		  break_var = 1;
                  /* check for common C-pt */
                  for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++)
                  {
                     k = S_diag_j[kS];

                     /* IMPORTANT: consider all dependencies */
                     if (S_diag_data[kS] && CF_marker[k] == COMMON_C_PT)
                     {
                        /* "remove" edge from S and update measure*/
                        S_diag_data[jS] = -S_diag_data[jS];
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

                        /* IMPORTANT: consider all dependencies */
                        if (S_offd_data[kS] &&
				CF_marker[k+num_variables] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_data[jS] = -S_diag_data[jS];
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               if (S_offd_data[jS] < 0)
               {
                  j = S_offd_j[jS];

                  /* check for common C-pt */
                  for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++)
                  {
                     k = S_ext_j[kS];
		     if (k >= col_1 && k < col_n)
		     {
			kc = k - col_1;

                        /* IMPORTANT: consider all dependencies */
                        if (S_ext_data[kS] && CF_marker[kc] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_data[jS] = -S_offd_data[jS];
                           measure_array[j+num_variables]--;
                           break;
                        }
                     }
		     else
		     {
		        kc = -1;
		        for (kk = 0; kk < num_cols_offd; kk++)
		        {
			   if (col_map_offd[kk] == k)
			   {
			      kc = kk;
			      break;
			   }
		        }
		        if (kc > -1 && S_ext_data[kS] && 
				CF_marker[kc+num_variables] == COMMON_C_PT)
		        {
                           /* "remove" edge from S and update measure*/
                           S_offd_data[jS] = -S_offd_data[jS];
                           measure_array[j+num_variables]--;
                           break;
		        }
                     }
                  }
               }
            }
 	 }
	 else /* boundary points */
 	 {
	    ic = i - num_variables;
            for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
            {
               j = S_ext_j[jS];
	       if (j >= col_1 && j < col_n)
	       {
	          jc = j - col_1;
                  if (CF_marker[jc] > 0)
                  {
                     if (S_ext_data[jS] < 0)
                     {
                        /* "remove" edge from S */
                        S_ext_data[jS] = -S_ext_data[jS];
                     }

                     /* IMPORTANT: consider all dependencies */
                     if (S_ext_data[jS])
                     {
                        /* temporarily modify CF_marker */
                        CF_marker[jc] = COMMON_C_PT;
                     }
                  }
	       }
	       else
	       {
		  jc = -1;
		  for (jj = 0; jj < num_cols_offd; jj++)
		  {
		     if (col_map_offd[jj] == j)
		     {
		        jc = jj + num_variables;
                  	if (CF_marker[jc] > 0)
                  	{
                     	   if (S_ext_data[jS] < 0)
                     	   {
                        	/* "remove" edge from S */
                        	S_ext_data[jS] = -S_ext_data[jS];
                     	   }

                     	   /* IMPORTANT: consider all dependencies */
                     	   if (S_ext_data[jS])
                     	   {
                        	/* temporarily modify CF_marker */
                        	CF_marker[jc] = COMMON_C_PT;
                     	   }
                  	}
			break;
                     }
		  }
	       }
            }

            /* unmarked dependencies */
            for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
            {
               if (S_ext_data[jS] < 0)
               {
                  j = S_ext_j[jS];

                  /* check for common C-pt */
		  if (j >= col_1 && j < col_n)
		  {
		    jc = j - col_1;
		    break_var = 1;
                    for (kS = S_diag_i[jc]; kS < S_diag_i[jc+1]; kS++)
                    {
                      k = S_diag_j[kS];

                      /* IMPORTANT: consider all dependencies */
                      if (S_diag_data[kS])
                      {
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_ext_data[jS] = -S_ext_data[jS];
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

                          /* IMPORTANT: consider all dependencies */
                          if (S_offd_data[kS])
                          {
                             if (CF_marker[k+num_variables] == COMMON_C_PT)
                             {
                                /* "remove" edge from S and update measure*/
                                S_ext_data[jS] = -S_ext_data[jS];
                                measure_array[jc]--;
                                break;
                             }
                          }
                       }
                    }
                  }
		  else
		  {
		    jc = -1;
		    for (jj = 0; jj < num_cols_offd; jj++)
		    {
		      if (col_map_offd[jj] == j)
		      {
			jc = jj;
			break;
		      }
		    }
		    if (jc > -1 )
                    {
		      for (kS = S_ext_i[jc]; kS < S_ext_i[jc+1]; kS++)
                      {
                      	k = S_ext_j[kS];

                      	/* IMPORTANT: consider all dependencies */
                      	if (k >= col_1 && k < col_n && S_ext_data[kS])
                      	{
                          if (CF_marker[k-col_1] == COMMON_C_PT)
                          {
                             /* "remove" edge from S and update measure*/
                             S_ext_data[jS] = -S_ext_data[jS];
                             measure_array[jc+num_variables]--;
                             break;
                          }
                        }
			else
			{
			  kc = -1;
			  for (kk = 0; kk < num_cols_offd; kk++)
			  {
			    if (col_map_offd[kk] == k)
			    {
			      kc = kk;
			      break;
			    }
			  }
			  if (kc > -1 &&
				CF_marker[kc+num_variables] == COMMON_C_PT)
			  {
                              /* "remove" edge from S and update measure*/
                              S_ext_data[jS] = -S_ext_data[jS];
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

          /* reset CF_marker */
          if (i < num_variables)
	  {
	     for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
	     {
                j = S_diag_j[jS];

                if (CF_marker[j] == COMMON_C_PT)
                {
                   CF_marker[j] = C_PT;
                }
             }
             for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
             {
                j = S_offd_j[jS]+num_variables;

                if (CF_marker[j] == COMMON_C_PT)
                {
                   CF_marker[j] = C_PT;
                }
             }
          }
	  else
	  {
	     ic = i - num_variables;
	     for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
	     {
                j = S_ext_j[jS];
		if (j >= col_1 && j < col_n &&
			CF_marker[j - col_1] == COMMON_C_PT)
                {
                   CF_marker[j - col_1] = C_PT;
                }
		else
		{
		   for (jj = 0; jj < num_cols_offd; jj++)
		   {
		      if (col_map_offd[jj] == j && 
			CF_marker[jj + num_variables] == COMMON_C_PT)
                      {
                   	 CF_marker[jj + num_variables] = C_PT;
			 break;
                      }
                   }
                }
             }
          }

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them, and F-pts
          * don't interpolate to neighbors they influence.
          *---------------------------------------------*/

         if ( (CF_marker[i] == C_PT) || (CF_marker[i] == F_PT) )
         {
            measure_array[i] = 0;

	    if (i < num_variables)
	    {  
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  if (S_diag_data[jS] < 0)
                  {
                     j = S_diag_j[jS];
               
                     /* "remove" edge from S */
                     S_diag_data[jS] = -S_diag_data[jS];
               
                     /* decrement measures of unmarked neighbors */
                     if (!CF_marker[j])
                     {
                        measure_array[j]--;
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  if (S_offd_data[jS] < 0)
                  {
                     j = S_offd_j[jS]+num_variables;
               
                     /* "remove" edge from S */
                     S_offd_data[jS] = -S_offd_data[jS];
               
                     /* decrement measures of unmarked neighbors */
                     if (!CF_marker[j])
                     {
                        measure_array[j]--;
                     }
                  }
               }
    	    }
	    else
	    {
	       ic = i - num_variables;
               for (jS = S_ext_i[ic]; jS < S_ext_i[ic+1]; jS++)
               {
                  if (S_ext_data[jS] < 0)
                  {
                     j = S_ext_j[jS];
               
                     /* "remove" edge from S */
                     S_ext_data[jS] = -S_ext_data[jS];

		     if (j >= col_1 && j < col_n)
		     {
			jc = j - col_1;
			               
                        /* decrement measures of unmarked neighbors */
                        if (!CF_marker[jc])
                        {
                           measure_array[jc]--;
                        }
                     }
		     else
		     {
			for (jj = 0; jj < num_cols_offd; jj++)
			{
			   if (col_map_offd[jj] == j && 
					!CF_marker[jj + num_variables])
			   {
                              /* decrement measures of unmarked neighbors */
                              measure_array[jj + num_variables]--;
                              break;
                           }
                        }
                     }
                  }
               }
	    }

            /* take point out of the subgraph */
            graph_size--;
            graph_array[ig] = graph_array[graph_size];
            graph_array[graph_size] = i;
            ig--;
         }
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_DestroyParVector(ones_vector);
   hypre_DestroyParVector(measure_vector);
   hypre_DestroyCSRMatrix(S_ext);
   hypre_TFree(graph_array);

   *S_ptr        = S;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

