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
   hypre_CommPkg      *comm_pkg_mS;
   hypre_CommHandle   *comm_handle;

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
   int		       num_recvs = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker;
   int                *CF_marker_offd;
   int                 coarse_size;
                      
   hypre_ParVector    *ones_vector;
   hypre_ParVector    *measure_vector;
   double             *measure_array;
   int                *graph_array;
   int                 graph_size;
   int                 global_graph_size;
                      
   double              diag, row_scale;
   int                 i, j, k, ic, jc, kc, jj, kk, jA, jS, kS, ig;
   int		       index, start, num_procs, jrow;
                      
   int                 ierr = 0;
   int                 break_var = 1;

   MPI_Datatype	       *recv_datatype, *send_datatype;
   MPI_Datatype	       types[2];
   MPI_Aint	       displs[2];
   int		       block_lens[2];
   int		       num_data, start_index;
   int		       *recv_vec_starts;

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
   num_recvs = hypre_CommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_CommPkgRecvVecStarts(comm_pkg);

   comm_pkg_mS = hypre_CTAlloc(hypre_CommPkg,1);
   hypre_CommPkgComm(comm_pkg_mS) = comm;
   hypre_CommPkgNumRecvs(comm_pkg_mS) = num_recvs;
   hypre_CommPkgRecvProcs(comm_pkg_mS) = hypre_CommPkgRecvProcs(comm_pkg);
   hypre_CommPkgRecvVecStarts(comm_pkg_mS) = recv_vec_starts; 
   hypre_CommPkgNumSends(comm_pkg_mS) = num_sends;
   hypre_CommPkgSendProcs(comm_pkg_mS) = hypre_CommPkgSendProcs(comm_pkg);
   hypre_CommPkgSendMapStarts(comm_pkg_mS) 
		= hypre_CommPkgSendMapStarts(comm_pkg);
   hypre_CommPkgSendMapElmts(comm_pkg_mS) = hypre_CommPkgSendMapElmts(comm_pkg);

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

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_variables+num_cols_offd; ig++)
      graph_array[ig] = ig;
/*   i = 0;
   if (num_procs > 1)
   {
      while (i < num_cols_offd && col_map_offd[i] < col_1)
      {
         graph_array[i] = i+num_variables;
         i++;
      }	
   }	

   for (ig = 0; ig < num_variables; ig++)
      graph_array[ig+i] = ig;
   
   for (ig = i; ig < num_cols_offd; ig++)
      graph_array[ig+num_variables] = ig+num_variables;
*/ 

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ... 
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(int, num_variables);
   CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
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
      S_ext      = hypre_ExtractBExt(S,A,1);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
      S_ext_data = hypre_CSRMatrixData(S_ext);
   }

   send_datatype = hypre_CTAlloc(MPI_Datatype,num_sends);
   recv_datatype = hypre_CTAlloc(MPI_Datatype,num_recvs);

   types[0] = MPI_DOUBLE;
   types[1] = MPI_DOUBLE;

   num_data = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_CommPkgSendMapStart(comm_pkg_mS, i);
      num_data += hypre_CommPkgSendMapStart(comm_pkg_mS, i+1)-start;
      for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg_mS, i+1); j++)
      {
         jrow = hypre_CommPkgSendMapElmt(comm_pkg_mS,j);
         num_data += S_diag_i[jrow+1] - S_diag_i[jrow]
         		+ S_offd_i[jrow+1] - S_offd_i[jrow];
      }
   }

   buf_data = hypre_CTAlloc(double, num_data);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      num_data = index;
      MPI_Address(&buf_data[index],&displs[0]);
      start = hypre_CommPkgSendMapStart(comm_pkg_mS, i);
      index += hypre_CommPkgSendMapStart(comm_pkg_mS, i+1) - start;
      for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg_mS, i+1); j++)
      {
         jrow = hypre_CommPkgSendMapElmt(comm_pkg_mS,j);
         index += S_diag_i[jrow+1] - S_diag_i[jrow]
	 		+ S_offd_i[jrow+1] - S_offd_i[jrow];
      }
      num_data = index-num_data;
      MPI_Type_struct(1,&num_data,displs,types,&send_datatype[i]);
      MPI_Type_commit(&send_datatype[i]); 
   }
   hypre_CommPkgSendMPITypes(comm_pkg_mS) = send_datatype;   

   for (i=0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
   {
      start_index = S_ext_i[recv_vec_starts[i]];
      block_lens[0] = recv_vec_starts[i+1] - recv_vec_starts[i];
      block_lens[1] = S_ext_i[recv_vec_starts[i+1]]-start_index; 
      MPI_Address(&measure_array[num_variables+recv_vec_starts[i]],&displs[0]);
      MPI_Address(&S_ext_data[start_index], &displs[1]);
      MPI_Type_struct(2,block_lens,displs,types,&recv_datatype[i]);
      MPI_Type_commit(&recv_datatype[i]); 
   }
   hypre_CommPkgRecvMPITypes(comm_pkg_mS) = recv_datatype;   
 
   
   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_CommPkgSendMapStart(comm_pkg_mS, i);
         for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg_mS, i+1); j++)
            buf_data[index++] 
                 = measure_array[hypre_CommPkgSendMapElmt(comm_pkg_mS,j)];
         for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg_mS, i+1); j++)
         {
            jrow = hypre_CommPkgSendMapElmt(comm_pkg_mS,j);
	    for (k = S_diag_i[jrow]; k < S_diag_i[jrow+1]; k++)
	    {
	       buf_data[index++] = S_diag_data[k];
            }
	    for (k = S_offd_i[jrow]; k < S_offd_i[jrow+1]; k++)
	    {
	       buf_data[index++] = S_offd_data[k];
            }
         }
      }
 
      comm_handle = hypre_InitializeCommunication( 0, comm_pkg_mS, NULL, NULL); 
 
      hypre_FinalizeCommunication(comm_handle);   
 
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
                  if (S_diag_data[jS] < 0)
                  {
                     CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  if (S_offd_data[jS] < 0)
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
                  if (S_ext_data[jS] < 0)
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

      MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

      if (global_graph_size == 0)
         break;

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/

      hypre_ParAMGIndepSet(S, S_ext, measure_array, graph_array, graph_size, 
				CF_marker, CF_marker_offd);

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
 
      comm_handle = hypre_InitializeCommunication(11, comm_pkg, int_buf_data, 
        CF_marker_offd);
 
      hypre_FinalizeCommunication(comm_handle);   
 
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

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
                     j = S_offd_j[jS];
               
                     /* "remove" edge from S */
                     S_offd_data[jS] = -S_offd_data[jS];
               
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
                  j = S_offd_j[jS];
   
                  if (CF_marker_offd[j] > 0)
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
                        CF_marker_offd[j] = COMMON_C_PT;
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
   				CF_marker_offd[k] == COMMON_C_PT)
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
   				CF_marker_offd[kc] == COMMON_C_PT)
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
                  if (S_ext_data[jS] < 0)
                  {
                     j = S_ext_j[jS];
	       	     if (j >= col_1 && j < col_n)
		     {
		        jj = j-col_1;
		     }
		     else
		     {
		        jj = -1;
			for (jc = 0; jc < num_cols_offd; jc++)
			{
			   if (col_map_offd[jc] == j) 
			   {
			      jj = jc+num_variables;
			      break;
			   }
			}
		     }
               
                     /* "remove" edge from S */
                     S_ext_data[jS] = -S_ext_data[jS];
               
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
		     for (jj = 0; jj < num_cols_offd; jj++)
		     {
		        if (col_map_offd[jj] == j)
		        {
                  	   if (CF_marker_offd[jj] > 0)
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
                        	 CF_marker_offd[jj] = COMMON_C_PT;
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
                                 if (CF_marker_offd[k] == COMMON_C_PT)
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
			         if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
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
               j = S_offd_j[jS];

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
			CF_marker_offd[jj] == COMMON_C_PT)
                     {
                        CF_marker_offd[jj] = C_PT;
		 	break;
                     }
                  }
               }
            }
         }
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_DestroyParVector(ones_vector);
   hypre_DestroyParVector(measure_vector);
   hypre_TFree(graph_array);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);

   for (i=0; i < num_sends; i++)
	MPI_Type_free(&send_datatype[i]);
   hypre_TFree(send_datatype);
   for (i=0; i < num_recvs; i++)
   	MPI_Type_free(&recv_datatype[i]);
   hypre_TFree(recv_datatype);
   hypre_TFree(comm_pkg_mS);
   if (num_procs > 1) hypre_DestroyCSRMatrix(S_ext);


   *S_ptr        = S;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/*==========================================================================*/
/* Ruge's coarsening algorithm 						    */
/*==========================================================================*/

int
hypre_ParAMGCoarsenRuge( hypre_ParCSRMatrix    *A,
                         double                 strength_threshold,
                         int                    measure_type,
                         int                    coarsen_type,
                         int                    debug_flag,
                         hypre_ParCSRMatrix   **S_ptr,
                         int                  **CF_marker_ptr,
                         int                   *coarse_size_ptr     )
{
   MPI_Comm         comm          = hypre_ParCSRMatrixComm(A);
   hypre_CommPkg   *comm_pkg      = hypre_ParCSRMatrixCommPkg(A);
   hypre_CommHandle *comm_handle;
   int		   *row_starts    = hypre_ParCSRMatrixRowStarts(A);
   int		   *col_map_offd_A= hypre_ParCSRMatrixColMapOffd(A);
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

   int             *measure_array;
   int             *graph_array;
   int             *graph_ptr;
   int              graph_size;
   int 	           *int_buf_data;
   int 	           *ci_array, *citilde_array;

   double           diag, row_scale;
   int              measure, max_measure;
   int              i, j, k, jA, jS, jS_offd, kS, ig;
   int		    ic, ji, jj, jk, jl, jm, index;
   int		    ci_size, ci_tilde_size;
   int		    ci_size_offd, ci_tilde_size_offd;
   int		    set_empty = 1;
   int		    C_i_nonempty = 0;
   int		    num_strong;
   int		    num_nonzeros;
   int		    num_procs, my_id;
   int		    num_sends = 0;
   int		    first_col, start;
   int		    col_0, col_n;

   int              ierr = 0;
   int              break_var = 0;
   double	    wall_time;

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
        hypre_GenerateMatvecCommunicationInfo(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_CommPkgNumSends(comm_pkg);
   num_strong = A_i[num_variables] - num_variables;

   S = hypre_CreateParCSRMatrix(comm, global_num_vars, global_num_vars,
			row_starts, row_starts,
			num_cols_offd, 0, 0);
/* row_starts is owned by A, col_starts = row_starts */
   hypre_SetParCSRMatrixRowStartsOwner(S,0);
   col_map_offd_S = hypre_CTAlloc(int,num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
	col_map_offd_S[i] = col_map_offd_A[i];

   hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;

   S_diag = hypre_ParCSRMatrixDiag(S);
   S_offd = hypre_ParCSRMatrixOffd(S);

   ST = hypre_CreateCSRMatrix(num_variables, num_variables, num_strong);

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
      if (diag < 0)
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            row_scale = max(row_scale, A_data[jA]);
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = max(row_scale, A_offd_data[jA]);
         }
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            row_scale = min(row_scale, A_data[jA]);
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            row_scale = min(row_scale, A_offd_data[jA]);
         }
      }

      /* compute row entries of S */
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
      S_ext      = hypre_ExtractBExt(S,A,0);
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
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables);
   graph_ptr   = hypre_CTAlloc(int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = i;
      graph_ptr[i] = i;
   }

   /*---------------------------------------------------
    * Initialize the C/F marker array
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(int, num_variables);
   for (i = 0; i < num_variables; i++)
   {
      CF_marker[i] = 0;
   }

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Coarsen Setup = %f\n",
                      my_id, wall_time); 
      fflush(NULL);
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   coarse_size = 0;
   graph_size = num_variables;

   /* first coarsening phase */

   while (graph_size > 0)
   {
      /*------------------------------------------------
       * pick an i with maximal measure
       *------------------------------------------------*/

      max_measure = -1;
      for (ic=0; ic < graph_size; ic++)
      {
	 measure = measure_array[graph_array[ic]];
	 if (measure > max_measure)
	 {
	    i = graph_array[ic];
	    ig = ic;
	    max_measure = measure;
	 }
      }

      /* make i a coarse point */

      CF_marker[i] = 1;
      measure_array[i] = -1;
      coarse_size++;
      graph_size--;
      graph_array[ig] = graph_array[graph_size];
      graph_ptr[graph_array[graph_size]] = ig;

      /* examine its connections, S_i^T */

      for (ji = ST_i[i]; ji < ST_i[i+1]; ji++)
      {
	 jj = ST_j[ji];
   	 if (measure_array[jj] != -1)
	 {
	    CF_marker[jj] = -1;
	    measure_array[jj] = -1;
	    graph_size--;
	    graph_array[graph_ptr[jj]] = graph_array[graph_size];
            graph_ptr[graph_array[graph_size]] = graph_ptr[jj];
	    for (jl = S_i[jj]; jl < S_i[jj+1]; jl++)
	    {
	       index = S_j[jl];
	       if (measure_array[index] != -1)
	          measure_array[index]++;
	    }
	 }
      }
      
      for (ji = S_i[i]; ji < S_i[i+1]; ji++)
      {
	 index = S_j[ji];
	 if (measure_array[index] != -1)
	    measure_array[index]--;
      }
   } 

   hypre_TFree(measure_array);
   hypre_DestroyCSRMatrix(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Coarsen 1st pass = %f\n",
                     my_id, wall_time); 
      fflush(NULL);
   }

   /* second pass, check fine points for coarse neighbors 
      for coarsen_type = 2, the second pass includes
      off-processore boundary points */

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   if (coarsen_type == 2)
   {
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/
    
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(int, hypre_CommPkgSendMapStart(comm_pkg,
                                                   num_sends));
    
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_CommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_CommPkgSendMapElmt(comm_pkg,j)];
      }
    
      comm_handle = hypre_InitializeCommunication(11, comm_pkg, int_buf_data,
        CF_marker_offd);
    
      hypre_FinalizeCommunication(comm_handle);
    
      ci_array = hypre_CTAlloc(int,num_cols_offd);
      citilde_array = hypre_CTAlloc(int,num_cols_offd);

      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] == -1)
         {
            ci_tilde_size = 0;
            ci_tilde_size_offd = 0;
            ci_size = 0;
            ci_size_offd = 0;
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i+1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
                  graph_array[ci_size++] = j;
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
                  ci_array[ci_size_offd++] = j;
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
                     for (jl=0; jl < ci_size; jl++)
                     {
                        if (graph_array[jl] == index)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                     if (!set_empty) break;
                  }
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = 1;
                        coarse_size++;
                        for (jj=0 ; jj < ci_tilde_size; jj++)
                        {
                           CF_marker[graph_ptr[jj]] = -1;
                           coarse_size--;
                        }
                        ci_tilde_size = 0;
                        break_var = 0;
                        break;
                     }
                     else
                     {
                        graph_ptr[ci_tilde_size++] = j;
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
                           for (jl=0; jl < ci_size; jl++)
                           {
                              if (graph_array[jl] == index)
                              {
                                 set_empty = 0;
                                 break;
                              }
                           }
                           if (!set_empty) break;
                        }
                        else
                        {
                           for (jk = 0; jk < num_cols_offd; jk++)
                           {
                              if (col_map_offd[jk] == index)
                              {
                                 for (jl=0; jl < ci_size_offd; jl++)
                                 {
                                    if (ci_array[jl] == jk)
                                    {
                                       set_empty = 0;
                                       break;
                                    }
                                 }
                              }
                              if (!set_empty) break;
                           }
                           if (!set_empty) break;
                        }
                     }
                     if (set_empty)
                     {
                        if (C_i_nonempty)
                        {
                           CF_marker[i] = 1;
                           coarse_size++;
                           for (jj=0 ; jj < ci_tilde_size; jj++)
                           {
                              CF_marker[graph_ptr[jj]] = -1;
                              coarse_size--;
                           }
                           for (jj=0 ; jj < ci_tilde_size_offd; jj++)
                           {
                              CF_marker_offd[citilde_array[jj]] = -1;
                           }
                           ci_tilde_size = 0;
                           ci_tilde_size_offd = 0;
                           break;
                        }
                        else
                        {
                           citilde_array[ci_tilde_size_offd++] = j;
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
   	    ci_tilde_size = 0;
   	    ci_size = 0;
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] > 0)
   	          graph_array[ci_size++] = j;
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
   		     for (jl=0; jl < ci_size; jl++)
   		     {
   		        if (graph_array[jl] == index)
   		        {
   		           set_empty = 0;
   		           break;
   		        }
   	             }
   	             if (!set_empty) break;
   	          }
   	          if (set_empty)
   	          {
   		     if (C_i_nonempty)
   		     {
   		        CF_marker[i] = 1;
   		        coarse_size++;
   		        for (jj=0 ; jj < ci_tilde_size; jj++)
   		        {
   			   CF_marker[graph_ptr[jj]] = -1;
   		           coarse_size--;
   		        }
   		        ci_tilde_size = 0;
   		        break;
   		     }
   		     else
   		     {
   		        graph_ptr[ci_tilde_size++] = j;
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
      fflush(NULL);
   }

   /* third pass, check boundary fine points for coarse neighbors */

   if (coarsen_type == 3)
   {
      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(int, hypre_CommPkgSendMapStart(comm_pkg,
                                                num_sends));

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
 
      comm_handle = hypre_InitializeCommunication(11, comm_pkg, int_buf_data, 
     		CF_marker_offd);
 
      hypre_FinalizeCommunication(comm_handle);   

      ci_array = hypre_CTAlloc(int,num_cols_offd);
      citilde_array = hypre_CTAlloc(int,num_cols_offd);
   }

   if (coarsen_type > 1)
   { 
      for (i=0; i < num_cols_offd; i++)
      {
         if (CF_marker_offd[i] == -1)
         {
   	    ci_tilde_size = 0;
   	    ci_tilde_size_offd = 0;
   	    ci_size = 0;
   	    ci_size_offd = 0;
   	    for (ji = S_ext_i[i]; ji < S_ext_i[i+1]; ji++)
   	    {
   	       j = S_ext_j[ji];
   	       if (j > col_0 && j < col_n)
   	       {
   	          j = j - first_col;
   	          if (CF_marker[j] > 0)
   	             graph_array[ci_size++] = j;
   	       }
   	       else
   	       {
   	          for (jj = 0; jj < num_cols_offd; jj++)
   	          {
   		     if (col_map_offd[jj] == j && CF_marker_offd[jj] > 0)
   	                ci_array[ci_size_offd++] = jj;
    	          }	
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
   		        for (jl=0; jl < ci_size; jl++)
   		        {
   		           if (graph_array[jl] == index)
   		           {
   		              set_empty = 0;
   		              break;
   		           }
   	                }
   	                if (!set_empty) break;
   	             }
   	             for (jj = S_offd_i[j]; jj < S_offd_i[j+1]; jj++)
   	             {
   		        index = S_offd_j[jj];
   		        for (jl=0; jl < ci_size_offd; jl++)
   		        {
   		           if (ci_array[jl] == index)
   		           {
   		              set_empty = 0;
   		              break;
   		           }
   	                }
   	                if (!set_empty) break;
   	             }
   	             if (set_empty)
   	             {
   		        if (C_i_nonempty)
   		        {
   		           CF_marker_offd[i] = 1;
   		           for (jj=0 ; jj < ci_tilde_size; jj++)
   		           {
   			      CF_marker[graph_ptr[jj]] = -1;
   		           }
   		           for (jj=0 ; jj < ci_tilde_size_offd; jj++)
   		           {
   			      CF_marker_offd[citilde_array[jj]] = -1;
   		           }
   		           ci_tilde_size = 0;
   		           ci_tilde_size_offd = 0;
   		           break;
   		        }
   		        else
   		        {
   		           graph_ptr[ci_tilde_size++] = j;
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
   	          for (jm = 0; jm < num_cols_offd; jm++)
   	          {
   		     if (col_map_offd[jm] == j && CF_marker_offd[jm] == -1)
   	             {
   	                set_empty = 1;
   	                for (jj = S_ext_i[jm]; jj < S_ext_i[jm+1]; jj++)
   	                {
   		           index = S_ext_j[jj];
   		           if (index > col_0 && index < col_n) 
   			   {
   			      for (jl=0; jl < ci_size; jl++)
   		              {
   		                 if (graph_array[jl] == index)
   		                 {
   		                    set_empty = 0;
   		                    break;
   		                 }
   	             	      }
   	              	      if (!set_empty) break;
   	                   }
   			   else
   			   {
   			      for (jk = 0; jk < num_cols_offd; jk++)
   			      {
   			         if (col_map_offd[jk] == index)
   			         {
   				    for (jl=0; jl < ci_size_offd; jl++)
   		           	    {
   		              	       if (ci_array[jl] == jk)
   		              	       {
   		                          set_empty = 0;
   		                          break;
   		                       }
   		                    }
   		                 }
   	              	         if (!set_empty) break;
   	             	      }
   	              	      if (!set_empty) break;
   	                   }
   	                }
   	                if (set_empty)
   	                {
   		           if (C_i_nonempty)
   		           {
   		              CF_marker_offd[i] = 1;
   		              for (jj=0 ; jj < ci_tilde_size; jj++)
   		              {
   			         CF_marker[graph_ptr[jj]] = -1;
   		              }
   		              for (jj=0 ; jj < ci_tilde_size_offd; jj++)
   		              {
   			         CF_marker_offd[citilde_array[jj]] = -1;
   		              }
   		              ci_tilde_size = 0;
   		              ci_tilde_size_offd = 0;
   		              break;
   		           }
   		           else
   		           {
   		              citilde_array[ci_tilde_size_offd++] = jm;
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
      }
      /*------------------------------------------------
       * Send boundary data for CF_marker back
       *------------------------------------------------*/

      comm_handle = hypre_InitializeCommunication(12, comm_pkg, CF_marker_offd, 
   			int_buf_data);
    
      hypre_FinalizeCommunication(comm_handle);   
    
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_CommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg, i+1); j++)
                CF_marker[hypre_CommPkgSendMapElmt(comm_pkg,j)] =
                   int_buf_data[index++]; 
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         if (coarsen_type == 3)
		printf("Proc = %d    Coarsen 3rd pass = %f\n",
                my_id, wall_time); 
         if (coarsen_type == 2)
		printf("Proc = %d    Coarsen 2nd pass = %f\n",
                my_id, wall_time); 
         fflush(NULL);
      }
   }
   if (coarsen_type == -2)
   {
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
    
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(int, hypre_CommPkgSendMapStart(comm_pkg,
                                                   num_sends));
    
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_CommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_CommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_CommPkgSendMapElmt(comm_pkg,j)];
      }
    
      comm_handle = hypre_InitializeCommunication(11, comm_pkg, int_buf_data,
        CF_marker_offd);
    
      hypre_FinalizeCommunication(comm_handle);
    
      ci_array = hypre_CTAlloc(int,num_cols_offd);
      citilde_array = hypre_CTAlloc(int,num_cols_offd);

      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] == -1 && (S_offd_i[i+1]-S_offd_i[i]) > 0)
         {
            ci_size = 0;
            ci_size_offd = 0;
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i+1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
                  graph_array[ci_size++] = j;
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
                  ci_array[ci_size_offd++] = j;
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
                        for (jl=0; jl < ci_size; jl++)
                        {
                           if (graph_array[jl] == index)
                           {
                              set_empty = 0;
                              break;
                           }
                        }
                        if (!set_empty) break;
                     }
                     else
                     {
                        for (jk = 0; jk < num_cols_offd; jk++)
                        {
                           if (col_map_offd[jk] == index)
                           {
                              for (jl=0; jl < ci_size_offd; jl++)
                              {
                                 if (ci_array[jl] == jk)
                                 {
                                    set_empty = 0;
                                    break;
                                 }
                              }
                           }
                           if (!set_empty) break;
                        }
                        if (!set_empty) break;
                     }
                  }
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = -2;
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
         fflush(NULL);
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
      hypre_TFree(citilde_array);
   }   
   hypre_TFree(graph_array);
   hypre_TFree(graph_ptr);
   if ((measure_type || coarsen_type != 1) && num_procs > 1)
   	hypre_DestroyCSRMatrix(S_ext); 
   
   *S_ptr           = S;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;
   
   return (ierr);
}
