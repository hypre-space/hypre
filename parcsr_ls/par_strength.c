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
  headers.h

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

int
hypre_BoomerAMGCreateS(hypre_ParCSRMatrix    *A,
                       double                 strength_threshold,
                       double                 max_row_sum,
                       int                    num_functions,
                       int                   *dof_func,
                       hypre_ParCSRMatrix   **S_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   double             *A_offd_data;
   int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   int 		      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   int                 global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
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
                 
   double              diag, row_scale, row_sum;
   int                 i, jA, jS;
                      
   int                 ierr = 0;

   int                 *dof_func_offd;
   int			num_sends;
   int		       *int_buf_data;
   int			index, start, j;
   
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
/* row_starts is owned by A, col_starts = row_starts */
   hypre_ParCSRMatrixSetRowStartsOwner(S,0);
   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(int, num_variables+1);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(int, num_nonzeros_diag);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(int, num_variables+1);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
        A_offd_data = hypre_CSRMatrixData(A_offd);
        hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(int, num_nonzeros_offd);
        S_offd_j = hypre_CSRMatrixJ(S_offd);
        hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(int, num_cols_offd);
        if (num_functions > 1)
	   dof_func_offd = hypre_CTAlloc(int, num_cols_offd);
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
      int_buf_data = hypre_CTAlloc(int,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
      hypre_TFree(int_buf_data);
   }

   /* give S same nonzero structure as A */
   hypre_ParCSRMatrixCopy(A,S,0);

#define HYPRE_SMP_PRIVATE i,diag,row_scale,row_sum,jA
#include "../utilities/hypre_smp_forloop.h"
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
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = hypre_max(row_scale, A_diag_data[jA]);
                  row_sum += A_diag_data[jA];
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
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
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = hypre_min(row_scale, A_diag_data[jA]);
                  row_sum += A_diag_data[jA];
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
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
         if (num_functions > 1)
         { 
            if (diag < 0) 
            { 
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
               {
                  if (A_diag_data[jA] <= strength_threshold * row_scale
                      || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
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
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
               {
                  if (A_diag_data[jA] >= strength_threshold * row_scale
                      || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
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

/* RDF: not sure if able to thread this loop */
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
   hypre_ParCSRMatrixCommPkg(S) = NULL;

   *S_ptr        = S;

   hypre_TFree(dof_func_offd);

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
  headers.h

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

int
hypre_BoomerAMGCreateSabs(hypre_ParCSRMatrix    *A,
                       double                 strength_threshold,
                       double                 max_row_sum,
                       int                    num_functions,
                       int                   *dof_func,
                       hypre_ParCSRMatrix   **S_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   double             *A_offd_data;
   int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   int 		      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   int                 global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
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
                 
   double              diag, row_scale, row_sum;
   int                 i, jA, jS;
                      
   int                 ierr = 0;

   int                 *dof_func_offd;
   int			num_sends;
   int		       *int_buf_data;
   int			index, start, j;
   
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
/* row_starts is owned by A, col_starts = row_starts */
   hypre_ParCSRMatrixSetRowStartsOwner(S,0);
   S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(int, num_variables+1);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(int, num_nonzeros_diag);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(int, num_variables+1);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
        A_offd_data = hypre_CSRMatrixData(A_offd);
        hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(int, num_nonzeros_offd);
        S_offd_j = hypre_CSRMatrixJ(S_offd);
        hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(int, num_cols_offd);
        if (num_functions > 1)
	   dof_func_offd = hypre_CTAlloc(int, num_cols_offd);
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
      int_buf_data = hypre_CTAlloc(int,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
	
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);   
      hypre_TFree(int_buf_data);
   }

   /* give S same nonzero structure as A */
   hypre_ParCSRMatrixCopy(A,S,0);

#define HYPRE_SMP_PRIVATE i,diag,row_scale,row_sum,jA
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < num_variables; i++)
   {
      diag = A_diag_data[A_diag_i[i]];

      /* compute scaling factor and row sum */
      row_scale = 0.0;
      row_sum = diag;
      if (num_functions > 1)
      {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = hypre_max(row_scale, fabs(A_diag_data[jA]));
                  row_sum += fabs(A_diag_data[jA]);
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
               {
                  row_scale = hypre_max(row_scale, fabs(A_offd_data[jA]));
                  row_sum += fabs(A_offd_data[jA]);
               }
            }
      }
      else
      {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               row_scale = hypre_max(row_scale, fabs(A_diag_data[jA]));
               row_sum += fabs(A_diag_data[jA]);
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               row_scale = hypre_max(row_scale, fabs(A_offd_data[jA]));
               row_sum += fabs(A_offd_data[jA]);
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
         if (num_functions > 1)
         { 
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
               {
                  if (fabs(A_diag_data[jA]) <= strength_threshold * row_scale
                      || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
               {
                  if (fabs(A_offd_data[jA]) <= strength_threshold * row_scale
                      || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                  {
                     S_offd_j[jA] = -1;
                  }
               }
         }
         else
         {
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
               {
                  if (fabs(A_diag_data[jA]) <= strength_threshold * row_scale)
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
               {
                  if (fabs(A_offd_data[jA]) <= strength_threshold * row_scale)
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

/* RDF: not sure if able to thread this loop */
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
   hypre_ParCSRMatrixCommPkg(S) = NULL;

   *S_ptr        = S;

   hypre_TFree(dof_func_offd);

   return (ierr);
}

/*--------------------------------------------------------------------------*/

int
hypre_BoomerAMGCreateSCommPkg(hypre_ParCSRMatrix *A, 
			      hypre_ParCSRMatrix *S,
			      int		 **col_offd_S_to_A_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   MPI_Status	      *status;
   MPI_Request	      *requests;
   hypre_ParCSRCommPkg     *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommPkg     *comm_pkg_S;
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(A);
   int    	      *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
                  
   hypre_CSRMatrix    *S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix    *S_offd = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j = hypre_CSRMatrixJ(S_offd);
   int    	      *col_map_offd_S;

   int                *recv_procs_A = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   int                *recv_vec_starts_A = 
				hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   int                *send_procs_A = 
				hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   int                *send_map_starts_A = 
				hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);
   int                *recv_procs_S;
   int                *recv_vec_starts_S;
   int                *send_procs_S;
   int                *send_map_starts_S;
   int                *send_map_elmts_S;
   int                *col_offd_S_to_A;

   int                *S_marker;
   int                *send_change;
   int                *recv_change;

   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int		       num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);                 
   int		       num_cols_offd_S;
   int                 i, j, jcol;
   int                 proc, cnt, proc_cnt, total_nz;
   int                 first_row;
                      
   int                 ierr = 0;

   int		       num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   int		       num_recvs_A = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   int		       num_sends_S;
   int		       num_recvs_S;
   int		       num_nonzeros;

   num_nonzeros = S_offd_i[num_variables];

   S_marker = NULL;
   if (num_cols_offd_A)
      S_marker = hypre_CTAlloc(int,num_cols_offd_A);

   for (i=0; i < num_cols_offd_A; i++)
      S_marker[i] = -1;

   for (i=0; i < num_nonzeros; i++)
   {
      jcol = S_offd_j[i];
      S_marker[jcol] = 0;
   }

   proc = 0;
   proc_cnt = 0;
   cnt = 0;
   num_recvs_S = 0;
   for (i=0; i < num_recvs_A; i++)
   {
      for (j=recv_vec_starts_A[i]; j < recv_vec_starts_A[i+1]; j++)
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
   col_map_offd_S = NULL;
   col_offd_S_to_A = NULL;
   if (num_recvs_A) recv_change = hypre_CTAlloc(int, num_recvs_A);
   if (num_sends_A) send_change = hypre_CTAlloc(int, num_sends_A);
   if (num_recvs_S) recv_procs_S = hypre_CTAlloc(int, num_recvs_S);
   recv_vec_starts_S = hypre_CTAlloc(int, num_recvs_S+1);
   if (num_cols_offd_S)
   {
      col_map_offd_S = hypre_CTAlloc(int,num_cols_offd_S);
      col_offd_S_to_A = hypre_CTAlloc(int,num_cols_offd_S);
   }
   if (num_cols_offd_S < num_cols_offd_A)
   {
      for (i=0; i < num_nonzeros; i++)
      {
         jcol = S_offd_j[i];
         S_offd_j[i] = S_marker[jcol];
      }

      proc = 0;
      proc_cnt = 0;
      cnt = 0;
      recv_vec_starts_S[0] = 0;
      for (i=0; i < num_recvs_A; i++)
      {
         for (j=recv_vec_starts_A[i]; j < recv_vec_starts_A[i+1]; j++)
         {
            if (S_marker[j] != -1)
            {
               col_map_offd_S[cnt] = col_map_offd_A[j];
               col_offd_S_to_A[cnt++] = j;
               proc = 1;
            }
         }
         recv_change[i] = j-cnt-recv_vec_starts_A[i]
				+recv_vec_starts_S[proc_cnt];
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
      for (i=0; i < num_recvs_A; i++)
      {
         for (j=recv_vec_starts_A[i]; j < recv_vec_starts_A[i+1]; j++)
         {
            col_map_offd_S[j] = col_map_offd_A[j];
            col_offd_S_to_A[j] = j;
         }
         recv_procs_S[i] = recv_procs_A[i];
         recv_vec_starts_S[i] = recv_vec_starts_A[i];
      }
      recv_vec_starts_S[num_recvs_A] = recv_vec_starts_A[num_recvs_A];
   } 

   requests = hypre_CTAlloc(MPI_Request,num_sends_A+num_recvs_A);
   j=0;
   for (i=0; i < num_sends_A; i++)
       MPI_Irecv(&send_change[i],1,MPI_INT,send_procs_A[i],
		0,comm,&requests[j++]);

   for (i=0; i < num_recvs_A; i++)
       MPI_Isend(&recv_change[i],1,MPI_INT,recv_procs_A[i],
		0,comm,&requests[j++]);

   status = hypre_CTAlloc(MPI_Status,j);
   MPI_Waitall(j,requests,status);
   hypre_TFree(status);

   num_sends_S = 0;
   total_nz = send_map_starts_A[num_sends_A];
   for (i=0; i < num_sends_A; i++)
   {
      if (send_change[i])
      {
	 if ((send_map_starts_A[i+1]-send_map_starts_A[i]) > send_change[i])
	    num_sends_S++;
      }
      else
	 num_sends_S++;
      total_nz -= send_change[i];
   }

   send_procs_S = NULL;
   if (num_sends_S)
      send_procs_S = hypre_CTAlloc(int,num_sends_S);
   send_map_starts_S = hypre_CTAlloc(int,num_sends_S+1);
   send_map_elmts_S = NULL;
   if (total_nz)
      send_map_elmts_S = hypre_CTAlloc(int,total_nz);


   proc = 0;
   proc_cnt = 0;
   for (i=0; i < num_sends_A; i++)
   {
      cnt = send_map_starts_A[i+1]-send_map_starts_A[i]-send_change[i];
      if (cnt)
      {
	 send_procs_S[proc_cnt++] = send_procs_A[i];
         send_map_starts_S[proc_cnt] = send_map_starts_S[proc_cnt-1]+cnt;
      }
   }

   comm_pkg_S = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(comm_pkg_S) = comm;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg_S) = num_recvs_S;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg_S) = recv_procs_S;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_S) = recv_vec_starts_S;
   hypre_ParCSRCommPkgNumSends(comm_pkg_S) = num_sends_S;
   hypre_ParCSRCommPkgSendProcs(comm_pkg_S) = send_procs_S;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_S) = send_map_starts_S;

   comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_S, col_map_offd_S,
			send_map_elmts_S);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   first_row = hypre_ParCSRMatrixFirstRowIndex(A);
   if (first_row)
      for (i=0; i < send_map_starts_S[num_sends_S]; i++)
          send_map_elmts_S[i] -= first_row;

   hypre_ParCSRCommPkgSendMapElmts(comm_pkg_S) = send_map_elmts_S;
  
   hypre_ParCSRMatrixCommPkg(S) = comm_pkg_S;
   hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;
   hypre_CSRMatrixNumCols(S_offd) = num_cols_offd_S;

   hypre_TFree(S_marker);
   hypre_TFree(send_change);
   hypre_TFree(recv_change);

   *col_offd_S_to_A_ptr = col_offd_S_to_A;

   return ierr;
} 
