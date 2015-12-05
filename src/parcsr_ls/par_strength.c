/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.15 $
 ***********************************************************************EHEADER*/




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

HYPRE_Int
hypre_BoomerAMGCreateS(hypre_ParCSRMatrix    *A,
                       double                 strength_threshold,
                       double                 max_row_sum,
                       HYPRE_Int                    num_functions,
                       HYPRE_Int                   *dof_func,
                       hypre_ParCSRMatrix   **S_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   double             *A_offd_data;
   HYPRE_Int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int 		      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int                 global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int 		       num_nonzeros_diag;
   HYPRE_Int 		       num_nonzeros_offd = 0;
   HYPRE_Int 		       num_cols_offd = 0;
                  
   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   HYPRE_Int                *S_diag_i;
   HYPRE_Int                *S_diag_j;
   /* double             *S_diag_data; */
   hypre_CSRMatrix    *S_offd;
   HYPRE_Int                *S_offd_i;
   HYPRE_Int                *S_offd_j;
   /* double             *S_offd_data; */
                 
   double              diag, row_scale, row_sum;
   HYPRE_Int                 i, jA, jS;
                      
   HYPRE_Int                 ierr = 0;

   HYPRE_Int                 *dof_func_offd;
   HYPRE_Int			num_sends;
   HYPRE_Int		       *int_buf_data;
   HYPRE_Int			index, start, j;
   
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
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(HYPRE_Int, num_variables+1);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_diag);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(HYPRE_Int, num_variables+1);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
        A_offd_data = hypre_CSRMatrixData(A_offd);
        hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_offd);
        S_offd_j = hypre_CSRMatrixJ(S_offd);
        hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
        if (num_functions > 1)
	   dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
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
      int_buf_data = hypre_CTAlloc(HYPRE_Int,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
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

HYPRE_Int
hypre_BoomerAMGCreateSabs(hypre_ParCSRMatrix    *A,
                       double                 strength_threshold,
                       double                 max_row_sum,
                       HYPRE_Int                    num_functions,
                       HYPRE_Int                   *dof_func,
                       hypre_ParCSRMatrix   **S_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   double             *A_offd_data;
   HYPRE_Int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int 		      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int                 global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int 		       num_nonzeros_diag;
   HYPRE_Int 		       num_nonzeros_offd = 0;
   HYPRE_Int 		       num_cols_offd = 0;
                  
   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   HYPRE_Int                *S_diag_i;
   HYPRE_Int                *S_diag_j;
   /* double             *S_diag_data; */
   hypre_CSRMatrix    *S_offd;
   HYPRE_Int                *S_offd_i;
   HYPRE_Int                *S_offd_j;
   /* double             *S_offd_data; */
                 
   double              diag, row_scale, row_sum;
   HYPRE_Int                 i, jA, jS;
                      
   HYPRE_Int                 ierr = 0;

   HYPRE_Int                 *dof_func_offd;
   HYPRE_Int			num_sends;
   HYPRE_Int		       *int_buf_data;
   HYPRE_Int			index, start, j;
   
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
   hypre_CSRMatrixI(S_diag) = hypre_CTAlloc(HYPRE_Int, num_variables+1);
   hypre_CSRMatrixJ(S_diag) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_diag);
   S_offd = hypre_ParCSRMatrixOffd(S);
   hypre_CSRMatrixI(S_offd) = hypre_CTAlloc(HYPRE_Int, num_variables+1);

   S_diag_i = hypre_CSRMatrixI(S_diag);
   S_diag_j = hypre_CSRMatrixJ(S_diag);
   S_offd_i = hypre_CSRMatrixI(S_offd);

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
        A_offd_data = hypre_CSRMatrixData(A_offd);
        hypre_CSRMatrixJ(S_offd) = hypre_CTAlloc(HYPRE_Int, num_nonzeros_offd);
        S_offd_j = hypre_CSRMatrixJ(S_offd);
        hypre_ParCSRMatrixColMapOffd(S) = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
        if (num_functions > 1)
	   dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
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
      int_buf_data = hypre_CTAlloc(HYPRE_Int,hypre_ParCSRCommPkgSendMapStart(comm_pkg,
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

HYPRE_Int
hypre_BoomerAMGCreateSCommPkg(hypre_ParCSRMatrix *A, 
			      hypre_ParCSRMatrix *S,
			      HYPRE_Int		 **col_offd_S_to_A_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_MPI_Status	      *status;
   hypre_MPI_Request	      *requests;
   hypre_ParCSRCommPkg     *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommPkg     *comm_pkg_S;
   hypre_ParCSRCommHandle  *comm_handle;
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int    	      *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
                  
   hypre_CSRMatrix    *S_diag = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix    *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int                *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int                *S_offd_j = hypre_CSRMatrixJ(S_offd);
   HYPRE_Int    	      *col_map_offd_S = hypre_ParCSRMatrixColMapOffd(S);

   HYPRE_Int                *recv_procs_A = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   HYPRE_Int                *recv_vec_starts_A = 
				hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   HYPRE_Int                *send_procs_A = 
				hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   HYPRE_Int                *send_map_starts_A = 
				hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);
   HYPRE_Int                *recv_procs_S;
   HYPRE_Int                *recv_vec_starts_S;
   HYPRE_Int                *send_procs_S;
   HYPRE_Int                *send_map_starts_S;
   HYPRE_Int                *send_map_elmts_S;
   HYPRE_Int                *col_offd_S_to_A;

   HYPRE_Int                *S_marker;
   HYPRE_Int                *send_change;
   HYPRE_Int                *recv_change;

   HYPRE_Int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int		       num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);                 
   HYPRE_Int		       num_cols_offd_S;
   HYPRE_Int                 i, j, jcol;
   HYPRE_Int                 proc, cnt, proc_cnt, total_nz;
   HYPRE_Int                 first_row;
                      
   HYPRE_Int                 ierr = 0;

   HYPRE_Int		       num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   HYPRE_Int		       num_recvs_A = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   HYPRE_Int		       num_sends_S;
   HYPRE_Int		       num_recvs_S;
   HYPRE_Int		       num_nonzeros;

   num_nonzeros = S_offd_i[num_variables];

   S_marker = NULL;
   if (num_cols_offd_A)
      S_marker = hypre_CTAlloc(HYPRE_Int,num_cols_offd_A);

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
   if (col_map_offd_S) hypre_TFree(col_map_offd_S);
   col_map_offd_S = NULL;
   col_offd_S_to_A = NULL;
   if (num_recvs_A) recv_change = hypre_CTAlloc(HYPRE_Int, num_recvs_A);
   if (num_sends_A) send_change = hypre_CTAlloc(HYPRE_Int, num_sends_A);
   if (num_recvs_S) recv_procs_S = hypre_CTAlloc(HYPRE_Int, num_recvs_S);
   recv_vec_starts_S = hypre_CTAlloc(HYPRE_Int, num_recvs_S+1);
   if (num_cols_offd_S)
   {
      col_map_offd_S = hypre_CTAlloc(HYPRE_Int,num_cols_offd_S);
      col_offd_S_to_A = hypre_CTAlloc(HYPRE_Int,num_cols_offd_S);
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

   requests = hypre_CTAlloc(hypre_MPI_Request,num_sends_A+num_recvs_A);
   j=0;
   for (i=0; i < num_sends_A; i++)
       hypre_MPI_Irecv(&send_change[i],1,HYPRE_MPI_INT,send_procs_A[i],
		0,comm,&requests[j++]);

   for (i=0; i < num_recvs_A; i++)
       hypre_MPI_Isend(&recv_change[i],1,HYPRE_MPI_INT,recv_procs_A[i],
		0,comm,&requests[j++]);

   status = hypre_CTAlloc(hypre_MPI_Status,j);
   hypre_MPI_Waitall(j,requests,status);
   hypre_TFree(status);
   hypre_TFree(requests);

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
      send_procs_S = hypre_CTAlloc(HYPRE_Int,num_sends_S);
   send_map_starts_S = hypre_CTAlloc(HYPRE_Int,num_sends_S+1);
   send_map_elmts_S = NULL;
   if (total_nz)
      send_map_elmts_S = hypre_CTAlloc(HYPRE_Int,total_nz);


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

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCreate2ndS : creates strength matrix on coarse points
 * for second coarsening pass in aggressive coarsening (S*S+2S)
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoomerAMGCreate2ndS( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker, 
	HYPRE_Int num_paths, HYPRE_Int *coarse_row_starts, hypre_ParCSRMatrix **C_ptr)
{
   MPI_Comm 	   comm = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommPkg *tmp_comm_pkg;
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   
   HYPRE_Int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   
   HYPRE_Int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

   HYPRE_Int	num_cols_diag_S = hypre_CSRMatrixNumCols(S_diag);
   HYPRE_Int	num_cols_offd_S = hypre_CSRMatrixNumCols(S_offd);
   
   hypre_ParCSRMatrix *S2;
   HYPRE_Int		      *col_map_offd_C = NULL;

   hypre_CSRMatrix *C_diag;

   HYPRE_Int          *C_diag_data = NULL;
   HYPRE_Int             *C_diag_i;
   HYPRE_Int             *C_diag_j = NULL;

   hypre_CSRMatrix *C_offd;

   HYPRE_Int          *C_offd_data=NULL;
   HYPRE_Int             *C_offd_i;
   HYPRE_Int             *C_offd_j=NULL;

   HYPRE_Int              C_diag_size;
   HYPRE_Int              C_offd_size;
   HYPRE_Int		    num_cols_offd_C = 0;
   
   HYPRE_Int             *S_ext_diag_i = NULL;
   HYPRE_Int             *S_ext_diag_j = NULL;
   HYPRE_Int              S_ext_diag_size = 0;

   HYPRE_Int             *S_ext_offd_i = NULL;
   HYPRE_Int             *S_ext_offd_j = NULL;
   HYPRE_Int              S_ext_offd_size = 0;

   HYPRE_Int		   *CF_marker_offd = NULL;

   HYPRE_Int		   *S_marker = NULL;
   HYPRE_Int		   *S_marker_offd = NULL;
   HYPRE_Int		   *temp = NULL;

   HYPRE_Int             *fine_to_coarse = NULL;
   HYPRE_Int             *fine_to_coarse_offd = NULL;
   HYPRE_Int		   *map_S_to_C = NULL;

   HYPRE_Int 	            num_sends = 0;
   HYPRE_Int 	            num_recvs = 0;
   HYPRE_Int 	           *send_map_starts;
   HYPRE_Int 	           *tmp_send_map_starts = NULL;
   HYPRE_Int 	           *send_map_elmts;
   HYPRE_Int 	           *recv_vec_starts;
   HYPRE_Int 	           *tmp_recv_vec_starts = NULL;
   HYPRE_Int 	           *int_buf_data = NULL;

   HYPRE_Int              i, j, k;
   HYPRE_Int              i1, i2, i3;
   HYPRE_Int              jj1, jj2, jcol, jrow, j_cnt;
   
   HYPRE_Int              jj_count_diag, jj_count_offd;
   HYPRE_Int              jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int              cnt, cnt_offd, cnt_diag;
   HYPRE_Int 		    num_procs, my_id;
   HYPRE_Int 		    value, index;
   HYPRE_Int		    num_coarse;
   HYPRE_Int		    num_coarse_offd;
   HYPRE_Int		    num_nonzeros;
   HYPRE_Int		    num_nonzeros_diag;
   HYPRE_Int		    num_nonzeros_offd;
   HYPRE_Int		    global_num_coarse;
   HYPRE_Int		    my_first_cpt, my_last_cpt;

   HYPRE_Int *S_int_i = NULL;
   HYPRE_Int *S_int_j = NULL;
   HYPRE_Int *S_ext_i = NULL;
   HYPRE_Int *S_ext_j = NULL;

   /*-----------------------------------------------------------------------
    *  Extract S_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product 
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = coarse_row_starts[0];
   my_last_cpt = coarse_row_starts[1]-1;
   if (my_id == (num_procs -1)) global_num_coarse = coarse_row_starts[1];
   hypre_MPI_Bcast(&global_num_coarse, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = coarse_row_starts[my_id];
   my_last_cpt = coarse_row_starts[my_id+1]-1;
   global_num_coarse = coarse_row_starts[num_procs];
#endif

   if (num_cols_offd_S)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_S);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_S);
   }

   if (num_cols_diag_S) fine_to_coarse = hypre_CTAlloc(HYPRE_Int, num_cols_diag_S);

   num_coarse = 0;
   for (i=0; i < num_cols_diag_S; i++)
   {
      if (CF_marker[i] > 0) 
      {
         fine_to_coarse[i] = num_coarse + my_first_cpt;
         num_coarse++;
      }
      else
      {
         fine_to_coarse[i] = -1;
      }
   }

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
      int_buf_data = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends]);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
            int_buf_data[index++]
                 = fine_to_coarse[send_map_elmts[j]];
      }
                                                                                
      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
           fine_to_coarse_offd);
                                                                                
      for (i=0; i < num_cols_diag_S; i++)
         if (CF_marker[i] > 0) 
            fine_to_coarse[i] -= my_first_cpt;

      hypre_ParCSRCommHandleDestroy(comm_handle);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
         {
            int_buf_data[index++] = CF_marker[send_map_elmts[j]];
         }
      }
                                                                                
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                CF_marker_offd);
                                                                                
      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data);

      S_int_i = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends]+1);
      S_ext_i = hypre_CTAlloc(HYPRE_Int, recv_vec_starts[num_recvs]+1);

/*--------------------------------------------------------------------------
 * generate S_int_i through adding number of coarse row-elements of offd and diag
 * for corresponding rows. S_int_i[j+1] contains the number of coarse elements of
 * a row j (which is determined through send_map_elmts)
 *--------------------------------------------------------------------------*/
      S_int_i[0] = 0;
      j_cnt = 0;
      num_nonzeros = 0;
      for (i=0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
         {
            jrow = send_map_elmts[j];
            index = 0;
            for (k = S_diag_i[jrow]; k < S_diag_i[jrow+1]; k++)
            {
	       if (CF_marker[S_diag_j[k]] > 0) index++;
	    }
            for (k = S_offd_i[jrow]; k < S_offd_i[jrow+1]; k++)
            {
	       if (CF_marker_offd[S_offd_j[k]] > 0) index++;
	    }
            S_int_i[++j_cnt] = index;
            num_nonzeros += S_int_i[j_cnt];
         }
      }
                                                                                
/*--------------------------------------------------------------------------
 * initialize communication
 *--------------------------------------------------------------------------*/
      if (num_procs > 1)
         comm_handle = 
		hypre_ParCSRCommHandleCreate(11,comm_pkg,&S_int_i[1],&S_ext_i[1]);

      if (num_nonzeros) S_int_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros);

      tmp_send_map_starts = hypre_CTAlloc(HYPRE_Int, num_sends+1);
      tmp_recv_vec_starts = hypre_CTAlloc(HYPRE_Int, num_recvs+1);
   
      tmp_send_map_starts[0] = 0;
      j_cnt = 0;
      for (i=0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
         {
            jrow = send_map_elmts[j];
            for (k=S_diag_i[jrow]; k < S_diag_i[jrow+1]; k++)
            {
               if (CF_marker[S_diag_j[k]] > 0)
		  S_int_j[j_cnt++] = fine_to_coarse[S_diag_j[k]]+my_first_cpt;
            }
            for (k=S_offd_i[jrow]; k < S_offd_i[jrow+1]; k++)
            {
               if (CF_marker_offd[S_offd_j[k]] > 0)
                  S_int_j[j_cnt++] = fine_to_coarse_offd[S_offd_j[k]];
            }
         }
         tmp_send_map_starts[i+1] = j_cnt;
      }
                                                                                
      tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
      hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
      hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
      hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
      hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = 
		hypre_ParCSRCommPkgSendProcs(comm_pkg);
      hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = 
		hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = tmp_send_map_starts;
                                                                                
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
/*--------------------------------------------------------------------------
 * after communication exchange S_ext_i[j+1] contains the number of coarse elements
 * of a row j !
 * evaluate S_ext_i and compute num_nonzeros for S_ext
 *--------------------------------------------------------------------------*/
                                                                                
      for (i=0; i < recv_vec_starts[num_recvs]; i++)
                S_ext_i[i+1] += S_ext_i[i];
                                                                                
      num_nonzeros = S_ext_i[recv_vec_starts[num_recvs]];
                                                                                
      if (num_nonzeros) S_ext_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros);

      tmp_recv_vec_starts[0] = 0;
      for (i=0; i < num_recvs; i++)
         tmp_recv_vec_starts[i+1] = S_ext_i[recv_vec_starts[i+1]];

      hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = tmp_recv_vec_starts;
                                                                                
      comm_handle = hypre_ParCSRCommHandleCreate(11,tmp_comm_pkg,S_int_j,S_ext_j);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      hypre_TFree(tmp_send_map_starts);
      hypre_TFree(tmp_recv_vec_starts);
      hypre_TFree(tmp_comm_pkg);

      hypre_TFree(S_int_i);
      hypre_TFree(S_int_j);

      S_ext_diag_size = 0;
      S_ext_offd_size = 0;

      for (i=0; i < num_cols_offd_S; i++)
      {
         for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
         {
            if (S_ext_j[j] < my_first_cpt || S_ext_j[j] > my_last_cpt)
               S_ext_offd_size++;
            else
               S_ext_diag_size++;
         }
      }
      S_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_S+1);
      S_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_S+1);

      if (S_ext_diag_size)
      {
         S_ext_diag_j = hypre_CTAlloc(HYPRE_Int, S_ext_diag_size);
      }
      if (S_ext_offd_size)
      {
         S_ext_offd_j = hypre_CTAlloc(HYPRE_Int, S_ext_offd_size);
      }

      cnt_offd = 0;
      cnt_diag = 0;
      cnt = 0;
      num_coarse_offd = 0;
      for (i=0; i < num_cols_offd_S; i++)
      {
         if (CF_marker_offd[i] > 0) num_coarse_offd++;

         for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
         {
            i1 = S_ext_j[j];
            if (i1 < my_first_cpt || i1 > my_last_cpt)
               S_ext_offd_j[cnt_offd++] = i1;
            else
               S_ext_diag_j[cnt_diag++] = i1 - my_first_cpt;
         }
         S_ext_diag_i[++cnt] = cnt_diag;
         S_ext_offd_i[cnt] = cnt_offd;
      }

      hypre_TFree(S_ext_i);
      hypre_TFree(S_ext_j);

      cnt = 0;
      if (S_ext_offd_size || num_coarse_offd)
      {
         temp = hypre_CTAlloc(HYPRE_Int, S_ext_offd_size+num_coarse_offd);
         for (i=0; i < S_ext_offd_size; i++)
            temp[i] = S_ext_offd_j[i];
         cnt = S_ext_offd_size;
         for (i=0; i < num_cols_offd_S; i++)
            if (CF_marker_offd[i] > 0) temp[cnt++] = fine_to_coarse_offd[i];
      }
      if (cnt)
      {
         qsort0(temp, 0, cnt-1);

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

      if (S_ext_offd_size || num_coarse_offd)
         hypre_TFree(temp);

      for (i=0 ; i < S_ext_offd_size; i++)
         S_ext_offd_j[i] = hypre_BinarySearch(col_map_offd_C,
                                           S_ext_offd_j[i],
                                           num_cols_offd_C);
      if (num_cols_offd_S)
      {
         map_S_to_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_S);

         cnt = 0;
         for (i=0; i < num_cols_offd_S; i++)
         {
            if (CF_marker_offd[i] > 0)
            {
               while (fine_to_coarse_offd[i] > col_map_offd_C[cnt])
               {
                  cnt++;
               }
               map_S_to_C[i] = cnt++;
            }
            else
            {
               map_S_to_C[i] = -1;
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate and initialize some stuff.
    *-----------------------------------------------------------------------*/

   if (num_coarse) S_marker = hypre_CTAlloc(HYPRE_Int, num_coarse);

   for (i1 = 0; i1 < num_coarse; i1++)
      S_marker[i1] = -1;

   S_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_C);

   for (i1 = 0; i1 < num_cols_offd_C; i1++)
      S_marker_offd[i1] = -1;

   C_diag_i = hypre_CTAlloc(HYPRE_Int, num_coarse+1);
   C_offd_i = hypre_CTAlloc(HYPRE_Int, num_coarse+1);

   /*-----------------------------------------------------------------------
    *  Loop over rows of S
    *-----------------------------------------------------------------------*/
   
   cnt = 0; 
   num_nonzeros_diag = 0; 
   num_nonzeros_offd = 0; 
   for (i1 = 0; i1 < num_cols_diag_S; i1++)
   {
      
      if (CF_marker[i1] > 0)
      {
         for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1++)
         {
             jcol = S_diag_j[jj1];
	     if (CF_marker[jcol] > 0)
	     {
	        S_marker[fine_to_coarse[jcol]] = i1;
	        num_nonzeros_diag++;
	     }
         }
         for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1++)
         {
             jcol = S_offd_j[jj1];
	     if (CF_marker_offd[jcol] > 0)
	     {
	        S_marker_offd[map_S_to_C[jcol]] = i1;
	        num_nonzeros_offd++;
	     }
         }
         for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1++)
         {
             i2 = S_diag_j[jj1];
             for (jj2 = S_diag_i[i2]; jj2 < S_diag_i[i2+1]; jj2++)
	     {
	        i3 = S_diag_j[jj2];
	        if (CF_marker[i3] > 0 && S_marker[fine_to_coarse[i3]] != i1)
	        {
                   S_marker[fine_to_coarse[i3]] = i1;
	           num_nonzeros_diag++;
                }
             }
             for (jj2 = S_offd_i[i2]; jj2 < S_offd_i[i2+1]; jj2++)
	     {
	        i3 = S_offd_j[jj2];
	        if (CF_marker_offd[i3] > 0 && 
			S_marker_offd[map_S_to_C[i3]] != i1)
	        {
                   S_marker_offd[map_S_to_C[i3]] = i1;
	           num_nonzeros_offd++;
                }
             }
         }
         for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1++)
         {
             i2 = S_offd_j[jj1];
             for (jj2 = S_ext_diag_i[i2]; jj2 < S_ext_diag_i[i2+1]; jj2++)
	     {
	        i3 = S_ext_diag_j[jj2];
	        if (S_marker[i3] != i1)
	        {
                   S_marker[i3] = i1;
	           num_nonzeros_diag++;
                }
             }
             for (jj2 = S_ext_offd_i[i2]; jj2 < S_ext_offd_i[i2+1]; jj2++)
	     {
	        i3 = S_ext_offd_j[jj2];
	        if (S_marker_offd[i3] != i1)
	        {
                   S_marker_offd[i3] = i1;
	           num_nonzeros_offd++;
                }
             }
         }
         C_diag_i[++cnt] = num_nonzeros_diag;
         C_offd_i[cnt] = num_nonzeros_offd;
      }
   }

   if (num_nonzeros_diag)
   {
      C_diag_j = hypre_CTAlloc(HYPRE_Int,num_nonzeros_diag);
      C_diag_data = hypre_CTAlloc(HYPRE_Int,num_nonzeros_diag);
   }
   if (num_nonzeros_offd)
   {
      C_offd_j = hypre_CTAlloc(HYPRE_Int,num_nonzeros_offd);
      C_offd_data = hypre_CTAlloc(HYPRE_Int,num_nonzeros_offd);
   }

   for (i1 = 0; i1 < num_coarse; i1++)
      S_marker[i1] = -1;

   for (i1 = 0; i1 < num_cols_offd_C; i1++)
      S_marker_offd[i1] = -1;

   jj_count_diag = 0;
   jj_count_offd = 0;

   for (i1 = 0; i1 < num_cols_diag_S; i1++)
   {
      
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, C_{i1,i1} (for square matrices). 
       *--------------------------------------------------------------------*/
 
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;

      if (CF_marker[i1] > 0)
      {
         for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1++)
         {
             jcol = S_diag_j[jj1];
	     if (CF_marker[jcol] > 0)
	     {
	        S_marker[fine_to_coarse[jcol]] = jj_count_diag;
	        C_diag_j[jj_count_diag] = fine_to_coarse[jcol];
	        C_diag_data[jj_count_diag] = 2;
	        jj_count_diag++;
	     }
         }
         for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1++)
         {
             jcol = S_offd_j[jj1];
	     if (CF_marker_offd[jcol] > 0)
	     {
	        index = map_S_to_C[jcol];
	        S_marker_offd[index] = jj_count_offd;
	        C_offd_j[jj_count_offd] = index;
	        C_offd_data[jj_count_offd] = 2;
	        jj_count_offd++;
	     }
         }
         for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1++)
         {
             i2 = S_diag_j[jj1];
             for (jj2 = S_diag_i[i2]; jj2 < S_diag_i[i2+1]; jj2++)
	     {
	        i3 = S_diag_j[jj2];
	        if (CF_marker[i3] > 0)
	        {
		   if (S_marker[fine_to_coarse[i3]] < jj_row_begin_diag)
	           {
                      S_marker[fine_to_coarse[i3]] = jj_count_diag;
	              C_diag_j[jj_count_diag] = fine_to_coarse[i3];
	              C_diag_data[jj_count_diag]++;
	              jj_count_diag++;
                   }
	           else
	           {
	              C_diag_data[S_marker[fine_to_coarse[i3]]]++;
	           }
                }
             }
             for (jj2 = S_offd_i[i2]; jj2 < S_offd_i[i2+1]; jj2++)
	     {
	        i3 = S_offd_j[jj2];
	        if (CF_marker_offd[i3] > 0)
                {
	           index = map_S_to_C[i3];
		   if (S_marker_offd[index] < jj_row_begin_offd)
	           {
                      S_marker_offd[index] = jj_count_offd;
	              C_offd_j[jj_count_offd] = index;
	              C_offd_data[jj_count_offd]++;
	              jj_count_offd++;
                   }
                   else
                   {
	              C_offd_data[S_marker_offd[index]]++;
                   }
                }
             }
         }
         for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1++)
         {
             i2 = S_offd_j[jj1];
             for (jj2 = S_ext_diag_i[i2]; jj2 < S_ext_diag_i[i2+1]; jj2++)
	     {
	        i3 = S_ext_diag_j[jj2];
	        if (S_marker[i3] < jj_row_begin_diag)
	        {
                   S_marker[i3] = jj_count_diag;
	           C_diag_j[jj_count_diag] = i3;
	           C_diag_data[jj_count_diag]++;
	           jj_count_diag++;
                }
	        else
	        {
	           C_diag_data[S_marker[i3]]++;
	        }
             }
             for (jj2 = S_ext_offd_i[i2]; jj2 < S_ext_offd_i[i2+1]; jj2++)
	     {
	        i3 = S_ext_offd_j[jj2];
	        if (S_marker_offd[i3] < jj_row_begin_offd)
	        {
                   S_marker_offd[i3] = jj_count_offd;
	           C_offd_j[jj_count_offd] = i3;
	           C_offd_data[jj_count_offd]++;
	           jj_count_offd++;
                }
                else
                {
	           C_offd_data[S_marker_offd[i3]]++;
                }
             }
         }
      }
   }

   cnt = 0;

   for (i=0; i < num_coarse; i++)
   {
      for (j=C_diag_i[i]; j < C_diag_i[i+1]; j++)
      {
         jcol = C_diag_j[j];
         if (C_diag_data[j] >= num_paths && jcol != i)
            C_diag_j[cnt++] = jcol;
      }
      C_diag_i[i] = cnt;
   }

   if (num_nonzeros_diag) hypre_TFree(C_diag_data);
   for (i=num_coarse; i > 0; i--)
      C_diag_i[i] = C_diag_i[i-1];

   C_diag_i[0] = 0;

   cnt = 0;
   for (i=0; i < num_coarse; i++)
   {
      for (j=C_offd_i[i]; j < C_offd_i[i+1]; j++)
      {
         jcol = C_offd_j[j];
         if (C_offd_data[j] >= num_paths)
            C_offd_j[cnt++] = jcol;
      }
      C_offd_i[i] = cnt;
   }

   if (num_nonzeros_offd) hypre_TFree(C_offd_data);

   for (i=num_coarse; i > 0; i--)
      C_offd_i[i] = C_offd_i[i-1];

   C_offd_i[0] = 0;

   cnt = 0;
   for (i=0; i < num_cols_diag_S; i++)
   {
      if (CF_marker[i] > 0)
      {
         if (!(C_diag_i[cnt+1]-C_diag_i[cnt]) && 
			!(C_offd_i[cnt+1]-C_offd_i[cnt]))
            CF_marker[i] = 2;
         cnt++;
      }
   }

   C_diag_size = C_diag_i[num_coarse];
   C_offd_size = C_offd_i[num_coarse];

   S2 = hypre_ParCSRMatrixCreate(comm, global_num_coarse, 
	global_num_coarse, coarse_row_starts,
	coarse_row_starts, num_cols_offd_C, C_diag_size, C_offd_size);

   hypre_ParCSRMatrixOwnsRowStarts(S2) = 0;

   C_diag = hypre_ParCSRMatrixDiag(S2);
   hypre_CSRMatrixI(C_diag) = C_diag_i; 
   if (num_nonzeros_diag) hypre_CSRMatrixJ(C_diag) = C_diag_j; 

   C_offd = hypre_ParCSRMatrixOffd(S2);
   hypre_CSRMatrixI(C_offd) = C_offd_i; 
   hypre_ParCSRMatrixOffd(S2) = C_offd;

   if (num_cols_offd_C)
   {
      if (num_nonzeros_offd) hypre_CSRMatrixJ(C_offd) = C_offd_j; 
      hypre_ParCSRMatrixColMapOffd(S2) = col_map_offd_C;

   }

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   hypre_TFree(S_marker);   
   hypre_TFree(S_marker_offd);   
   hypre_TFree(S_ext_diag_i);
   hypre_TFree(fine_to_coarse);
   if (S_ext_diag_size)
   {
      hypre_TFree(S_ext_diag_j);
   }
   hypre_TFree(S_ext_offd_i);
   if (S_ext_offd_size)
   {
      hypre_TFree(S_ext_offd_j);
   }
   if (num_cols_offd_S) 
   {
      hypre_TFree(map_S_to_C);
      hypre_TFree(CF_marker_offd);
      hypre_TFree(fine_to_coarse_offd);
   }

   *C_ptr = S2;

   return 0;
   
}            

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCorrectCFMarker : corrects CF_marker after aggr. coarsening
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCorrectCFMarker(HYPRE_Int *CF_marker, HYPRE_Int num_var, HYPRE_Int *new_CF_marker)
{
   HYPRE_Int i, cnt;

   cnt = 0;
   for (i=0; i < num_var; i++)
   {
      if (CF_marker[i] > 0 )
      {
         if (CF_marker[i] == 1) CF_marker[i] = new_CF_marker[cnt++];
         else { CF_marker[i] = 1; cnt++;}
      }
   }

   return 0;
}
/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCorrectCFMarker2 : corrects CF_marker after aggr. coarsening,
 * but marks new F-points (previous C-points) as -2
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCorrectCFMarker2(HYPRE_Int *CF_marker, HYPRE_Int num_var, HYPRE_Int *new_CF_marker)
{
   HYPRE_Int i, cnt;

   cnt = 0;
   for (i=0; i < num_var; i++)
   {
      if (CF_marker[i] > 0 )
      {
         if (new_CF_marker[cnt] == -1) CF_marker[i] = -2;
         else CF_marker[i] = 1;
         cnt++;
      }
   }

   return 0;
}
