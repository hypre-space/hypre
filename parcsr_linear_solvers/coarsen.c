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
  \item The underlying matrix storage scheme is a hypre_CSR matrix.
  \item We define the following temporary storage:
  \begin{itemize}
  \item S - a CSR matrix representing the "strength matrix".
  \item measure\_array - a double array containing the "measures" for
  each of the fine-grid points
  \item work\_array - a double array containing used in both the
  independent set routine and here to help with one of the heuristics
  \item graph\_array - an integer array containing the list of points
  in the "current subgraph" being considered in the coarsening process.
  \end{itemize}
  \item The graph of the "strength matrix" for A is a subgraph of the
  graph of A, but requires nonsymmetric storage even if A is
  symmetric.  This is because of the directional nature of the
  "strengh of dependence" notion (see below).  Since we are using
  nonsymmetric storage for A right now, this is not a problem.  If we
  ever add the ability to store A symmetrically, then we could store
  the strength graph as floats instead of doubles to save space.
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
  @param CF_marker_ptr [OUT]
  array indicating C/F points
  @param S_ptr [OUT]
  strength matrix
  
  @see */
/*--------------------------------------------------------------------------*/

int
hypre_AMGCoarsen( hypre_CSRMatrix    *A,
                  double              strength_threshold,
                  int               **CF_marker_ptr,
                  hypre_CSRMatrix   **S_ptr              )
{
   int             *CF_marker;
   hypre_CSRMatrix *S;

   int             *A_i           = hypre_CSRMatrixI(A);
   double          *A_data        = hypre_CSRMatrixData(A);
   int              num_variables = hypre_CSRMatrixNumRows(A);
                  
   int             *S_i;
   int             *S_j;
   double          *S_data;
                 
   double          *measure_array;
   double          *work_array;
   int             *graph_array;
   int              graph_start, IS_size;

   double           diag, row_scale;
   int              i, j, k, jA, jS, kS, ig;

   int              ierr = 0;

   /*---------------------------------------------------
    * Allocate memory.
    *---------------------------------------------------*/

   measure_array = hypre_CTAlloc(double, num_variables);
   work_array    = hypre_CTAlloc(double, num_variables);
   graph_array   = hypre_CTAlloc(int, num_variables);

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
    * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
    *----------------------------------------------------------------*/

   S = hypre_CreateCSRMatrix(num_variables, num_variables,
                             A_i[num_variables]);
   hypre_InitializeCSRMatrix(S);

   S_i           = hypre_CSRMatrixI(S);
   S_j           = hypre_CSRMatrixJ(S);
   S_data        = hypre_CSRMatrixData(S);

   for (i = 0; i < num_variables; i++)
   {
      diag = A_data[A_i[i]];

      /* compute scaling factor */
      row_scale = 0.0;
      for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
      {
         if (diag < 0) 
         { 
            row_scale = max(row_scale, A_data[jA]);
         }
         else
         {
            row_scale = min(row_scale, A_data[jA]);
         }
      }

      /* compute row entries of S */
      for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
      {
         if (diag < 0) 
         { 
            if (A_data[jA] > strength_threshold * row_scale)
            {
               S_data[jA] = 1;
            }
         }
         else
         {
            if (A_data[jA] < strength_threshold * row_scale)
            {
               S_data[jA] = 1;
            }
         }
      }
   }

   /*----------------------------------------------------------
    * Compute "measures" for the fine-grid points,
    * and initialize graph_array
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the *number* of influences
    * of variable i.
    *----------------------------------------------------------*/

   /* intialize measure array */
   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = 0.0;
      graph_array[i] = i;
   }

   for (i = 0; i < num_variables; i++)
   {
      for (jS = S_i[i]+1; jS < S_i[i+1]; jS++)
      {
         j = S_j[jS];
         measure_array[j] += S_data[jA];
      }
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   graph_start = 0;
   while (1)
   {
      /*------------------------------------------------
       * Pick an independent set (maybe maximal) of
       * points with maximal measure.
       *
       * NOTE: Each independent set is tacked onto the
       * end of the array, graph_array.
       *------------------------------------------------*/

      hypre_AMGIndepSet(S, measure_array, work_array,
                        &graph_array[graph_start],
                        (num_variables - graph_start),
                        &IS_size);

      if (IS_size == 0)  
         break;

      /*------------------------------------------------
       * Mark C-pts with zero measure array entries.
       *------------------------------------------------*/

      for (ig = graph_start; ig < graph_start + IS_size; ig++)
      {
         i = graph_array[ig];
         measure_array[i] = 0.0;
      }

      /*------------------------------------------------
       * zero out work array to help with heuristics
       *------------------------------------------------*/

      for (ig = graph_start; ig < num_variables; ig++)
      {
         i = graph_array[ig];

         for (jS = S_i[i]+1; jS < S_i[i+1]; jS++)
         {
            j = S_j[jS];
            work_array[j] = 0.0;
         }
      }

      /*------------------------------------------------
       * Update strength matrix and measure array.
       *------------------------------------------------*/

      for (ig = graph_start; ig < num_variables; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors they depend on.
          *---------------------------------------------*/

         if (measure_array[i] == 0)
         {
            for (jS = S_i[i]+1; jS < S_i[i+1]; jS++)
            {
               if (S_data[jS] > 0)
               {
                  j = S_j[jS];

                  /* "remove" edge from S */
                  S_data[jS] = -S_data[jS];

                  /* decrement measures of non-C-pt neighbors */
                  if (measure_array[j] > 0)
                  {
                     measure_array[j]--;
                  }
               }
            }
         }

         /*---------------------------------------------
          * Heuristic: F-pts that interpolate from a
          * common C-pt are less dependent on each other.
          *---------------------------------------------*/

         else
         {
            /* mark C-pts in work array */
            for (jS = S_i[i]+1; jS < S_i[i+1]; jS++)
            {
               if (S_data[jS] > 0)
               {
                  j = S_j[jS];
                  work_array[j] = 1.0;
               }
            }

            /* apply heuristic */
            for (jS = S_i[i]+1; jS < S_i[i+1]; jS++)
            {
               if (S_data[jS] > 0)
               {
                  j = S_j[jS];

                  /* C-pt dependence: "remove" edge from S */
                  if (measure_array[j] == 0)
                  {
                     S_data[jS] = -S_data[jS];
                  }

                  /* F-pt dependence: check for common C-pt */
                  else
                  {
                     for (kS = S_i[j]+1; kS < S_i[j+1]; kS++)
                     {
                        k = S_j[kS];

                        /* common C-pt: "remove" edge & update measure */
                        /* NOTE: need to also consider edges that have */
                        /* previously been removed.                    */
                        if ( (S_data[kS]) && (work_array[k]) )
                        {
                           S_data[jS] = -S_data[jS];
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }

            /* reset work array */
            for (jS = S_i[i]+1; jS < S_i[i+1]; jS++)
            {
               j = S_j[jS];
               work_array[j] = 0.0;
            }
         }
      }

      graph_start += IS_size;
   }

   /*---------------------------------------------------
    * Set CF marker array.
    *---------------------------------------------------*/

   CF_marker = graph_array;
   for (i = 0; i < num_variables; i++)
   {
      /* mark as C-pt */
      if (measure_array[i] == 0)
      {
         CF_marker[i] = 1;
      }

      /* mark as F-pt */
      else
      {
         CF_marker[i] = -1;
      }
   }

   *CF_marker_ptr = CF_marker;

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array);
   hypre_TFree(work_array);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * Debugging stuff
 *--------------------------------------------------------------------------*/

void debug_out(ST_data,ST_i,ST_j,num_variables)

int   *ST_data;
int   *ST_i;
int   *ST_j;
int    num_variables;

{
   hypre_CSRMatrix *ST;
   double  *matx_data;
   int      i;

   ST = hypre_CreateCSRMatrix(num_variables, num_variables, NULL);

   matx_data = hypre_CTAlloc(double,ST_i[num_variables]);
   for (i=0;i<ST_i[num_variables];i++)
   {
      matx_data[i] = (double) ST_data[i];
   }

   hypre_CSRMatrixData(ST) = matx_data;
   hypre_CSRMatrixI(ST)    = ST_i;
   hypre_CSRMatrixJ(ST)    = ST_j;
   hypre_CSRMatrixOwnsData(ST) = 0;  /* matrix does NOT own data */

   hypre_PrintCSRMatrix(ST,"st.matx");

   hypre_DestroyCSRMatrix(ST);
   hypre_TFree(matx_data);

}
