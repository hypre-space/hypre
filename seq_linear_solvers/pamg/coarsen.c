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
  \item The routine returns the following:
  \begin{itemize}
  \item S - a CSR matrix representing the "strength matrix".  This is
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
hypre_AMGCoarsen( hypre_CSRMatrix    *A,
                  double              strength_threshold,
                  hypre_CSRMatrix   **S_ptr,
                  int               **CF_marker_ptr,
                  int                *coarse_size_ptr     )
{
   int             *A_i           = hypre_CSRMatrixI(A);
   int             *A_j           = hypre_CSRMatrixJ(A);
   double          *A_data        = hypre_CSRMatrixData(A);
   int              num_variables = hypre_CSRMatrixNumRows(A);
                  
   hypre_CSRMatrix *S;
   int             *S_i;
   int             *S_j;
   double          *S_data;
                 
   int             *CF_marker;
   int              coarse_size;

   double          *measure_array;
   int             *graph_array;
   int              graph_size;

   double           diag, row_scale;
   int              i, j, k, jA, jS, kS, ig;

   int              ierr = 0;

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

   S = hypre_CreateCSRMatrix(num_variables, num_variables,
                             A_i[num_variables]);
   hypre_InitializeCSRMatrix(S);

   S_i           = hypre_CSRMatrixI(S);
   S_j           = hypre_CSRMatrixJ(S);
   S_data        = hypre_CSRMatrixData(S);

   /* give S same nonzero structure as A */
   for (i = 0; i < num_variables; i++)
   {
      S_i[i] = A_i[i];
      for (jA = A_i[i]; jA < A_i[i+1]; jA++)
      {
         S_j[jA] = A_j[jA];
      }
   }
   S_i[num_variables] = A_i[num_variables];

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
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            row_scale = min(row_scale, A_data[jA]);
         }
      }

      /* compute row entries of S */
      S_data[A_i[i]] = 0;
      if (diag < 0) 
      { 
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            S_data[jA] = 0;
            if (A_data[jA] > strength_threshold * row_scale)
            {
               S_data[jA] = -1;
            }
         }
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            S_data[jA] = 0;
            if (A_data[jA] < strength_threshold * row_scale)
            {
               S_data[jA] = -1;
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
      S_i[i] = jS;
      for (jA = A_i[i]; jA < A_i[i+1]; jA++)
      {
         if (S_data[jA])
         {
            S_j[jS]    = S_j[jA];
            S_data[jS] = S_data[jA];
            jS++;
         }
      }
   }
   S_i[num_variables] = jS;
   hypre_CSRMatrixNumNonzeros(S) = jS;

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

   measure_array = hypre_CTAlloc(double, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      for (jS = S_i[i]; jS < S_i[i+1]; jS++)
      {
         j = S_j[jS];
         measure_array[j] -= S_data[jS];
      }
   }

   /* this augments the measures */
   hypre_InitAMGIndepSet(S, measure_array);

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array   = hypre_CTAlloc(int, num_variables);

   /* intialize measure array and graph array */
   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = i;
   }

   /*---------------------------------------------------
    * Initialize the C/F marker array
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(int, num_variables);

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   coarse_size = 0;
   graph_size = num_variables;
   while (1)
   {
      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) )
         {
            /* set to be an F-pt */
            CF_marker[i] = F_PT;

            /* make sure all dependencies have been accounted for */
            for (jS = S_i[i]; jS < S_i[i+1]; jS++)
            {
               if (S_data[jS] < 0)
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
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/

      hypre_AMGIndepSet(S, measure_array,
                        graph_array, graph_size, CF_marker);

      /*------------------------------------------------
       * Set C-pts and apply heuristics.
       *------------------------------------------------*/

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
            coarse_size++;

            for (jS = S_i[i]; jS < S_i[i+1]; jS++)
            {
               if (S_data[jS] < 0)
               {
                  j = S_j[jS];
               
                  /* "remove" edge from S */
                  S_data[jS] = -S_data[jS];
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j]--;
                  }
               }
            }
         }

         /*---------------------------------------------
          * Heuristic: points that depend on a common
          * C-pt are less dependent on each other.
          *
          * NOTE: CF_marker is used to help check for
          * common C-pt's in the heuristic.
          *---------------------------------------------*/

         else
         {
            /* marked dependences */
            for (jS = S_i[i]; jS < S_i[i+1]; jS++)
            {
               j = S_j[jS];

               if (CF_marker[j] > 0)
               {
                  if (S_data[jS] < 0)
                  {
                     /* "remove" edge from S */
                     S_data[jS] = -S_data[jS];
                  }

                  /* IMPORTANT: consider all dependencies */
                  if (S_data[jS])
                  {
                     /* temporarily modify CF_marker */
                     CF_marker[j] = COMMON_C_PT;
                  }
               }
            }

            /* unmarked dependences */
            for (jS = S_i[i]; jS < S_i[i+1]; jS++)
            {
               if (S_data[jS] < 0)
               {
                  j = S_j[jS];

                  /* check for common C-pt */
                  for (kS = S_i[j]; kS < S_i[j+1]; kS++)
                  {
                     k = S_j[kS];

                     /* IMPORTANT: consider all dependencies */
                     if (S_data[kS])
                     {
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_data[jS] = -S_data[jS];
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }

            /* reset CF_marker */
            for (jS = S_i[i]; jS < S_i[i+1]; jS++)
            {
               j = S_j[jS];

               if (CF_marker[j] == COMMON_C_PT)
               {
                  CF_marker[j] = C_PT;
               }
            }
         }
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);

   *S_ptr           = S;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/*==========================================================================*/
/* Ruge's coarsening algorithm 						    */
/*==========================================================================*/

int
hypre_AMGCoarsenRuge( hypre_CSRMatrix    *A,
                      double              strength_threshold,
                      hypre_CSRMatrix   **S_ptr,
                      int               **CF_marker_ptr,
                      int                *coarse_size_ptr     )
{
   int             *A_i           = hypre_CSRMatrixI(A);
   int             *A_j           = hypre_CSRMatrixJ(A);
   double          *A_data        = hypre_CSRMatrixData(A);
   int              num_variables = hypre_CSRMatrixNumRows(A);
                  
   hypre_CSRMatrix *S;
   int             *S_i;
   int             *S_j;
                 
   hypre_CSRMatrix *ST;
   int             *ST_i;
   int             *ST_j;
                 
   int             *CF_marker;
   int              coarse_size;

   int             *measure_array;
   int             *graph_array;
   int             *graph_ptr;
   int              graph_size;

   double           diag, row_scale;
   int              measure, max_measure;
   int              i, j, k, jA, jS, kS, ig;
   int		    ic, ji, jj, jl, index;
   int		    max_ci_size, ci_size, ci_tilde_size;
   int		    set_empty = 1;
   int		    C_i_nonempty = 0;
   int		    num_strong;

   int              ierr = 0;

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

   num_strong = A_i[num_variables] - num_variables;
   S = hypre_CreateCSRMatrix(num_variables, num_variables, num_strong);
   ST = hypre_CreateCSRMatrix(num_variables, num_variables, num_strong);

   S_i = hypre_CTAlloc(int,num_variables+1);
   hypre_CSRMatrixI(S) = S_i;

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
   }
   ST_i[num_variables] = A_i[num_variables];

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
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            row_scale = min(row_scale, A_data[jA]);
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
      }
   }

   S_j = hypre_CTAlloc(int,num_strong);
   hypre_CSRMatrixJ(S) = S_j;

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
      S_i[i] = jS;
      for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
      {
         if (ST_j[jA] != -1)
         {
            S_j[jS]    = ST_j[jA];
            jS++;
         }
      }
   }
   S_i[num_variables] = jS;
   hypre_CSRMatrixNumNonzeros(S) = jS;
   hypre_CSRMatrixNumNonzeros(ST) = jS;

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
    *
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = ST_i[i+1]-ST_i[i];
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

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

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

   /* second pass, check fine points for coarse neighbors */

   for (i=0; i < num_variables; i++)
   {
      if (CF_marker[i] == -1)
      {
	 max_ci_size = S_i[i+1]-S_i[i];
	 ci_tilde_size = max_ci_size;
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
		     for (jj=max_ci_size ; jj < ci_tilde_size; jj++)
		     {
			CF_marker[graph_array[jj]] = -1;
		     }
		     ci_tilde_size = max_ci_size;
		  }
		  else
		  {
		     graph_array[ci_tilde_size++] = index;
		     CF_marker[index] = 2;
		     C_i_nonempty = 1;
		     i--;
		     break;
		  }
	       }
	    }
	 }
      }
   }

		  	       


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   hypre_TFree(graph_ptr);
   hypre_DestroyCSRMatrix(ST);

   *S_ptr           = S;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

