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
hypre_ParAMGCoarsen( hypre_ParCSRMatrix    *parA,
                     double                 strength_threshold,
                     hypre_ParCSRMatrix   **parS_ptr,
                     int                  **CF_marker_ptr,
                     int                   *coarse_size_ptr     )
{
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(parA);
   int                *A_i             = hypre_CSRMatrixI(A_diag);
   int                *A_j             = hypre_CSRMatrixJ(A_diag);
   double             *A_data          = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(parA);
   int                *A_offd_i;

   int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   int                 global_num_vars = hypre_ParCSRMatrixGlobalNumRows(parA);
   int 		       num_nonzeros_diag;
   int 		       num_nonzeros_offd = 0;
   int 		       num_cols_offd = 0;
                  
   hypre_ParCSRMatrix *parS;
   hypre_CSRMatrix    *S_diag;
   int                *S_i;
   int                *S_j;
   double             *S_data;
                 
   int                *CF_marker;
   int                 coarse_size;
                      
   double             *measure_array;
   int                *graph_array;
   int                 graph_size;
                      
   double              diag, row_scale;
   int                 i, j, k, jA, jS, kS, ig;
                      
   int                 ierr = 0;

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

   num_nonzeros_diag = A_i[num_variables];
   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   if (num_cols_offd) 
   {
	A_offd_i = hypre_CSRMatrixI(A_offd);
	num_nonzeros_offd = A_offd_i[num_variables];
   }

   parS = hypre_CreateParCSRMatrix(hypre_ParCSRMatrixComm(parA), 
			global_num_vars, global_num_vars,
			hypre_ParCSRMatrixRowStarts(parA),
			hypre_ParCSRMatrixColStarts(parA),
			num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
/* row_starts is owned by parA, col_starts = row_starts */
   hypre_SetParCSRMatrixRowStartsOwner(parS,0);
   hypre_InitializeParCSRMatrix(parS);

   S_diag = hypre_ParCSRMatrixDiag(parS);
   S_i           = hypre_CSRMatrixI(S_diag);
   S_j           = hypre_CSRMatrixJ(S_diag);
   S_data        = hypre_CSRMatrixData(S_diag);

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
   hypre_CSRMatrixNumNonzeros(S_diag) = jS;

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
   hypre_InitParAMGIndepSet(parS, measure_array);

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

      hypre_ParAMGIndepSet(parS, measure_array,
                           graph_array, graph_size, CF_marker);

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
            coarse_size++;
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

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them, and F-pts
          * don't interpolate to neighbors they influence.
          *---------------------------------------------*/

         if ( (CF_marker[i] == C_PT) || (CF_marker[i] == F_PT) )
         {
            measure_array[i] = 0;

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

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);

   *parS_ptr        = parS;
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

