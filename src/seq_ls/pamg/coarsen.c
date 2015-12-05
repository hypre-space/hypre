/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/




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

#define CPOINT 1
#define FPOINT -1
#define UNDECIDED 0




/**************************************************************
 *
 *      Coarsening routine
 *
 **************************************************************/
HYPRE_Int
hypre_AMGCoarsen( hypre_CSRMatrix    *A,
                  double              strength_threshold,
                  hypre_CSRMatrix    *S,
                  HYPRE_Int               **CF_marker_ptr,
                  HYPRE_Int                *coarse_size_ptr     )
{
   HYPRE_Int              num_variables = hypre_CSRMatrixNumRows(A);
/*
    HYPRE_Int             *A_i           = hypre_CSRMatrixI(A);
    HYPRE_Int             *A_j           = hypre_CSRMatrixJ(A);
    double          *A_data        = hypre_CSRMatrixData(A);
*/                
   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;
   double          *S_data;
                 
   HYPRE_Int             *CF_marker;
   HYPRE_Int              coarse_size;

   double          *measure_array;
   HYPRE_Int             *graph_array;
   HYPRE_Int              graph_size;

/*  double           diag, row_scale;
    HYPRE_Int              jS; */

   HYPRE_Int              i, j, k, jS, kS, ig;

   HYPRE_Int              ierr = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   S_i           = hypre_CSRMatrixI(S);
   S_j           = hypre_CSRMatrixJ(S);
   S_data        = hypre_CSRMatrixData(S);

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
         measure_array[j]++;
      }
   }

   /* this augments the measures */
   hypre_InitAMGIndepSet(S, measure_array, 1.0);

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array   = hypre_CTAlloc(HYPRE_Int, num_variables);

   /* intialize measure array and graph array */
   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = i;
   }

   /*---------------------------------------------------
    * Initialize the C/F marker array
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);

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

      if (graph_size == 0)
         break;

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/

      hypre_AMGIndepSet(S, measure_array, 1.0,
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

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/*==========================================================================*/
/* Ruge's coarsening algorithm 						    */
/*==========================================================================*/

HYPRE_Int
hypre_AMGCoarsenRuge( hypre_CSRMatrix    *A,
                      double              strength_threshold,
                      hypre_CSRMatrix    *S,
                      HYPRE_Int               **CF_marker_ptr,
                      HYPRE_Int                *coarse_size_ptr     )
{

   HYPRE_Int              num_variables = hypre_CSRMatrixNumRows(A);

/*
    HYPRE_Int             *A_i           = hypre_CSRMatrixI(A);
    HYPRE_Int             *A_j           = hypre_CSRMatrixJ(A);
    double          *A_data        = hypre_CSRMatrixData(A);
*/
                  
   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;
                 
   hypre_CSRMatrix *ST;
   HYPRE_Int             *ST_i;
   HYPRE_Int             *ST_j;
                 
   HYPRE_Int             *CF_marker;
   HYPRE_Int              coarse_size;

   HYPRE_Int             *measure_array;
   HYPRE_Int             *graph_array;
   HYPRE_Int             *graph_ptr;
   HYPRE_Int              graph_size;

/*   double           diag, row_scale;

    HYPRE_Int              jA, jS;
*/

   HYPRE_Int              measure, max_measure;
   HYPRE_Int              i, j, ig;
   HYPRE_Int		    ic, ji, jj, jl, index;
   HYPRE_Int		    ci_tilde = -1;
   HYPRE_Int		    ci_tilde_mark = -1;
   HYPRE_Int		    set_empty = 1;
   HYPRE_Int		    C_i_nonempty = 0;
   HYPRE_Int		    num_strong;

   HYPRE_Int              ierr = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   S_i = hypre_CSRMatrixI(S);
   S_j = hypre_CSRMatrixJ(S);

   num_strong = S_i[num_variables];
   ST = hypre_CSRMatrixCreate(num_variables, num_variables, num_strong);

   ST_i = hypre_CTAlloc(HYPRE_Int,num_variables+1);
   hypre_CSRMatrixI(ST) = ST_i;

   ST_j = hypre_CTAlloc(HYPRE_Int,num_strong);
   hypre_CSRMatrixJ(ST) = ST_j;

   hypre_CSRMatrixNumNonzeros(ST) = num_strong;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i=0; i <= num_variables; i++)
      ST_i[i] = 0;
 
   for (i=0; i < num_strong; i++)
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

   measure_array = hypre_CTAlloc(HYPRE_Int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = ST_i[i+1]-ST_i[i];
   }


   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(HYPRE_Int, num_variables);
   graph_ptr   = hypre_CTAlloc(HYPRE_Int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = i;
      graph_ptr[i] = i;
   }

   /*---------------------------------------------------
    * Initialize the C/F marker array
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);
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

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }
   for (i=0; i < num_variables; i++)
   {
      if (ci_tilde_mark |= i) ci_tilde = -1;
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
		     ci_tilde_mark = i;
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

		  	       


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   hypre_TFree(graph_ptr);
   hypre_CSRMatrixDestroy(ST);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/***************************************************************************
 *
 *             Ruge's coarsening algorithm using List of lists		   
 *
 ***************************************************************************/

HYPRE_Int
hypre_AMGCoarsenRugeLoL( hypre_CSRMatrix    *A,
			 double              strength_threshold,
			 hypre_CSRMatrix    *S,
			 HYPRE_Int               **CF_marker_ptr,
			 HYPRE_Int                *coarse_size_ptr     )
{
   HYPRE_Int              num_variables = hypre_CSRMatrixNumRows(A);

/*
    HYPRE_Int             *A_i           = hypre_CSRMatrixI(A);
    HYPRE_Int             *A_j           = hypre_CSRMatrixJ(A);
    double          *A_data        = hypre_CSRMatrixData(A);
*/

                  
   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;
                 
   hypre_CSRMatrix *ST;
   HYPRE_Int             *ST_i;
   HYPRE_Int             *ST_j;
                 
   HYPRE_Int             *CF_marker;
   HYPRE_Int              coarse_size;

   HYPRE_Int             *measure_array;
   HYPRE_Int             *graph_array;
/*
    double           diag, row_scale;
    HYPRE_Int              jA, jS;
*/

   HYPRE_Int              measure;
   HYPRE_Int              i, j, k;
   HYPRE_Int		    ji, jj, index;
   HYPRE_Int		    ci_tilde = -1;
   HYPRE_Int		    ci_tilde_mark = -1;
   HYPRE_Int		    set_empty = 1;
   HYPRE_Int		    C_i_nonempty = 0;
   HYPRE_Int		    num_strong;

   HYPRE_Int              ierr = 0;

   hypre_LinkList LoL_head;
   hypre_LinkList LoL_tail;

   HYPRE_Int *lists, *where;
   HYPRE_Int  new_meas;
   HYPRE_Int  num_left;
   HYPRE_Int  nabor, nabor_two;

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/


   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(HYPRE_Int, num_variables);
   where = hypre_CTAlloc(HYPRE_Int, num_variables);


   CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);
   for (j = 0; j < num_variables; j++)
   {
      CF_marker[j] = UNDECIDED;
   }

   S_i = hypre_CSRMatrixI(S);
   S_j = hypre_CSRMatrixJ(S);

   num_strong = S_i[num_variables];
   ST = hypre_CSRMatrixCreate(num_variables, num_variables, num_strong);

   ST_i = hypre_CTAlloc(HYPRE_Int,num_variables+1);
   hypre_CSRMatrixI(ST) = ST_i;

   ST_j = hypre_CTAlloc(HYPRE_Int,num_strong);
   hypre_CSRMatrixJ(ST) = ST_j;

   hypre_CSRMatrixNumNonzeros(ST) = num_strong;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i=0; i <= num_variables; i++)
      ST_i[i] = 0;
 
   for (i=0; i < num_strong; i++)
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
    * Compute the measures and initialize the lists
    *
    * The measures are given by the row sums of ST.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(HYPRE_Int, num_variables);
   for (j = 0; j < num_variables; j++) 
   {    
      measure_array[j] = ST_i[j+1]-ST_i[j];
   }    

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
         if (measure < 0) hypre_printf("negative measure!\n");
         CF_marker[j] = FPOINT;
	 for (k = S_i[j]; k < S_i[j+1]; k++)
	 {
	    nabor = S_j[k];
	    if (nabor < j)
	    {
	       new_meas = measure_array[nabor];
	       if (new_meas > 0)
		  remove_point(&LoL_head, &LoL_tail, new_meas,
				nabor, lists, where);
	       new_meas = ++(measure_array[nabor]);
	       enter_on_lists(&LoL_head, &LoL_tail, new_meas,
				nabor, lists, where);
	    }
	    else
	    {
	       new_meas = ++(measure_array[nabor]);
	    }
  	 }
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

            if (measure > 0)
               enter_on_lists(&LoL_head, &LoL_tail, measure, nabor,
                                lists, where);
            else
            {
               CF_marker[nabor] = F_PT;
               --num_left;

               for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     remove_point(&LoL_head, &LoL_tail, new_meas,
                               nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);

                     enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor_two, lists, where);
                  }
               }
            }
         }
      }
   }

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   hypre_TFree(lists);
   hypre_TFree(where);
   hypre_TFree(LoL_head);
   hypre_TFree(LoL_tail);
   hypre_TFree(measure_array);
   hypre_CSRMatrixDestroy(ST);


   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(HYPRE_Int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }



   /* second pass, check fine points for coarse neighbors */

   for (i=0; i < num_variables; i++)
   {
      if (ci_tilde_mark |= i) ci_tilde = -1;
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
		     ci_tilde_mark = i;
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

		  	       


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(graph_array);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/* begin HANS added */
/**************************************************************
 *
 *      weighted LJP Coarsening routine
 *          (don't worry about strong F-F connections
 *           without a common C point)
 *
 **************************************************************/
HYPRE_Int
hypre_AMGCoarsenwLJP( hypre_CSRMatrix    *A,
                  double              strength_threshold,
                  hypre_CSRMatrix    *S,
                  /*hypre_CSRMatrix   **S_ptr,*/
                  HYPRE_Int               **CF_marker_ptr,
                  HYPRE_Int                *coarse_size_ptr     )
{
   HYPRE_Int              num_variables = hypre_CSRMatrixNumRows(A);

/*
    HYPRE_Int             *A_i           = hypre_CSRMatrixI(A);
    HYPRE_Int             *A_j           = hypre_CSRMatrixJ(A);
    double          *A_data        = hypre_CSRMatrixData(A);
*/

                  
   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;
   double          *S_data;
                 
   HYPRE_Int             *CF_marker;
   HYPRE_Int              coarse_size;

   double          *measure_array;
   HYPRE_Int             *graph_array;
   HYPRE_Int              graph_size;
/*
    double           diag, row_scale;
    HYPRE_Int              i, j, k, jA, jS, kS, ig;
*/
   HYPRE_Int              i, j, jS, ig;
   HYPRE_Int              jjS;

   HYPRE_Int              ierr = 0;

   #if 0  
   char  filename[256];
   FILE *fp;
   #endif
   HYPRE_Int   iter = 0;

   hypre_printf("\n");
   hypre_printf("wLJP coarsening...\n");

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

   /*S = hypre_CSRMatrixCreate(num_variables, num_variables,
                             A_i[num_variables]);
   hypre_CSRMatrixInitialize(S);*/

   S_i           = hypre_CSRMatrixI(S);
   S_j           = hypre_CSRMatrixJ(S);
   S_data        = hypre_CSRMatrixData(S);

   /* give S same nonzero structure as A */
   /*for (i = 0; i < num_variables; i++)
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
      diag = A_data[A_i[i]];*/

      /* compute scaling factor */
      /*row_scale = 0.0;
      if (diag < 0)
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
	   if (dof_func[i] == dof_func[A_j[jA]])
            row_scale = hypre_max(row_scale, A_data[jA]);
         }
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
	   if (dof_func[i] == dof_func[A_j[jA]])
            row_scale = hypre_min(row_scale, A_data[jA]);
         }
      }*/

      /* compute row entries of S */
      /*S_data[A_i[i]] = 0;
      if (diag < 0) 
      { 
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            S_data[jA] = 0;
            if (A_data[jA] > strength_threshold * row_scale
		&& dof_func[i] == dof_func[A_j[jA]])            {
               S_data[jA] = -1;
            }
         }
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            S_data[jA] = 0;
            if (A_data[jA] < strength_threshold * row_scale
		&&dof_func[i] == dof_func[A_j[jA]])
            {
               S_data[jA] = -1;
            }
         }
      }
   }*/

   /*--------------------------------------------------------------
    * "Compress" the strength matrix.
    *
    * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
    *
    * NOTE: This "compression" section of code may be removed, and
    * coarsening will still be done correctly.  However, the routine
    * that builds interpolation would have to be modified first.
    *----------------------------------------------------------------*/

   /*jS = 0;
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
   hypre_CSRMatrixNumNonzeros(S) = jS;*/

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
   hypre_InitAMGIndepSet(S, measure_array, 1.0);

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array   = hypre_CTAlloc(HYPRE_Int, num_variables);

   /* intialize measure array and graph array */
   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = i;
   }

   /*---------------------------------------------------
    * Initialize the C/F marker array
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   coarse_size = 0;
   graph_size = num_variables;

   while (1) {

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
     
     if(graph_size == 0) break;

     hypre_printf("*** wLJP iteration %d\n",iter);
     hypre_printf("graph_size remaining %d\n",graph_size);
     hypre_printf("coarse_size selected %d\n",coarse_size);
     
     /*------------------------------------------------
      * Pick an independent set of points with
      * maximal measure.
      *------------------------------------------------*/
     
     /*     for (i = 0; i < num_variables; i++) {
       hypre_printf("CF_marker[i] %d\n",CF_marker[i]);
       hypre_printf("measure_array[i] %e\n",measure_array[i]);
     } 
     hypre_printf("CALL \n");*/

     hypre_AMGIndepSet(S, measure_array, 1.0,
		       graph_array, graph_size, CF_marker);
     
     /*    for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];
       hypre_printf("CF_marker[i] %d\n",CF_marker[i]);
       hypre_printf("measure_array[i] %e\n",measure_array[i]);
       } */

     /*------------------------------------------------
      * Set C-pts and F-pts.
      *------------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];

       /*---------------------------------------------
	* If the measure of i is smaller than 1, then
        * make i and F point (because it does not influence
        * any other point), and remove all edges of
	* equation i.
	*---------------------------------------------*/

       if(measure_array[i]<1.){
	 /* make point i an F point*/
	 CF_marker[i]= F_PT;
         /* remove the edges in equation i */
	 for (jS = S_i[i]; jS < S_i[i+1]; jS++) {
	   if (S_data[jS] < 0){
	     S_data[jS] = -S_data[jS];
	   }
	 }
       }

       /*---------------------------------------------
	* First treat the case where point i is in the
	* independent set: make i a C point, and all
        * points j that influence equation i become F
        * points; take out all the graph edges for
        * equation i.
        * All the edges of the new F-equations j need
        * to be taken out as well.          
	*---------------------------------------------*/
       
       if (CF_marker[i] > 0) {
	 /* set to be a C-pt */
	 CF_marker[i] = C_PT;
	 coarse_size++;

	 for (jS = S_i[i]; jS < S_i[i+1]; jS++) {
	   /*j = S_j[jS];
	   CF_marker[j]= F_PT*/
	   /* "remove" edge from S in equation i */
	   if (S_data[jS] < 0){
	     S_data[jS] = -S_data[jS];
	   }
	 }
       }  
       else  
       {  
	 for (jS = S_i[i]; jS < S_i[i+1]; jS++) {
	   j = S_j[jS];
	   if (CF_marker[j] > 0)
	   {
	      CF_marker[i] = F_PT;
	   /* now loop over the columns of equation j */
	      for (jjS = S_i[j]; jjS < S_i[j+1]; jjS++) {
	     /* "remove" edge from S in equation j */
	         if (S_data[jjS] < 0){
	            S_data[jjS] = -S_data[jjS];
	         }
	      }
	   }
	 }
       }  
     }

     /*---------------------------------------------
      * Now loop over the points in the unassigned
      * graph again. For all points i that are no new C or
      * F points, remove the edges in equation i that
      * connect to C or F points.
      * (We have removed the rows for the new C and F
      * points above; now remove the columns.)
      *---------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];

       if(CF_marker[i]==0) {

	 for (jS = S_i[i]; jS < S_i[i+1]; jS++){
	   j = S_j[jS];
	   if (CF_marker[j]!=0 && S_data[jS] < 0){
	       /* "remove" edge from S */
	       S_data[jS] = -S_data[jS];
	   }
	 }
       }
     }

     /*------------------------------------------------
      * Update subgraph
      *------------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];
       
       if (CF_marker[i]==C_PT || CF_marker[i]==F_PT)
	 {
	   /* the independent set subroutine needs measure 0 for
              removed nodes */
	   measure_array[i] = 0;
	   /* take point out of the subgraph */
	   graph_size--;
	   graph_array[ig] = graph_array[graph_size];
	   graph_array[graph_size] = i;
	   ig--;
	 }
     }

   } /* end while */

   hypre_printf("*** wLJP iteration %d\n",iter);
   hypre_printf("graph_size remaining %d\n",graph_size);
   hypre_printf("coarse_size selected %d\n",coarse_size);

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/*==========================================================================*/
/* Ruge's coarsening algorithm modified: only one pass			    */
/*==========================================================================*/

HYPRE_Int
hypre_AMGCoarsenRugeOnePass( hypre_CSRMatrix    *A,
                      double              strength_threshold,
                      /*hypre_CSRMatrix   **S_ptr,*/
                      hypre_CSRMatrix    *S,
                      HYPRE_Int               **CF_marker_ptr,
                      HYPRE_Int                *coarse_size_ptr     )
{

   HYPRE_Int             *A_i           = hypre_CSRMatrixI(A);
/*
    HYPRE_Int             *A_j           = hypre_CSRMatrixJ(A);
    double          *A_data        = hypre_CSRMatrixData(A);
*/

   HYPRE_Int              num_variables = hypre_CSRMatrixNumRows(A);
                  
   /*hypre_CSRMatrix *S;*/
   HYPRE_Int             *S_i;
   HYPRE_Int             *S_j;
                 
   hypre_CSRMatrix *ST;
   HYPRE_Int             *ST_i;
   HYPRE_Int             *ST_j;
                 
   HYPRE_Int             *CF_marker;
   HYPRE_Int              coarse_size;

   HYPRE_Int             *measure_array;
   HYPRE_Int             *graph_array;
   HYPRE_Int             *graph_ptr;
   HYPRE_Int              graph_size;
/*
   double           diag, row_scale;
    HYPRE_Int              i, j, jA, jS, ig;
*/
   HYPRE_Int              measure, max_measure;
   HYPRE_Int              i, j, jS, ig;
   HYPRE_Int		    ic, ji, jj, jl, index;
   HYPRE_Int		    num_strong;
/*
    HYPRE_Int                    ci_tilde = -1;
    HYPRE_Int                    ci_tilde_mark = -1;
    HYPRE_Int                    set_empty = 1;
    HYPRE_Int                    C_i_nonempty = 0;
*/
   HYPRE_Int              ierr = 0;

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

   num_strong = A_i[num_variables] - num_variables;
   /*S = hypre_CSRMatrixCreate(num_variables, num_variables, num_strong);*/
   ST = hypre_CSRMatrixCreate(num_variables, num_variables, num_strong);
   S_i = hypre_CSRMatrixI(S);
   S_j = hypre_CSRMatrixJ(S);

   /*S_i = hypre_CTAlloc(HYPRE_Int,num_variables+1);
   hypre_CSRMatrixI(S) = S_i;*/

   ST_i = hypre_CTAlloc(HYPRE_Int,num_variables+1);
   hypre_CSRMatrixI(ST) = ST_i;

   ST_j = hypre_CTAlloc(HYPRE_Int,A_i[num_variables]);
   hypre_CSRMatrixJ(ST) = ST_j;

   /* give S same nonzero structure as A, store in ST*/
   /*for (i = 0; i < num_variables; i++)
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
      diag = A_data[A_i[i]];*/

      /* compute scaling factor */
      /*row_scale = 0.0;
      if (diag < 0)
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            row_scale = hypre_max(row_scale, A_data[jA]);
         }
      }
      else
      {
         for (jA = A_i[i]+1; jA < A_i[i+1]; jA++)
         {
            row_scale = hypre_min(row_scale, A_data[jA]);
         }
      }*/

      /* compute row entries of S */
      /*if (diag < 0) 
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

   S_j = hypre_CTAlloc(HYPRE_Int,num_strong);
   hypre_CSRMatrixJ(S) = S_j;*/

   /*--------------------------------------------------------------
    * "Compress" the strength matrix.
    *
    * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
    *
    * NOTE: This "compression" section of code may be removed, and
    * coarsening will still be done correctly.  However, the routine
    * that builds interpolation would have to be modified first.
    *----------------------------------------------------------------*/

   /*jS = 0;
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
   hypre_CSRMatrixNumNonzeros(ST) = jS;*/
   jS = hypre_CSRMatrixNumNonzeros(S);

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

   measure_array = hypre_CTAlloc(HYPRE_Int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = ST_i[i+1]-ST_i[i];
   }


   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(HYPRE_Int, num_variables);
   graph_ptr   = hypre_CTAlloc(HYPRE_Int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = i;
      graph_ptr[i] = i;
   }

   /*---------------------------------------------------
    * Initialize the C/F marker array
    *---------------------------------------------------*/

   CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);
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


   /*   
   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }
   for (i=0; i < num_variables; i++)
   {
      if (ci_tilde_mark |= i) ci_tilde = -1;
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
		     ci_tilde_mark = i;
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
   */	  	       


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   hypre_TFree(graph_ptr);
   hypre_CSRMatrixDestroy(ST);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/* end HANS added */
