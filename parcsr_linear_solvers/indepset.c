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
  Partition a graph into a maximal independent set, and a remainder set.
  The graph to be partitioned is actually a subgraph of some parent
  graph.  The parent graph is described as a matrix in compressed sparse
  row format, where edges in the graph are represented by positive
  matrix coefficients.  A non-negative measure is assigned to each node
  of the parent graph, and this measure is used to pick the independent
  set.  The subgraph is a collection of nodes in the parent graph.

  When the routine returns, the first `IS\_size' entries of
  `graph\_array' are the nodes in the independent set.  The entries from
  `IS_size'+1 to `graph\_array\_size'-1 represent the remainder set.

  The algorithm uses the `work\_array' entries in two different ways:
  \begin{itemize}
  \item the value of each subgraph entry is the corresponding value in
  `measure\_array', augmented by some random value between 0 and 1;
  \item positive entries represent nodes in the independent set and
  negative entries represent nodes not in the independent set.
  \end{itemize}

  The algorithm proceeds by first copying the `measure\_array' values
  into the `work\_array', and augmenting them by some random value, so
  that initially all subgraph nodes are assumed to be in the independent
  set.  The nodes in `graph\_array' are then looped over, and nodes are
  removed from the independent set by simply comparing the measures of
  adjacent nodes, and negating the appropriate `work\_array' entry.

  {\bf Input files:}
  headers.h

  @return Error code.

  @param S [IN]
  parent graph matrix in CSR format
  @param measure_array [IN]
  measures assigned to each node of the parent graph
  @param work_array [IN/OUT]
  work array of the same size as `measure\_array'
  @param graph_array [IN/OUT]
  node numbers in the subgraph to be partitioned
  @param graph_array_size [IN]
  number of nodes in the subgraph to be partitioned
  @param IS_size_ptr [OUT]
  size of the independent set

  @see
  */
/*--------------------------------------------------------------------------*/

int
hypre_AMGIndepSet( hypre_CSRMatrix *S,
                   double          *measure_array,
                   double          *work_array,
                   int             *graph_array,
                   int              graph_array_size,
                   int             *IS_size_ptr      )
{
   int    *S_i         = hypre_CSRMatrixI(S);
   int    *S_j         = hypre_CSRMatrixJ(S);
   double *S_data      = hypre_CSRMatrixData(S);
   int     S_num_nodes = hypre_CSRMatrixNumRows(S);
         
   int     IS_last_start;
   int     IS_start;
   int     RS_start;
         
   double  s;
   int     i, j, ig, jS;

   /*-------------------------------------------------------
    * Initialize work_array.
    *-------------------------------------------------------*/

   for (i = 0; i < S_num_nodes; i++)
   {
      work_array[j] = 0.0;
   }

   hypre_SeedRand(2747);
   for (ig = 0; ig < graph_array_size; ig++)
   {
      i = graph_array[ig];

      s = hypre_Rand();
      work_array[i] = measure_array[i] + s;
   }

   /*-------------------------------------------------------
    * Partition the graph
    *-------------------------------------------------------*/

   IS_start = 0;

   do
   {
      IS_last_start = IS_start;

      /* Remove nodes from the initial independent set */
      for (ig = IS_start; ig < graph_array_size; ig++)
      {
         i = graph_array[ig];

         /* only consider nodes that have not been removed */
         if (work_array[i] > 0)
         {
            for (jS = S_i[i]; jS < S_i[i+1]; jS++)
            {
               /* only consider valid graph edges */
               if (S_data[jS] > 0)
               {
                  j = S_j[jS];
                  
                  /* only consider nodes that have not been removed */
                  if (work_array[j] > 0)
                  {
                     if (work_array[i] > work_array[j])
                     {
                        work_array[j] = -work_array[j];
                     }
                     else if (work_array[j] > work_array[i])
                     {
                        work_array[i] = -work_array[i];
                        break;
                     }
                  }
               }
            }
         }
      }
            
      /* Modify the graph_array to represent the new partition */
      RS_start = graph_array_size;
      while (IS_start < RS_start)
      {
         i = graph_array[IS_start];

         if (work_array[i] > 0)
         {
            /* the node is in the independent set */
            IS_start++;
         }
         else
         {
            /* the node is in the remainder set */
            RS_start--;

            /* swap node entries in graph_array */
            graph_array[IS_start] = graph_array[RS_start];
            graph_array[RS_start] = i;

            /* re-initialize remainder set to be in the independent set */
            work_array[i] = -work_array[i];
         }
      }

   } while (IS_start > IS_last_start);

   *IS_size_ptr = IS_start;

   return(0);
}
         
