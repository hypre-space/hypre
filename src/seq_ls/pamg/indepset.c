/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *****************************************************************************/

#include "headers.h"

/*==========================================================================*/
/*==========================================================================*/
/**
  Augments measures by some random value between 0 and 1.

  {\bf Input files:}
  headers.h

  @return Error code.

  @param S [IN]
  parent graph matrix in CSR format
  @param measure_array [IN/OUT]
  measures assigned to each node of the parent graph

  @see hypre_AMGIndepSet */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_InitAMGIndepSet( hypre_CSRMatrix *S,
                       double          *measure_array, 
                       double           cconst)
{
   HYPRE_Int     S_num_nodes = hypre_CSRMatrixNumRows(S);
   HYPRE_Int     i;
   HYPRE_Int     ierr = 0;

   hypre_SeedRand(2747);
   for (i = 0; i < S_num_nodes; i++)
   {
      measure_array[i] += hypre_Rand()*cconst;
   }

   return (ierr);
}

/*==========================================================================*/
/*==========================================================================*/
/**
  Select an independent set from a graph.  This graph is actually a
  subgraph of some parent graph.  The parent graph is described as a
  matrix in compressed sparse row format, where edges in the graph are
  represented by nonzero matrix coefficients (zero coefficients are
  ignored).  A positive measure is given for each node in the
  subgraph, and this is used to pick the independent set.  A measure
  of zero must be given for all other nodes in the parent graph.  The
  subgraph is a collection of nodes in the parent graph.

  Positive entries in the `IS\_marker' array indicate nodes in the
  independent set.  All other entries are zero.

  The algorithm proceeds by first setting all nodes in `graph\_array'
  to be in the independent set.  Nodes are then removed from the
  independent set by simply comparing the measures of adjacent nodes.

  {\bf Input files:}
  headers.h

  @return Error code.

  @param S [IN]
  parent graph matrix in CSR format
  @param measure_array [IN]
  measures assigned to each node of the parent graph
  @param graph_array [IN]
  node numbers in the subgraph to be partitioned
  @param graph_array_size [IN]
  number of nodes in the subgraph to be partitioned
  @param IS_marker [IN/OUT]
  marker array for independent set

  @see hypre_InitAMGIndepSet */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGIndepSet( hypre_CSRMatrix *S,
                   double          *measure_array,
                   double           cconst,
                   HYPRE_Int             *graph_array,
                   HYPRE_Int              graph_array_size,
                   HYPRE_Int             *IS_marker        )
{
   HYPRE_Int    *S_i         = hypre_CSRMatrixI(S);
   HYPRE_Int    *S_j         = hypre_CSRMatrixJ(S);
   double *S_data      = hypre_CSRMatrixData(S);
         
   HYPRE_Int     i, j, ig, jS;

   HYPRE_Int     ierr = 0;

   /*-------------------------------------------------------
    * Initialize IS_marker by putting all nodes in
    * the independent set.
    *-------------------------------------------------------*/

   for (ig = 0; ig < graph_array_size; ig++)
   {
      i = graph_array[ig];
      if (measure_array[i] > cconst) 
      {
         IS_marker[i] = 1;
      }
   }

   /*-------------------------------------------------------
    * Remove nodes from the initial independent set
    *-------------------------------------------------------*/

   for (ig = 0; ig < graph_array_size; ig++)
   {
      i = graph_array[ig];

      if (measure_array[i] > cconst)
      {
         for (jS = S_i[i]; jS < S_i[i+1]; jS++)
         {
            j = S_j[jS];
            
            /* only consider valid graph edges */
            if ( (measure_array[j] > cconst) && (S_data[jS]) ) 
            {
               if (measure_array[i] > measure_array[j])
               {
                  IS_marker[j] = 0;
               }
               else if (measure_array[j] > measure_array[i])
               {
                  IS_marker[i] = 0;
               }
            }
         }
      }
   }
            
   return (ierr);
}
         
