/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include <math.h>

// Data structure for stack
typedef struct
{
   HYPRE_Int maxsize;    // define max capacity of stack
   HYPRE_Int top;
   HYPRE_Int *items;
} stack;

// Utility function to initialize stack
static inline stack*
newStack(HYPRE_Int capacity)
{
   stack *pt = hypre_TAlloc(stack, 1, HYPRE_MEMORY_HOST);

   pt->maxsize = capacity;
   pt->top = -1;
   pt->items = hypre_TAlloc(HYPRE_Int, capacity, HYPRE_MEMORY_HOST);

   return pt;
}

static inline HYPRE_Int
isEmpty(stack *pt)
{
   return (pt->top == -1);
}

static inline HYPRE_Int
isFull(stack *pt)
{
   return (pt->top == (pt->maxsize - 1));
}

static inline void
push(stack *pt, HYPRE_Int x)
{
   // check if stack is full
   if (isFull(pt))
   {
      printf("OverFlow\nProgram Terminated\n");
      exit(EXIT_FAILURE);
   }

   // add element and increment top index
   pt->items[++pt->top] = x;
}

// Utility function to return top element in a stack
static inline HYPRE_Int
peek(stack *pt)
{
   // check for empty stack
   if (!isEmpty(pt))
   {
      return pt->items[pt->top];
   }

   return -1;
}

static inline HYPRE_Int
pop(stack *pt)
{
   // decrement stack size by 1 and return popped element
   return pt->items[pt->top--];
}

static inline void
deleteStack(stack *pt)
{
   if (pt)
   {
      hypre_TFree(pt->items, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(pt, HYPRE_MEMORY_HOST);
}

/**
 * Solves the feedback arc set problem using the heuristics of Eades et al.
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.7745&rep=rep1&type=pdf
 */
void
hypre_solve_fas( HYPRE_Int      n,
                 HYPRE_Complex  A[],
                 HYPRE_Int     *ordering)
{
   HYPRE_Int i, k, v, nodes_left;
   HYPRE_Int order_next_pos = 0;
   HYPRE_Int order_next_neg = -1;
   HYPRE_Complex val, diff, maxdiff;
   HYPRE_Real tol = 1e-12;

   // TODO : delete later too
   stack *sources = newStack(n);
   stack *sinks = newStack(n);

   // Fill in degree/strength of nodes
   HYPRE_Int* indegrees = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   HYPRE_Int* outdegrees = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   HYPRE_Complex* instrengths = hypre_CTAlloc(HYPRE_Complex, n, HYPRE_MEMORY_HOST);
   HYPRE_Complex* outstrengths = hypre_CTAlloc(HYPRE_Complex, n, HYPRE_MEMORY_HOST);

   for (i = 0; i < n; i++)
   {
      for (k = 0; k < n; k++)
      {
         // Skip diagonal entries i=k
         if (i == k)
         {
            continue;
         }

         // A(i,k) != 0
         val = fabs(A[i + k*n]);
         if (val > tol)
         {
            indegrees[k]++;
            instrengths[k] += val;
            outdegrees[i]++;
            outstrengths[i] += val;
         }
      }
   }

   /* Find initial sources and sinks */
   nodes_left = n;
   for (i = 0; i < n; i++)
   {
      if (indegrees[i] == 0)
      {
         if (outdegrees[i] == 0)
         {
            // Isolated vertex, we simply ignore it
            nodes_left--;
            ordering[i] = order_next_pos++;
            indegrees[i] = -1;
            outdegrees[i] = -1;
         }
         else
         {
            // This is a source
            push(sources, i);
            //printf("Source %d\n",i);
         }
      }
      else if (outdegrees[i] == 0)
      {
         // This is a sink
         push(sinks, i);
         //printf("Sink %d\n",i);
      }
   }

   /* While we have any nodes left... */
   //HYPRE_Int it=0;
   while (nodes_left > 0)
   {
      /* (1) Remove the sources one by one */
      while (!isEmpty(sources))
      {
         i = pop(sources);
         // Add the node to the ordering
         //printf("source node %d, order %d\n", i, order_next_pos);
         ordering[i] = order_next_pos++;

         // Exclude the node from further searches
         indegrees[i] = -1;
         outdegrees[i] = -1;

         // Find outgoing neighbors and decrease their degrees
         for (k = 0; k < n; k++)
         {
            if (i == k)
            {
               continue;
            }
            val = fabs(A[i + n*k]);
            if (val < tol)
            {
               continue; // No such edge, continue
            }

            if (indegrees[k] <= 0)
            {
               continue; // Already removed, continue
            }

            indegrees[k]--;
            instrengths[k] -= val;
            if (indegrees[k] == 0)
            {
               push(sources, k);
               //printf("Source %d\n",k);
            }
            A[i + n*k] = 0.0;
         }
         nodes_left--;
      }

      /* (2) Remove the sinks one by one */
      while (!isEmpty(sinks))
      {
         i = pop(sinks);
         // Check if removed in previous iteration
         if (indegrees[i] < 0)
         {
            continue;
         }
         // Add the node to the ordering
         //printf("sink node %d, order %d\n", v, order_next_neg);
         ordering[i] = order_next_neg--;
         // Exclude the node from further searches
         indegrees[i] = outdegrees[i] = -1;

         // Find incoming neighbors and decrease their degrees
         for (k = 0; k < n; k++)
         {
            if (i == k)
            {
               continue;
            }
            val = fabs(A[k + n*i]);
            if (val < tol)
            {
               continue; // No such edge, continue
            }

            if (outdegrees[k] <= 0)
            {
               continue; // Already removed, continue
            }

            outdegrees[k]--;
            outstrengths[k] -= val;
            if (outdegrees[k] == 0)
            {
               push(sinks, k);
            }
            A[k + n*i] = 0.0;
         }
         nodes_left--;
      }

      /* (3) No more sources or sinks. Find the node with the largest
       * difference between its out-strength and in-strength */
      v = -1;
      maxdiff = -(1e50);
      for (i = 0; i < n; i++)
      {
         if (outdegrees[i] < 0)
         {
            continue;
         }
         diff = outstrengths[i] - instrengths[i];
         if (diff > maxdiff)
         {
            maxdiff = diff;
            v = i;
         }
      }
      if (v >= 0)
      {
         // Remove vertex v
         //printf("vertex node %d, order %d\n", v, order_next_pos);
         ordering[v] = order_next_pos++;

         // Update vertices from outgoing edges
         for (k = 0; k < n; k++)
         {
            if (v == k)
            {
               continue;
            }
            val = fabs(A[v + n*k]);
            if (val < tol)
            {
               continue; // No such edge, continue
            }

            if (indegrees[k] <= 0)
            {
               continue; // Already removed, continue
            }

            indegrees[k]--;
            instrengths[k] -= val;
            if (indegrees[k] == 0)
            {
               push(sources, k);
            }
            A[v + n*k] = 0.0;
         }

         // Update vertices from incoming edges
         for (k = 0; k < n; k++)
         {
            if (v == k)
            {
               continue;
            }
            val = fabs(A[k + n*v]);
            if (val < tol)
            {
               continue; // No such edge, continue
            }

            if (outdegrees[k] <= 0)
            {
               continue; // Already removed, continue
            }

            outdegrees[k]--;
            outstrengths[k] -= val;
            if (outdegrees[k] == 0)
            {
               push(sinks, k);
            }
            A[k + n*v] = 0.0;
         }

         outdegrees[v] = -1;
         indegrees[v] = -1;
         nodes_left--;
      }
   }

   // Move sink nodes to end of ordering
   for (i = 0; i < n; i++)
   {
      if (ordering[i] < 0)
      {
         ordering[i] += n;
      }
   }

   // Convert ordering from solve_fas to correct format (store
   // in indegrees to avoid allocating another array).
   for (i=0; i<n; i++)
   {
      for (k=0; k<n; k++)
      {
         if (ordering[k] == i)
         {
            indegrees[i] = k;
         }
      }
   }
   for (i=0; i<n; i++)
   {
      ordering[i] = indegrees[i];
   }

   // Clean up
   free(indegrees);
   free(instrengths);
   free(outdegrees);
   free(outstrengths);

   // free structs
   deleteStack(sources);
   deleteStack(sinks);
}

HYPRE_Real
hypre_getOrderedNormRatio( HYPRE_Int      n,
                           HYPRE_Complex *A,
                           HYPRE_Int     *ordering,
                           HYPRE_Int      pow)
{
   HYPRE_Real tol = 1e-12;
   HYPRE_Real outlier = 2.0;

   HYPRE_Real normL = 0;
   HYPRE_Real normU = 0;
   HYPRE_Int i,j;
   HYPRE_Int nnzL = 0;
   HYPRE_Int nnzU = 0;
   for (i = 0; i < n; i++)
   {
      for (j = 0; j < n; j++)
      {
         if (i < j)
         {
            if (fabs(A[ordering[i]+n*ordering[j]]) > tol)
            {
               nnzU++;
            }

            if (pow == 2)
            {
               normU += A[ordering[i]+n*ordering[j]]*A[ordering[i]+n*ordering[j]];
            }
            else
            {
               normU += fabs(A[ordering[i]+n*ordering[j]]);
            }
         }
         else if (i > j)
         {
            if (fabs(A[ordering[i]+n*ordering[j]]) > tol)
            {
               nnzL++;
            }
            if (pow == 2)
            {
               normL += A[ordering[i]+n*ordering[j]]*A[ordering[i]+n*ordering[j]];
            }
            else
            {
               normL += fabs(A[ordering[i]+n*ordering[j]]);
            }
         }
      }
   }

   if (pow == 2)
   {
      normL = sqrt(normL);
      normU = sqrt(normU);
   }

   // Check for trivial returns here
   if (normU < tol && normL < tol)
   {
      return 0.0;
   }
   else if (normU < tol || normL < tol)
   {
      return 0.0;
   }

   // standard deviation
   HYPRE_Real meanL = normL / nnzL;
   HYPRE_Real meanU = normU / nnzU;
   HYPRE_Real stdL = 0;
   HYPRE_Real stdU = 0;
   for (i = 0; i < n; i++)
   {
      for (j = 0; j < n; j++)
      {
         if (i < j)
         {
            if (fabs(A[ordering[i] + n*ordering[j]]) > tol)
            {
               HYPRE_Real temp = fabs(A[ordering[i]+n*ordering[j]]) - meanU;
               stdU += temp * temp;
            }
         }
         else if (i > j)
         {
            if (fabs(A[ordering[i]+n*ordering[j]]) > tol)
            {
               HYPRE_Real temp = fabs(A[ordering[i]+n*ordering[j]]) - meanL;
               stdL += temp * temp;
            }
         }
      }
   }

   stdL = sqrt(stdL / nnzL);
   stdU = sqrt(stdU / nnzU);

   //printf("Mean(L) = %1.3f, Std(L) = %1.3f\n",meanL,stdL);
   //printf("Mean(U) = %1.3f, Std(U) = %1.3f\n",meanU,stdU);

   // Mean without outliers
   HYPRE_Int nnzL_0 = 0;
   HYPRE_Int nnzU_0 = 0;
   HYPRE_Real normU_0 = 0;
   HYPRE_Real normL_0 = 0;
   for (i = 0; i < n; i++)
   {
      for (j = 0; j < n; j++)
      {
         HYPRE_Real temp = fabs(A[ordering[i]+n*ordering[j]]);
         if (i < j)
         {
            if ( (temp > tol) && (fabs(temp - meanU) < outlier * stdU) )
            {
               normU_0 += temp;
               nnzU_0++;
            }
         }
         else if (i > j)
         {
            if ( (temp > tol) && (fabs(temp - meanL) < outlier * stdL) )
            {
               normL_0 += temp;
               nnzL_0++;
            }
         }
      }
   }

   /*
   printf("Mod mean(L) = %1.3f\n",normU_0/nnzU_0);
   printf("Mod mean(U) = %1.3f\n",normL_0/nnzL_0);
   printf("Ratio mod means = %1.3f\n\n",(normL_0/nnzL_0)/(normU_0/nnzU_0));
   */
   HYPRE_Real modU = normU_0/nnzU_0;
   HYPRE_Real modL = normL_0/nnzL_0;
   if (normU_0 < tol && normL_0 < tol)
   {
      if (normU > normL) return normL / normU;
      else return normU / normL;
   }
   else if (normU_0 < tol || normL_0 < tol) 
   {
      return 0.0;
   }
   else if (modU > modL)
   {
      return modL / modU;
   }
   else
   {
      return modU / modL;
   }
}

