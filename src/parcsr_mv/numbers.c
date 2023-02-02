/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_parcsr_mv.h"

hypre_NumbersNode * hypre_NumbersNewNode( void )
/* makes a new node for a tree representing numbers */
{
   HYPRE_Int i;
   hypre_NumbersNode * newnode = hypre_CTAlloc( hypre_NumbersNode,  1, HYPRE_MEMORY_HOST);
   for ( i = 0; i <= 10; ++i ) { newnode->digit[i] = NULL; }
   return newnode;
}

void hypre_NumbersDeleteNode( hypre_NumbersNode * node )
/* deletes a node and the tree of which it is root */
{
   HYPRE_Int i;
   for ( i = 0; i <= 10; ++i ) if ( node->digit[i] != NULL )
      {
         hypre_NumbersDeleteNode( node->digit[i] );
         node->digit[i] = NULL;
      };
   hypre_TFree( node, HYPRE_MEMORY_HOST);
}

HYPRE_Int hypre_NumbersEnter( hypre_NumbersNode * node, const HYPRE_Int n )
/* enters a number in the tree starting with 'node'. */
{
   HYPRE_Int newN = 0;
   HYPRE_Int q = n / 10;
   HYPRE_Int r = n % 10;
   hypre_assert( n >= 0 );
   if ( node->digit[r] == NULL )
   {
      node->digit[r] = hypre_NumbersNewNode();
      newN = 1;
   };
   if ( q < 10 )  /* q is a one-digit number; point to terminal object */
   {
      if ( (node->digit[r])->digit[10] == NULL )
      {
         (node->digit[r])->digit[10] = hypre_NumbersNewNode();
      }
   }
   else    /* multidigit number; place for this digit points to next node */
   {
      newN = hypre_NumbersEnter(node->digit[r], q );
   }
   return newN;
}

HYPRE_Int hypre_NumbersNEntered( hypre_NumbersNode * node )
/* returns the number of numbers represented by the tree whose root is 'node' */
{
   HYPRE_Int i;
   HYPRE_Int count = 0;
   if ( node == NULL ) { return 0; }
   for ( i = 0; i < 10; ++i ) if ( node->digit[i] != NULL )
      {
         count += hypre_NumbersNEntered( node->digit[i] );
      }
   if ( node->digit[10] != NULL ) { ++count; }
   return count;
}

HYPRE_Int hypre_NumbersQuery( hypre_NumbersNode * node, const HYPRE_Int n )
/* returns 1 if n is on the tree with root 'node', 0 otherwise */
{
   HYPRE_Int q = n / 10;
   HYPRE_Int r = n % 10;
   hypre_assert( n >= 0 );
   if ( node->digit[r] == NULL )   /* low order digit of n not on tree */
   {
      return 0;
   }
   else if ( q < 10 ) /* q is a one-digit number; check terminal object */
   {
      if ( (node->digit[r])->digit[10] == NULL )
      {
         return 0;
      }
      else
      {
         return 1;
      }
   }
   else    /* look for higher order digits of n on tree of its low order digit r */
   {
      return hypre_NumbersQuery( node->digit[r], q );
   }
}

HYPRE_Int * hypre_NumbersArray( hypre_NumbersNode * node )
/* allocates and returns an unordered array of ints as a simpler representation
   of the contents of the Numbers tree.
   For the array length, call hypre_NumbersNEntered */
{
   HYPRE_Int i, j, Ntemp;
   HYPRE_Int k = 0;
   HYPRE_Int N = hypre_NumbersNEntered(node);
   HYPRE_Int * array, * temp;
   array = hypre_CTAlloc( HYPRE_Int,  N, HYPRE_MEMORY_HOST);
   if ( node == NULL ) { return array; }
   for ( i = 0; i < 10; ++i ) if ( node->digit[i] != NULL )
      {
         Ntemp = hypre_NumbersNEntered( node->digit[i] );
         temp = hypre_NumbersArray( node->digit[i] );
         for ( j = 0; j < Ntemp; ++j )
         {
            array[k++] = temp[j] * 10 + i;
         }
         hypre_TFree(temp, HYPRE_MEMORY_HOST);
      }
   if ( node->digit[10] != NULL ) { array[k++] = 0; }
   hypre_assert( k == N );
   return array;
}
