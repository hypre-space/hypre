/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/



#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "utilities.h"

#include "numbers.h"

hypre_NumbersNode * hypre_NumbersNewNode()
/* makes a new node for a tree representing numbers */
{
   int i;
   hypre_NumbersNode * newnode = hypre_CTAlloc( hypre_NumbersNode, 1 );
   for ( i=0; i<=10; ++i ) newnode->digit[i] = NULL;
   return newnode;
}

void hypre_NumbersDeleteNode( hypre_NumbersNode * node )
/* deletes a node and the tree of which it is root */
{
   int i;
   for ( i=0; i<=10; ++i ) if ( node->digit[i] != NULL ) {
      hypre_NumbersDeleteNode( node->digit[i] );
      node->digit[i] = NULL;
   };
   hypre_TFree( node );
}

int hypre_NumbersEnter( hypre_NumbersNode * node, const int n )
/* enters a number in the tree starting with 'node'. */
{
   int new = 0;
   int q = n/10;
   int r = n%10;
   hypre_assert( n>=0 );
   if ( node->digit[r] == NULL ) {
      node->digit[r] = hypre_NumbersNewNode();
      new = 1;
   };
   if ( q<10 ) {  /* q is a one-digit number; point to terminal object */
      if ( ((hypre_NumbersNode *)node->digit[r])->digit[10] == NULL )
         ((hypre_NumbersNode *)node->digit[r])->digit[10] = hypre_NumbersNewNode();
   }
   else {  /* multidigit number; place for this digit points to next node */
      new = hypre_NumbersEnter( node->digit[r], q );
   }
   return new;
}

int hypre_NumbersNEntered( hypre_NumbersNode * node )
/* returns the number of numbers represented by the tree whose root is 'node' */
{
   int i;
   int count = 0;
   if ( node==NULL ) return 0;
   for ( i=0; i<10; ++i ) if ( node->digit[i] != NULL )
      count += hypre_NumbersNEntered( node->digit[i] );
   if ( node->digit[10] != NULL ) ++count;
   return count;
}

int hypre_NumbersQuery( hypre_NumbersNode * node, const int n )
/* returns 1 if n is on the tree with root 'node', 0 otherwise */
{
   int q = n/10;
   int r = n%10;
   hypre_assert( n>=0 );
   if ( node->digit[r] == NULL ) { /* low order digit of n not on tree */
      return 0;
   }
   else if ( q<10 ) { /* q is a one-digit number; check terminal object */
      if ( ((hypre_NumbersNode *)node->digit[r])->digit[10] == NULL )
         return 0;
      else
         return 1;
   }
   else {  /* look for higher order digits of n on tree of its low order digit r */
      return hypre_NumbersQuery( node->digit[r], q );
   }
}

int * hypre_NumbersArray( hypre_NumbersNode * node )
/* allocates and returns an unordered array of ints as a simpler representation
   of the contents of the Numbers tree.
   For the array length, call hypre_NumbersNEntered */
{
   int i, j, Ntemp;
   int k = 0;
   int N = hypre_NumbersNEntered(node);
   int * array, * temp;
   array = hypre_CTAlloc( int, N );
   if ( node==NULL ) return array;
   for ( i=0; i<10; ++i ) if ( node->digit[i] != NULL ) {
      Ntemp = hypre_NumbersNEntered( node->digit[i] );
      temp = hypre_NumbersArray( node->digit[i] );
      for ( j=0; j<Ntemp; ++j )
         array[k++] = temp[j]*10 + i;
      hypre_TFree(temp);
   }
   if ( node->digit[10] != NULL ) array[k++] = 0;
   hypre_assert( k==N );
   return array;
}
