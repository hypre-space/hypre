#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
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
   assert( n>=0 );
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
   int i, new=0;
   int q = n/10;
   int r = n%10;
   assert( n>=0 );
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
   assert( k==N );
   return array;
}
