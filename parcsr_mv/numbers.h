/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Tree structure for keeping track of numbers (e.g. column numbers) -
 * when you get them one at a time, in no particular order, possibly very
 * sparse.  In a scalable manner you want to be able to store them and find
 * out whether a number has been stored.
 * All decimal numbers will fit in a tree with 10 branches (digits)
 * off each node.  We also have a terminal "digit" to indicate that the entire
 * number has been seen.  E.g., 1234 would be entered in a tree as:
 * (numbering the digits off a node as 0 1 2 3 4 5 6 7 8 9 TERM )
 *                          root
 *                           |
 *                   - - - - 4 - - - - - -
 *                           |
 *                     - - - 3 - - - - - - -
 *                           |
 *                       - - 2 - - - - - - - -
 *                           |
 *                         - 1 - - - - - - - - -
 *                           |
 *       - - - - - - - - - - T
 *
 *
 * This tree represents a number through its decimal expansion, but if needed
 * this code can be changed to a different base, e.g. binary.  The appropriate
 * base depends on how the numbers encountered are distributed.  Totally
 * random (independent, equally likely in a large range) calls for binary.
 * The more clustered, the larger the base should be in my judgement.
 *
 *****************************************************************************/

#ifndef hypre_NUMBERS_HEADER
#define hypre_NUMBERS_HEADER

struct hypre_NumbersNode;

typedef struct {
   void * digit[11];
/* ... should be   hypre_NumbersNode * digit[11]; */
} hypre_NumbersNode;

#endif


hypre_NumbersNode * hypre_NumbersNewNode();
void hypre_NumbersDeleteNode( hypre_NumbersNode * node );
int hypre_NumbersEnter( hypre_NumbersNode * node, const int n );
int hypre_NumbersNEntered( hypre_NumbersNode * node );
int hypre_NumbersQuery( hypre_NumbersNode * node, const int n );
int * hypre_NumbersArray( hypre_NumbersNode * node );


