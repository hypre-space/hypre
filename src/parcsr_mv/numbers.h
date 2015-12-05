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
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



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
 * base depends on how the numbers encountered are distributed.  Totally
 * The more clustered, the larger the base should be in my judgement.
 *
 *****************************************************************************/

#ifndef hypre_NUMBERS_HEADER
#define hypre_NUMBERS_HEADER

typedef struct {
   void * digit[11];
/* ... should be   hypre_NumbersNode * digit[11]; */
} hypre_NumbersNode;


hypre_NumbersNode * hypre_NumbersNewNode(void);
void hypre_NumbersDeleteNode( hypre_NumbersNode * node );
int hypre_NumbersEnter( hypre_NumbersNode * node, const int n );
int hypre_NumbersNEntered( hypre_NumbersNode * node );
int hypre_NumbersQuery( hypre_NumbersNode * node, const int n );
int * hypre_NumbersArray( hypre_NumbersNode * node );


#endif
