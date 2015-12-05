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
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Member functions for hypre_Box class:
 *   Functions to check for (physical) boundaries, adjacency, etc.
 * This is experimental code.  Whatever functions I find useful will be
 * copied into some other file.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * Take away from the boxes in boxes1 whatever is adjacent to boxes in boxes2,
 * as well as the boxes themselves.  But ignore "box" if it appears in boxes2.
 * The result is returned in boxes1.  The last argument, thick, is the number
 * of layers around a box considered to be "adjacent", typically 1.
 *--------------------------------------------------------------------------*/

int
hypre_BoxArraySubtractAdjacentBoxArray( hypre_BoxArray *boxes1,
                                        hypre_BoxArray *boxes2,
                                        hypre_Box *box, int thick )
{
   int ierr = 0;
   int i;
   int numexp[6];
   hypre_Box *box2e;
   hypre_Box *boxe = hypre_BoxDuplicate( box );
   hypre_BoxArray *boxes2e = hypre_BoxArrayDuplicate( boxes2 );
   hypre_BoxArray *tmp_box_array = hypre_BoxArrayCreate( 0 );
   for ( i=0; i<6; ++i ) numexp[i] = thick;
   hypre_ForBoxI(i, boxes2e)
      {
         box2e = hypre_BoxArrayBox(boxes2e, i);
         ierr += hypre_BoxExpand( box2e, numexp );
      }
   ierr += hypre_BoxExpand( boxe, numexp );
   ierr += hypre_SubtractBoxArraysExceptBoxes( boxes1, boxes2e, tmp_box_array, box, boxe );

   ierr += hypre_BoxArrayDestroy( boxes2e );
   ierr += hypre_BoxArrayDestroy( tmp_box_array );
   ierr += hypre_BoxDestroy( boxe );

   return ierr;
}

/*--------------------------------------------------------------------------
 * Take away from the boxes in boxes1 whatever is adjacent to boxes in boxes2,
 * in the signed direction ds only (ds=0,1,2,3,4,5);
 * as well as the boxes themselves.  But ignore "box" if it appears in boxes2.
 * The result is returned in boxes1.  The last argument, thick, is the number
 * of layers around a box considered to be "adjacent", typically 1.
 *--------------------------------------------------------------------------*/

int
hypre_BoxArraySubtractAdjacentBoxArrayD( hypre_BoxArray *boxes1,
                                        hypre_BoxArray *boxes2,
                                        hypre_Box *box, int ds, int thick )
{
   int ierr = 0;
   int i;
   int numexp[6];
   hypre_Box *box2e;
   hypre_Box *boxe = hypre_BoxDuplicate( box );
   hypre_BoxArray *boxes2e = hypre_BoxArrayDuplicate( boxes2 );
   hypre_BoxArray *tmp_box_array = hypre_BoxArrayCreate( 0 );
   for ( i=0; i<6; ++i ) numexp[i] = 0;
   numexp[ds] = thick;
   hypre_ForBoxI(i, boxes2e)
      {
         box2e = hypre_BoxArrayBox(boxes2e, i);
         ierr += hypre_BoxExpand( box2e, numexp );
      }
   ierr += hypre_BoxExpand( boxe, numexp );
   ierr += hypre_SubtractBoxArraysExceptBoxes( boxes1, boxes2e, tmp_box_array, box, boxe );

   ierr += hypre_BoxArrayDestroy( boxes2e );
   ierr += hypre_BoxArrayDestroy( tmp_box_array );
   ierr += hypre_BoxDestroy( boxe );

   return ierr;
}

/*--------------------------------------------------------------------------
 * Find the parts of the given box which lie on a (physical) boundary, in
 * the supplied signed direction (ds=0,1,2,3,4,5; for unsigned directions
 * d=0,0,1,1,2,2).  Boundary thickness is provided.
 * Stick them into the user-provided box array boundary (any input contents
 * of this box array may get changed).
 * The second input argument is a list of all neighbor boxes.
 *--------------------------------------------------------------------------*/

int
hypre_BoxBoundaryDNT( hypre_Box *box, hypre_BoxArray *neighbor_boxes,
                      hypre_BoxArray *boundary, int ds, int thick )
{
   int i;
   int numexp[6];
   int ierr = 0;
   hypre_Box *boxe = hypre_BoxDuplicate( box );
   for ( i=0; i<6; ++i ) numexp[i] = 0;
   numexp[ds] = -thick;

   ierr += hypre_BoxExpand( boxe, numexp );  /* shrink box away from boundary */
   ierr += hypre_SubtractBoxes( box, boxe, boundary );

   /* Now boundary contains the surface of the original box, in direction ds.
      Subtract out the neighbor boxes, and anything adjacent to a neighbor box
      in the opposite direction.
      Anything left will belong to the physical boundary. */

   switch(ds)
   {
   case 0:
      ds = 1;
      break;
   case 1:
      ds = 0;
      break;
   case 2:
      ds = 3;
      break;
   case 3:
      ds = 2;
      break;
   case 4:
      ds = 5;
      break;
   case 5:
      ds = 4;
   }
   ierr += hypre_BoxArraySubtractAdjacentBoxArrayD(
      boundary, neighbor_boxes, box, ds, thick );

   ierr += hypre_BoxDestroy( boxe );

   return ierr;
}

/*--------------------------------------------------------------------------
 * Find the parts of the given box which lie on a (physical) boundary.
 * Stick them into the user-provided box array boundary (it is recommended that
 * this box array be empty on input).
 * The second input argument is a list of all neighbor boxes.
 * The last argument has 6 values to denote the boundary thickness in each direction.
 *--------------------------------------------------------------------------*/

int
hypre_BoxBoundaryNT( hypre_Box *box, hypre_BoxArray *neighbor_boxes,
                    hypre_BoxArray *boundary, int* thickness )
{
   int ds;
   int ierr = 0;
   hypre_BoxArray *boundary_d;

   /* We'll find the physical boundary in one direction at a time.
      This is so that we don't lose boundary points which are adjacent
      to boundary points of the neighbor boxes. */
   for ( ds=0; ds<6; ++ds )
   {
      boundary_d = hypre_BoxArrayCreate( 0 );
      ierr += hypre_BoxBoundaryDNT( box, neighbor_boxes, boundary_d,
                                    ds, thickness[ds] );
      ierr += hypre_AppendBoxArray( boundary_d, boundary );
      hypre_BoxArrayDestroy( boundary_d );
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * Find the parts of the given box which lie on a (physical) boundary.
 * Stick them into the user-provided box array boundary (any input contents
 * of this box array may get changed).
 * The second input argument is the grid.
 * The boundary thickness is set to the ghost layer thickness, regardless
 * of whether the computed boundary will consist of ghost zones.
 *--------------------------------------------------------------------------*/

int
hypre_BoxBoundaryG( hypre_Box *box, hypre_StructGrid *g,
                    hypre_BoxArray *boundary )
{
   hypre_BoxNeighbors  *neighbors = hypre_StructGridNeighbors(g);
   hypre_BoxArray *neighbor_boxes = hypre_BoxNeighborsBoxes( neighbors );
   /* neighbor_boxes are this processor's neighbors, not this box's
      neighbors.  But it's likely to be cheaper to use them all in the
      next step than to try to shrink it to just this box's neighbors. */
   int * thickness = hypre_StructGridNumGhost(g);
   return hypre_BoxBoundaryNT( box, neighbor_boxes, boundary, thickness );
}

/*--------------------------------------------------------------------------
 * Find the parts of the given box which lie on a (physical) boundary, only
 * in the (unsigned) direction of d (d=0,1,2).
 * Stick them into the user-provided box arrays boundarym (for the minus direction)
 * and boundaryp (for the plus direction).  (Any input contents of these box
 * arrays may get changed).
 * The second input argument is the grid the box is in (hypre_BoxBoundaryG).
 * The boundary thickness is set to 1.
 *--------------------------------------------------------------------------*/

int
hypre_BoxBoundaryDG( hypre_Box *box, hypre_StructGrid *g,
                     hypre_BoxArray *boundarym, hypre_BoxArray *boundaryp,
                     int d )
{
   int ierr = 0;
   hypre_BoxNeighbors  *neighbors = hypre_StructGridNeighbors(g);
   hypre_BoxArray *neighbor_boxes = hypre_BoxNeighborsBoxes( neighbors );
   int i;
   int thickness[6];
   for ( i=0; i<6; ++i ) thickness[i] = 1;
   /* neighbor_boxes are this processor's neighbors, not this box's
      neighbors.  But it's likely to be cheaper to use them all in the
      next step than to try to shrink it to just this box's neighbors. */
   ierr += hypre_BoxBoundaryDNT( box, neighbor_boxes, boundarym, 2*d, thickness[2*d] );
   ierr += hypre_BoxBoundaryDNT( box, neighbor_boxes, boundaryp, 2*d+1, thickness[2*d] );
   return ierr;
}

