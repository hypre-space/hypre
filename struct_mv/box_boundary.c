/*BHEADER**********************************************************************
 * (c) 2004   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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
 * Find the parts of the given box which lie on a (physical) boundary.
 * Stick them into the user-provided box array boundary (any input contents
 * of this box array may get changed).
 * The second input argument is a list of all neighbor boxes.
 * The last argument has 6 values to denote the boundary thickness in each direction.
 *--------------------------------------------------------------------------*/

int
hypre_BoxBoundaryNT( hypre_Box *box, hypre_BoxArray *neighbor_boxes,
                    hypre_BoxArray *boundary, int* thickness )
{
   int i, size;
   int numexp[6];
   int ierr = 0;
   hypre_Box *boxe = hypre_BoxDuplicate( box );
   hypre_Box *boxi;
   int thick = 1;
   for ( i=0; i<6; ++i ) {
      numexp[i] = -thickness[i];
      thick = thickness[i]>thick ? thickness[i] : thick;
   }

   ierr += hypre_BoxExpand( boxe, numexp );  /* shrink box away from boundary */
   ierr += hypre_SubtractBoxes( box, boxe, boundary );

   /* Now boundary contains the surface of the original box.
      Subtract out the neighbor boxes, and anything adjacent to a neighbor box.
      Anything left will belong to the physical boundary. */

   ierr += hypre_BoxArraySubtractAdjacentBoxArray( boundary, neighbor_boxes, box, thick );

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
   hypre_Box *boxi;
   for ( i=0; i<6; ++i ) numexp[i] = 0;
   numexp[ds] = -thick;

   ierr += hypre_BoxExpand( boxe, numexp );  /* shrink box away from boundary */
   ierr += hypre_SubtractBoxes( box, boxe, boundary );

   /* Now boundary contains the surface of the original box, in direction ds.
      Subtract out the neighbor boxes, and anything adjacent to a neighbor box.
      Anything left will belong to the physical boundary. */

   ierr += hypre_BoxArraySubtractAdjacentBoxArray( boundary, neighbor_boxes, box, thick );

   ierr += hypre_BoxDestroy( boxe );

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

