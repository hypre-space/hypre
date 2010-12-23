/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
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

HYPRE_Int
hypre_BoxArraySubtractAdjacentBoxArray( hypre_BoxArray *boxes1,
                                        hypre_BoxArray *boxes2,
                                        hypre_Box *box, HYPRE_Int thick )
{
   HYPRE_Int ierr = 0;
   HYPRE_Int i;
   HYPRE_Int numexp[6];
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

HYPRE_Int
hypre_BoxArraySubtractAdjacentBoxArrayD( hypre_BoxArray *boxes1,
                                        hypre_BoxArray *boxes2,
                                        hypre_Box *box, HYPRE_Int ds, HYPRE_Int thick )
{
   HYPRE_Int ierr = 0;
   HYPRE_Int i;
   HYPRE_Int numexp[6];
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

HYPRE_Int
hypre_BoxBoundaryDNT( hypre_Box *box, hypre_BoxArray *neighbor_boxes,
                      hypre_BoxArray *boundary, HYPRE_Int ds, HYPRE_Int thick )
{
   HYPRE_Int i;
   HYPRE_Int numexp[6];
   HYPRE_Int ierr = 0;
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

HYPRE_Int
hypre_BoxBoundaryNT( hypre_Box *box, hypre_BoxArray *neighbor_boxes,
                    hypre_BoxArray *boundary, HYPRE_Int* thickness )
{
   HYPRE_Int ds;
   HYPRE_Int ierr = 0;
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

HYPRE_Int
hypre_BoxBoundaryG( hypre_Box *box, hypre_StructGrid *g,
                    hypre_BoxArray *boundary )
{

    

   hypre_BoxManager *boxman;
   hypre_BoxArray   *neighbor_boxes = NULL;
   HYPRE_Int        *thickness = hypre_StructGridNumGhost(g);
 
   /* neighbor_boxes are this processor's neighbors, not this box's
      neighbors.  But it's likely to be cheaper to use them all in the
      next step than to try to shrink it to just this box's neighbors. */

   /* get the boxes out of the box manager - use these as the neighbor boxes */
   boxman = hypre_StructGridBoxMan(g);
   neighbor_boxes = hypre_BoxArrayCreate(0);
   hypre_BoxManGetAllEntriesBoxes( boxman, neighbor_boxes);
      
   hypre_BoxBoundaryNT( box, neighbor_boxes, boundary, thickness );

   /* clean up */
   hypre_BoxArrayDestroy(neighbor_boxes);

   return hypre_error_flag;
   


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

HYPRE_Int
hypre_BoxBoundaryDG( hypre_Box *box, hypre_StructGrid *g,
                     hypre_BoxArray *boundarym, hypre_BoxArray *boundaryp,
                     HYPRE_Int d )
{
   hypre_BoxManager *boxman;
   hypre_BoxArray *neighbor_boxes = NULL;
   HYPRE_Int i;
   HYPRE_Int thickness[6];

   /* neighbor_boxes are this processor's neighbors, not this box's
      neighbors.  But it's likely to be cheaper to use them all in the
      next step than to try to shrink it to just this box's neighbors. */
   
   /* get the boxes out of the box manager - use these as the neighbor boxes */
   boxman = hypre_StructGridBoxMan(g);
   neighbor_boxes = hypre_BoxArrayCreate(0);
   hypre_BoxManGetAllEntriesBoxes( boxman, neighbor_boxes);
   
   for ( i=0; i<6; ++i ) thickness[i] = 1;
   
   hypre_BoxBoundaryDNT( box, neighbor_boxes, boundarym, 2*d, thickness[2*d] );
   hypre_BoxBoundaryDNT( box, neighbor_boxes, boundaryp, 2*d+1, thickness[2*d] );

   hypre_BoxArrayDestroy(neighbor_boxes);

   return hypre_error_flag;
}

