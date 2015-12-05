/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * NOTE: The following routines are currently only used as follows in hypre, and
 * also appear in '_hypre_struct_mv.h':
 * 
 * hypre_BoxBoundaryG
 * struct_mv/box_boundary.c
 * struct_mv/struct_vector.c
 * sstruct_ls/maxwell_grad.c
 * sstruct_ls/maxwell_TV_setup.c
 * 
 * hypre_BoxBoundaryDG
 * struct_mv/box_boundary.c
 * sstruct_ls/maxwell_grad.c
 * sstruct_ls/maxwell_PNedelec_bdy.c
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * Intersect a surface of 'box' with the physical boundary.  The surface is
 * given by (d,dir), where 'dir' is a direction (+-1) in dimension 'd'.
 *
 * The result will be returned in the box array 'boundary'.  Any boxes already
 * in 'boundary' will be overwritten.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBoundaryIntersect( hypre_Box *box,
                            hypre_StructGrid *grid,
                            HYPRE_Int d,
                            HYPRE_Int dir,
                            hypre_BoxArray *boundary )
{
   hypre_BoxManager   *boxman;
   hypre_BoxManEntry **entries;
   hypre_BoxArray     *int_boxes, *tmp_boxes;
   hypre_Box          *bbox, *ibox;
   HYPRE_Int           nentries, i;

   /* set bbox to the box surface of interest */
   hypre_BoxArraySetSize(boundary, 1);
   bbox = hypre_BoxArrayBox(boundary, 0);
   hypre_CopyBox(box, bbox);
   if (dir > 0)
   {
      hypre_BoxIMinD(bbox, d) = hypre_BoxIMaxD(bbox, d);
   }
   else if (dir < 0)
   {
      hypre_BoxIMaxD(bbox, d) = hypre_BoxIMinD(bbox, d);
   }

   /* temporarily shift bbox in direction dir and intersect with the grid */
   hypre_BoxIMinD(bbox, d) += dir;
   hypre_BoxIMaxD(bbox, d) += dir;
   boxman = hypre_StructGridBoxMan(grid);
   hypre_BoxManIntersect(boxman, hypre_BoxIMin(bbox), hypre_BoxIMax(bbox),
                         &entries, &nentries);
   hypre_BoxIMinD(bbox, d) -= dir;
   hypre_BoxIMaxD(bbox, d) -= dir;

   /* shift intersected boxes in direction -dir and subtract from bbox */
   int_boxes  = hypre_BoxArrayCreate(nentries);
   tmp_boxes  = hypre_BoxArrayCreate(0);
   for (i = 0; i < nentries; i++)
   {
      ibox = hypre_BoxArrayBox(int_boxes, i);
      hypre_BoxManEntryGetExtents(
         entries[i], hypre_BoxIMin(ibox), hypre_BoxIMax(ibox));
      hypre_BoxIMinD(ibox, d) -= dir;
      hypre_BoxIMaxD(ibox, d) -= dir;
   }
   hypre_SubtractBoxArrays(boundary, int_boxes, tmp_boxes);

   hypre_BoxArrayDestroy(int_boxes);
   hypre_BoxArrayDestroy(tmp_boxes);
   hypre_TFree(entries);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Find the parts of the given box which lie on a (physical) boundary of grid g.
 * Stick them into the user-provided box array boundary.  Any input contents of
 * this box array will get overwritten.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBoundaryG( hypre_Box *box,
                    hypre_StructGrid *g,
                    hypre_BoxArray *boundary )
{
   hypre_BoxArray *boundary_d;
   HYPRE_Int       d;
 
   boundary_d = hypre_BoxArrayCreate(0);
   for (d = 0; d < 3; d++)
   {
      hypre_BoxBoundaryIntersect(box, g, d, -1, boundary_d);
      hypre_AppendBoxArray(boundary_d, boundary);
      hypre_BoxBoundaryIntersect(box, g, d,  1, boundary_d);
      hypre_AppendBoxArray(boundary_d, boundary);
   }
   hypre_BoxArrayDestroy(boundary_d);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Find the parts of the given box which lie on a (physical) boundary of grid g,
 * only in the (unsigned) direction of d (d=0,1,2).  Stick them into the
 * user-provided box arrays boundarym (minus direction) and boundaryp (plus
 * direction).  Any input contents of these box arrays will get overwritten.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBoundaryDG( hypre_Box *box,
                     hypre_StructGrid *g,
                     hypre_BoxArray *boundarym,
                     hypre_BoxArray *boundaryp,
                     HYPRE_Int d )
{
   hypre_BoxBoundaryIntersect(box, g, d, -1, boundarym);
   hypre_BoxBoundaryIntersect(box, g, d,  1, boundaryp);

   return hypre_error_flag;
}

