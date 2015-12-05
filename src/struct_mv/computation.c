/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.17 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 * 
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoCreate( hypre_CommInfo       *comm_info,
                         hypre_BoxArrayArray  *indt_boxes,
                         hypre_BoxArrayArray  *dept_boxes,
                         hypre_ComputeInfo   **compute_info_ptr )
{
   HYPRE_Int  ierr = 0;
   hypre_ComputeInfo  *compute_info;

   compute_info = hypre_TAlloc(hypre_ComputeInfo, 1);

   hypre_ComputeInfoCommInfo(compute_info)  = comm_info;
   hypre_ComputeInfoIndtBoxes(compute_info) = indt_boxes;
   hypre_ComputeInfoDeptBoxes(compute_info) = dept_boxes;

   hypre_SetIndex(hypre_ComputeInfoStride(compute_info), 1, 1, 1);

   *compute_info_ptr = compute_info;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoProjectSend( hypre_ComputeInfo  *compute_info,
                              hypre_Index         index,
                              hypre_Index         stride )
{
   HYPRE_Int  ierr = 0;

   hypre_CommInfoProjectSend(hypre_ComputeInfoCommInfo(compute_info),
                             index, stride);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoProjectRecv( hypre_ComputeInfo  *compute_info,
                              hypre_Index         index,
                              hypre_Index         stride )
{
   HYPRE_Int  ierr = 0;

   hypre_CommInfoProjectRecv(hypre_ComputeInfoCommInfo(compute_info),
                             index, stride);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoProjectComp( hypre_ComputeInfo  *compute_info,
                              hypre_Index         index,
                              hypre_Index         stride )
{
   HYPRE_Int  ierr = 0;

   hypre_ProjectBoxArrayArray(hypre_ComputeInfoIndtBoxes(compute_info),
                              index, stride);
   hypre_ProjectBoxArrayArray(hypre_ComputeInfoDeptBoxes(compute_info),
                              index, stride);
   hypre_CopyIndex(stride, hypre_ComputeInfoStride(compute_info));

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputeInfoDestroy( hypre_ComputeInfo  *compute_info )
{
   HYPRE_Int  ierr = 0;

   hypre_TFree(compute_info);

   return ierr;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications and computations patterns for
 * a given grid-stencil computation.  If HYPRE\_OVERLAP\_COMM\_COMP is
 * defined, then the patterns are computed to allow for overlapping
 * communications and computations.  The default is no overlap.
 *
 * Note: This routine assumes that the grid boxes do not overlap.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateComputeInfo( hypre_StructGrid      *grid,
                         hypre_StructStencil   *stencil,
                         hypre_ComputeInfo    **compute_info_ptr )
{
   HYPRE_Int                ierr = 0;

   hypre_CommInfo          *comm_info;
   hypre_BoxArrayArray     *indt_boxes;
   hypre_BoxArrayArray     *dept_boxes;

   hypre_BoxArray          *boxes;

   hypre_BoxArray          *cbox_array;
   hypre_Box               *cbox;

   HYPRE_Int                i;

#ifdef HYPRE_OVERLAP_COMM_COMP
   hypre_Box               *rembox;
   hypre_Index             *stencil_shape;
   HYPRE_Int                border[3][2] = {{0, 0}, {0, 0}, {0, 0}};
   HYPRE_Int                cbox_array_size;
   HYPRE_Int                s, d;
#endif

   /*------------------------------------------------------
    * Extract needed grid info
    *------------------------------------------------------*/

   boxes = hypre_StructGridBoxes(grid);

   /*------------------------------------------------------
    * Get communication info
    *------------------------------------------------------*/

   hypre_CreateCommInfoFromStencil(grid, stencil, &comm_info);

#ifdef HYPRE_OVERLAP_COMM_COMP

   /*------------------------------------------------------
    * Compute border info
    *------------------------------------------------------*/

   stencil_shape = hypre_StructStencilShape(stencil);
   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
   {
      for (d = 0; d < 3; d++)
      {
         i = hypre_IndexD(stencil_shape[s], d);
         if (i < 0)
         {
            border[d][0] = hypre_max(border[d][0], -i);
         }
         else if (i > 0)
         {
            border[d][1] = hypre_max(border[d][1], i);
         }
      }
   }

   /*------------------------------------------------------
    * Set up the dependent boxes
    *------------------------------------------------------*/

   dept_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));

   rembox = hypre_BoxCreate();
   hypre_ForBoxI(i, boxes)
      {
         cbox_array = hypre_BoxArrayArrayBoxArray(dept_boxes, i);
         hypre_BoxArraySetSize(cbox_array, 6);

         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), rembox);
         cbox_array_size = 0;
         for (d = 0; d < 3; d++)
         {
            if ( (hypre_BoxVolume(rembox)) && (border[d][0]) )
            {
               cbox = hypre_BoxArrayBox(cbox_array, cbox_array_size);
               hypre_CopyBox(rembox, cbox);
               hypre_BoxIMaxD(cbox, d) =
                  hypre_BoxIMinD(cbox, d) + border[d][0] - 1;
               hypre_BoxIMinD(rembox, d) =
                  hypre_BoxIMinD(cbox, d) + border[d][0];
               cbox_array_size++;
            }
            if ( (hypre_BoxVolume(rembox)) && (border[d][1]) )
            {
               cbox = hypre_BoxArrayBox(cbox_array, cbox_array_size);
               hypre_CopyBox(rembox, cbox);
               hypre_BoxIMinD(cbox, d) =
                  hypre_BoxIMaxD(cbox, d) - border[d][1] + 1;
               hypre_BoxIMaxD(rembox, d) =
                  hypre_BoxIMaxD(cbox, d) - border[d][1];
               cbox_array_size++;
            }
         }
         hypre_BoxArraySetSize(cbox_array, cbox_array_size);
      }
   hypre_BoxDestroy(rembox);

   /*------------------------------------------------------
    * Set up the independent boxes
    *------------------------------------------------------*/

   indt_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));

   hypre_ForBoxI(i, boxes)
      {
         cbox_array = hypre_BoxArrayArrayBoxArray(indt_boxes, i);
         hypre_BoxArraySetSize(cbox_array, 1);
         cbox = hypre_BoxArrayBox(cbox_array, 0);
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), cbox);

         for (d = 0; d < 3; d++)
         {
            if ( (border[d][0]) )
            {
               hypre_BoxIMinD(cbox, d) += border[d][0];
            }
            if ( (border[d][1]) )
            {
               hypre_BoxIMaxD(cbox, d) -= border[d][1];
            }
         }
      }

#else

   /*------------------------------------------------------
    * Set up the independent boxes
    *------------------------------------------------------*/

   indt_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));

   /*------------------------------------------------------
    * Set up the dependent boxes
    *------------------------------------------------------*/

   dept_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));

   hypre_ForBoxI(i, boxes)
      {
         cbox_array = hypre_BoxArrayArrayBoxArray(dept_boxes, i);
         hypre_BoxArraySetSize(cbox_array, 1);
         cbox = hypre_BoxArrayBox(cbox_array, 0);
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), cbox);
      }

#endif

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   hypre_ComputeInfoCreate(comm_info, indt_boxes, dept_boxes,
                           compute_info_ptr);

   return ierr;
}

/*--------------------------------------------------------------------------
 * Create a computation package from a grid-based description of a
 * communication-computation pattern.
 *
 * Note: The input boxes and processes are destroyed.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputePkgCreate( hypre_ComputeInfo     *compute_info,
                        hypre_BoxArray        *data_space,
                        HYPRE_Int              num_values,
                        hypre_StructGrid      *grid,
                        hypre_ComputePkg     **compute_pkg_ptr )
{
   HYPRE_Int          ierr = 0;
   hypre_ComputePkg  *compute_pkg;
   hypre_CommPkg     *comm_pkg;

   compute_pkg = hypre_CTAlloc(hypre_ComputePkg, 1);

   hypre_CommPkgCreate(hypre_ComputeInfoCommInfo(compute_info),
                       data_space, data_space, num_values, NULL, 0,
                       hypre_StructGridComm(grid), &comm_pkg);
   hypre_CommInfoDestroy(hypre_ComputeInfoCommInfo(compute_info));
   hypre_ComputePkgCommPkg(compute_pkg) = comm_pkg;

   hypre_ComputePkgIndtBoxes(compute_pkg) = 
      hypre_ComputeInfoIndtBoxes(compute_info);
   hypre_ComputePkgDeptBoxes(compute_pkg) =
      hypre_ComputeInfoDeptBoxes(compute_info);
   hypre_CopyIndex(hypre_ComputeInfoStride(compute_info),
                   hypre_ComputePkgStride(compute_pkg));

   hypre_StructGridRef(grid, &hypre_ComputePkgGrid(compute_pkg));
   hypre_ComputePkgDataSpace(compute_pkg) = data_space;
   hypre_ComputePkgNumValues(compute_pkg) = num_values;

   hypre_ComputeInfoDestroy(compute_info);

   *compute_pkg_ptr = compute_pkg;

   return ierr;
}

/*--------------------------------------------------------------------------
 * Destroy a computation package.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ComputePkgDestroy( hypre_ComputePkg *compute_pkg )
{
   HYPRE_Int ierr = 0;

   if (compute_pkg)
   {
      hypre_CommPkgDestroy(hypre_ComputePkgCommPkg(compute_pkg));

      hypre_BoxArrayArrayDestroy(hypre_ComputePkgIndtBoxes(compute_pkg));
      hypre_BoxArrayArrayDestroy(hypre_ComputePkgDeptBoxes(compute_pkg));

      hypre_StructGridDestroy(hypre_ComputePkgGrid(compute_pkg));

      hypre_TFree(compute_pkg);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * Initialize a non-blocking communication exchange.  The independent
 * computations may be done after a call to this routine, to allow for
 * overlap of communications and computations.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_InitializeIndtComputations( hypre_ComputePkg  *compute_pkg,
                                  double            *data,
                                  hypre_CommHandle **comm_handle_ptr )
{
   HYPRE_Int      ierr = 0;
   hypre_CommPkg *comm_pkg = hypre_ComputePkgCommPkg(compute_pkg);

   ierr = hypre_InitializeCommunication(comm_pkg, data, data, 0, 0, comm_handle_ptr);

   return ierr;
}

/*--------------------------------------------------------------------------
 * Finalize a communication exchange.  The dependent computations may
 * be done after a call to this routine.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FinalizeIndtComputations( hypre_CommHandle *comm_handle )
{
   HYPRE_Int ierr = 0;

   ierr = hypre_FinalizeCommunication(comm_handle );

   return ierr;
}
