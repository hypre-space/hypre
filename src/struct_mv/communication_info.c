/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * See comments for CreateCommInfo() routine for more info on CommStencil
 *--------------------------------------------------------------------------*/

hypre_CommStencil *
hypre_CommStencilCreate( HYPRE_Int  ndim )
{
   hypre_CommStencil  *comm_stencil;
   hypre_Box          *csbox;
   HYPRE_Int          *csdata;

   comm_stencil = hypre_CTAlloc(hypre_CommStencil, 1, HYPRE_MEMORY_HOST);

   csbox = hypre_BoxCreate(ndim);
   hypre_SetIndex(hypre_BoxIMin(csbox), 0);
   hypre_SetIndex(hypre_BoxIMax(csbox), 2);
   csdata = hypre_CTAlloc(HYPRE_Int, hypre_BoxVolume(csbox), HYPRE_MEMORY_HOST);

   hypre_CommStencilNDim(comm_stencil) = ndim;
   hypre_CommStencilBox(comm_stencil)  = csbox;
   hypre_CommStencilData(comm_stencil) = csdata;
   hypre_SetIndex(hypre_CommStencilStride(comm_stencil), 1);

   return comm_stencil;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommStencilSetEntry( hypre_CommStencil  *comm_stencil,
                           hypre_Index         offset )
{
   HYPRE_Int              ndim   = hypre_CommStencilNDim(comm_stencil);
   hypre_Box             *csbox  = hypre_CommStencilBox(comm_stencil);
   HYPRE_Int             *csdata = hypre_CommStencilData(comm_stencil);
   hypre_IndexRef         stride = hypre_CommStencilStride(comm_stencil);
   HYPRE_Int             *mgrow  = hypre_CommStencilMGrow(comm_stencil);
   HYPRE_Int             *pgrow  = hypre_CommStencilPGrow(comm_stencil);

   hypre_Box              boxmem;
   hypre_Box             *box = &boxmem;
   hypre_IndexRef         imin = hypre_BoxIMin(box);
   hypre_IndexRef         imax = hypre_BoxIMax(box);
   hypre_Index            loop_size;
   HYPRE_Int              d, m;

   hypre_BoxInit(box, ndim);

   for (d = 0; d < ndim; d++)
   {
      m = offset[d];

      imin[d] = 1;
      imax[d] = 1;

      if (m < 0)
      {
         imin[d] = 0;
         mgrow[d] = hypre_max(mgrow[d], -m);
      }
      else if (m > 0)
      {
         imax[d] = 2;
         pgrow[d] = hypre_max(pgrow[d],  m);
      }
   }

   /* update comm-stencil data */
   hypre_BoxGetSize(box, loop_size);
   hypre_SerialBoxLoop1Begin(ndim, loop_size,
                             csbox, imin, stride, ii);
   {
      csdata[ii] = 1;
   }
   hypre_SerialBoxLoop1End(ii);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommStencilDestroy( hypre_CommStencil  *comm_stencil )
{
   hypre_BoxDestroy(hypre_CommStencilBox(comm_stencil));
   hypre_TFree(hypre_CommStencilData(comm_stencil), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_stencil, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommStencilCreateNumGhost( hypre_CommStencil  *comm_stencil,
                                 HYPRE_Int         **num_ghost_ptr )
{
   HYPRE_Int      *num_ghost;
   HYPRE_Int       ndim   = hypre_CommStencilNDim(comm_stencil);
   HYPRE_Int      *mgrow  = hypre_CommStencilMGrow(comm_stencil);
   HYPRE_Int      *pgrow  = hypre_CommStencilPGrow(comm_stencil);
   HYPRE_Int       d;

   num_ghost = hypre_CTAlloc(HYPRE_Int, 2 * ndim, HYPRE_MEMORY_HOST);

   for (d = 0; d < ndim; d++)
   {
      num_ghost[2 * d]     = mgrow[d];
      num_ghost[2 * d + 1] = pgrow[d];
   }

   *num_ghost_ptr = num_ghost;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note that send_coords, recv_coords, send_dirs, recv_dirs may be NULL to
 * represent an identity transform.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoCreate( hypre_BoxArrayArray  *send_boxes,
                      hypre_BoxArrayArray  *recv_boxes,
                      HYPRE_Int           **send_procs,
                      HYPRE_Int           **recv_procs,
                      HYPRE_Int           **send_rboxnums,
                      HYPRE_Int           **recv_rboxnums,
                      hypre_BoxArrayArray  *send_rboxes,
                      hypre_BoxArrayArray  *recv_rboxes,
                      HYPRE_Int             boxes_match,
                      hypre_CommInfo      **comm_info_ptr )
{
   hypre_CommInfo  *comm_info;

   comm_info = hypre_TAlloc(hypre_CommInfo, 1, HYPRE_MEMORY_HOST);

   hypre_CommInfoNDim(comm_info)          = hypre_BoxArrayArrayNDim(send_boxes);
   hypre_CommInfoSendBoxes(comm_info)     = send_boxes;
   hypre_CommInfoRecvBoxes(comm_info)     = recv_boxes;
   hypre_CommInfoSendProcesses(comm_info) = send_procs;
   hypre_CommInfoRecvProcesses(comm_info) = recv_procs;
   hypre_CommInfoSendRBoxnums(comm_info)  = send_rboxnums;
   hypre_CommInfoRecvRBoxnums(comm_info)  = recv_rboxnums;
   hypre_CommInfoSendRBoxes(comm_info)    = send_rboxes;
   hypre_CommInfoRecvRBoxes(comm_info)    = recv_rboxes;

   hypre_CommInfoNumTransforms(comm_info)  = 0;
   hypre_CommInfoCoords(comm_info)         = NULL;
   hypre_CommInfoDirs(comm_info)           = NULL;
   hypre_CommInfoSendTransforms(comm_info) = NULL;
   hypre_CommInfoRecvTransforms(comm_info) = NULL;

   hypre_CommInfoBoxesMatch(comm_info)    = boxes_match;
   hypre_SetIndex(hypre_CommInfoSendStride(comm_info), 1);
   hypre_SetIndex(hypre_CommInfoRecvStride(comm_info), 1);

   *comm_info_ptr = comm_info;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoSetTransforms( hypre_CommInfo  *comm_info,
                             HYPRE_Int        num_transforms,
                             hypre_Index     *coords,
                             hypre_Index     *dirs,
                             HYPRE_Int      **send_transforms,
                             HYPRE_Int      **recv_transforms )
{
   hypre_CommInfoNumTransforms(comm_info)  = num_transforms;
   hypre_CommInfoCoords(comm_info)         = coords;
   hypre_CommInfoDirs(comm_info)           = dirs;
   hypre_CommInfoSendTransforms(comm_info) = send_transforms;
   hypre_CommInfoRecvTransforms(comm_info) = recv_transforms;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoGetTransforms( hypre_CommInfo  *comm_info,
                             HYPRE_Int       *num_transforms,
                             hypre_Index    **coords,
                             hypre_Index    **dirs )
{
   *num_transforms = hypre_CommInfoNumTransforms(comm_info);
   *coords         = hypre_CommInfoCoords(comm_info);
   *dirs           = hypre_CommInfoDirs(comm_info);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoProjectSend( hypre_CommInfo  *comm_info,
                           hypre_Index      index,
                           hypre_Index      stride )
{
   hypre_ProjectBoxArrayArray(hypre_CommInfoSendBoxes(comm_info), index, stride);
   hypre_ProjectBoxArrayArray(hypre_CommInfoSendRBoxes(comm_info), index, stride);
   hypre_CopyIndex(stride, hypre_CommInfoSendStride(comm_info));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoProjectRecv( hypre_CommInfo  *comm_info,
                           hypre_Index      index,
                           hypre_Index      stride )
{
   hypre_ProjectBoxArrayArray(hypre_CommInfoRecvBoxes(comm_info), index, stride);
   hypre_ProjectBoxArrayArray(hypre_CommInfoRecvRBoxes(comm_info), index, stride);
   hypre_CopyIndex(stride, hypre_CommInfoRecvStride(comm_info));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoCoarsenSend( hypre_CommInfo     *comm_info,
                           hypre_Index         index,
                           hypre_Index         stride )
{
   hypre_CoarsenBoxArrayArray(hypre_CommInfoSendBoxes(comm_info), index, stride);
   hypre_CoarsenBoxArrayArray(hypre_CommInfoSendRBoxes(comm_info), index, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoCoarsenRecv( hypre_CommInfo     *comm_info,
                           hypre_Index         index,
                           hypre_Index         stride )
{
   hypre_CoarsenBoxArrayArray(hypre_CommInfoRecvBoxes(comm_info), index, stride);
   hypre_CoarsenBoxArrayArray(hypre_CommInfoRecvRBoxes(comm_info), index, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoRefineSend( hypre_CommInfo     *comm_info,
                          hypre_Index         index,
                          hypre_Index         stride )
{
   hypre_RefineBoxArrayArray(hypre_CommInfoSendBoxes(comm_info), index, stride);
   hypre_RefineBoxArrayArray(hypre_CommInfoSendRBoxes(comm_info), index, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoRefineRecv( hypre_CommInfo     *comm_info,
                          hypre_Index         index,
                          hypre_Index         stride )
{
   hypre_RefineBoxArrayArray(hypre_CommInfoRecvBoxes(comm_info), index, stride);
   hypre_RefineBoxArrayArray(hypre_CommInfoRecvRBoxes(comm_info), index, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoCoarsen( hypre_CommInfo     *comm_info,
                       hypre_Index         index,
                       hypre_Index         stride )
{
   hypre_CommInfoCoarsenSend(comm_info, index, stride);
   hypre_CommInfoCoarsenRecv(comm_info, index, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoRefine( hypre_CommInfo     *comm_info,
                      hypre_Index         index,
                      hypre_Index         stride )
{
   hypre_CommInfoRefineSend(comm_info, index, stride);
   hypre_CommInfoRefineRecv(comm_info, index, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoDestroy( hypre_CommInfo  *comm_info )
{
   HYPRE_Int           **processes;
   HYPRE_Int           **rboxnums;
   HYPRE_Int           **transforms;
   HYPRE_Int             i, size;

   if (comm_info)
   {
      size = hypre_BoxArrayArraySize(hypre_CommInfoSendBoxes(comm_info));
      hypre_BoxArrayArrayDestroy(hypre_CommInfoSendBoxes(comm_info));
      processes = hypre_CommInfoSendProcesses(comm_info);
      for (i = 0; i < size; i++)
      {
         hypre_TFree(processes[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(processes, HYPRE_MEMORY_HOST);
      rboxnums = hypre_CommInfoSendRBoxnums(comm_info);
      if (rboxnums != NULL)
      {
         for (i = 0; i < size; i++)
         {
            hypre_TFree(rboxnums[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(rboxnums, HYPRE_MEMORY_HOST);
      }
      hypre_BoxArrayArrayDestroy(hypre_CommInfoSendRBoxes(comm_info));
      transforms = hypre_CommInfoSendTransforms(comm_info);
      if (transforms != NULL)
      {
         for (i = 0; i < size; i++)
         {
            hypre_TFree(transforms[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(transforms, HYPRE_MEMORY_HOST);
      }

      size = hypre_BoxArrayArraySize(hypre_CommInfoRecvBoxes(comm_info));
      hypre_BoxArrayArrayDestroy(hypre_CommInfoRecvBoxes(comm_info));
      processes = hypre_CommInfoRecvProcesses(comm_info);
      for (i = 0; i < size; i++)
      {
         hypre_TFree(processes[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(processes, HYPRE_MEMORY_HOST);
      rboxnums = hypre_CommInfoRecvRBoxnums(comm_info);
      if (rboxnums != NULL)
      {
         for (i = 0; i < size; i++)
         {
            hypre_TFree(rboxnums[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(rboxnums, HYPRE_MEMORY_HOST);
      }
      hypre_BoxArrayArrayDestroy(hypre_CommInfoRecvRBoxes(comm_info));
      transforms = hypre_CommInfoRecvTransforms(comm_info);
      if (transforms != NULL)
      {
         for (i = 0; i < size; i++)
         {
            hypre_TFree(transforms[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(transforms, HYPRE_MEMORY_HOST);
      }

      hypre_TFree(hypre_CommInfoCoords(comm_info), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommInfoDirs(comm_info), HYPRE_MEMORY_HOST);

      hypre_TFree(comm_info, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Clone a CommInfo structure.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommInfoClone( hypre_CommInfo   *comm_info,
                     hypre_CommInfo  **clone_ptr )
{
   hypre_CommInfo       *clone;
   hypre_BoxArrayArray  *comm_boxes,       *clone_boxes;
   hypre_IndexRef        comm_stride,       clone_stride;
   HYPRE_Int           **comm_processes,  **clone_processes;
   HYPRE_Int           **comm_rboxnums,   **clone_rboxnums;
   hypre_BoxArrayArray  *comm_rboxes,      *clone_rboxes;
   HYPRE_Int           **comm_transforms, **clone_transforms;
   hypre_Index          *comm_coords,      *clone_coords;
   hypre_Index          *comm_dirs,        *clone_dirs;
   HYPRE_Int             i, j, k, size_aa, size_a, num_transforms;

   clone = hypre_TAlloc(hypre_CommInfo, 1, HYPRE_MEMORY_HOST);

   /* ndim */
   hypre_CommInfoNDim(clone) = hypre_CommInfoNDim(comm_info);

   for (k = 0; k < 2; k++)
   {
      switch (k)
      {
         case 0: /* Clone send info */
            comm_boxes      = hypre_CommInfoSendBoxes(comm_info);
            comm_stride     = hypre_CommInfoSendStride(comm_info);
            clone_stride    = hypre_CommInfoSendStride(clone); /* Needs to be here, not below */
            comm_processes  = hypre_CommInfoSendProcesses(comm_info);
            comm_rboxnums   = hypre_CommInfoSendRBoxnums(comm_info);
            comm_rboxes     = hypre_CommInfoSendRBoxes(comm_info);
            comm_transforms = hypre_CommInfoSendTransforms(comm_info);
            break;

         case 1: /* Clone recv info */
            comm_boxes      = hypre_CommInfoRecvBoxes(comm_info);
            comm_stride     = hypre_CommInfoRecvStride(comm_info);
            clone_stride    = hypre_CommInfoRecvStride(clone); /* Needs to be here, not below */
            comm_processes  = hypre_CommInfoRecvProcesses(comm_info);
            comm_rboxnums   = hypre_CommInfoRecvRBoxnums(comm_info);
            comm_rboxes     = hypre_CommInfoRecvRBoxes(comm_info);
            comm_transforms = hypre_CommInfoRecvTransforms(comm_info);
            break;
      }

      size_aa = hypre_BoxArrayArraySize(comm_boxes);
      clone_boxes = hypre_BoxArrayArrayClone(comm_boxes);
      hypre_CopyIndex(comm_stride, clone_stride);
      {
         clone_processes = hypre_CTAlloc(HYPRE_Int *, size_aa, HYPRE_MEMORY_HOST);
         for (i = 0; i < size_aa; i++)
         {
            size_a = hypre_BoxArraySize(hypre_BoxArrayArrayBoxArray(comm_boxes, i));
            clone_processes[i] = hypre_CTAlloc(HYPRE_Int, size_a, HYPRE_MEMORY_HOST);
            for (j = 0; j < size_a; j++)
            {
               clone_processes[i][j] = comm_processes[i][j];
            }
         }
      }
      clone_rboxnums = NULL;
      if (comm_rboxnums != NULL)
      {
         clone_rboxnums = hypre_CTAlloc(HYPRE_Int *, size_aa, HYPRE_MEMORY_HOST);
         for (i = 0; i < size_aa; i++)
         {
            size_a = hypre_BoxArraySize(hypre_BoxArrayArrayBoxArray(comm_boxes, i));
            clone_rboxnums[i] = hypre_CTAlloc(HYPRE_Int, size_a, HYPRE_MEMORY_HOST);
            for (j = 0; j < size_a; j++)
            {
               clone_rboxnums[i][j] = comm_rboxnums[i][j];
            }
         }
      }
      clone_rboxes = hypre_BoxArrayArrayClone(comm_rboxes);
      clone_transforms = NULL;
      if (comm_transforms != NULL)
      {
         clone_transforms = hypre_CTAlloc(HYPRE_Int *, size_aa, HYPRE_MEMORY_HOST);
         for (i = 0; i < size_aa; i++)
         {
            size_a = hypre_BoxArraySize(hypre_BoxArrayArrayBoxArray(comm_boxes, i));
            clone_transforms[i] = hypre_CTAlloc(HYPRE_Int, size_a, HYPRE_MEMORY_HOST);
            for (j = 0; j < size_a; j++)
            {
               clone_transforms[i][j] = comm_transforms[i][j];
            }
         }
      }

      switch (k)
      {
         case 0: /* Clone send info */
            hypre_CommInfoSendBoxes(clone)      = clone_boxes;
            hypre_CommInfoSendProcesses(clone)  = clone_processes;
            hypre_CommInfoSendRBoxnums(clone)   = clone_rboxnums;
            hypre_CommInfoSendRBoxes(clone)     = clone_rboxes;
            hypre_CommInfoSendTransforms(clone) = clone_transforms;
            break;

         case 1: /* Clone recv info */
            hypre_CommInfoRecvBoxes(clone)      = clone_boxes;
            hypre_CommInfoRecvProcesses(clone)  = clone_processes;
            hypre_CommInfoRecvRBoxnums(clone)   = clone_rboxnums;
            hypre_CommInfoRecvRBoxes(clone)     = clone_rboxes;
            hypre_CommInfoRecvTransforms(clone) = clone_transforms;
            break;
      }
   }

   num_transforms = hypre_CommInfoNumTransforms(comm_info);
   comm_coords    = hypre_CommInfoCoords(comm_info);
   comm_dirs      = hypre_CommInfoDirs(comm_info);
   clone_coords = NULL;
   clone_dirs   = NULL;
   if (num_transforms > 0)
   {
      clone_coords = hypre_CTAlloc(hypre_Index, num_transforms, HYPRE_MEMORY_HOST);
      clone_dirs   = hypre_CTAlloc(hypre_Index, num_transforms, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_transforms; i++)
      {
         hypre_CopyIndex(comm_coords[i], clone_coords[i]);
         hypre_CopyIndex(comm_dirs[i], clone_dirs[i]);
      }
   }
   hypre_CommInfoNumTransforms(clone) = num_transforms;
   hypre_CommInfoCoords(clone)        = clone_coords;
   hypre_CommInfoDirs(clone)          = clone_dirs;

   hypre_CommInfoBoxesMatch(clone) = hypre_CommInfoBoxesMatch(comm_info);

   *clone_ptr = clone;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications patterns for a given grid-stencil
 * computation.  These patterns are defined by intersecting the data
 * dependencies of each grid box (including data dependencies within the box)
 * with its neighbor boxes.
 *
 * An inconsistent ordering of the boxes in the send/recv data regions is
 * returned.  That is, the ordering of the boxes on process p for receives from
 * process q is not guaranteed to be the same as the ordering of the boxes on
 * process q for sends to process p.
 *
 * The grid is defined by the triple (bgrid, origin, stride) where bgrid is a
 * base grid coarsened by origin and stride.  For now, the origin is assumed to
 * be the zero origin.
 *
 * The comm_info structure is set up in terms of the box numbering in bgrid,
 * even if a coarsened box is empty and does not participate in the computation.
 * The main reason for doing this is because we need to use the box manager to
 * get box number information for neighbors, and that is associated with bgrid.
 * The overhead in the rest of the code for looping over these empty boxes is
 * extremely small.
 *
 * The routine works on the index space for bgrid by using (origin, stride) to
 * adjust bgrid box growth and ensure that box extents line up with the grid.
 * The resulting comm_info is given on the index space for the grid.  Note that
 * the period of a periodic dimension must be evenly divisible by the stride in
 * that dimension, so it is okay to adjust boxes before shifting by the period.
 *
 * The routine uses a grow-the-box-and-intersect-with-neighbors style algorithm.
 *
 * 1. The basic algorithm:
 *
 * The basic algorithm is as follows, with one additional optimization discussed
 * below that helps to minimize the number of communications that are done with
 * neighbors (e.g., consider a 7-pt stencil and the difference between doing 26
 * communications versus 6):
 *
 * To compute send/recv regions, do
 *
 *   for i = local box
 *   {
 *      gbox_i = grow box i according to stencil
 *
 *      //find neighbors of i
 *      call BoxManIntersect on gbox_i (and periodic gbox_i)
 *
 *      // receives
 *      for j = neighbor box of i
 *      {
 *         intersect gbox_i with box j and add to recv region
 *      }
 *
 *      // sends
 *      for j = neighbor box of i
 *      {
 *         gbox_j = grow box j according to stencil
 *         intersect gbox_j with box i and add to send region
 *      }
 *   }
 *
 *   (Note: no ordering is assumed)
 *
 * 2. Optimization on basic algorithm:
 *
 * Before looping over the neighbors in the above algorithm, do a preliminary
 * sweep through the neighbors to select a subset of neighbors to do the
 * intersections with.  To select the subset, compute a "distance index" and
 * check the corresponding entry in the comm-stencil (described next) to decide
 * whether or not to use the box.
 *
 * The comm-stencil consists of a 3x3x3 array in 3D that is built from the
 * stencil as follows:
 *
 *   // assume for simplicity that i,j,k are -1, 0, or 1
 *   for each stencil entry (i,j,k)
 *   {
 *      mark all comm-stencil entries in (1,1,1) x (1+i,1+j,1+k)
 *      // here (1,1,1) is the "center" entry in the comm-stencil
 *   }
 *
 * 3. Complications with periodicity:
 *
 * When periodicity is on, it is possible to have a box-pair region (the
 * description of a communication pattern between two boxes) that consists of
 * more than one box.
 *
 * 4. Box Manager (added by AHB on 9/2006)
 *
 * The box manager is used to determine neighbors.  It is assumed that the
 * bgrid's box manager contains sufficient neighbor information.
 *
 * NOTES:
 *
 * A. No concept of data ownership is assumed.  As a result, redundant
 *    communication patterns can be produced when the grid boxes overlap.
 *
 * B. Boxes in the send and recv regions do not need to be in any particular
 *    order (including those that are periodic).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateCommInfo( hypre_StructGrid   *bgrid,
                      hypre_Index         stride,
                      hypre_CommStencil  *comm_stencil,
                      hypre_CommInfo    **comm_info_ptr )
{
   HYPRE_Int              ndim   = hypre_StructGridNDim(bgrid);
   hypre_Box             *csbox  = hypre_CommStencilBox(comm_stencil);
   HYPRE_Int             *csdata = hypre_CommStencilData(comm_stencil);
   HYPRE_Int             *mgrow  = hypre_CommStencilMGrow(comm_stencil);
   HYPRE_Int             *pgrow  = hypre_CommStencilPGrow(comm_stencil);
   hypre_Index            csindex;

   HYPRE_Int              i, j, k, d, m, s, si;

   hypre_BoxArrayArray   *send_boxes;
   hypre_BoxArrayArray   *recv_boxes;

   HYPRE_Int            **send_procs;
   HYPRE_Int            **recv_procs;
   HYPRE_Int            **send_rboxnums;
   HYPRE_Int            **recv_rboxnums;
   hypre_BoxArrayArray   *send_rboxes;
   hypre_BoxArrayArray   *recv_rboxes;

   hypre_BoxArray        *local_boxes;
   HYPRE_Int              num_boxes;

   hypre_BoxManager      *boxman;
   hypre_IndexRef         pshift;

   hypre_Box             *box;
   hypre_Box             *hood_box;
   hypre_Box             *grow_box;
   hypre_Box             *extend_box;
   hypre_Box             *int_box;
   hypre_Box             *periodic_box;

   hypre_BoxManEntry    **entries;
   hypre_BoxManEntry     *entry;

   HYPRE_Int              num_entries;
   hypre_BoxArray        *neighbor_boxes = NULL;
   HYPRE_Int             *neighbor_procs = NULL;
   HYPRE_Int             *neighbor_ids = NULL;
   HYPRE_Int             *neighbor_shifts = NULL;
   HYPRE_Int              neighbor_count;
   HYPRE_Int              neighbor_alloc;

   hypre_Index            ilower, iupper;

   hypre_BoxArray        *send_box_array;
   hypre_BoxArray        *recv_box_array;
   hypre_BoxArray        *send_rbox_array;
   hypre_BoxArray        *recv_rbox_array;

   hypre_Box            **cboxes;
   hypre_Box             *cboxes_mem;
   HYPRE_Int             *cboxes_neighbor_location;
   HYPRE_Int              num_cboxes, cbox_alloc;

   HYPRE_Int              num_periods, loc, box_id, id, proc_id, myid;

   MPI_Comm               comm;

   /*------------------------------------------------------
    * Initializations
    *------------------------------------------------------*/

   local_boxes = hypre_StructGridBoxes(bgrid);
   num_boxes   = hypre_BoxArraySize(local_boxes);
   num_periods = hypre_StructGridNumPeriods(bgrid);

   boxman = hypre_StructGridBoxMan(bgrid);
   comm   = hypre_StructGridComm(bgrid);

   hypre_MPI_Comm_rank(comm, &myid);

   /*------------------------------------------------------
    * Compute send/recv boxes and procs for each local box
    *------------------------------------------------------*/

   /* initialize: for each local box, we create an array of send/recv info */

   send_boxes = hypre_BoxArrayArrayCreate(num_boxes, ndim);
   recv_boxes = hypre_BoxArrayArrayCreate(num_boxes, ndim);
   send_procs = hypre_CTAlloc(HYPRE_Int *, num_boxes, HYPRE_MEMORY_HOST);
   recv_procs = hypre_CTAlloc(HYPRE_Int *, num_boxes, HYPRE_MEMORY_HOST);

   /* Remote boxnums and boxes describe data on the opposing processor, so some
      shifting of boxes is needed below for periodic neighbor boxes.  Remote box
      info is also needed for receives to allow for reverse communication. */
   send_rboxnums = hypre_CTAlloc(HYPRE_Int *, num_boxes, HYPRE_MEMORY_HOST);
   send_rboxes   = hypre_BoxArrayArrayCreate(num_boxes, ndim);
   recv_rboxnums = hypre_CTAlloc(HYPRE_Int *, num_boxes, HYPRE_MEMORY_HOST);
   recv_rboxes   = hypre_BoxArrayArrayCreate(num_boxes, ndim);

   grow_box = hypre_BoxCreate(hypre_StructGridNDim(bgrid));
   extend_box = hypre_BoxCreate(hypre_StructGridNDim(bgrid));
   int_box  = hypre_BoxCreate(hypre_StructGridNDim(bgrid));
   periodic_box =  hypre_BoxCreate(hypre_StructGridNDim(bgrid));

   /* storage we will use and keep track of the neighbors */
   neighbor_alloc = 30; /* initial guess at max size */
   neighbor_boxes = hypre_BoxArrayCreate(neighbor_alloc, ndim);
   neighbor_procs = hypre_CTAlloc(HYPRE_Int, neighbor_alloc, HYPRE_MEMORY_HOST);
   neighbor_ids = hypre_CTAlloc(HYPRE_Int, neighbor_alloc, HYPRE_MEMORY_HOST);
   neighbor_shifts = hypre_CTAlloc(HYPRE_Int, neighbor_alloc, HYPRE_MEMORY_HOST);

   /* storage we will use to collect all of the intersected boxes (the send and
      recv regions for box i (this may not be enough in the case of periodic
      boxes, so we will have to check) */
   cbox_alloc =  hypre_BoxManNEntries(boxman);

   cboxes_neighbor_location = hypre_CTAlloc(HYPRE_Int, cbox_alloc, HYPRE_MEMORY_HOST);
   cboxes = hypre_CTAlloc(hypre_Box *, cbox_alloc, HYPRE_MEMORY_HOST);
   cboxes_mem = hypre_CTAlloc(hypre_Box, cbox_alloc, HYPRE_MEMORY_HOST);

   /******* loop through each local box **************/

   for (i = 0; i < num_boxes; i++)
   {
      /* get the box */
      box = hypre_BoxArrayBox(local_boxes, i);
      box_id = i;

      /* grow_box - grow the local box according to the stencil */
      hypre_CopyBox(box, grow_box);
      hypre_ProjectBox(grow_box, NULL, stride);  /* ensure box extents line up with the grid */
      /* check for an empty grid box (coarsened or projected bgrid box) */
      if (hypre_BoxVolume(grow_box) == 0)
      {
         /* skip the rest of this loop - no communication needed for this box */
         continue;
      }
      for (d = 0; d < ndim; d++)
      {
         /* adjust growth by stride */
         hypre_BoxIMinD(grow_box, d) -= stride[d] * mgrow[d];
         hypre_BoxIMaxD(grow_box, d) += stride[d] * pgrow[d];
      }

      /* extend_box - to find the list of potential neighbors, we need to grow
         the local box a bit differently in case, for example, the stencil grows
         in one dimension [0] and not the other [1] */
      hypre_CopyBox(box, extend_box);
      hypre_ProjectBox(extend_box, NULL, stride);  /* ensure box extents line up with the grid */
      for (d = 0; d < ndim; d++)
      {
         /* adjust growth by stride */
         hypre_BoxIMinD(extend_box, d) -= stride[d] * hypre_max(mgrow[d], pgrow[d]);
         hypre_BoxIMaxD(extend_box, d) += stride[d] * hypre_max(mgrow[d], pgrow[d]);
      }

      /*------------------------------------------------
       * Determine the neighbors of box i
       *------------------------------------------------*/

      /* Do this by intersecting the extend box with the BoxManager, and also
         check for periodic neighbors */

      neighbor_count = 0;
      hypre_BoxArraySetSize(neighbor_boxes, 0);
      /* shift the box by each period (k=0 is original box) */
      for (k = 0; k < num_periods; k++)
      {
         hypre_CopyBox(extend_box, periodic_box);
         pshift = hypre_StructGridPShift(bgrid, k);
         hypre_BoxShiftPos(periodic_box, pshift);

         /* get the intersections */
         hypre_BoxManIntersect(boxman, hypre_BoxIMin(periodic_box), hypre_BoxIMax(periodic_box),
                               &entries, &num_entries);

         /* note: do we need to remove the intersection with our original box?
            no if periodic, yes if non-periodic (k=0) */

         /* unpack entries (first check storage) */
         if (neighbor_count + num_entries > neighbor_alloc)
         {
            neighbor_alloc = neighbor_count + num_entries + 5;
            neighbor_procs =
               hypre_TReAlloc(neighbor_procs, HYPRE_Int, neighbor_alloc, HYPRE_MEMORY_HOST);
            neighbor_ids =
               hypre_TReAlloc(neighbor_ids, HYPRE_Int, neighbor_alloc, HYPRE_MEMORY_HOST);
            neighbor_shifts =
               hypre_TReAlloc(neighbor_shifts, HYPRE_Int, neighbor_alloc, HYPRE_MEMORY_HOST);
         }
         /* check storage for the array */
         hypre_BoxArraySetSize(neighbor_boxes, neighbor_count + num_entries);
         /* now unpack */
         for (j = 0; j < num_entries; j++)
         {
            entry = entries[j];
            proc_id = hypre_BoxManEntryProc(entry);
            id = hypre_BoxManEntryId(entry);
            /* don't keep box i in the non-periodic case*/
            if (!k)
            {
               if ((myid == proc_id) && (box_id == id))
               {
                  continue;
               }
            }

            hypre_BoxManEntryGetExtents(entry, ilower, iupper);
            hood_box = hypre_BoxArrayBox(neighbor_boxes, neighbor_count);
            hypre_BoxSetExtents(hood_box, ilower, iupper);
            /* shift the periodic boxes (needs to be the opposite of above) */
            if (k)
            {
               hypre_BoxShiftNeg(hood_box, pshift);
            }

            hypre_ProjectBox(hood_box, NULL, stride);  /* ensure box extents line up with the grid */
            /* check for an empty hood_box */
            if (hypre_BoxVolume(hood_box) == 0)
            {
               /* don't keep this hood_box - no communication needed with it */
               continue;
            }

            neighbor_procs[neighbor_count] = proc_id;
            neighbor_ids[neighbor_count] = id;
            neighbor_shifts[neighbor_count] = k;
            neighbor_count++;
         }
         hypre_BoxArraySetSize(neighbor_boxes, neighbor_count);

         hypre_TFree(entries, HYPRE_MEMORY_HOST);

      } /* end of loop through periods k */

      /* Now we have a list of all of the neighbors for box i! */

      /* note: we don't want/need to remove duplicates - they should have
         different intersections (TO DO: put more thought into if there are ever
         any exceptions to this? - the intersection routine already eliminates
         duplicates - so what i mean is eliminating duplicates from multiple
         intersection calls in periodic case)  */

      /*------------------------------------------------
       * Compute recv_box_array for box i
       *------------------------------------------------*/

      /* check size of storage for cboxes */
      /* let's make sure that we have enough storage in case each neighbor
         produces a send/recv region */
      if (neighbor_count > cbox_alloc)
      {
         cbox_alloc = neighbor_count;
         cboxes_neighbor_location = hypre_TReAlloc(cboxes_neighbor_location,
                                                   HYPRE_Int, cbox_alloc, HYPRE_MEMORY_HOST);
         cboxes = hypre_TReAlloc(cboxes, hypre_Box *, cbox_alloc, HYPRE_MEMORY_HOST);
         cboxes_mem = hypre_TReAlloc(cboxes_mem, hypre_Box, cbox_alloc, HYPRE_MEMORY_HOST);
      }

      /* Loop through each neighbor box.  If the neighbor box intersects the
         grown box i (grown according to our stencil), then the intersection is
         a recv region.  If the neighbor box was shifted to handle periodicity,
         we need to (positive) shift it back. */

      num_cboxes = 0;

      for (k = 0; k < neighbor_count; k++)
      {
         hood_box = hypre_BoxArrayBox(neighbor_boxes, k);
         /* check the comm stencil to see if it makes sense to intersect */
         for (d = 0; d < ndim; d++)
         {
            csindex[d] = 1;

            s = hypre_BoxIMinD(hood_box, d) - hypre_BoxIMaxD(box, d);
            if (s > 0)
            {
               csindex[d] = 2;
            }
            s = hypre_BoxIMinD(box, d) - hypre_BoxIMaxD(hood_box, d);
            if (s > 0)
            {
               csindex[d] = 0;
            }
         }
         /* it makes sense only if we have at least one non-zero entry */
         si = hypre_BoxIndexRank(csbox, csindex);
         if (csdata[si])
         {
            /* intersect - result is int_box - don't need to project hood_box */
            hypre_IntersectBoxes(grow_box, hood_box, int_box);
            /* if we have a positive volume box, this is a recv region */
            if (hypre_BoxVolume(int_box))
            {
               /* keep track of which neighbor: k... */
               cboxes_neighbor_location[num_cboxes] = k;
               cboxes[num_cboxes] = &cboxes_mem[num_cboxes];
               /* keep the intersected box */
               hypre_CopyBox(int_box, cboxes[num_cboxes]);
               num_cboxes++;
            }
         }
      } /* end of loop through each neighbor */

      /* create recv_box_array and recv_procs for box i - coarsen comm boxes here */
      recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      hypre_BoxArraySetSize(recv_box_array, num_cboxes);
      recv_procs[i] = hypre_CTAlloc(HYPRE_Int, num_cboxes, HYPRE_MEMORY_HOST);
      recv_rboxnums[i] = hypre_CTAlloc(HYPRE_Int, num_cboxes, HYPRE_MEMORY_HOST);
      recv_rbox_array = hypre_BoxArrayArrayBoxArray(recv_rboxes, i);
      hypre_BoxArraySetSize(recv_rbox_array, num_cboxes);

      for (m = 0; m < num_cboxes; m++)
      {
         loc = cboxes_neighbor_location[m];
         recv_procs[i][m] = neighbor_procs[loc];
         recv_rboxnums[i][m] = neighbor_ids[loc];
         hypre_CopyBox(cboxes[m], hypre_BoxArrayBox(recv_box_array, m));
         hypre_CoarsenBox(hypre_BoxArrayBox(recv_box_array, m), NULL, stride);

         /* if periodic, positive shift before copying to the rbox_array */
         if (neighbor_shifts[loc]) /* periodic if shift != 0 */
         {
            pshift = hypre_StructGridPShift(bgrid, neighbor_shifts[loc]);
            hypre_BoxShiftPos(cboxes[m], pshift);
         }
         hypre_CopyBox(cboxes[m], hypre_BoxArrayBox(recv_rbox_array, m));
         hypre_CoarsenBox(hypre_BoxArrayBox(recv_rbox_array, m), NULL, stride);

         cboxes[m] = NULL;
      }

      /*------------------------------------------------
       * Compute send_box_array for box i
       *------------------------------------------------*/

      /* Loop through each neighbor box.  If the grown neighbor box intersects
         box i, then the intersection is a send region.  If the neighbor box was
         shifted to handle periodicity, we need to (positive) shift it back. */

      num_cboxes = 0;

      for (k = 0; k < neighbor_count; k++)
      {
         hood_box = hypre_BoxArrayBox(neighbor_boxes, k);
         /* check the comm stencil to see if it makes sense to intersect */
         for (d = 0; d < ndim; d++)
         {
            csindex[d] = 1;

            s = hypre_BoxIMinD(box, d) - hypre_BoxIMaxD(hood_box, d);
            if (s > 0)
            {
               csindex[d] = 2;
            }
            s = hypre_BoxIMinD(hood_box, d) - hypre_BoxIMaxD(box, d);
            if (s > 0)
            {
               csindex[d] = 0;
            }
         }
         /* it makes sense only if we have at least one non-zero entry */
         si = hypre_BoxIndexRank(csbox, csindex);
         if (csdata[si])
         {
            /* grow the neighbor box and intersect */
            hypre_CopyBox(hood_box, grow_box);
            hypre_ProjectBox(grow_box, NULL, stride);  /* ensure box extents line up with the grid */
            for (d = 0; d < ndim; d++)
            {
               /* adjust growth by stride */
               hypre_BoxIMinD(grow_box, d) -= stride[d] * mgrow[d];
               hypre_BoxIMaxD(grow_box, d) += stride[d] * pgrow[d];
            }
            /* intersect - result is int_box - don't need to project box */
            hypre_IntersectBoxes(box, grow_box, int_box);
            /* if we have a positive volume box, this is a send region */
            if (hypre_BoxVolume(int_box))
            {
               /* keep track of which neighbor: k... */
               cboxes_neighbor_location[num_cboxes] = k;
               cboxes[num_cboxes] = &cboxes_mem[num_cboxes];
               /* keep the intersected box */
               hypre_CopyBox(int_box, cboxes[num_cboxes]);
               num_cboxes++;
            }
         }
      }/* end of loop through neighbors */

      /* create send_box_array and send_procs for box i - coarsen comm boxes here */
      send_box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      hypre_BoxArraySetSize(send_box_array, num_cboxes);
      send_procs[i] = hypre_CTAlloc(HYPRE_Int, num_cboxes, HYPRE_MEMORY_HOST);
      send_rboxnums[i] = hypre_CTAlloc(HYPRE_Int, num_cboxes, HYPRE_MEMORY_HOST);
      send_rbox_array = hypre_BoxArrayArrayBoxArray(send_rboxes, i);
      hypre_BoxArraySetSize(send_rbox_array, num_cboxes);

      for (m = 0; m < num_cboxes; m++)
      {
         loc = cboxes_neighbor_location[m];
         send_procs[i][m] = neighbor_procs[loc];
         send_rboxnums[i][m] = neighbor_ids[loc];
         hypre_CopyBox(cboxes[m], hypre_BoxArrayBox(send_box_array, m));
         hypre_CoarsenBox(hypre_BoxArrayBox(send_box_array, m), NULL, stride);

         /* if periodic, positive shift before copying to the rbox_array */
         if (neighbor_shifts[loc]) /* periodic if shift != 0 */
         {
            pshift = hypre_StructGridPShift(bgrid, neighbor_shifts[loc]);
            hypre_BoxShiftPos(cboxes[m], pshift);
         }
         hypre_CopyBox(cboxes[m], hypre_BoxArrayBox(send_rbox_array, m));
         hypre_CoarsenBox(hypre_BoxArrayBox(send_rbox_array, m), NULL, stride);

         cboxes[m] = NULL;
      }
   } /* end of loop through each local box */

   /* clean up */
   hypre_TFree(neighbor_procs, HYPRE_MEMORY_HOST);
   hypre_TFree(neighbor_ids, HYPRE_MEMORY_HOST);
   hypre_TFree(neighbor_shifts, HYPRE_MEMORY_HOST);
   hypre_BoxArrayDestroy(neighbor_boxes);

   hypre_TFree(cboxes, HYPRE_MEMORY_HOST);
   hypre_TFree(cboxes_mem, HYPRE_MEMORY_HOST);
   hypre_TFree(cboxes_neighbor_location, HYPRE_MEMORY_HOST);

   hypre_BoxDestroy(grow_box);
   hypre_BoxDestroy(int_box);
   hypre_BoxDestroy(periodic_box);
   hypre_BoxDestroy(extend_box);

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   hypre_CommInfoCreate(send_boxes, recv_boxes, send_procs, recv_procs,
                        send_rboxnums, recv_rboxnums, send_rboxes, recv_rboxes,
                        1, comm_info_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return communication-pattern descriptions for a grid-stencil computation.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateCommInfoFromStencil( hypre_StructGrid      *grid,
                                 hypre_Index            stride,
                                 hypre_StructStencil   *stencil,
                                 hypre_CommInfo       **comm_info_ptr )
{
   HYPRE_Int           ndim          = hypre_StructGridNDim(grid);
   hypre_Index        *stencil_shape = hypre_StructStencilShape(stencil);
   hypre_IndexRef      stencil_offset;
   hypre_CommStencil  *comm_stencil;
   HYPRE_Int           s;

   /* Set up the comm-stencil */
   comm_stencil = hypre_CommStencilCreate(ndim);
   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
   {
      stencil_offset = stencil_shape[s];
      hypre_CommStencilSetEntry(comm_stencil, stencil_offset);
   }

   hypre_CreateCommInfo(grid, stride, comm_stencil, comm_info_ptr );
   hypre_CommStencilDestroy(comm_stencil);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return communication-pattern descriptions for a given grid based on a
 * specified number of "ghost zones".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateCommInfoFromNumGhost( hypre_StructGrid      *grid,
                                  hypre_Index            stride,
                                  HYPRE_Int             *num_ghost,
                                  hypre_CommInfo       **comm_info_ptr )
{
   HYPRE_Int           ndim = hypre_StructGridNDim(grid);
   hypre_CommStencil  *comm_stencil;
   hypre_Box          *csbox;
   HYPRE_Int          *csdata;
   HYPRE_Int          *mgrow;
   HYPRE_Int          *pgrow;
   HYPRE_Int           d, ii;

   /* Set up the comm-stencil */
   comm_stencil = hypre_CommStencilCreate(ndim);
   csbox  = hypre_CommStencilBox(comm_stencil);
   csdata = hypre_CommStencilData(comm_stencil);
   mgrow  = hypre_CommStencilMGrow(comm_stencil);
   pgrow  = hypre_CommStencilPGrow(comm_stencil);
   for (ii = 0; ii < hypre_BoxVolume(csbox); ii++)
   {
      csdata[ii] = 1;
   }
   for (d = 0; d < ndim; d++)
   {
      mgrow[d] = num_ghost[2 * d];
      pgrow[d] = num_ghost[2 * d + 1];
   }

   hypre_CreateCommInfo(grid, stride, comm_stencil, comm_info_ptr);
   hypre_CommStencilDestroy(comm_stencil);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications patterns for migrating data
 * from one grid distribution to another.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateCommInfoFromGrids( hypre_StructGrid      *from_grid,
                               hypre_StructGrid      *to_grid,
                               hypre_CommInfo       **comm_info_ptr )
{
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   HYPRE_Int              **send_procs;
   HYPRE_Int              **recv_procs;
   HYPRE_Int              **send_rboxnums;
   HYPRE_Int              **recv_rboxnums;
   hypre_BoxArrayArray     *send_rboxes;
   hypre_BoxArrayArray     *recv_rboxes;

   hypre_BoxArrayArray     *comm_boxes;
   HYPRE_Int              **comm_procs;
   HYPRE_Int              **comm_boxnums;
   hypre_BoxArray          *comm_box_array;
   hypre_Box               *comm_box;

   hypre_StructGrid        *local_grid;
   hypre_StructGrid        *remote_grid;

   hypre_BoxArray          *local_boxes;
   hypre_BoxArray          *remote_boxes;
   hypre_BoxArray          *remote_all_boxes;
   HYPRE_Int               *remote_all_procs;
   HYPRE_Int               *remote_all_boxnums;
   HYPRE_Int                remote_first_local;

   hypre_Box               *local_box;
   hypre_Box               *remote_box;

   HYPRE_Int                i, j, k, r, ndim;

   /*------------------------------------------------------
    * Set up communication info
    *------------------------------------------------------*/

   ndim = hypre_StructGridNDim(from_grid);

   for (r = 0; r < 2; r++)
   {
      switch (r)
      {
         case 0:
            local_grid  = from_grid;
            remote_grid = to_grid;
            break;

         case 1:
            local_grid  = to_grid;
            remote_grid = from_grid;
            break;
      }

      /*---------------------------------------------------
       * Compute comm_boxes and comm_procs
       *---------------------------------------------------*/

      local_boxes  = hypre_StructGridBoxes(local_grid);
      remote_boxes = hypre_StructGridBoxes(remote_grid);
      hypre_GatherAllBoxes(hypre_StructGridComm(remote_grid), remote_boxes, ndim,
                           &remote_all_boxes,
                           &remote_all_procs,
                           &remote_first_local);
      hypre_ComputeBoxnums(remote_all_boxes, remote_all_procs,
                           &remote_all_boxnums);

      comm_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(local_boxes), ndim);
      comm_procs = hypre_CTAlloc(HYPRE_Int *, hypre_BoxArraySize(local_boxes), HYPRE_MEMORY_HOST);
      comm_boxnums = hypre_CTAlloc(HYPRE_Int *, hypre_BoxArraySize(local_boxes),
                                   HYPRE_MEMORY_HOST);

      comm_box = hypre_BoxCreate(ndim);
      hypre_ForBoxI(i, local_boxes)
      {
         local_box = hypre_BoxArrayBox(local_boxes, i);

         comm_box_array = hypre_BoxArrayArrayBoxArray(comm_boxes, i);
         comm_procs[i] =
            hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(remote_all_boxes), HYPRE_MEMORY_HOST);
         comm_boxnums[i] =
            hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(remote_all_boxes), HYPRE_MEMORY_HOST);

         hypre_ForBoxI(j, remote_all_boxes)
         {
            remote_box = hypre_BoxArrayBox(remote_all_boxes, j);

            hypre_IntersectBoxes(local_box, remote_box, comm_box);
            if (hypre_BoxVolume(comm_box))
            {
               k = hypre_BoxArraySize(comm_box_array);
               comm_procs[i][k] = remote_all_procs[j];
               comm_boxnums[i][k] = remote_all_boxnums[j];

               hypre_AppendBox(comm_box, comm_box_array);
            }
         }

         comm_procs[i] =
            hypre_TReAlloc(comm_procs[i],
                           HYPRE_Int, hypre_BoxArraySize(comm_box_array), HYPRE_MEMORY_HOST);
         comm_boxnums[i] =
            hypre_TReAlloc(comm_boxnums[i],
                           HYPRE_Int, hypre_BoxArraySize(comm_box_array), HYPRE_MEMORY_HOST);
      }
      hypre_BoxDestroy(comm_box);

      hypre_BoxArrayDestroy(remote_all_boxes);
      hypre_TFree(remote_all_procs, HYPRE_MEMORY_HOST);
      hypre_TFree(remote_all_boxnums, HYPRE_MEMORY_HOST);

      switch (r)
      {
         case 0:
            send_boxes = comm_boxes;
            send_procs = comm_procs;
            send_rboxnums = comm_boxnums;
            send_rboxes = hypre_BoxArrayArrayClone(comm_boxes);
            break;

         case 1:
            recv_boxes = comm_boxes;
            recv_procs = comm_procs;
            recv_rboxnums = comm_boxnums;
            recv_rboxes = hypre_BoxArrayArrayClone(comm_boxes);
            break;
      }
   }

   hypre_CommInfoCreate(send_boxes, recv_boxes, send_procs, recv_procs,
                        send_rboxnums, recv_rboxnums, send_rboxes, recv_rboxes,
                        1, comm_info_ptr);

   return hypre_error_flag;
}
