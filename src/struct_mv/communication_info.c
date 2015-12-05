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

#include "headers.h"

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

   comm_info = hypre_TAlloc(hypre_CommInfo, 1);

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
   hypre_SetIndex(hypre_CommInfoSendStride(comm_info), 1, 1, 1);
   hypre_SetIndex(hypre_CommInfoRecvStride(comm_info), 1, 1, 1);

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
   hypre_ProjectBoxArrayArray(hypre_CommInfoSendBoxes(comm_info),
                              index, stride);
   hypre_ProjectBoxArrayArray(hypre_CommInfoSendRBoxes(comm_info),
                              index, stride);
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
   hypre_ProjectBoxArrayArray(hypre_CommInfoRecvBoxes(comm_info),
                              index, stride);
   hypre_CopyIndex(stride, hypre_CommInfoRecvStride(comm_info));

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

   size = hypre_BoxArrayArraySize(hypre_CommInfoSendBoxes(comm_info));
   hypre_BoxArrayArrayDestroy(hypre_CommInfoSendBoxes(comm_info));
   processes = hypre_CommInfoSendProcesses(comm_info);
   for (i = 0; i < size; i++)
   {
      hypre_TFree(processes[i]);
   }
   hypre_TFree(processes);
   rboxnums = hypre_CommInfoSendRBoxnums(comm_info);
   if (rboxnums != NULL)
   {
      for (i = 0; i < size; i++)
      {
         hypre_TFree(rboxnums[i]);
      }
      hypre_TFree(rboxnums);
   }
   hypre_BoxArrayArrayDestroy(hypre_CommInfoSendRBoxes(comm_info));
   transforms = hypre_CommInfoSendTransforms(comm_info);
   if (transforms != NULL)
   {
      for (i = 0; i < size; i++)
      {
         hypre_TFree(transforms[i]);
      }
      hypre_TFree(transforms);
   }

   size = hypre_BoxArrayArraySize(hypre_CommInfoRecvBoxes(comm_info));
   hypre_BoxArrayArrayDestroy(hypre_CommInfoRecvBoxes(comm_info));
   processes = hypre_CommInfoRecvProcesses(comm_info);
   for (i = 0; i < size; i++)
   {
      hypre_TFree(processes[i]);
   }
   hypre_TFree(processes);
   rboxnums = hypre_CommInfoRecvRBoxnums(comm_info);
   if (rboxnums != NULL)
   {
      for (i = 0; i < size; i++)
      {
         hypre_TFree(rboxnums[i]);
      }
      hypre_TFree(rboxnums);
   }
   hypre_BoxArrayArrayDestroy(hypre_CommInfoRecvRBoxes(comm_info));
   transforms = hypre_CommInfoRecvTransforms(comm_info);
   if (transforms != NULL)
   {
      for (i = 0; i < size; i++)
      {
         hypre_TFree(transforms[i]);
      }
      hypre_TFree(transforms);
   }

   hypre_TFree(hypre_CommInfoCoords(comm_info));
   hypre_TFree(hypre_CommInfoDirs(comm_info));

   hypre_TFree(comm_info);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NEW version that uses the box manager to find neighbors boxes. 
 * AHB 9/06
 *
 * Return descriptions of communications patterns for a given
 * grid-stencil computation.  These patterns are defined by
 * intersecting the data dependencies of each box (including data
 * dependencies within the box) with its neighbor boxes.
 *
 * An inconsistent ordering of the boxes in the send/recv data regions
 * is returned.  That is, the ordering of the boxes on process p for
 * receives from process q is not guaranteed to be the same as the
 * ordering of the boxes on process q for sends to process p.
 *
 * The routine uses a grow-the-box-and-intersect-with-neighbors style
 * algorithm.
 *
 * 1. The basic algorithm:
 *
 * The basic algorithm is as follows, with one additional optimization
 * discussed below that helps to minimize the number of communications
 * that are done with neighbors (e.g., consider a 7-pt stencil and the
 * difference between doing 26 communications versus 6):
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
 * Before looping over the neighbors in the above algorithm, do a
 * preliminary sweep through the neighbors to select a subset of
 * neighbors to do the intersections with.  To select the subset,
 * compute a so-called "distance index" and check the corresponding
 * entry in the so-called "stencil grid" to decide whether or not to
 * use the box.
 * 
 * The "stencil grid" is a 3x3x3 grid in 3D that is built from the
 * stencil as follows:
 * 
 *   // assume for simplicity that i,j,k are -1, 0, or 1
 *   for each stencil entry (i,j,k)
 *   {
 *      mark all stencil grid entries in (1,1,1) x (1+i,1+j,1+k)
 *      // here (1,1,1) is the "center" entry in the stencil grid
 *   }
 *
 *
 * 3. Complications with periodicity:
 * 
 * When periodicity is on, it is possible to have a box-pair region
 * (the description of a communication pattern between two boxes) that
 * consists of more than one box.
 * 
 * 4.  Box Manager
 *
 *   The box manager is used to determine neighbors.  It is assumed 
 *   that the grid's box manager contains sufficient neighbor 
 *   information.
 *
 * NOTES: 
 *
 *    A. No concept of data ownership is assumed.  As a result,
 *       redundant communication patterns can be produced when the grid
 *       boxes overlap.
 *
 *    B. Boxes in the send and recv regions do not need to be in any
 *       particular order (including those that are periodic).  
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateCommInfoFromStencil( hypre_StructGrid      *grid,
                                 hypre_StructStencil   *stencil,
                                 hypre_CommInfo       **comm_info_ptr )
{
   HYPRE_Int              i,j,k, d, m, s;   

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

   HYPRE_Int             *local_ids;

   hypre_BoxManager      *boxman;
                       
   hypre_Index           *stencil_shape;
   hypre_IndexRef         stencil_offset;
   hypre_IndexRef         pshift;
                          
   hypre_Box             *box;
   hypre_Box             *hood_box;
   hypre_Box             *grow_box;
   hypre_Box             *extend_box;
   hypre_Box             *int_box;
   hypre_Box             *periodic_box;
   
   HYPRE_Int              stencil_grid[3][3][3];
   HYPRE_Int              grow[3][2];
                       
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
                       
   HYPRE_Int              istart[3], istop[3];
   HYPRE_Int              sgindex[3];               

   HYPRE_Int              num_periods, loc, box_id, id, proc_id;
   HYPRE_Int              myid;
   
   MPI_Comm               comm;

   /*------------------------------------------------------
    * Initializations
    *------------------------------------------------------*/

   local_boxes  = hypre_StructGridBoxes(grid);
   local_ids    = hypre_StructGridIDs(grid);
   num_boxes    = hypre_BoxArraySize(local_boxes);
   num_periods  = hypre_StructGridNumPeriods(grid);
   
   boxman    = hypre_StructGridBoxMan(grid);
   comm      =  hypre_StructGridComm(grid);
   
   hypre_MPI_Comm_rank(comm, &myid);
  
   for (k = 0; k < 3; k++)
   {
      for (j = 0; j < 3; j++)
      {
         for (i = 0; i < 3; i++)
         {
            stencil_grid[i][j][k] = 0;
         }
      }
   }

   /*------------------------------------------------------
    * Compute the "grow" information from the stencil
    *------------------------------------------------------*/

   stencil_shape = hypre_StructStencilShape(stencil);

   for (d = 0; d < 3; d++)
   {
      grow[d][0] = 0;
      grow[d][1] = 0;
   }

   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
   {
      stencil_offset = stencil_shape[s];

      for (d = 0; d < 3; d++)
      {
         m = stencil_offset[d];

         istart[d] = 1;
         istop[d]  = 1;

         if (m < 0)
         {
            istart[d] = 0;
            grow[d][0] = hypre_max(grow[d][0], -m);
         }
         else if (m > 0)
         {
            istop[d] = 2;
            grow[d][1] = hypre_max(grow[d][1],  m);
         }
      }

      /* update stencil grid from the grow_stencil */
      for (k = istart[2]; k <= istop[2]; k++)
      {
         for (j = istart[1]; j <= istop[1]; j++)
         {
            for (i = istart[0]; i <= istop[0]; i++)
            {
               stencil_grid[i][j][k] = 1;
            }
         }
      }
   }

   /*------------------------------------------------------
    * Compute send/recv boxes and procs for each local box
    *------------------------------------------------------*/

   /* initialize: for each local box, we create an array of send/recv info */

   send_boxes = hypre_BoxArrayArrayCreate(num_boxes);
   recv_boxes = hypre_BoxArrayArrayCreate(num_boxes);
   send_procs = hypre_CTAlloc(HYPRE_Int *, num_boxes);
   recv_procs = hypre_CTAlloc(HYPRE_Int *, num_boxes);

   /* Remote boxnums and boxes describe data on the opposing processor, so some
      shifting of boxes is needed below for periodic neighbor boxes.  Remote box
      info is also needed for receives to allow for reverse communication. */
   send_rboxnums = hypre_CTAlloc(HYPRE_Int *, num_boxes);
   send_rboxes   = hypre_BoxArrayArrayCreate(num_boxes);
   recv_rboxnums = hypre_CTAlloc(HYPRE_Int *, num_boxes);
   recv_rboxes   = hypre_BoxArrayArrayCreate(num_boxes);

   grow_box = hypre_BoxCreate();
   extend_box = hypre_BoxCreate();
   int_box  = hypre_BoxCreate();
   periodic_box =  hypre_BoxCreate();
 
   /* storage we will use and keep track of the neighbors */
   neighbor_alloc = 30; /* initial guess at max size */
   neighbor_boxes = hypre_BoxArrayCreate(neighbor_alloc);
   neighbor_procs = hypre_CTAlloc(HYPRE_Int, neighbor_alloc);
   neighbor_ids = hypre_CTAlloc(HYPRE_Int, neighbor_alloc);
   neighbor_shifts = hypre_CTAlloc(HYPRE_Int, neighbor_alloc);

   /* storage we will use to collect all of the intersected boxes (the send and
      recv regions for box i (this may not be enough in the case of periodic
      boxes, so we will have to check) */
   cbox_alloc =  hypre_BoxManNEntries(boxman);

   cboxes_neighbor_location = hypre_CTAlloc(HYPRE_Int, cbox_alloc);
   cboxes = hypre_CTAlloc(hypre_Box *, cbox_alloc);
   cboxes_mem = hypre_CTAlloc(hypre_Box, cbox_alloc);

   /******* loop through each local box **************/

   for (i = 0; i < num_boxes; i++)
   {
      /* get the box */
      box = hypre_BoxArrayBox(local_boxes, i);
      /* box_id = local_ids[i]; the box id in the Box Manager is the box number,
       * and we use this to find out if a box has intersected with itself */
      box_id = i;
      
      /* grow box local i according to the stencil*/
      hypre_CopyBox(box, grow_box);
      for (d = 0; d < 3; d++)
      {
         hypre_BoxIMinD(grow_box, d) -= grow[d][0];
         hypre_BoxIMaxD(grow_box, d) += grow[d][1];
      }

      /* extend_box - to find the list of potential neighbors, we need to grow
         the local box a bit differently in case, for example, the stencil grows
         in one dimension [0] and not the other [1] */
      hypre_CopyBox(box, extend_box);
      for (d = 0; d < 3; d++)
      { 
         hypre_BoxIMinD(extend_box, d) -= hypre_max(grow[d][0],grow[d][1]);
         hypre_BoxIMaxD(extend_box, d) += hypre_max(grow[d][0],grow[d][1]);
      }

      /*------------------------------------------------
       * Determine the neighbors of box i
       *------------------------------------------------*/
     
      /* Do this by intersecting the extend box with the BoxManager. 
         We must also check for periodic neighbors. */

      neighbor_count = 0;
      hypre_BoxArraySetSize(neighbor_boxes, 0);
      /* shift the box by each period (k=0 is original box) */
      for (k = 0; k < num_periods; k++)
      {
         hypre_CopyBox(extend_box, periodic_box);
         pshift = hypre_StructGridPShift(grid, k);
         hypre_BoxShiftPos(periodic_box, pshift);
         
         /* get the intersections */
         hypre_BoxManIntersect(boxman, hypre_BoxIMin(periodic_box) , 
                               hypre_BoxIMax(periodic_box) , 
                               &entries , &num_entries);
      
         /* note: do we need to remove the intersection with our original box?
            no if periodic, yes if non-periodic (k=0) */ 

         /* unpack entries (first check storage) */
         if (neighbor_count + num_entries > neighbor_alloc)
         {
            neighbor_alloc = neighbor_count + num_entries + 5;
            neighbor_procs = hypre_TReAlloc(neighbor_procs, HYPRE_Int,
                                            neighbor_alloc);
            neighbor_ids = hypre_TReAlloc(neighbor_ids, HYPRE_Int, neighbor_alloc);
            neighbor_shifts = hypre_TReAlloc(neighbor_shifts, HYPRE_Int,
                                             neighbor_alloc);
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
               if((myid == proc_id) && (box_id == id))
               {
                  continue;
               }
            }

            hypre_BoxManEntryGetExtents(entry, ilower, iupper);        
            hypre_BoxSetExtents(hypre_BoxArrayBox(neighbor_boxes, neighbor_count),
                                ilower, iupper);
            /* shift the periodic boxes (needs to be the opposite of above) */
            if (k)
            {
               hypre_BoxShiftNeg(
                  hypre_BoxArrayBox(neighbor_boxes, neighbor_count), pshift);
            }
            
            neighbor_procs[neighbor_count] = proc_id;
            neighbor_ids[neighbor_count] = id;
            neighbor_shifts[neighbor_count] = k;
            neighbor_count++;
         }
         hypre_BoxArraySetSize(neighbor_boxes, neighbor_count);

         hypre_TFree(entries);
  
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
                                                   HYPRE_Int, cbox_alloc);
         cboxes = hypre_TReAlloc(cboxes, hypre_Box *, cbox_alloc);
         cboxes_mem = hypre_TReAlloc(cboxes_mem, hypre_Box, cbox_alloc);
      }

      /* Loop through each neighbor box.  If the neighbor box intersects the
         grown box i (grown according to our stencil), then the intersection is
         a recv region.  If the neighbor box was shifted to handle periodicity,
         we need to (positive) shift it back. */

      num_cboxes = 0;
      
      for (k = 0; k < neighbor_count; k++)
      {
         hood_box = hypre_BoxArrayBox(neighbor_boxes, k);
         /* check the stencil grid to see if it makes sense to intersect */
         for (d = 0; d < 3; d++)
         {
            sgindex[d] = 1;
               
            s = hypre_BoxIMinD(hood_box, d) - hypre_BoxIMaxD(box, d);
            if (s > 0)
            {
               sgindex[d] = 2;
            }
            s = hypre_BoxIMinD(box, d) - hypre_BoxIMaxD(hood_box, d);
            if (s > 0)
            {
               sgindex[d] = 0;
            }
         }
         /* it makes sense only if we have at least one non-zero entry */   
         if (stencil_grid[sgindex[0]][sgindex[1]][sgindex[2]])
         {
            /* intersect - result is int_box */
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

      /* create recv_box_array and recv_procs for box i */
      recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      hypre_BoxArraySetSize(recv_box_array, num_cboxes);
      recv_procs[i] = hypre_CTAlloc(HYPRE_Int, num_cboxes);
      recv_rboxnums[i] = hypre_CTAlloc(HYPRE_Int, num_cboxes);
      recv_rbox_array = hypre_BoxArrayArrayBoxArray(recv_rboxes, i);
      hypre_BoxArraySetSize(recv_rbox_array, num_cboxes);

      for (m = 0; m < num_cboxes; m++)
      {
         loc = cboxes_neighbor_location[m];
         recv_procs[i][m] = neighbor_procs[loc];
         recv_rboxnums[i][m] = neighbor_ids[loc];
         hypre_CopyBox(cboxes[m], hypre_BoxArrayBox(recv_box_array, m));

         /* if periodic, positive shift before copying to the rbox_array */
         if (neighbor_shifts[loc]) /* periodic if shift != 0 */
         {
            pshift = hypre_StructGridPShift(grid, neighbor_shifts[loc]);
            hypre_BoxShiftPos(cboxes[m], pshift);
         }
         hypre_CopyBox(cboxes[m], hypre_BoxArrayBox(recv_rbox_array, m));

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
         /* check the stencil grid to see if it makes sense to intersect */
         for (d = 0; d < 3; d++)
         {
            sgindex[d] = 1;
            
            s = hypre_BoxIMinD(box, d) - hypre_BoxIMaxD(hood_box, d);
            if (s > 0)
            {
               sgindex[d] = 2;
            }
            s = hypre_BoxIMinD(hood_box, d) - hypre_BoxIMaxD(box, d);
            if (s > 0)
            {
               sgindex[d] = 0;
            }
         }
         /* it makes sense only if we have at least one non-zero entry */   
         if (stencil_grid[sgindex[0]][sgindex[1]][sgindex[2]])
         {
            /* grow the neighbor box and intersect */
            hypre_CopyBox(hood_box, grow_box);
            for (d = 0; d < 3; d++)
            {
               hypre_BoxIMinD(grow_box, d) -= grow[d][0];
               hypre_BoxIMaxD(grow_box, d) += grow[d][1];
            }
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

      /* create send_box_array and send_procs for box i */
      send_box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      hypre_BoxArraySetSize(send_box_array, num_cboxes);
      send_procs[i] = hypre_CTAlloc(HYPRE_Int, num_cboxes);
      send_rboxnums[i] = hypre_CTAlloc(HYPRE_Int, num_cboxes);
      send_rbox_array = hypre_BoxArrayArrayBoxArray(send_rboxes, i);
      hypre_BoxArraySetSize(send_rbox_array, num_cboxes);

      for (m = 0; m < num_cboxes; m++)
      {
         loc = cboxes_neighbor_location[m];
         send_procs[i][m] = neighbor_procs[loc];
         send_rboxnums[i][m] = neighbor_ids[loc];
         hypre_CopyBox(cboxes[m], hypre_BoxArrayBox(send_box_array, m));

         /* if periodic, positive shift before copying to the rbox_array */
         if (neighbor_shifts[loc]) /* periodic if shift != 0 */
         {
            pshift = hypre_StructGridPShift(grid, neighbor_shifts[loc]);
            hypre_BoxShiftPos(cboxes[m], pshift);
         }
         hypre_CopyBox(cboxes[m], hypre_BoxArrayBox(send_rbox_array, m));

         cboxes[m] = NULL;
      }
   } /* end of loop through each local box */

   /* clean up */
   hypre_TFree(neighbor_procs);
   hypre_TFree(neighbor_ids);
   hypre_TFree(neighbor_shifts);
   hypre_BoxArrayDestroy(neighbor_boxes);

   hypre_TFree(cboxes);
   hypre_TFree(cboxes_mem);
   hypre_TFree(cboxes_neighbor_location);

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
 * Return descriptions of communications patterns for a given grid
 * based on a specified number of "ghost zones".  These patterns are
 * defined by building a stencil and calling CommInfoFromStencil.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CreateCommInfoFromNumGhost( hypre_StructGrid      *grid,
                                  HYPRE_Int             *num_ghost,
                                  hypre_CommInfo       **comm_info_ptr )
{
   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             startstop[6], ii[3], i, d, size;

   stencil_shape = hypre_CTAlloc(hypre_Index, 27);
   for (i = 0; i < 6; i++)
   {
      startstop[i] = num_ghost[i] ? 1 : 0;
   }
   size = 0;
   for (ii[2] = -startstop[4]; ii[2] <= startstop[5]; ii[2]++)
   {
      for (ii[1] = -startstop[2]; ii[1] <= startstop[3]; ii[1]++)
      {
         for (ii[0] = -startstop[0]; ii[0] <= startstop[1]; ii[0]++)
         {
            for (d = 0; d < 3; d++)
            {
               if (ii[d] < 0)
               {
                  stencil_shape[size][d] = -num_ghost[2*d];
               }
               else if (ii[d] > 0)
               {
                  stencil_shape[size][d] =  num_ghost[2*d+1];
               }
            }
            size++;
         }
      }
   }
   stencil = hypre_StructStencilCreate(3, size, stencil_shape);

   hypre_CreateCommInfoFromStencil(grid, stencil, comm_info_ptr);
   
   hypre_StructStencilDestroy(stencil);

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

   HYPRE_Int                i, j, k, r;

   /*------------------------------------------------------
    * Set up communication info
    *------------------------------------------------------*/
 
   for (r = 0; r < 2; r++)
   {
      switch(r)
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
      hypre_GatherAllBoxes(hypre_StructGridComm(remote_grid), remote_boxes,
                           &remote_all_boxes,
                           &remote_all_procs,
                           &remote_first_local);
      hypre_ComputeBoxnums(remote_all_boxes, remote_all_procs,
                           &remote_all_boxnums);

      comm_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(local_boxes));
      comm_procs = hypre_CTAlloc(HYPRE_Int *, hypre_BoxArraySize(local_boxes));
      comm_boxnums = hypre_CTAlloc(HYPRE_Int *, hypre_BoxArraySize(local_boxes));

      comm_box = hypre_BoxCreate();
      hypre_ForBoxI(i, local_boxes)
      {
         local_box = hypre_BoxArrayBox(local_boxes, i);

         comm_box_array = hypre_BoxArrayArrayBoxArray(comm_boxes, i);
         comm_procs[i] =
            hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(remote_all_boxes));
         comm_boxnums[i] =
            hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(remote_all_boxes));

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
                           HYPRE_Int, hypre_BoxArraySize(comm_box_array));
         comm_boxnums[i] =
            hypre_TReAlloc(comm_boxnums[i],
                           HYPRE_Int, hypre_BoxArraySize(comm_box_array));
      }
      hypre_BoxDestroy(comm_box);

      hypre_BoxArrayDestroy(remote_all_boxes);
      hypre_TFree(remote_all_procs);
      hypre_TFree(remote_all_boxnums);

      switch(r)
      {
         case 0:
            send_boxes = comm_boxes;
            send_procs = comm_procs;
            send_rboxnums = comm_boxnums;
            send_rboxes = hypre_BoxArrayArrayDuplicate(comm_boxes);
            break;

         case 1:
            recv_boxes = comm_boxes;
            recv_procs = comm_procs;
            recv_rboxnums = comm_boxnums;
            recv_rboxes = hypre_BoxArrayArrayDuplicate(comm_boxes);
            break;
      }
   }

   hypre_CommInfoCreate(send_boxes, recv_boxes, send_procs, recv_procs,
                        send_rboxnums, recv_rboxnums, send_rboxes, recv_rboxes,
                        1, comm_info_ptr);

   return hypre_error_flag;
}
