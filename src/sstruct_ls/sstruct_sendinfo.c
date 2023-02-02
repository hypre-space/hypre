/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * hypre_SStructSendInfo: Given a fgrid, coarsen each fbox and find the
 * coarsened boxes that must be sent, the procs that they must be sent to,
 * and the remote boxnums of these sendboxes.
 *--------------------------------------------------------------------------*/

hypre_SStructSendInfoData *
hypre_SStructSendInfo( hypre_StructGrid      *fgrid,
                       hypre_BoxManager      *cboxman,
                       hypre_Index            rfactor )
{
   hypre_SStructSendInfoData *sendinfo_data;

   MPI_Comm                   comm = hypre_StructGridComm(fgrid);
   HYPRE_Int                  ndim = hypre_StructGridNDim(fgrid);

   hypre_BoxArray            *grid_boxes;
   hypre_Box                 *grid_box, cbox;
   hypre_Box                 *intersect_box, boxman_entry_box;

   hypre_BoxManEntry        **boxman_entries;
   HYPRE_Int                  nboxman_entries;

   hypre_BoxArrayArray       *send_boxes;
   HYPRE_Int                **send_processes;
   HYPRE_Int                **send_remote_boxnums;

   hypre_Index                ilower, iupper, index;

   HYPRE_Int                  myproc, proc;

   HYPRE_Int                  cnt;
   HYPRE_Int                  i, j;

   hypre_BoxInit(&cbox, ndim);
   hypre_BoxInit(&boxman_entry_box, ndim);

   hypre_ClearIndex(index);
   hypre_MPI_Comm_rank(comm, &myproc);

   sendinfo_data = hypre_CTAlloc(hypre_SStructSendInfoData,  1, HYPRE_MEMORY_HOST);

   /*------------------------------------------------------------------------
    * Create the structured sendbox patterns.
    *
    *   send_boxes are obtained by intersecting this proc's fgrid boxes
    *   with cgrid's box_man. Intersecting BoxManEntries not on this proc
    *   will give boxes that we will need to send data to- i.e., we scan
    *   through the boxes of grid and find the processors that own a chunk
    *   of it.
    *------------------------------------------------------------------------*/
   intersect_box = hypre_BoxCreate(ndim);
   grid_boxes   = hypre_StructGridBoxes(fgrid);

   send_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(grid_boxes), ndim);
   send_processes = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(grid_boxes), HYPRE_MEMORY_HOST);
   send_remote_boxnums = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(grid_boxes),
                                       HYPRE_MEMORY_HOST);

   hypre_ForBoxI(i, grid_boxes)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);

      /*---------------------------------------------------------------------
       * Find the boxarray that must be sent. BoxManIntersect returns
       * the full extents of the boxes that intersect with the given box.
       * We further need to intersect each box in the list with the given
       * box to determine the actual box that needs to be sent.
       *---------------------------------------------------------------------*/
      hypre_SStructIndexScaleF_C(hypre_BoxIMin(grid_box), index,
                                 rfactor, hypre_BoxIMin(&cbox));
      hypre_SStructIndexScaleF_C(hypre_BoxIMax(grid_box), index,
                                 rfactor, hypre_BoxIMax(&cbox));

      hypre_BoxManIntersect(cboxman, hypre_BoxIMin(&cbox), hypre_BoxIMax(&cbox),
                            &boxman_entries, &nboxman_entries);

      cnt = 0;
      for (j = 0; j < nboxman_entries; j++)
      {
         hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);
         if (proc != myproc)
         {
            cnt++;
         }
      }
      send_processes[i]     = hypre_CTAlloc(HYPRE_Int,  cnt, HYPRE_MEMORY_HOST);
      send_remote_boxnums[i] = hypre_CTAlloc(HYPRE_Int,  cnt, HYPRE_MEMORY_HOST);

      cnt = 0;
      for (j = 0; j < nboxman_entries; j++)
      {
         hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);

         /* determine the chunk of the boxman_entries[j] box that is needed */
         hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
         hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
         hypre_IntersectBoxes(&boxman_entry_box, &cbox, &boxman_entry_box);

         if (proc != myproc)
         {
            send_processes[i][cnt]     = proc;
            hypre_SStructBoxManEntryGetBoxnum(boxman_entries[j],
                                              &send_remote_boxnums[i][cnt]);
            hypre_AppendBox(&boxman_entry_box,
                            hypre_BoxArrayArrayBoxArray(send_boxes, i));
            cnt++;
         }
      }
      hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);
   }  /* hypre_ForBoxI(i, grid_boxes) */

   hypre_BoxDestroy(intersect_box);

   (sendinfo_data -> size)               = hypre_BoxArraySize(grid_boxes);
   (sendinfo_data -> send_boxes)         = send_boxes;
   (sendinfo_data -> send_procs)         = send_processes;
   (sendinfo_data -> send_remote_boxnums) = send_remote_boxnums;

   return sendinfo_data;
}

/*--------------------------------------------------------------------------
 * hypre_SStructSendInfoDataDestroy
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructSendInfoDataDestroy(hypre_SStructSendInfoData *sendinfo_data)
{
   HYPRE_Int ierr = 0;
   HYPRE_Int i;

   if (sendinfo_data)
   {
      if (sendinfo_data -> send_boxes)
      {
         hypre_BoxArrayArrayDestroy( (sendinfo_data -> send_boxes) );
      }

      for (i = 0; i < (sendinfo_data -> size); i++)
      {
         if (sendinfo_data -> send_procs[i])
         {
            hypre_TFree(sendinfo_data -> send_procs[i], HYPRE_MEMORY_HOST);
         }

         if (sendinfo_data -> send_remote_boxnums[i])
         {
            hypre_TFree(sendinfo_data -> send_remote_boxnums[i], HYPRE_MEMORY_HOST);
         }
      }
      hypre_TFree(sendinfo_data -> send_procs, HYPRE_MEMORY_HOST);
      hypre_TFree(sendinfo_data -> send_remote_boxnums, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(sendinfo_data, HYPRE_MEMORY_HOST);

   return ierr;
}

