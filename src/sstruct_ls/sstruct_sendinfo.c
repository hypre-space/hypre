/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




#include "headers.h"

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

   MPI_Comm                   comm= hypre_SStructVectorComm(fgrid);

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

   hypre_ClearIndex(index); 
   hypre_MPI_Comm_rank(comm, &myproc);

   sendinfo_data= hypre_CTAlloc(hypre_SStructSendInfoData, 1);

   /*------------------------------------------------------------------------
    * Create the structured sendbox patterns. 
    *
    *   send_boxes are obtained by intersecting this proc's fgrid boxes
    *   with cgrid's box_man. Intersecting BoxManEntries not on this proc
    *   will give boxes that we will need to send data to- i.e., we scan
    *   through the boxes of grid and find the processors that own a chunk
    *   of it.
    *------------------------------------------------------------------------*/
   intersect_box = hypre_CTAlloc(hypre_Box, 1);
   grid_boxes   = hypre_StructGridBoxes(fgrid);

   send_boxes= hypre_BoxArrayArrayCreate(hypre_BoxArraySize(grid_boxes));
   send_processes= hypre_CTAlloc(HYPRE_Int *, hypre_BoxArraySize(grid_boxes));
   send_remote_boxnums= hypre_CTAlloc(HYPRE_Int *, hypre_BoxArraySize(grid_boxes));

   hypre_ForBoxI(i, grid_boxes)
   {
       grid_box= hypre_BoxArrayBox(grid_boxes, i);

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

       cnt= 0;
       for (j= 0; j< nboxman_entries; j++)
       {
          hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);
          if (proc != myproc)
          {
             cnt++;
          }
       }
       send_processes[i]     = hypre_CTAlloc(HYPRE_Int, cnt);
       send_remote_boxnums[i]= hypre_CTAlloc(HYPRE_Int, cnt);

       cnt= 0;
       for (j= 0; j< nboxman_entries; j++)
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
      hypre_TFree(boxman_entries);
   }  /* hypre_ForBoxI(i, grid_boxes) */ 

   hypre_TFree(intersect_box);

   (sendinfo_data -> size)               = hypre_BoxArraySize(grid_boxes);
   (sendinfo_data -> send_boxes)         = send_boxes;
   (sendinfo_data -> send_procs)         = send_processes;
   (sendinfo_data -> send_remote_boxnums)= send_remote_boxnums;

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

      for (i= 0; i< (sendinfo_data -> size); i++)
      {
         if (sendinfo_data -> send_procs[i])
         {
             hypre_TFree(sendinfo_data -> send_procs[i]);
         }

         if (sendinfo_data -> send_remote_boxnums[i])
         {
             hypre_TFree(sendinfo_data -> send_remote_boxnums[i]);
         }
      }
      hypre_TFree(sendinfo_data -> send_procs);
      hypre_TFree(sendinfo_data -> send_remote_boxnums);
   }

   hypre_TFree(sendinfo_data);

   return ierr;
}

