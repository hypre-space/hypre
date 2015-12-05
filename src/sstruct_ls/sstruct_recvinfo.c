#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructRecvInfo: For each processor, for each cbox of its cgrid,
 * refine it and find out which processors owe this cbox. Coarsen these
 * fine recv boxes and store them.
 *--------------------------------------------------------------------------*/

hypre_SStructRecvInfoData *
hypre_SStructRecvInfo( hypre_StructGrid  *cgrid,
                       hypre_BoxMap      *fmap,
                       hypre_Index        rfactor )
{
   hypre_SStructRecvInfoData *recvinfo_data;

   MPI_Comm                   comm= hypre_SStructVectorComm(cgrid);

   hypre_BoxArray            *grid_boxes;
   hypre_Box                 *grid_box, fbox;
   hypre_Box                 *intersect_box, map_entry_box;

   hypre_BoxMapEntry        **map_entries;
   int                        nmap_entries;

   hypre_BoxArrayArray       *recv_boxes;
   int                      **recv_processes;

   hypre_Index                ilower, iupper, index1, index2;

   int                        myproc, proc;

   int                        cnt;
   int                        i, j;

   hypre_ClearIndex(index1); 
   hypre_SetIndex(index2, rfactor[0]-1, rfactor[1]-1, rfactor[2]-1);

   MPI_Comm_rank(comm, &myproc);

   recvinfo_data= hypre_CTAlloc(hypre_SStructRecvInfoData, 1);

   /*------------------------------------------------------------------------
    * Create the structured recvbox patterns. 
    *   recv_boxes are obtained by intersecting this proc's cgrid boxes
    *   with the fine fmap. Intersecting BoxMapEntries not on this proc
    *   will give the boxes that we will be receiving some data from. To
    *   get the exact receiving box extents, we need to take an intersection.
    *   Since only coarse data is communicated, these intersection boxes
    *   must be coarsened.
    *------------------------------------------------------------------------*/
   intersect_box = hypre_CTAlloc(hypre_Box, 1);
   grid_boxes   = hypre_StructGridBoxes(cgrid);

   recv_boxes= hypre_BoxArrayArrayCreate(hypre_BoxArraySize(grid_boxes));
   recv_processes= hypre_CTAlloc(int *, hypre_BoxArraySize(grid_boxes));

   hypre_ForBoxI(i, grid_boxes)
   {
       grid_box= hypre_BoxArrayBox(grid_boxes, i);

       hypre_SStructIndexScaleC_F(hypre_BoxIMin(grid_box), index1,
                                  rfactor, hypre_BoxIMin(&fbox));
       hypre_SStructIndexScaleC_F(hypre_BoxIMax(grid_box), index2,
                                  rfactor, hypre_BoxIMax(&fbox));

       hypre_BoxMapIntersect(fmap, hypre_BoxIMin(&fbox), hypre_BoxIMax(&fbox),
                            &map_entries, &nmap_entries);

       cnt= 0;
       for (j= 0; j< nmap_entries; j++)
       {
          hypre_SStructMapEntryGetProcess(map_entries[j], &proc);
          if (proc != myproc)
          {
             cnt++;
          }
       }
       recv_processes[i]     = hypre_CTAlloc(int, cnt);

       cnt= 0;
       for (j= 0; j< nmap_entries; j++)
       {
          hypre_SStructMapEntryGetProcess(map_entries[j], &proc);

          /* determine the chunk of the map_entries[j] box that is needed */
          hypre_BoxMapEntryGetExtents(map_entries[j], ilower, iupper);
          hypre_BoxSetExtents(&map_entry_box, ilower, iupper);
          hypre_IntersectBoxes(&map_entry_box, &fbox, &map_entry_box);

          if (proc != myproc)
          {
             recv_processes[i][cnt]= proc;
             hypre_SStructIndexScaleF_C(hypre_BoxIMin(&map_entry_box), index1,
                                        rfactor, hypre_BoxIMin(&map_entry_box));
             hypre_SStructIndexScaleF_C(hypre_BoxIMax(&map_entry_box), index1,
                                        rfactor, hypre_BoxIMax(&map_entry_box));
             hypre_AppendBox(&map_entry_box, 
                              hypre_BoxArrayArrayBoxArray(recv_boxes, i));
             cnt++;
          }
      } 
      hypre_TFree(map_entries);
   }  /* hypre_ForBoxI(i, grid_boxes) */ 

   hypre_TFree(intersect_box);

   (recvinfo_data -> size)      = hypre_BoxArraySize(grid_boxes);
   (recvinfo_data -> recv_boxes)= recv_boxes;
   (recvinfo_data -> recv_procs)= recv_processes;

   return recvinfo_data;
}

/*--------------------------------------------------------------------------
 * hypre_SStructRecvInfoDataDestroy
 *--------------------------------------------------------------------------*/
int
hypre_SStructRecvInfoDataDestroy(hypre_SStructRecvInfoData *recvinfo_data)
{
   int ierr = 0;
   int i;

   if (recvinfo_data)
   {
      if (recvinfo_data -> recv_boxes)
      {
         hypre_BoxArrayArrayDestroy( (recvinfo_data -> recv_boxes) );
      }

      for (i= 0; i< (recvinfo_data -> size); i++)
      {
         if (recvinfo_data -> recv_procs[i])
         {
             hypre_TFree(recvinfo_data -> recv_procs[i]);
         }

      }
      hypre_TFree(recvinfo_data -> recv_procs);
   }

   hypre_TFree(recvinfo_data);

   return ierr;
}

