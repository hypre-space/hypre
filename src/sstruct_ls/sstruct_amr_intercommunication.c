/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * hypre_SStructAMRInterCommunication: Given the sendinfo, recvinfo, etc.,
 * a communication pkg is formed. This pkg may be used for amr inter_level
 * communication.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructAMRInterCommunication( hypre_SStructSendInfoData *sendinfo,
                                    hypre_SStructRecvInfoData *recvinfo,
                                    hypre_BoxArray            *send_data_space,
                                    hypre_BoxArray            *recv_data_space,
                                    HYPRE_Int                  num_values,
                                    MPI_Comm                   comm,
                                    hypre_CommPkg            **comm_pkg_ptr )
{
   hypre_CommInfo         *comm_info;
   hypre_CommPkg          *comm_pkg;

   hypre_BoxArrayArray    *sendboxes;
   HYPRE_Int             **sprocesses;
   hypre_BoxArrayArray    *send_rboxes;
   HYPRE_Int             **send_rboxnums;

   hypre_BoxArrayArray    *recvboxes;
   HYPRE_Int             **rprocesses;
   hypre_BoxArrayArray    *recv_rboxes;
   HYPRE_Int             **recv_rboxnums;

   hypre_BoxArray         *boxarray;

   HYPRE_Int               i, j;
   HYPRE_Int               ierr = 0;

   /*------------------------------------------------------------------------
    *  The communication info is copied from sendinfo & recvinfo.
    *------------------------------------------------------------------------*/
   sendboxes  = hypre_BoxArrayArrayDuplicate(sendinfo -> send_boxes);
   send_rboxes = hypre_BoxArrayArrayDuplicate(sendinfo -> send_boxes);

   sprocesses   = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArrayArraySize(send_rboxes), HYPRE_MEMORY_HOST);
   send_rboxnums = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArrayArraySize(send_rboxes),
                                 HYPRE_MEMORY_HOST);

   hypre_ForBoxArrayI(i, sendboxes)
   {
      boxarray = hypre_BoxArrayArrayBoxArray(sendboxes, i);
      sprocesses[i]   = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(boxarray), HYPRE_MEMORY_HOST);
      send_rboxnums[i] = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(boxarray), HYPRE_MEMORY_HOST);

      hypre_ForBoxI(j, boxarray)
      {
         sprocesses[i][j]   = (sendinfo -> send_procs)[i][j];
         send_rboxnums[i][j] = (sendinfo -> send_remote_boxnums)[i][j];
      }
   }

   recvboxes   = hypre_BoxArrayArrayDuplicate(recvinfo -> recv_boxes);
   recv_rboxes = hypre_BoxArrayArrayDuplicate(recvinfo -> recv_boxes);
   rprocesses  = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArrayArraySize(recvboxes), HYPRE_MEMORY_HOST);

   /* dummy pointer for CommInfoCreate */
   recv_rboxnums = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArrayArraySize(recvboxes), HYPRE_MEMORY_HOST);

   hypre_ForBoxArrayI(i, recvboxes)
   {
      boxarray = hypre_BoxArrayArrayBoxArray(recvboxes, i);
      rprocesses[i] = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(boxarray), HYPRE_MEMORY_HOST);
      recv_rboxnums[i] = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(boxarray), HYPRE_MEMORY_HOST);

      hypre_ForBoxI(j, boxarray)
      {
         rprocesses[i][j]   = (recvinfo -> recv_procs)[i][j];
      }
   }


   hypre_CommInfoCreate(sendboxes, recvboxes, sprocesses, rprocesses,
                        send_rboxnums, recv_rboxnums, send_rboxes,
                        recv_rboxes, 1, &comm_info);

   hypre_CommPkgCreate(comm_info,
                       send_data_space,
                       recv_data_space,
                       num_values, NULL, 0, comm,
                       &comm_pkg);
   hypre_CommInfoDestroy(comm_info);

   *comm_pkg_ptr = comm_pkg;

   return ierr;
}


