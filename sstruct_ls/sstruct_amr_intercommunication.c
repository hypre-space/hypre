/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructAMRInterCommunication: Given the sendinfo, recvinfo, etc.,
 * a communication pkg is formed. This pkg may be used for amr inter_level
 * communication.
 *--------------------------------------------------------------------------*/

int
hypre_SStructAMRInterCommunication( hypre_SStructSendInfoData *sendinfo,
                                    hypre_SStructRecvInfoData *recvinfo,
                                    hypre_BoxArray            *send_data_space,
                                    hypre_BoxArray            *recv_data_space,
                                    int                        num_values,
                                    MPI_Comm                   comm,
                                    hypre_CommPkg            **comm_pkg_ptr )
{
   hypre_CommInfo         *comm_info;
   hypre_CommPkg          *comm_pkg;

   hypre_BoxArrayArray    *sendboxes;
   int                   **sprocesses;
   hypre_BoxArrayArray    *send_rboxes;
   int                   **send_rboxnums;

   hypre_BoxArrayArray    *recvboxes;
   int                   **rprocesses;
   int                   **recv_rboxnums;

   hypre_BoxArray         *boxarray;

   int                     i, j;
   int                     ierr = 0;

   /*------------------------------------------------------------------------
    *  The communication info is copied from sendinfo & recvinfo.
    *------------------------------------------------------------------------*/
   sendboxes  = hypre_BoxArrayArrayDuplicate(sendinfo -> send_boxes);
   send_rboxes= hypre_BoxArrayArrayDuplicate(sendinfo -> send_boxes);

   sprocesses   = hypre_CTAlloc(int *, hypre_BoxArrayArraySize(send_rboxes));
   send_rboxnums= hypre_CTAlloc(int *, hypre_BoxArrayArraySize(send_rboxes));

   hypre_ForBoxArrayI(i, sendboxes)
   {
      boxarray= hypre_BoxArrayArrayBoxArray(sendboxes, i);
      sprocesses[i]   = hypre_CTAlloc(int, hypre_BoxArraySize(boxarray));
      send_rboxnums[i]= hypre_CTAlloc(int, hypre_BoxArraySize(boxarray));

      hypre_ForBoxI(j, boxarray)
      {
         sprocesses[i][j]   = (sendinfo -> send_procs)[i][j];
         send_rboxnums[i][j]= (sendinfo -> send_remote_boxnums)[i][j];
      }
   }

   recvboxes  = hypre_BoxArrayArrayDuplicate(recvinfo -> recv_boxes);
   rprocesses = hypre_CTAlloc(int *, hypre_BoxArrayArraySize(recvboxes));

   /* dummy pointer for CommInfoCreate */
   recv_rboxnums = hypre_CTAlloc(int *, hypre_BoxArrayArraySize(recvboxes));

   hypre_ForBoxArrayI(i, recvboxes)
   {
      boxarray= hypre_BoxArrayArrayBoxArray(recvboxes, i);
      rprocesses[i]= hypre_CTAlloc(int, hypre_BoxArraySize(boxarray));
      recv_rboxnums[i]= hypre_CTAlloc(int, hypre_BoxArraySize(boxarray));

      hypre_ForBoxI(j, boxarray)
      {
         rprocesses[i][j]   = (recvinfo -> recv_procs)[i][j];
      }
   }


   hypre_CommInfoCreate(sendboxes, recvboxes, sprocesses, rprocesses,
                        send_rboxnums, recv_rboxnums, send_rboxes, NULL,
                        1, &comm_info);

   hypre_CommPkgCreate(comm_info,
                       send_data_space,
                       recv_data_space,
                       num_values, NULL, 0, comm,
                       &comm_pkg);
   hypre_CommInfoDestroy(comm_info);

  *comm_pkg_ptr = comm_pkg;

   return ierr;
}


