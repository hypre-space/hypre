/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/



#ifndef hypre_COMMUNICATION_HEADER
#define hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_CommInfo:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommInfo_struct
{
   hypre_BoxArrayArray   *send_boxes;
   hypre_BoxArrayArray   *recv_boxes;
   hypre_Index            send_stride;
   hypre_Index            recv_stride;
   int                  **send_processes;
   int                  **recv_processes;
   int                  **send_rboxnums;
   int                  **recv_rboxnums; /* required for "inverse" communication */
   hypre_BoxArrayArray   *send_rboxes;

} hypre_CommInfo;

/*--------------------------------------------------------------------------
 * hypre_CommEntryType:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommEntryType_struct
{
   int  offset;           /* offset for the data */
   int  dim;              /* dimension of the communication */
   int  length_array[4];
   int  stride_array[4];

} hypre_CommEntryType;

/*--------------------------------------------------------------------------
 * hypre_CommType:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommType_struct
{
   int                   proc;
   int                   bufsize;     /* message buffer size (in doubles) */
   int                   num_entries;
   hypre_CommEntryType  *entries;

   int                  *loc_boxnums; /* entry local box numbers */
   int                  *rem_boxnums; /* entry remote box numbers */
   hypre_Box            *loc_boxes;   /* entry local boxes */
   hypre_Box            *rem_boxes;   /* entry remote boxes */

} hypre_CommType;

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommPkg_struct
{
   MPI_Comm          comm;

   int               first_send; /* is this the first send? */
   int               first_recv; /* is this the first recv? */
                   
   int               num_values;
   hypre_Index       send_stride;
   hypre_Index       recv_stride;
   int               send_bufsize; /* total send buffer size (in doubles) */
   int               recv_bufsize; /* total recv buffer size (in doubles) */

   int               num_sends;
   int               num_recvs;
   hypre_CommType   *send_types;
   hypre_CommType   *recv_types;

   hypre_CommType   *copy_from_type;
   hypre_CommType   *copy_to_type;

   int              *recv_data_offsets; /* offsets into recv data (by box) */
   hypre_BoxArray   *recv_data_space;   /* recv data dimensions (by box) */

} hypre_CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommHandle_struct
{
   hypre_CommPkg  *comm_pkg;
   double         *send_data;
   double         *recv_data;

   int             num_requests;
   MPI_Request    *requests;
   MPI_Status     *status;

   double        **send_buffers;
   double        **recv_buffers;

} hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommInto
 *--------------------------------------------------------------------------*/
 
#define hypre_CommInfoSendBoxes(info)     (info -> send_boxes)
#define hypre_CommInfoRecvBoxes(info)     (info -> recv_boxes)
#define hypre_CommInfoSendStride(info)    (info -> send_stride)
#define hypre_CommInfoRecvStride(info)    (info -> recv_stride)
#define hypre_CommInfoSendProcesses(info) (info -> send_processes)
#define hypre_CommInfoRecvProcesses(info) (info -> recv_processes)
#define hypre_CommInfoSendRBoxnums(info)  (info -> send_rboxnums)
#define hypre_CommInfoRecvRBoxnums(info)  (info -> recv_rboxnums)
#define hypre_CommInfoSendRBoxes(info)    (info -> send_rboxes)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommEntryType
 *--------------------------------------------------------------------------*/
 
#define hypre_CommEntryTypeOffset(entry)       (entry -> offset)
#define hypre_CommEntryTypeDim(entry)          (entry -> dim)
#define hypre_CommEntryTypeLengthArray(entry)  (entry -> length_array)
#define hypre_CommEntryTypeStrideArray(entry)  (entry -> stride_array)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommType
 *--------------------------------------------------------------------------*/
 
#define hypre_CommTypeProc(type)          (type -> proc)
#define hypre_CommTypeBufsize(type)       (type -> bufsize)
#define hypre_CommTypeNumEntries(type)    (type -> num_entries)
#define hypre_CommTypeEntries(type)       (type -> entries)
#define hypre_CommTypeEntry(type, i)     &(type -> entries[i])
#define hypre_CommTypeLocBoxnums(type)    (type -> loc_boxnums)
#define hypre_CommTypeLocBoxnum(type, i)  (type -> loc_boxnums[i])
#define hypre_CommTypeRemBoxnums(type)    (type -> rem_boxnums)
#define hypre_CommTypeRemBoxnum(type, i)  (type -> rem_boxnums[i])
#define hypre_CommTypeLocBoxes(type)      (type -> loc_boxes)
#define hypre_CommTypeLocBox(type, i)    &(type -> loc_boxes[i])
#define hypre_CommTypeRemBoxes(type)      (type -> rem_boxes)
#define hypre_CommTypeRemBox(type, i)    &(type -> rem_boxes[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgComm(comm_pkg)            (comm_pkg -> comm)

#define hypre_CommPkgFirstSend(comm_pkg)       (comm_pkg -> first_send)
#define hypre_CommPkgFirstRecv(comm_pkg)       (comm_pkg -> first_recv)

#define hypre_CommPkgNumValues(comm_pkg)       (comm_pkg -> num_values)
#define hypre_CommPkgSendStride(comm_pkg)      (comm_pkg -> send_stride)
#define hypre_CommPkgRecvStride(comm_pkg)      (comm_pkg -> recv_stride)
#define hypre_CommPkgSendBufsize(comm_pkg)     (comm_pkg -> send_bufsize)
#define hypre_CommPkgRecvBufsize(comm_pkg)     (comm_pkg -> recv_bufsize)
                                               
#define hypre_CommPkgNumSends(comm_pkg)        (comm_pkg -> num_sends)
#define hypre_CommPkgNumRecvs(comm_pkg)        (comm_pkg -> num_recvs)
#define hypre_CommPkgSendTypes(comm_pkg)       (comm_pkg -> send_types)
#define hypre_CommPkgSendType(comm_pkg, i)    &(comm_pkg -> send_types[i])
#define hypre_CommPkgRecvTypes(comm_pkg)       (comm_pkg -> recv_types)
#define hypre_CommPkgRecvType(comm_pkg, i)    &(comm_pkg -> recv_types[i])

#define hypre_CommPkgCopyFromType(comm_pkg)    (comm_pkg -> copy_from_type)
#define hypre_CommPkgCopyToType(comm_pkg)      (comm_pkg -> copy_to_type)

#define hypre_CommPkgRecvDataOffsets(comm_pkg) (comm_pkg -> recv_data_offsets)
#define hypre_CommPkgRecvDataSpace(comm_pkg)   (comm_pkg -> recv_data_space)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_CommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_CommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_CommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_CommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_CommHandleStatus(comm_handle)      (comm_handle -> status)
#define hypre_CommHandleSendBuffers(comm_handle) (comm_handle -> send_buffers)
#define hypre_CommHandleRecvBuffers(comm_handle) (comm_handle -> recv_buffers)

#endif
