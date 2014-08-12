/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef hypre_COMMUNICATION_HEADER
#define hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 *
 * For "reverse" communication, the following are not needed (may be NULL)
 *    send_rboxnums, send_rboxes, send_transforms
 *
 * For "forward" communication, the following are not needed (may be NULL)
 *    recv_rboxnums, recv_rboxes, recv_transforms
 *
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommInfo_struct
{
   HYPRE_Int              ndim;
   hypre_BoxArrayArray   *send_boxes;
   hypre_Index            send_stride;
   HYPRE_Int            **send_processes;
   HYPRE_Int            **send_rboxnums;
   hypre_BoxArrayArray   *send_rboxes;  /* send_boxes, some with periodic shift */

   hypre_BoxArrayArray   *recv_boxes;
   hypre_Index            recv_stride;
   HYPRE_Int            **recv_processes;
   HYPRE_Int            **recv_rboxnums;
   hypre_BoxArrayArray   *recv_rboxes;  /* recv_boxes, some with periodic shift */

   HYPRE_Int              num_transforms;  /* may be 0    = identity transform */
   hypre_Index           *coords;          /* may be NULL = identity transform */
   hypre_Index           *dirs;            /* may be NULL = identity transform */
   HYPRE_Int            **send_transforms; /* may be NULL = identity transform */
   HYPRE_Int            **recv_transforms; /* may be NULL = identity transform */

   HYPRE_Int              boxes_match;  /* true (>0) if each send box has a
                                         * matching box on the recv processor */

} hypre_CommInfo;

/*--------------------------------------------------------------------------
 * Note: The dimension of the data represented by CommEntry will often be
 * smaller than the original problem dimension.
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommEntry_struct
{
   HYPRE_Int  offset;                       /* offset for the data */
   HYPRE_Int  dim;                          /* dimension of the data */
   HYPRE_Int  length_array[HYPRE_MAXDIM+1];
   HYPRE_Int  stride_array[HYPRE_MAXDIM+1];
   HYPRE_Int *imap;                         /* index map for last dim values */

} hypre_CommEntry;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommBlock_struct
{
   HYPRE_Int         bufsize;     /* message buffer size (in doubles) */
   HYPRE_Int         ndim;
   HYPRE_Int         num_values;
   HYPRE_Int         num_entries;
   hypre_CommEntry  *entries;
   HYPRE_Int        *imaps;       /* length = (num_entries*num_values) */

   /* This is only needed until first send buffer prefix is packed */
   HYPRE_Int        *rem_boxnums; /* entry remote box numbers */
   hypre_Box        *rem_boxes;   /* entry remote boxes */
   HYPRE_Int        *rem_orders;  /* length = (num_entries*num_values) */

} hypre_CommBlock;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommType_struct
{
   HYPRE_Int         first_comm;  /* is this the first communication? */
   HYPRE_Int         proc;
   HYPRE_Int         bufsize;     /* message buffer size (in doubles) */
   HYPRE_Int         ndim;
   HYPRE_Int         num_blocks;
   hypre_CommBlock  *blocks;

} hypre_CommType;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommPkg_struct
{
   MPI_Comm          comm;
                   
   HYPRE_Int         ndim;
   HYPRE_Int         send_bufsize; /* total send buffer size (in doubles) */
   HYPRE_Int         recv_bufsize; /* total recv buffer size (in doubles) */

   HYPRE_Int         num_sends;
   HYPRE_Int         num_recvs;
   hypre_CommType   *send_types;
   hypre_CommType   *recv_types;

   hypre_CommType   *copy_from_type;
   hypre_CommType   *copy_to_type;

   /* needed for setting recv entries after the first communication */
   HYPRE_Int         num_blocks;  /* arrays below are num_blocks x ... */
   hypre_Index      *recv_strides;
   hypre_BoxArray  **recv_data_spaces;  /* recv data dimensions (by box) */
   HYPRE_Int       **recv_data_offsets; /* offsets into recv data (by box) */

} hypre_CommPkg;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommHandle_struct
{
   hypre_CommPkg     *comm_pkg;
   HYPRE_Complex    **send_data;
   HYPRE_Complex    **recv_data;

   HYPRE_Int          num_requests;
   hypre_MPI_Request *requests;
   hypre_MPI_Status  *status;

   HYPRE_Complex    **send_buffers;
   HYPRE_Complex    **recv_buffers;

   /* set = 0, add = 1 */
   HYPRE_Int          action;

} hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommInto
 *--------------------------------------------------------------------------*/
 
#define hypre_CommInfoNDim(info)           (info -> ndim)
#define hypre_CommInfoSendBoxes(info)      (info -> send_boxes)
#define hypre_CommInfoSendStride(info)     (info -> send_stride)
#define hypre_CommInfoSendProcesses(info)  (info -> send_processes)
#define hypre_CommInfoSendRBoxnums(info)   (info -> send_rboxnums)
#define hypre_CommInfoSendRBoxes(info)     (info -> send_rboxes)
                                           
#define hypre_CommInfoRecvBoxes(info)      (info -> recv_boxes)
#define hypre_CommInfoRecvStride(info)     (info -> recv_stride)
#define hypre_CommInfoRecvProcesses(info)  (info -> recv_processes)
#define hypre_CommInfoRecvRBoxnums(info)   (info -> recv_rboxnums)
#define hypre_CommInfoRecvRBoxes(info)     (info -> recv_rboxes)
                                           
#define hypre_CommInfoNumTransforms(info)  (info -> num_transforms)
#define hypre_CommInfoCoords(info)         (info -> coords)
#define hypre_CommInfoDirs(info)           (info -> dirs)
#define hypre_CommInfoSendTransforms(info) (info -> send_transforms)
#define hypre_CommInfoRecvTransforms(info) (info -> recv_transforms)
                                           
#define hypre_CommInfoBoxesMatch(info)     (info -> boxes_match)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommEntry
 *--------------------------------------------------------------------------*/
 
#define hypre_CommEntryOffset(entry)       (entry -> offset)
#define hypre_CommEntryDim(entry)          (entry -> dim)
#define hypre_CommEntryLengthArray(entry)  (entry -> length_array)
#define hypre_CommEntryStrideArray(entry)  (entry -> stride_array)
#define hypre_CommEntryIMap(entry)         (entry -> imap)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommBlock
 *--------------------------------------------------------------------------*/
 
#define hypre_CommBlockBufsize(blk)       (blk -> bufsize)
#define hypre_CommBlockNDim(blk)          (blk -> ndim)
#define hypre_CommBlockNumValues(blk)     (blk -> num_values)
#define hypre_CommBlockNumEntries(blk)    (blk -> num_entries)
#define hypre_CommBlockEntries(blk)       (blk -> entries)
#define hypre_CommBlockEntry(blk, i)     &(blk -> entries[i])
#define hypre_CommBlockIMaps(blk)         (blk -> imaps)
#define hypre_CommBlockIMap(blk, i)      &(blk -> imaps[i*(blk->num_values)])

#define hypre_CommBlockRemBoxnums(blk)    (blk -> rem_boxnums)
#define hypre_CommBlockRemBoxnum(blk, i)  (blk -> rem_boxnums[i])
#define hypre_CommBlockRemBoxes(blk)      (blk -> rem_boxes)
#define hypre_CommBlockRemBox(blk, i)    &(blk -> rem_boxes[i])
#define hypre_CommBlockRemOrders(blk)     (blk -> rem_orders)
#define hypre_CommBlockRemOrder(blk, i)  &(blk -> rem_orders[i*(blk->num_values)])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommType
 *--------------------------------------------------------------------------*/
 
#define hypre_CommTypeFirstComm(type)     (type -> first_comm)
#define hypre_CommTypeProc(type)          (type -> proc)
#define hypre_CommTypeBufsize(type)       (type -> bufsize)
#define hypre_CommTypeNDim(type)          (type -> ndim)
#define hypre_CommTypeNumBlocks(type)     (type -> num_blocks)
#define hypre_CommTypeBlocks(type)        (type -> blocks)
#define hypre_CommTypeBlock(type, i)     &(type -> blocks[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgComm(comm_pkg)            (comm_pkg -> comm)

#define hypre_CommPkgNDim(comm_pkg)            (comm_pkg -> ndim)
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

#define hypre_CommPkgNumBlocks(comm_pkg)       (comm_pkg -> num_blocks)
#define hypre_CommPkgRecvStrides(comm_pkg)     (comm_pkg -> recv_strides)
#define hypre_CommPkgRecvDataSpaces(comm_pkg)  (comm_pkg -> recv_data_spaces)
#define hypre_CommPkgRecvDataOffsets(comm_pkg) (comm_pkg -> recv_data_offsets)

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
#define hypre_CommHandleAction(comm_handle)      (comm_handle -> action)

#endif
