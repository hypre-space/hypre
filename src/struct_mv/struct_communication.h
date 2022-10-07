/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_COMMUNICATION_HEADER
#define hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_CommInfo:
 *
 * For "reverse" communication, send_transforms is not needed (may be NULL).
 * For "forward" communication, recv_transforms is not needed (may be NULL).
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
 * hypre_CommEntryType:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommEntryType_struct
{
   HYPRE_Int  offset;                       /* offset for the data */
   HYPRE_Int  dim;                          /* dimension of the communication */
   HYPRE_Int  length_array[HYPRE_MAXDIM];   /* last dim has length num_values */
   HYPRE_Int  stride_array[HYPRE_MAXDIM + 1];
   HYPRE_Int *order;                        /* order of last dim values */

} hypre_CommEntryType;

/*--------------------------------------------------------------------------
 * hypre_CommType:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommType_struct
{
   HYPRE_Int             proc;
   HYPRE_Int             bufsize;     /* message buffer size (in doubles) */
   HYPRE_Int             num_entries;
   hypre_CommEntryType  *entries;

   /* this is only needed until first send buffer prefix is packed */
   HYPRE_Int            *rem_boxnums; /* entry remote box numbers */
   hypre_Box            *rem_boxes;   /* entry remote boxes */

} hypre_CommType;

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommPkg_struct
{
   MPI_Comm             comm;

   /* is this the first communication? */
   HYPRE_Int            first_comm;

   HYPRE_Int            ndim;
   HYPRE_Int            num_values;
   hypre_Index          send_stride;
   hypre_Index          recv_stride;

   /* total send buffer size (in doubles) */
   HYPRE_Int            send_bufsize;
   /* total recv buffer size (in doubles) */
   HYPRE_Int            recv_bufsize;
   /* total send buffer size (in doubles) at the first comm. */
   HYPRE_Int            send_bufsize_first_comm;
   /* total recv buffer size (in doubles) at the first comm. */
   HYPRE_Int            recv_bufsize_first_comm;

   HYPRE_Int            num_sends;
   HYPRE_Int            num_recvs;
   hypre_CommType      *send_types;
   hypre_CommType      *recv_types;

   hypre_CommType      *copy_from_type;
   hypre_CommType      *copy_to_type;

   /* these pointers are just to help free up memory for send/from types */
   hypre_CommEntryType *entries;
   HYPRE_Int           *rem_boxnums;
   hypre_Box           *rem_boxes;

   HYPRE_Int            num_orders;
   /* num_orders x num_values */
   HYPRE_Int          **orders;

   /* offsets into recv data (by box) */
   HYPRE_Int           *recv_data_offsets;
   /* recv data dimensions (by box) */
   hypre_BoxArray      *recv_data_space;

   hypre_Index          identity_coord;
   hypre_Index          identity_dir;
   HYPRE_Int           *identity_order;
} hypre_CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct hypre_CommHandle_struct
{
   hypre_CommPkg     *comm_pkg;
   HYPRE_Complex     *send_data;
   HYPRE_Complex     *recv_data;

   HYPRE_Int          num_requests;
   hypre_MPI_Request *requests;
   hypre_MPI_Status  *status;

   HYPRE_Complex    **send_buffers;
   HYPRE_Complex    **recv_buffers;

   HYPRE_Complex    **send_buffers_mpi;
   HYPRE_Complex    **recv_buffers_mpi;

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
 * Accessor macros: hypre_CommEntryType
 *--------------------------------------------------------------------------*/

#define hypre_CommEntryTypeOffset(entry)       (entry -> offset)
#define hypre_CommEntryTypeDim(entry)          (entry -> dim)
#define hypre_CommEntryTypeLengthArray(entry)  (entry -> length_array)
#define hypre_CommEntryTypeStrideArray(entry)  (entry -> stride_array)
#define hypre_CommEntryTypeOrder(entry)        (entry -> order)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommType
 *--------------------------------------------------------------------------*/

#define hypre_CommTypeProc(type)          (type -> proc)
#define hypre_CommTypeBufsize(type)       (type -> bufsize)
#define hypre_CommTypeNumEntries(type)    (type -> num_entries)
#define hypre_CommTypeEntries(type)       (type -> entries)
#define hypre_CommTypeEntry(type, i)    (&(type -> entries[i]))

#define hypre_CommTypeRemBoxnums(type)    (type -> rem_boxnums)
#define hypre_CommTypeRemBoxnum(type, i)  (type -> rem_boxnums[i])
#define hypre_CommTypeRemBoxes(type)      (type -> rem_boxes)
#define hypre_CommTypeRemBox(type, i)   (&(type -> rem_boxes[i]))

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/

#define hypre_CommPkgComm(comm_pkg)                       (comm_pkg -> comm)

#define hypre_CommPkgFirstComm(comm_pkg)                  (comm_pkg -> first_comm)

#define hypre_CommPkgNDim(comm_pkg)                       (comm_pkg -> ndim)
#define hypre_CommPkgNumValues(comm_pkg)                  (comm_pkg -> num_values)
#define hypre_CommPkgSendStride(comm_pkg)                 (comm_pkg -> send_stride)
#define hypre_CommPkgRecvStride(comm_pkg)                 (comm_pkg -> recv_stride)
#define hypre_CommPkgSendBufsize(comm_pkg)                (comm_pkg -> send_bufsize)
#define hypre_CommPkgRecvBufsize(comm_pkg)                (comm_pkg -> recv_bufsize)
#define hypre_CommPkgSendBufsizeFirstComm(comm_pkg)       (comm_pkg -> send_bufsize_first_comm)
#define hypre_CommPkgRecvBufsizeFirstComm(comm_pkg)       (comm_pkg -> recv_bufsize_first_comm)

#define hypre_CommPkgNumSends(comm_pkg)                   (comm_pkg -> num_sends)
#define hypre_CommPkgNumRecvs(comm_pkg)                   (comm_pkg -> num_recvs)
#define hypre_CommPkgSendTypes(comm_pkg)                  (comm_pkg -> send_types)
#define hypre_CommPkgSendType(comm_pkg, i)              (&(comm_pkg -> send_types[i]))
#define hypre_CommPkgRecvTypes(comm_pkg)                  (comm_pkg -> recv_types)
#define hypre_CommPkgRecvType(comm_pkg, i)              (&(comm_pkg -> recv_types[i]))

#define hypre_CommPkgCopyFromType(comm_pkg)               (comm_pkg -> copy_from_type)
#define hypre_CommPkgCopyToType(comm_pkg)                 (comm_pkg -> copy_to_type)

#define hypre_CommPkgEntries(comm_pkg)                    (comm_pkg -> entries)
#define hypre_CommPkgRemBoxnums(comm_pkg)                 (comm_pkg -> rem_boxnums)
#define hypre_CommPkgRemBoxes(comm_pkg)                   (comm_pkg -> rem_boxes)

#define hypre_CommPkgNumOrders(comm_pkg)                  (comm_pkg -> num_orders)
#define hypre_CommPkgOrders(comm_pkg)                     (comm_pkg -> orders)

#define hypre_CommPkgRecvDataOffsets(comm_pkg)            (comm_pkg -> recv_data_offsets)
#define hypre_CommPkgRecvDataSpace(comm_pkg)              (comm_pkg -> recv_data_space)

#define hypre_CommPkgIdentityCoord(comm_pkg)              (comm_pkg -> identity_coord)
#define hypre_CommPkgIdentityDir(comm_pkg)                (comm_pkg -> identity_dir)
#define hypre_CommPkgIdentityOrder(comm_pkg)              (comm_pkg -> identity_order)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/

#define hypre_CommHandleCommPkg(comm_handle)              (comm_handle -> comm_pkg)
#define hypre_CommHandleSendData(comm_handle)             (comm_handle -> send_data)
#define hypre_CommHandleRecvData(comm_handle)             (comm_handle -> recv_data)
#define hypre_CommHandleNumRequests(comm_handle)          (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)             (comm_handle -> requests)
#define hypre_CommHandleStatus(comm_handle)               (comm_handle -> status)
#define hypre_CommHandleSendBuffers(comm_handle)          (comm_handle -> send_buffers)
#define hypre_CommHandleRecvBuffers(comm_handle)          (comm_handle -> recv_buffers)
#define hypre_CommHandleAction(comm_handle)               (comm_handle -> action)
#define hypre_CommHandleSendBuffersMPI(comm_handle)       (comm_handle -> send_buffers_mpi)
#define hypre_CommHandleRecvBuffersMPI(comm_handle)       (comm_handle -> recv_buffers_mpi)

#endif
