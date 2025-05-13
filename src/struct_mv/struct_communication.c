/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

#define DEBUG 0
#define DEBUG_COMM_MAT 0

#if DEBUG
char       filename[255];
FILE      *file;
#endif

/* this computes a (large enough) size (in doubles) for the message prefix */
#define hypre_CommPrefixSize(ne,nv)                                     \
   ( (((1+ne+ne*nv)*sizeof(HYPRE_Int) + ne*sizeof(hypre_Box))/sizeof(HYPRE_Complex)) + 1 )

/*--------------------------------------------------------------------------
 * Create a communication package.  A grid-based description of a communication
 * exchange is passed in.  This description is then compiled into an
 * intermediate processor-based description of the communication.  The
 * intermediate processor-based description is used directly to pack and unpack
 * buffers during the communications.
 *
 * The 'orders' argument is a number-of-orders x num_values array of integers.
 * If orders is NULL, the incremental order from 0 to (num_values-1) is used.
 * If orders is not NULL and there are no transforms in comm_info, then
 * orders[0] is used.  Otherwise, number-of-orders must equal the number of
 * transforms and there should be a one-to-one correspondence with the transform
 * data in comm_info.  Negative order numbers indicate skipped indices, which
 * allows a subset of values to be communicated.
 *
 * If 'reverse' is > 0, then the meaning of send/recv is reversed
 *
 * RDF NOTE: The buffer size will be too large in the case where a subset of
 * values is communicated using negative order numbers.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgCreate( hypre_CommInfo        *comm_info,
                     hypre_BoxArray        *send_data_space,
                     hypre_BoxArray        *recv_data_space,
                     HYPRE_Int              num_values,
                     HYPRE_Int            **orders,
                     HYPRE_Int              reverse,
                     MPI_Comm               comm,
                     HYPRE_MemoryLocation   memory_location,
                     hypre_CommPkg        **comm_pkg_ptr )
{
   HYPRE_Int             ndim = hypre_CommInfoNDim(comm_info);
   hypre_BoxArrayArray  *send_boxes;
   hypre_BoxArrayArray  *recv_boxes;
   hypre_BoxArrayArray  *send_rboxes;
   hypre_BoxArrayArray  *recv_rboxes;
   hypre_IndexRef        send_stride;
   hypre_IndexRef        recv_stride;
   HYPRE_Int           **send_processes;
   HYPRE_Int           **recv_processes;
   HYPRE_Int           **send_rboxnums;

   HYPRE_Int             num_transforms;
   hypre_Index          *coords;
   hypre_Index          *dirs;
   HYPRE_Int           **send_transforms;

   hypre_CommPkg        *comm_pkg;
   hypre_CommType       *comm_types;
   hypre_CommType       *comm_type  = NULL;
   hypre_CommBlock      *comm_block = NULL;
   HYPRE_Int            *cb_num_entries;
   HYPRE_Int            *comm_boxes_p, *comm_boxes_i, *comm_boxes_j;
   HYPRE_Int             num_boxes, num_entries, num_comms, comm_bufsize;

   hypre_BoxArray       *box_array;
   hypre_Box            *box;
   hypre_BoxArray       *rbox_array;
   hypre_Box            *rbox;
   hypre_Box            *data_box;
   HYPRE_Int            *data_offsets;
   HYPRE_Int             data_offset;
   hypre_Index           identity_coord, identity_dir;
   hypre_IndexRef        send_coord, send_dir;
   HYPRE_Int            *send_order;

   HYPRE_Int             i, j, k, p, m, n, size, p_old, my_proc;

   /*------------------------------------------------------
    *------------------------------------------------------*/

   if (reverse > 0)
   {
      /* reverse the meaning of send and recv */
      send_boxes      = hypre_CommInfoRecvBoxes(comm_info);
      recv_boxes      = hypre_CommInfoSendBoxes(comm_info);
      send_stride     = hypre_CommInfoRecvStride(comm_info);
      recv_stride     = hypre_CommInfoSendStride(comm_info);
      send_processes  = hypre_CommInfoRecvProcesses(comm_info);
      recv_processes  = hypre_CommInfoSendProcesses(comm_info);
      send_rboxnums   = hypre_CommInfoRecvRBoxnums(comm_info);
      send_rboxes     = hypre_CommInfoRecvRBoxes(comm_info);
      recv_rboxes     = hypre_CommInfoSendRBoxes(comm_info);
      send_transforms = hypre_CommInfoRecvTransforms(comm_info); /* may be NULL */

      box_array = send_data_space;
      send_data_space = recv_data_space;
      recv_data_space = box_array;
   }
   else
   {
      send_boxes      = hypre_CommInfoSendBoxes(comm_info);
      recv_boxes      = hypre_CommInfoRecvBoxes(comm_info);
      send_stride     = hypre_CommInfoSendStride(comm_info);
      recv_stride     = hypre_CommInfoRecvStride(comm_info);
      send_processes  = hypre_CommInfoSendProcesses(comm_info);
      recv_processes  = hypre_CommInfoRecvProcesses(comm_info);
      send_rboxnums   = hypre_CommInfoSendRBoxnums(comm_info);
      send_rboxes     = hypre_CommInfoSendRBoxes(comm_info);
      recv_rboxes     = hypre_CommInfoRecvRBoxes(comm_info);
      send_transforms = hypre_CommInfoSendTransforms(comm_info); /* may be NULL */
   }
   num_transforms = hypre_CommInfoNumTransforms(comm_info);
   coords         = hypre_CommInfoCoords(comm_info); /* may be NULL */
   dirs           = hypre_CommInfoDirs(comm_info);   /* may be NULL */

   hypre_MPI_Comm_rank(comm, &my_proc);

   /*------------------------------------------------------
    * Set up various entries in CommPkg
    *------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1, HYPRE_MEMORY_HOST);
   hypre_CommPkgComm(comm_pkg) = comm;
   hypre_CommPkgNDim(comm_pkg) = ndim;
   hypre_CommPkgMemoryLocation(comm_pkg) = memory_location;

   /* set up identity transform and order */
   for (i = 0; i < ndim; i++)
   {
      identity_coord[i] = i;
      identity_dir[i]   = 1;
   }

   /*------------------------------------------------------
    * Set up send CommType information
    *------------------------------------------------------*/

   /* set the default send transform and order (may be changed below) */
   send_coord = identity_coord;
   send_dir   = identity_dir;
   send_order = NULL;
   if (orders != NULL)
   {
      /* use the order passed in */
      send_order = orders[0];
   }

   /* set data_offsets and compute num_boxes */
   data_offsets = hypre_TAlloc(HYPRE_Int, hypre_BoxArraySize(send_data_space), HYPRE_MEMORY_HOST);
   data_offset  = num_boxes = 0;
   hypre_ForBoxI(i, send_data_space)
   {
      data_offsets[i] = data_offset;
      data_box = hypre_BoxArrayBox(send_data_space, i);
      data_offset += hypre_BoxVolume(data_box) * num_values;

      /* RDF: This should always be true, but it's not for FAC.  Find out why. */
      if (i < hypre_BoxArrayArraySize(send_boxes))
      {
         box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
         num_boxes += hypre_BoxArraySize(box_array);
      }
   }

   /* set up comm_boxes_[pij] */
   comm_boxes_p = hypre_TAlloc(HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_i = hypre_TAlloc(HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_j = hypre_TAlloc(HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   num_boxes    = 0;
   hypre_ForBoxArrayI(i, send_boxes)
   {
      box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      hypre_ForBoxI(j, box_array)
      {
         comm_boxes_p[num_boxes] = send_processes[i][j];
         comm_boxes_i[num_boxes] = i;
         comm_boxes_j[num_boxes] = j;
         num_boxes++;
      }
   }
   hypre_qsort3i(comm_boxes_p, comm_boxes_i, comm_boxes_j, 0, num_boxes - 1);

   /* count cb_num_entries */
   cb_num_entries = hypre_TAlloc(HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   n = p_old = -1;
   for (m = 0; m < num_boxes; m++)
   {
      i = comm_boxes_i[m];
      j = comm_boxes_j[m];
      box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      box = hypre_BoxArrayBox(box_array, j);

      if (hypre_BoxVolume(box) != 0)
      {
         p = comm_boxes_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            n++;
            cb_num_entries[n] = 0;
            p_old = p;
         }

         cb_num_entries[n] ++;
      }
   }

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = hypre_CTAlloc(hypre_CommType, (num_boxes + 1), HYPRE_MEMORY_HOST);
   comm_type  = &comm_types[0];
   hypre_CommTypeBufsize(comm_type)   = 0;
   hypre_CommTypeNDim(comm_type)      = ndim;
   hypre_CommTypeNumBlocks(comm_type) = 1;
   hypre_CommTypeBlocks(comm_type)    = hypre_CTAlloc(hypre_CommBlock, 1, HYPRE_MEMORY_HOST);

   k = 0;
   n = p_old = -1;
   num_comms = 0;
   comm_bufsize = 0;
   comm_type = &comm_types[0];
   for (m = 0; m < num_boxes; m++)
   {
      i = comm_boxes_i[m];
      j = comm_boxes_j[m];
      box_array  = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      rbox_array = hypre_BoxArrayArrayBoxArray(send_rboxes, i);
      box  = hypre_BoxArrayBox(box_array, j);
      rbox = hypre_BoxArrayBox(rbox_array, j);

      if ((hypre_BoxVolume(box) != 0) && (hypre_BoxVolume(rbox) != 0))
      {
         p = comm_boxes_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            n++;
            k = 0;
            num_entries = cb_num_entries[n];
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms + 1];
               hypre_CommTypeFirstComm(comm_type) = 1;
               hypre_CommTypeProc(comm_type)      = p;
               hypre_CommTypeBufsize(comm_type)   = 0;
               hypre_CommTypeNDim(comm_type)      = ndim;
               hypre_CommTypeNumBlocks(comm_type) = 1;
               hypre_CommTypeBlocks(comm_type)    = hypre_CTAlloc(hypre_CommBlock, 1,
                                                                  HYPRE_MEMORY_HOST);
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }
            comm_block = hypre_CommTypeBlock(comm_type, 0);

            hypre_CommBlockBufsize(comm_block)    = 0;
            hypre_CommBlockNDim(comm_block)       = ndim;
            hypre_CommBlockNumValues(comm_block)  = num_values;
            hypre_CommBlockNumEntries(comm_block) = num_entries;
            hypre_CommBlockEntries(comm_block)    = hypre_TAlloc(hypre_CommEntry, num_entries,
                                                                 HYPRE_MEMORY_HOST);
            hypre_CommBlockIMaps(comm_block)      = hypre_TAlloc(HYPRE_Int,
                                                                 num_entries * num_values,
                                                                 HYPRE_MEMORY_HOST);
            hypre_CommBlockRemBoxnums(comm_block) = hypre_TAlloc(HYPRE_Int, num_entries,
                                                                 HYPRE_MEMORY_HOST);
            hypre_CommBlockRemBoxes(comm_block)   = hypre_TAlloc(hypre_Box, num_entries,
                                                                 HYPRE_MEMORY_HOST);
            hypre_CommBlockRemOrders(comm_block)  = hypre_TAlloc(HYPRE_Int,
                                                                 num_entries * num_values,
                                                                 HYPRE_MEMORY_HOST);
            p_old = p;
         }

         hypre_BoxGetStrideVolume(box, send_stride, &size);
         hypre_CommBlockBufsize(comm_block) += (size * num_values);
         hypre_CommTypeBufsize(comm_type)   += (size * num_values);
         comm_bufsize                       += (size * num_values);
         rbox_array = hypre_BoxArrayArrayBoxArray(send_rboxes, i);
         data_box = hypre_BoxArrayBox(send_data_space, i);
         if (num_transforms != 0)
         {
            send_coord = coords[send_transforms[i][j]];
            send_dir   = dirs[send_transforms[i][j]];
            if (orders != NULL)
            {
               send_order = orders[send_transforms[i][j]];
            }
         }
         hypre_CommBlockSetEntry(comm_block, k,
                                 box, send_stride, send_coord, send_dir,
                                 send_order, hypre_CommBlockRemOrder(comm_block, k),
                                 data_box, data_offsets[i]);
         hypre_CommBlockRemBoxnum(comm_block, k) = send_rboxnums[i][j];
         hypre_CopyBox(hypre_BoxArrayBox(rbox_array, j),
                       hypre_CommBlockRemBox(comm_block, k));
         k++;
      }
   }

   /* set send info in comm_pkg */
   comm_types = hypre_TReAlloc(comm_types, hypre_CommType, (num_comms + 1), HYPRE_MEMORY_HOST);
   hypre_CommPkgSendBufsize(comm_pkg)  = comm_bufsize;
   hypre_CommPkgNumSends(comm_pkg)     = num_comms;
   hypre_CommPkgSendTypes(comm_pkg)    = &comm_types[1];
   hypre_CommPkgCopyFromType(comm_pkg) = &comm_types[0];

   /* free up data_offsets and cb_num_entries */
   hypre_TFree(data_offsets, HYPRE_MEMORY_HOST);
   hypre_TFree(cb_num_entries, HYPRE_MEMORY_HOST);

   /*------------------------------------------------------
    * Set up recv CommType information
    *------------------------------------------------------*/

   /* set data_offsets and compute num_boxes */
   data_offsets = hypre_TAlloc(HYPRE_Int, hypre_BoxArraySize(recv_data_space), HYPRE_MEMORY_HOST);
   data_offset  = num_boxes = 0;
   hypre_ForBoxI(i, recv_data_space)
   {
      data_offsets[i] = data_offset;
      data_box = hypre_BoxArrayBox(recv_data_space, i);
      data_offset += hypre_BoxVolume(data_box) * num_values;

      /* RDF: This should always be true, but it's not for FAC.  Find out why. */
      if (i < hypre_BoxArrayArraySize(recv_boxes))
      {
         box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
         num_boxes += hypre_BoxArraySize(box_array);
      }
   }

   {
      hypre_Index      *recv_strides;
      hypre_BoxArray  **recv_data_spaces;
      HYPRE_Int       **recv_data_offsets;
      HYPRE_Int        *boxes_match;

      recv_strides      = hypre_TAlloc(hypre_Index, 1, HYPRE_MEMORY_HOST);
      recv_data_spaces  = hypre_TAlloc(hypre_BoxArray *, 1, HYPRE_MEMORY_HOST);
      recv_data_offsets = hypre_TAlloc(HYPRE_Int *, 1, HYPRE_MEMORY_HOST);
      boxes_match       = hypre_TAlloc(HYPRE_Int, 1, HYPRE_MEMORY_HOST);

      hypre_CopyIndex(recv_stride, recv_strides[0]);
      recv_data_spaces[0]  = hypre_BoxArrayClone(recv_data_space);
      recv_data_offsets[0] = data_offsets;
      boxes_match[0]       = hypre_CommInfoBoxesMatch(comm_info);

      hypre_CommPkgNumBlocks(comm_pkg)       = 1;
      hypre_CommPkgRecvStrides(comm_pkg)     = recv_strides;
      hypre_CommPkgRecvDataSpaces(comm_pkg)  = recv_data_spaces;
      hypre_CommPkgRecvDataOffsets(comm_pkg) = recv_data_offsets;
      hypre_CommPkgBoxesMatch(comm_pkg)      = boxes_match;
   }

   /* set up comm_boxes_[pij] */
   comm_boxes_p = hypre_TReAlloc(comm_boxes_p, HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_i = hypre_TReAlloc(comm_boxes_i, HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_j = hypre_TReAlloc(comm_boxes_j, HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   num_boxes    = 0;
   hypre_ForBoxArrayI(i, recv_boxes)
   {
      box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      hypre_ForBoxI(j, box_array)
      {
         comm_boxes_p[num_boxes] = recv_processes[i][j];
         comm_boxes_i[num_boxes] = i;
         comm_boxes_j[num_boxes] = j;
         num_boxes++;
      }
   }
   hypre_qsort3i(comm_boxes_p, comm_boxes_i, comm_boxes_j, 0, num_boxes - 1);

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = hypre_CTAlloc(hypre_CommType, (num_boxes + 1), HYPRE_MEMORY_HOST);
   comm_type = &comm_types[0];
   hypre_CommTypeBufsize(comm_type)   = 0;
   hypre_CommTypeNDim(comm_type)      = ndim;
   hypre_CommTypeNumBlocks(comm_type) = 1;
   hypre_CommTypeBlocks(comm_type) = hypre_CTAlloc(hypre_CommBlock, 1, HYPRE_MEMORY_HOST);

   p_old = -1;
   num_comms    = 0;
   comm_bufsize = 0;
   comm_type    = &comm_types[0];
   for (m = 0; m < num_boxes; m++)
   {
      i = comm_boxes_i[m];
      j = comm_boxes_j[m];
      box_array  = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      rbox_array = hypre_BoxArrayArrayBoxArray(recv_rboxes, i);
      box  = hypre_BoxArrayBox(box_array, j);
      rbox = hypre_BoxArrayBox(rbox_array, j);

      if ((hypre_BoxVolume(box) != 0) && (hypre_BoxVolume(rbox) != 0))
      {
         p = comm_boxes_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms + 1];
               hypre_CommTypeFirstComm(comm_type)  = 1;
               hypre_CommTypeProc(comm_type)       = p;
               hypre_CommTypeBufsize(comm_type)    = 0;
               hypre_CommTypeNDim(comm_type)       = ndim;
               hypre_CommTypeNumBlocks(comm_type)  = 1;
               hypre_CommTypeBlocks(comm_type)     = hypre_CTAlloc(hypre_CommBlock, 1,
                                                                   HYPRE_MEMORY_HOST);
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }
            comm_block = hypre_CommTypeBlock(comm_type, 0);

            hypre_CommBlockBufsize(comm_block)    = 0;
            hypre_CommBlockNDim(comm_block)       = ndim;
            hypre_CommBlockNumValues(comm_block)  = num_values;
            hypre_CommBlockNumEntries(comm_block) = 0;
            p_old = p;
         }

         k = hypre_CommBlockNumEntries(comm_block);
         hypre_BoxGetStrideVolume(box, recv_stride, &size);
         hypre_CommBlockBufsize(comm_block) += (size * num_values);
         hypre_CommTypeBufsize(comm_type)   += (size * num_values);
         comm_bufsize                       += (size * num_values);
         hypre_CommBlockNumEntries(comm_block) ++;
      }
   }

   /* set recv info in comm_pkg */
   comm_types = hypre_TReAlloc(comm_types, hypre_CommType, (num_comms + 1), HYPRE_MEMORY_HOST);
   hypre_CommPkgRecvBufsize(comm_pkg) = comm_bufsize;
   hypre_CommPkgNumRecvs(comm_pkg)    = num_comms;
   hypre_CommPkgRecvTypes(comm_pkg)   = &comm_types[1];
   hypre_CommPkgCopyToType(comm_pkg)  = &comm_types[0];

   /* set up CopyToType */
   {
      hypre_CommType  *from_type,  *to_type;
      hypre_CommBlock *from_block, *to_block;

      from_type   = hypre_CommPkgCopyFromType(comm_pkg);
      to_type     = hypre_CommPkgCopyToType(comm_pkg);
      from_block  = hypre_CommTypeBlock(from_type, 0);
      to_block    = hypre_CommTypeBlock(to_type, 0);
      num_entries = hypre_CommBlockNumEntries(from_block);
      hypre_CommBlockNumEntries(to_block) = num_entries;
      hypre_CommBlockEntries(to_block)    = hypre_TAlloc(hypre_CommEntry, num_entries,
                                                         HYPRE_MEMORY_HOST);
      hypre_CommBlockIMaps(to_block)      = hypre_TAlloc(HYPRE_Int, (num_entries * num_values),
                                                         HYPRE_MEMORY_HOST);
      hypre_CommBlockSetEntries(to_block,
                                hypre_CommBlockRemBoxnums(from_block),
                                hypre_CommBlockRemBoxes(from_block),
                                hypre_CommBlockRemOrders(from_block),
                                recv_stride, recv_data_space, data_offsets);
      hypre_TFree(hypre_CommBlockRemBoxnums(from_block), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommBlockRemBoxes(from_block), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommBlockRemOrders(from_block), HYPRE_MEMORY_HOST);
   }

   /*------------------------------------------------------
    * Set buffer prefix sizes
    *------------------------------------------------------*/

   hypre_CommPkgSetPrefixSizes(comm_pkg);

   /*------------------------------------------------------
    * Debugging stuff - ONLY WORKS FOR 3D
    * - Also needs to be updated to use new data structures
    *------------------------------------------------------*/

#if DEBUG
   {
      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_proc);

      hypre_sprintf(filename, "zcommboxes.%05d", my_proc);

      if ((file = fopen(filename, "a")) == NULL)
      {
         hypre_printf("Error: can't open output file %s\n", filename);
         exit(1);
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "SEND boxes:\n\n");

      hypre_fprintf(file, "Stride = (%d,%d,%d)\n",
                    hypre_IndexD(send_stride, 0),
                    hypre_IndexD(send_stride, 1),
                    hypre_IndexD(send_stride, 2));
      hypre_fprintf(file, "BoxArrayArraySize = %d\n",
                    hypre_BoxArrayArraySize(send_boxes));
      hypre_ForBoxArrayI(i, send_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);

         hypre_fprintf(file, "BoxArraySize = %d\n", hypre_BoxArraySize(box_array));
         hypre_ForBoxI(j, box_array)
         {
            box = hypre_BoxArrayBox(box_array, j);
            hypre_fprintf(file, "(%d,%d): (%d,%d,%d) x (%d,%d,%d)\n",
                          i, j,
                          hypre_BoxIMinD(box, 0),
                          hypre_BoxIMinD(box, 1),
                          hypre_BoxIMinD(box, 2),
                          hypre_BoxIMaxD(box, 0),
                          hypre_BoxIMaxD(box, 1),
                          hypre_BoxIMaxD(box, 2));
            hypre_fprintf(file, "(%d,%d): %d,%d\n",
                          i, j, send_processes[i][j], send_rboxnums[i][j]);
         }
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "RECV boxes:\n\n");

      hypre_fprintf(file, "Stride = (%d,%d,%d)\n",
                    hypre_IndexD(recv_stride, 0),
                    hypre_IndexD(recv_stride, 1),
                    hypre_IndexD(recv_stride, 2));
      hypre_fprintf(file, "BoxArrayArraySize = %d\n",
                    hypre_BoxArrayArraySize(recv_boxes));
      hypre_ForBoxArrayI(i, recv_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);

         hypre_fprintf(file, "BoxArraySize = %d\n", hypre_BoxArraySize(box_array));
         hypre_ForBoxI(j, box_array)
         {
            box = hypre_BoxArrayBox(box_array, j);
            hypre_fprintf(file, "(%d,%d): (%d,%d,%d) x (%d,%d,%d)\n",
                          i, j,
                          hypre_BoxIMinD(box, 0),
                          hypre_BoxIMinD(box, 1),
                          hypre_BoxIMinD(box, 2),
                          hypre_BoxIMaxD(box, 0),
                          hypre_BoxIMaxD(box, 1),
                          hypre_BoxIMaxD(box, 2));
            hypre_fprintf(file, "(%d,%d): %d\n",
                          i, j, recv_processes[i][j]);
         }
      }

      fflush(file);
      fclose(file);
   }
#endif

#if DEBUG
   {
      hypre_CommEntry  *comm_entry;
      HYPRE_Int         offset, ndim;
      HYPRE_Int        *length;
      HYPRE_Int        *stride;

      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_proc);

      hypre_sprintf(filename, "zcommentries.%05d", my_proc);

      if ((file = fopen(filename, "a")) == NULL)
      {
         hypre_printf("Error: can't open output file %s\n", filename);
         exit(1);
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "SEND entries:\n\n");

      hypre_fprintf(file, "num_sends = %d\n", hypre_CommPkgNumSends(comm_pkg));

      comm_types = hypre_CommPkgCopyFromType(comm_pkg);
      for (m = 0; m < (hypre_CommPkgNumSends(comm_pkg) + 1); m++)
      {
         comm_type = &comm_types[m];
         hypre_fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
         hypre_fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
         for (i = 0; i < hypre_CommTypeNumEntries(comm_type); i++)
         {
            comm_entry = hypre_CommTypeEntry(comm_type, i);
            offset = hypre_CommEntryOffset(comm_entry);
            dnim   = hypre_CommEntryNDim(comm_entry);
            length = hypre_CommEntryLengthArray(comm_entry);
            stride = hypre_CommEntryStrideArray(comm_entry);
            hypre_fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                          i, offset, ndim,
                          length[0], length[1], length[2], length[3],
                          stride[0], stride[1], stride[2], stride[3]);
         }
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "RECV entries:\n\n");

      hypre_fprintf(file, "num_recvs = %d\n", hypre_CommPkgNumRecvs(comm_pkg));

      comm_types = hypre_CommPkgCopyToType(comm_pkg);

      comm_type = &comm_types[0];
      hypre_fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
      hypre_fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
      for (i = 0; i < hypre_CommTypeNumEntries(comm_type); i++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, i);
         offset = hypre_CommEntryOffset(comm_entry);
         ndim   = hypre_CommEntryNDim(comm_entry);
         length = hypre_CommEntryLengthArray(comm_entry);
         stride = hypre_CommEntryStrideArray(comm_entry);
         hypre_fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                       i, offset, ndim,
                       length[0], length[1], length[2], length[3],
                       stride[0], stride[1], stride[2], stride[3]);
      }

      for (m = 1; m < (hypre_CommPkgNumRecvs(comm_pkg) + 1); m++)
      {
         comm_type = &comm_types[m];
         hypre_fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
         hypre_fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
      }

      fflush(file);
      fclose(file);
   }
#endif

   /*------------------------------------------------------
    * Clean up
    *------------------------------------------------------*/

   hypre_TFree(comm_boxes_p, HYPRE_MEMORY_HOST);
   hypre_TFree(comm_boxes_i, HYPRE_MEMORY_HOST);
   hypre_TFree(comm_boxes_j, HYPRE_MEMORY_HOST);

   *comm_pkg_ptr = comm_pkg;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note that this routine uses an identity coordinate transform.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommBlockSetEntries( hypre_CommBlock  *comm_block,
                           HYPRE_Int        *boxnums,
                           hypre_Box        *boxes,
                           HYPRE_Int        *orders,
                           hypre_Index       stride,
                           hypre_BoxArray   *data_space,
                           HYPRE_Int        *data_offsets )
{
   HYPRE_Int             ndim        = hypre_CommBlockNDim(comm_block);
   HYPRE_Int             num_values  = hypre_CommBlockNumValues(comm_block);
   HYPRE_Int             num_entries = hypre_CommBlockNumEntries(comm_block);
   hypre_Box            *box;
   hypre_Box            *data_box;
   hypre_Index           coord, dir;
   HYPRE_Int            *order;
   HYPRE_Int             i, j, k;

   /* set identity transform */
   for (i = 0; i < ndim; i++)
   {
      coord[i] = i;
      dir[i] = 1;
   }

   for (j = 0; j < num_entries; j++)
   {
      k = boxnums[j];
      box = &boxes[j];
      order = &orders[j * num_values];
      data_box = hypre_BoxArrayBox(data_space, k);

      hypre_CommBlockSetEntry(comm_block, j, box, stride, coord, dir,
                              order, NULL, data_box, data_offsets[k]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * If order is NULL, a natural incremental order is used.  If rem_order is not
 * NULL, then it is set to the equivalent remote ordering.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommBlockSetEntry( hypre_CommBlock  *comm_block,
                         HYPRE_Int         comm_num,
                         hypre_Box        *box,
                         hypre_Index       stride,
                         hypre_Index       coord,
                         hypre_Index       dir,
                         HYPRE_Int        *order,
                         HYPRE_Int        *rem_order,
                         hypre_Box        *data_box,
                         HYPRE_Int         data_box_offset )
{
   hypre_CommEntry *comm_entry = hypre_CommBlockEntry(comm_block, comm_num);
   HYPRE_Int        num_values = hypre_CommBlockNumValues(comm_block);

   HYPRE_Int        ndim = hypre_BoxNDim(box);
   HYPRE_Int        dim;
   HYPRE_Int        offset;
   HYPRE_Int       *length_array, tmp_length_array[HYPRE_MAXDIM];
   HYPRE_Int       *stride_array, tmp_stride_array[HYPRE_MAXDIM];
   HYPRE_Int       *entry_imap;
   hypre_Index      size;
   HYPRE_Int        i, j;

   length_array = hypre_CommEntryLengthArray(comm_entry);
   stride_array = hypre_CommEntryStrideArray(comm_entry);

   /* initialize offset */
   offset = data_box_offset + hypre_BoxIndexRank(data_box, hypre_BoxIMin(box));

   /* initialize length_array and stride_array */
   hypre_BoxGetStrideSize(box, stride, size);
   for (i = 0; i < ndim; i++)
   {
      length_array[i] = hypre_IndexD(size, i);
      stride_array[i] = hypre_IndexD(stride, i);
      for (j = 0; j < i; j++)
      {
         stride_array[i] *= hypre_BoxSizeD(data_box, j);
      }
   }
   length_array[ndim] = num_values;
   stride_array[ndim] = hypre_BoxVolume(data_box);

   /* make adjustments for dir */
   for (i = 0; i < ndim; i++)
   {
      if (dir[i] < 0)
      {
         offset += (length_array[i] - 1) * stride_array[i];
         stride_array[i] = -stride_array[i];
      }
   }

   /* make adjustments for coord */
   for (i = 0; i < ndim; i++)
   {
      tmp_length_array[i] = length_array[i];
      tmp_stride_array[i] = stride_array[i];
   }
   for (i = 0; i < ndim; i++)
   {
      j = coord[i];
      length_array[j] = tmp_length_array[i];
      stride_array[j] = tmp_stride_array[i];
   }

   /* eliminate dimensions with length_array = 1 */
   dim = ndim;
   i = 0;
   while ((dim > 1) && (i < dim)) /* make sure dim is at least one */
   {
      if (length_array[i] == 1)
      {
         for (j = i; j < dim; j++)
         {
            length_array[j] = length_array[j + 1];
            stride_array[j] = stride_array[j + 1];
         }
         length_array[dim] = 1;
         stride_array[dim] = 1;
         dim--;
      }
      else
      {
         i++;
      }
   }

#if 0
   /* sort the array according to length_array (largest to smallest) */
   for (i = (dim - 1); i > 0; i--)
   {
      for (j = 0; j < i; j++)
      {
         if (length_array[j] < length_array[j + 1])
         {
            i_tmp             = length_array[j];
            length_array[j]   = length_array[j + 1];
            length_array[j + 1] = i_tmp;

            i_tmp             = stride_array[j];
            stride_array[j]   = stride_array[j + 1];
            stride_array[j + 1] = i_tmp;
         }
      }
   }
#endif

   hypre_CommEntryOffset(comm_entry) = offset;
   hypre_CommEntryNDim(comm_entry)   = dim;
   entry_imap = hypre_CommBlockIMap(comm_block, comm_num);
   if (order != NULL)
   {
      /* Set imap to order and compress */
      j = 0;
      for (i = 0; i < num_values; i++)
      {
         if (order[i] > -1)
         {
            entry_imap[j++] = order[i];
         }
         if (rem_order != NULL)
         {
            rem_order[i] = -1;
            if (order[i] > -1)
            {
               rem_order[i] = i;
            }
         }
      }
      length_array[dim] = j;
   }
   else
   {
      /* Set imap to natural incremental order */
      for (i = 0; i < num_values; i++)
      {
         entry_imap[i] = i;
         if (rem_order != NULL)
         {
            rem_order[i] = i;
         }
      }
   }
   hypre_CommEntryIMap(comm_entry) = entry_imap;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgSetPrefixSizes( hypre_CommPkg  *comm_pkg )
{
   HYPRE_Int            num_sends   = hypre_CommPkgNumSends(comm_pkg);
   HYPRE_Int            num_recvs   = hypre_CommPkgNumRecvs(comm_pkg);
   HYPRE_Int            num_blocks  = hypre_CommPkgNumBlocks(comm_pkg);
   HYPRE_Int           *boxes_match = hypre_CommPkgBoxesMatch(comm_pkg);

   hypre_CommType      *comm_type;
   hypre_CommBlock     *comm_block;
   HYPRE_Int            i, b, num_values, num_entries;
   HYPRE_Int            buffer_size;

   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      for (b = 0; b < num_blocks; b++)
      {
         comm_block  = hypre_CommTypeBlock(comm_type, b);
         num_values  = hypre_CommBlockNumValues(comm_block);
         num_entries = hypre_CommBlockNumEntries(comm_block);
         buffer_size = hypre_CommPrefixSize(num_entries, num_values);

         hypre_CommBlockPfxsize(comm_block) = buffer_size;
         hypre_CommPkgSendBufsize(comm_pkg) += buffer_size;
      }
   }

   for (i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      for (b = 0; b < num_blocks; b++)
      {
         comm_block  = hypre_CommTypeBlock(comm_type, b);
         num_values  = hypre_CommBlockNumValues(comm_block);
         num_entries = hypre_CommBlockNumEntries(comm_block);
         if ( !boxes_match[b] )
         {
            num_entries = hypre_CommBlockBufsize(comm_block);
         }
         buffer_size = hypre_CommPrefixSize(num_entries, num_values);
         hypre_CommBlockPfxsize(comm_block) = buffer_size;
         hypre_CommPkgRecvBufsize(comm_pkg) += buffer_size;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Create a new CommPkg out of several others
 *
 * This assumes that the MPI communicators are all the same and that the
 * first_comm arguments are all 1.  The ordering of the agglomerated blocks
 * starts with the blocks in comm_pkg[0] and continues with comm_pkg[1], etc.
 * The original comm_pkgs are left intact so that they can be used separately.
 *
 * See companion functions CommPkgAgglomData() and CommPkgAgglomDestroy().
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgAgglomerate( HYPRE_Int        num_comm_pkgs,
                          hypre_CommPkg  **comm_pkgs,
                          hypre_CommPkg  **agg_comm_pkg_ptr )
{
   hypre_CommPkg    *agg_comm_pkg;

   MPI_Comm          comm = hypre_CommPkgComm(comm_pkgs[0]);
   HYPRE_Int         ndim = hypre_CommPkgNDim(comm_pkgs[0]);

   hypre_CommType   *comm_types;
   hypre_CommType   *comm_type = NULL;
   hypre_CommType   *in_comm_type;
   hypre_CommBlock  *comm_block, *in_comm_block;
   hypre_CommEntry  *comm_entry;
   HYPRE_Int        *ctype_p, *ctype_i, *ctype_j;
   HYPRE_Int         num_ctypes, num_comms;
   HYPRE_Int         num_values, num_entries, num_ev;
   HYPRE_Int         num_blocks, *block_starts, in_num_blocks;
   HYPRE_Int         buffer_size;

   HYPRE_Int         i, j, b, k, m, p, p_old, srcase, my_proc;

   /*------------------------------------------------------
    * TODO:
    * - check possible NULL pointers in rem_boxnums, copy types, etc.
    *------------------------------------------------------*/

   /* Sanity check */
   if (num_comm_pkgs < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Cannot agglomerate less than 1 comm. pkg");
      return hypre_error_flag;
   }

   hypre_MPI_Comm_rank(comm, &my_proc);

   agg_comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1, HYPRE_MEMORY_HOST);
   hypre_CommPkgComm(agg_comm_pkg) = comm;
   hypre_CommPkgNDim(agg_comm_pkg) = ndim;
   hypre_CommPkgMemoryLocation(agg_comm_pkg) = hypre_CommPkgMemoryLocation(comm_pkgs[0]);

   block_starts = hypre_TAlloc(HYPRE_Int, num_comm_pkgs, HYPRE_MEMORY_HOST);
   num_blocks = 0;
   for (i = 0; i < num_comm_pkgs; i++)
   {
      block_starts[i] = num_blocks;
      num_blocks += hypre_CommPkgNumBlocks(comm_pkgs[i]);
      if (hypre_CommPkgMemoryLocation(agg_comm_pkg) !=
          hypre_CommPkgMemoryLocation(comm_pkgs[i]))
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Cannot agglomerate comm. pkgs with different memory locations");
         return hypre_error_flag;
      }
   }

   /*------------------------------------------------------
    * Set up send/recv CommType information
    *------------------------------------------------------*/

   for (srcase = 0; srcase < 2; srcase++)
   {
      /* compute num_ctypes */
      num_ctypes = 0;
      for (i = 0; i < num_comm_pkgs; i++)
      {
         switch (srcase)
         {
            case 0:
               num_ctypes += hypre_CommPkgNumSends(comm_pkgs[i]) + 1;
               break;

            case 1:
               num_ctypes += hypre_CommPkgNumRecvs(comm_pkgs[i]) + 1;
               break;
         }
      }

      /* set up ctypes_[pij] */
      ctype_p = hypre_TAlloc(HYPRE_Int, num_ctypes, HYPRE_MEMORY_HOST);
      ctype_i = hypre_TAlloc(HYPRE_Int, num_ctypes, HYPRE_MEMORY_HOST);
      ctype_j = hypre_TAlloc(HYPRE_Int, num_ctypes, HYPRE_MEMORY_HOST);
      num_ctypes = 0;
      for (i = 0; i < num_comm_pkgs; i++)
      {
         switch (srcase)
         {
            case 0:
               num_comms  = hypre_CommPkgNumSends(comm_pkgs[i]);
               comm_types = hypre_CommPkgSendTypes(comm_pkgs[i]);
               break;

            case 1:
               num_comms  = hypre_CommPkgNumRecvs(comm_pkgs[i]);
               comm_types = hypre_CommPkgRecvTypes(comm_pkgs[i]);
               break;
         }
         for (j = 0; j < num_comms; j++)
         {
            comm_type = &comm_types[j];
            ctype_p[num_ctypes] = hypre_CommTypeProc(comm_type);
            ctype_i[num_ctypes] = i;
            ctype_j[num_ctypes] = j;
            num_ctypes++;
         }
      }
      hypre_qsort3i(ctype_p, ctype_i, ctype_j, 0, num_ctypes - 1);

      /* Append copy_[from|to]_type here to simplify things below */
      for (i = 0; i < num_comm_pkgs; i++)
      {
         ctype_p[num_ctypes] = my_proc;
         ctype_i[num_ctypes] = i;
         ctype_j[num_ctypes] = -1;
         num_ctypes++;
      }

      /* Allocate memory for comm_types */
      comm_types = hypre_TAlloc(hypre_CommType, num_ctypes, HYPRE_MEMORY_HOST);

      p_old = -1;
      num_comms = 0;
      buffer_size = 0;
      for (m = 0; m < num_ctypes; m++)
      {
         i = ctype_i[m];
         j = ctype_j[m];
         p = ctype_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms + 1];
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }

            hypre_CommTypeFirstComm(comm_type) = 1;
            hypre_CommTypeProc(comm_type)      = p;
            hypre_CommTypeBufsize(comm_type)   = 0;
            hypre_CommTypeNDim(comm_type)      = ndim;
            hypre_CommTypeNumBlocks(comm_type) = num_blocks;
            hypre_CommTypeBlocks(comm_type)    = hypre_CTAlloc(hypre_CommBlock, num_blocks,
                                                               HYPRE_MEMORY_HOST);
            p_old = p;
         }

         switch (srcase)
         {
            case 0:
               in_comm_type = hypre_CommPkgSendType(comm_pkgs[i], j);
               break;

            case 1:
               in_comm_type = hypre_CommPkgRecvType(comm_pkgs[i], j);
               break;
         }

         in_num_blocks = hypre_CommTypeNumBlocks(in_comm_type);
         for (b = 0; b < in_num_blocks; b++)
         {
            in_comm_block = hypre_CommTypeBlock(in_comm_type, b);
            num_values    = hypre_CommBlockNumValues(in_comm_block);
            num_entries   = hypre_CommBlockNumEntries(in_comm_block);
            num_ev        = num_entries * num_values;

            comm_block = hypre_CommTypeBlock(comm_type, block_starts[i] + b);
            hypre_CommBlockBufsize(comm_block)    = hypre_CommBlockBufsize(in_comm_block);
            hypre_CommBlockNDim(comm_block)       = ndim;
            hypre_CommBlockNumValues(comm_block)  = num_values;
            hypre_CommBlockNumEntries(comm_block) = num_entries;
            if (hypre_CommBlockEntries(in_comm_block) != NULL)
            {
               /* allocate arrays */
               hypre_CommBlockEntries(comm_block) =
                  hypre_TAlloc(hypre_CommEntry, num_entries, HYPRE_MEMORY_HOST);
               hypre_CommBlockIMaps(comm_block) =
                  hypre_TAlloc(HYPRE_Int, num_ev, HYPRE_MEMORY_HOST);

               /* copy data into arrays */
               hypre_TMemcpy(hypre_CommBlockEntries(comm_block),
                             hypre_CommBlockEntries(in_comm_block),
                             hypre_CommEntry, num_entries,
                             HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
               hypre_TMemcpy(hypre_CommBlockIMaps(comm_block),
                             hypre_CommBlockIMaps(in_comm_block),
                             HYPRE_Int, num_ev,
                             HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

               /* fix imap pointers */
               for (k = 0; k < num_entries; k++)
               {
                  comm_entry = hypre_CommBlockEntry(comm_block, k);
                  hypre_CommEntryIMap(comm_entry) =
                     hypre_CommBlockIMap(comm_block, k);
               }
            }
            if (hypre_CommBlockRemBoxnums(in_comm_block) != NULL)
            {
               /* allocate arrays */
               hypre_CommBlockRemBoxnums(comm_block) =
                  hypre_TAlloc(HYPRE_Int, num_entries, HYPRE_MEMORY_HOST);
               hypre_CommBlockRemBoxes(comm_block) =
                  hypre_TAlloc(hypre_Box, num_entries, HYPRE_MEMORY_HOST);
               hypre_CommBlockRemOrders(comm_block) =
                  hypre_TAlloc(HYPRE_Int, num_ev, HYPRE_MEMORY_HOST);

               /* copy data into arrays */
               hypre_TMemcpy(hypre_CommBlockRemBoxnums(comm_block),
                             hypre_CommBlockRemBoxnums(in_comm_block),
                             HYPRE_Int, num_entries,
                             HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
               hypre_TMemcpy(hypre_CommBlockRemBoxes(comm_block),
                             hypre_CommBlockRemBoxes(in_comm_block),
                             hypre_Box, num_entries,
                             HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
               hypre_TMemcpy(hypre_CommBlockRemOrders(comm_block),
                             hypre_CommBlockRemOrders(in_comm_block),
                             HYPRE_Int, num_ev,
                             HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
            }

            hypre_CommTypeBufsize(comm_type) += hypre_CommBlockBufsize(comm_block);
            buffer_size += hypre_CommBlockBufsize(comm_block);
         }
      }

      hypre_TFree(ctype_p, HYPRE_MEMORY_HOST);
      hypre_TFree(ctype_i, HYPRE_MEMORY_HOST);
      hypre_TFree(ctype_j, HYPRE_MEMORY_HOST);

      /* set send/recv info in comm_pkg */
      comm_types = hypre_TReAlloc(comm_types, hypre_CommType, (num_comms + 1), HYPRE_MEMORY_HOST);
      switch (srcase)
      {
         case 0:
            hypre_CommPkgNumSends(agg_comm_pkg)     = num_comms;
            hypre_CommPkgSendBufsize(agg_comm_pkg)  = buffer_size;
            hypre_CommPkgSendTypes(agg_comm_pkg)    = &comm_types[1];
            hypre_CommPkgCopyFromType(agg_comm_pkg) = &comm_types[0];
            break;

         case 1:
            hypre_CommPkgNumRecvs(agg_comm_pkg)     = num_comms;
            hypre_CommPkgRecvBufsize(agg_comm_pkg)  = buffer_size;
            hypre_CommPkgRecvTypes(agg_comm_pkg)    = &comm_types[1];
            hypre_CommPkgCopyToType(agg_comm_pkg)   = &comm_types[0];
            break;
      }
   }

   {
      hypre_Index      *recv_strides,       *in_recv_strides;
      hypre_BoxArray  **recv_data_spaces,  **in_recv_data_spaces;
      HYPRE_Int       **recv_data_offsets, **in_recv_data_offsets, size;
      HYPRE_Int        *boxes_match,        *in_boxes_match;

      recv_strides      = hypre_TAlloc(hypre_Index, num_blocks, HYPRE_MEMORY_HOST);
      recv_data_spaces  = hypre_TAlloc(hypre_BoxArray *, num_blocks, HYPRE_MEMORY_HOST);
      recv_data_offsets = hypre_TAlloc(HYPRE_Int *, num_blocks, HYPRE_MEMORY_HOST);
      boxes_match       = hypre_TAlloc(HYPRE_Int, num_blocks, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_comm_pkgs; i++)
      {
         in_num_blocks = hypre_CommPkgNumBlocks(comm_pkgs[i]);
         for (b = 0; b < in_num_blocks; b++)
         {
            in_recv_strides      = hypre_CommPkgRecvStrides(comm_pkgs[i]);
            in_recv_data_spaces  = hypre_CommPkgRecvDataSpaces(comm_pkgs[i]);
            in_recv_data_offsets = hypre_CommPkgRecvDataOffsets(comm_pkgs[i]);
            in_boxes_match       = hypre_CommPkgBoxesMatch(comm_pkgs[i]);

            j = block_starts[i] + b;
            hypre_CopyIndex(in_recv_strides[b], recv_strides[j]);
            recv_data_spaces[j] = hypre_BoxArrayClone(in_recv_data_spaces[b]);
            size = hypre_BoxArraySize(in_recv_data_spaces[b]);

            recv_data_offsets[j] = hypre_TAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(recv_data_offsets[j], in_recv_data_offsets[b],
                          HYPRE_Int, size, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

            boxes_match[j] = in_boxes_match[b];
         }
      }
      hypre_CommPkgNumBlocks(agg_comm_pkg)       = num_blocks;
      hypre_CommPkgRecvStrides(agg_comm_pkg)     = recv_strides;
      hypre_CommPkgRecvDataSpaces(agg_comm_pkg)  = recv_data_spaces;
      hypre_CommPkgRecvDataOffsets(agg_comm_pkg) = recv_data_offsets;
      hypre_CommPkgBoxesMatch(agg_comm_pkg)      = boxes_match;
   }

   /*------------------------------------------------------
    * Set buffer prefix sizes
    *------------------------------------------------------*/

   hypre_CommPkgSetPrefixSizes(agg_comm_pkg);

   hypre_TFree(block_starts, HYPRE_MEMORY_HOST);

   *agg_comm_pkg_ptr = agg_comm_pkg;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Companion function to CommPkgAgglomerate() for agglomerating data pointers
 * before communication.  For any given agglomerated CommPkg, this function can
 * be used multiple times to agglomerate new data pointers.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgAgglomData( HYPRE_Int         num_comm_pkgs,
                         hypre_CommPkg   **comm_pkg_a,
                         HYPRE_Complex  ***comm_data_a,
                         hypre_CommPkg    *comm_pkg,
                         HYPRE_Complex  ***agg_comm_data_ptr )
{
   HYPRE_Complex  **agg_comm_data = NULL;
   HYPRE_Int        i, j, nb;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (num_comm_pkgs > 0)
   {
      agg_comm_data = hypre_TAlloc(HYPRE_Complex *, hypre_CommPkgNumBlocks(comm_pkg),
                                   HYPRE_MEMORY_HOST);
      nb = 0;
      for (i = 0; i < num_comm_pkgs; i++)
      {
         for (j = 0; j < hypre_CommPkgNumBlocks(comm_pkg_a[i]); j++)
         {
            agg_comm_data[nb++] = comm_data_a[i][j];
         }
      }
   }

   *agg_comm_data_ptr = agg_comm_data;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Free up CommPkg and/or CommPkg data arrays used in Agglomerate() function.
 * Use NULL arguments to control which arrays to free.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgAgglomDestroy( HYPRE_Int         num_comm_pkgs,
                            hypre_CommPkg   **comm_pkg_a,
                            HYPRE_Complex  ***comm_data_a )
{
   HYPRE_Int  i;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (num_comm_pkgs > 0)
   {
      if (comm_pkg_a)
      {
         for (i = 0; i < num_comm_pkgs; i++)
         {
            hypre_CommPkgDestroy(comm_pkg_a[i]);
         }
         hypre_TFree(comm_pkg_a, HYPRE_MEMORY_HOST);
      }
      if (comm_data_a)
      {
         for (i = 0; i < num_comm_pkgs; i++)
         {
            hypre_TFree(comm_data_a[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(comm_data_a, HYPRE_MEMORY_HOST);
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Complex *
hypre_StructCommunicationGetBuffer(HYPRE_MemoryLocation memory_location,
                                   HYPRE_Int            size,
                                   HYPRE_Int            is_send)
{
   HYPRE_Complex *ptr;

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int      buffer_size;
   HYPRE_Int      new_size;

   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      buffer_size = is_send ?
        hypre_HandleStructCommSendBufferSize(hypre_handle()) :
        hypre_HandleStructCommRecvBufferSize(hypre_handle());

      if (size > buffer_size)
      {
         new_size = 5 * size;

         if (is_send)
         {
            hypre_HandleStructCommSendBufferSize(hypre_handle()) = new_size;
            hypre_TFree(hypre_HandleStructCommSendBuffer(hypre_handle()), memory_location);
            hypre_HandleStructCommSendBuffer(hypre_handle()) =
              hypre_CTAlloc(HYPRE_Complex, new_size, memory_location);
         }
         else
         {
            hypre_HandleStructCommRecvBufferSize(hypre_handle()) = new_size;
            hypre_TFree(hypre_HandleStructCommRecvBuffer(hypre_handle()), memory_location);
            hypre_HandleStructCommRecvBuffer(hypre_handle()) =
              hypre_CTAlloc(HYPRE_Complex, new_size, memory_location);
         }
      }

      ptr = is_send ?
        hypre_HandleStructCommSendBuffer(hypre_handle()) :
        hypre_HandleStructCommRecvBuffer(hypre_handle());
   }
   else
#else
   HYPRE_UNUSED_VAR(is_send);
#endif
   {
      ptr = hypre_CTAlloc(HYPRE_Complex, size, memory_location);
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructCommunicationReleaseBuffer(HYPRE_Complex       *buffer,
                                       HYPRE_MemoryLocation memory_location)
{
   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_HOST)
   {
      hypre_TFree(buffer, memory_location);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Initialize a non-blocking communication exchange.
 *
 * The communication buffers are created, the send buffer is manually
 * packed, and the communication requests are posted.
 *
 * Different "actions" are possible when the buffer data is unpacked:
 *   action = 0    - copy the data over existing values in memory
 *   action = 1    - add the data to existing values in memory
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructCommunicationInitialize( hypre_CommPkg     *comm_pkg,
                                     HYPRE_Complex    **send_data,
                                     HYPRE_Complex    **recv_data,
                                     HYPRE_Int          action,
                                     HYPRE_Int          tag,
                                     hypre_CommHandle **comm_handle_ptr )
{
   MPI_Comm             comm                = hypre_CommPkgComm(comm_pkg);
   HYPRE_Int            num_sends           = hypre_CommPkgNumSends(comm_pkg);
   HYPRE_Int            num_recvs           = hypre_CommPkgNumRecvs(comm_pkg);
   HYPRE_Int            num_blocks          = hypre_CommPkgNumBlocks(comm_pkg);
   HYPRE_MemoryLocation memory_location     = hypre_CommPkgMemoryLocation(comm_pkg);
   HYPRE_MemoryLocation memory_location_mpi;

   HYPRE_Int            num_requests;
   hypre_MPI_Request   *requests;
   hypre_MPI_Status    *status;

   HYPRE_Int           *send_bufsizes;
   HYPRE_Int           *recv_bufsizes;
   HYPRE_Complex      **send_buffers;
   HYPRE_Complex      **recv_buffers;
   HYPRE_Complex      **send_buffers_mpi;
   HYPRE_Complex      **recv_buffers_mpi;
   HYPRE_Complex      **sdata;
   HYPRE_Complex      **rdata;

   hypre_CommHandle    *comm_handle;
   hypre_CommType      *comm_type;
   hypre_CommBlock     *comm_block;
   hypre_CommEntry     *comm_entry;
   HYPRE_Int            num_values, num_entries, offset;

   HYPRE_Int           *length_array;
   HYPRE_Int           *stride_array;
   HYPRE_Int            unitst_array[HYPRE_MAXDIM + 1];
   HYPRE_Int           *imap;

   HYPRE_Complex       *dptr, *kptr, *lptr;
   HYPRE_Int           *qptr;

   HYPRE_Int            i, j, b, d, ll, ndim, size;

   /* Set memory location for MPI buffers */
   memory_location_mpi = hypre_GetGpuAwareMPI() ? memory_location : HYPRE_MEMORY_HOST;

#if DEBUG_COMM_MAT
   /*--------------------------------------------------------------------
    * Check if communication matrix is symmetric
    *--------------------------------------------------------------------*/
   HYPRE_Int           *recvs, *sends;

   /* Set receives */
   recvs = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      recvs[i] = hypre_CommTypeProc(comm_type);
   }

   /* Set sends */
   sends = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      sends[i] = hypre_CommTypeProc(comm_type);
   }

   hypre_MPI_CheckCommMatrix(comm, num_recvs, recvs, num_sends, sends);

   hypre_TFree(recvs, HYPRE_MEMORY_HOST);
   hypre_TFree(sends, HYPRE_MEMORY_HOST);
#endif

   /*--------------------------------------------------------------------
    * allocate requests and status
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests     = hypre_CTAlloc(hypre_MPI_Request, num_requests, HYPRE_MEMORY_HOST);
   status       = hypre_CTAlloc(hypre_MPI_Status,  num_requests, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------
    * allocate buffers
    *--------------------------------------------------------------------*/

   /* allocate send buffers */
   send_buffers  = hypre_TAlloc(HYPRE_Complex *, num_sends, HYPRE_MEMORY_HOST);
   send_bufsizes = hypre_TAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
   if (num_sends > 0)
   {
      size = hypre_CommPkgSendBufsize(comm_pkg);
      send_buffers[0] = hypre_StructCommunicationGetBuffer(memory_location, size, 1);
      for (i = 0; i < num_sends; i++)
      {
         comm_type = hypre_CommPkgSendType(comm_pkg, i);
         size = hypre_CommTypeBufsize(comm_type);
         if (hypre_CommTypeFirstComm(comm_type))
         {
            for (b = 0; b < num_blocks; b++)
            {
               comm_block = hypre_CommTypeBlock(comm_type, b);
               size += hypre_CommBlockPfxsize(comm_block);
            }
         }
         send_bufsizes[i] = size;
         if (i < num_sends - 1)
         {
            send_buffers[i + 1] = send_buffers[i] + size;
         }
      }
   }

   /* allocate recv buffers */
   recv_buffers  = hypre_TAlloc(HYPRE_Complex *, num_recvs, HYPRE_MEMORY_HOST);
   recv_bufsizes = hypre_TAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST);
   if (num_recvs > 0)
   {
      size = hypre_CommPkgRecvBufsize(comm_pkg);
      recv_buffers[0] = hypre_StructCommunicationGetBuffer(memory_location, size, 0);
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);
         size = hypre_CommTypeBufsize(comm_type);
         if (hypre_CommTypeFirstComm(comm_type))
         {
            for (b = 0; b < num_blocks; b++)
            {
               comm_block = hypre_CommTypeBlock(comm_type, b);
               size += hypre_CommBlockPfxsize(comm_block);
            }
         }
         recv_bufsizes[i] = size;
         if (i < num_recvs - 1)
         {
            recv_buffers[i + 1] = recv_buffers[i] + size;
         }
      }
   }

   /*--------------------------------------------------------------------
    * pack send buffers
    *--------------------------------------------------------------------*/

   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);

      dptr = (HYPRE_Complex *) send_buffers[i];
      if (hypre_CommTypeFirstComm(comm_type))
      {
         for (b = 0; b < num_blocks; b++)
         {
            comm_block  = hypre_CommTypeBlock(comm_type, b);
            num_values  = hypre_CommBlockNumValues(comm_block);
            num_entries = hypre_CommBlockNumEntries(comm_block);

            /* Shift dptr to the next block */
            dptr += hypre_CommPrefixSize(num_entries, num_values);
         }
      }

      for (b = 0; b < num_blocks; b++)
      {
         comm_block  = hypre_CommTypeBlock(comm_type, b);
         num_values  = hypre_CommBlockNumValues(comm_block);
         num_entries = hypre_CommBlockNumEntries(comm_block);

         for (j = 0; j < num_entries; j++)
         {
            comm_entry   = hypre_CommBlockEntry(comm_block, j);
            ndim         = hypre_CommEntryNDim(comm_entry);
            length_array = hypre_CommEntryLengthArray(comm_entry);
            stride_array = hypre_CommEntryStrideArray(comm_entry);
            imap         = hypre_CommEntryIMap(comm_entry);
            offset       = hypre_CommEntryOffset(comm_entry);

            unitst_array[0] = 1;
            for (d = 0; d < ndim; d++)
            {
               unitst_array[d + 1] = unitst_array[d] * length_array[d];
            }

            lptr = send_data[b] + offset;
            for (ll = 0; ll < length_array[ndim]; ll++)
            {
               if (imap[ll] > -1)
               {
                  kptr = lptr + imap[ll] * stride_array[ndim];

#define DEVICE_VAR is_device_ptr(dptr, kptr)
                  hypre_BasicBoxLoop2Begin(ndim, length_array,
                                           stride_array, ki,
                                           unitst_array, di);
                  {
                     dptr[di] = kptr[ki];
                  }
                  hypre_BoxLoop2End(ki, di);
#undef DEVICE_VAR

                  dptr += unitst_array[ndim];
               }
               else
               {
                  size = 1;
                  for (d = 0; d < ndim; d++)
                  {
                     size *= length_array[d];
                  }

                  hypre_Memset(dptr, 0, size * sizeof(HYPRE_Complex), memory_location);
                  dptr += size;
               }
            }
         }
      }
   }

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream();

         send_buffers_mpi = send_buffers;
         recv_buffers_mpi = recv_buffers;
      }
      else
      {
         send_buffers_mpi = hypre_TAlloc(HYPRE_Complex *, num_sends, HYPRE_MEMORY_HOST);
         if (num_sends > 0)
         {
            size = hypre_CommPkgSendBufsize(comm_pkg);
            send_buffers_mpi[0] = hypre_CTAlloc(HYPRE_Complex, size, memory_location_mpi);
            for (i = 1; i < num_sends; i++)
            {
               send_buffers_mpi[i] = send_buffers_mpi[i - 1] + send_bufsizes[i - 1];
            }
            hypre_TMemcpy(send_buffers_mpi[0], send_buffers[0], HYPRE_Complex, size,
                          HYPRE_MEMORY_HOST, memory_location);
         }

         recv_buffers_mpi = hypre_TAlloc(HYPRE_Complex *, num_recvs, HYPRE_MEMORY_HOST);
         if (num_recvs > 0)
         {
            size = hypre_CommPkgRecvBufsize(comm_pkg);
            recv_buffers_mpi[0] = hypre_CTAlloc(HYPRE_Complex, size, memory_location_mpi);
            for (i = 1; i < num_recvs; i++)
            {
               recv_buffers_mpi[i] = recv_buffers_mpi[i - 1] + recv_bufsizes[i - 1];
            }
         }
      }
   }
   else
#endif /* if defined HYPRE_USING_GPU */
   {
      send_buffers_mpi = send_buffers;
      recv_buffers_mpi = recv_buffers;
   }

   /* Pack prefix data */
   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);

      dptr = (HYPRE_Complex *) send_buffers_mpi[i];
      if (hypre_CommTypeFirstComm(comm_type))
      {
         for (b = 0; b < num_blocks; b++)
         {
            comm_block  = hypre_CommTypeBlock(comm_type, b);
            num_values  = hypre_CommBlockNumValues(comm_block);
            num_entries = hypre_CommBlockNumEntries(comm_block);

            qptr = (HYPRE_Int *) dptr;
            hypre_TMemcpy(qptr, &num_entries, HYPRE_Int, 1,
                          memory_location_mpi, HYPRE_MEMORY_HOST);
            qptr ++;
            hypre_TMemcpy(qptr, hypre_CommBlockRemOrders(comm_block),
                          HYPRE_Int, num_entries * num_values,
                          memory_location_mpi, HYPRE_MEMORY_HOST);
            qptr += num_entries * num_values;
            hypre_TMemcpy(qptr, hypre_CommBlockRemBoxnums(comm_block),
                          HYPRE_Int, num_entries,
                          memory_location_mpi, HYPRE_MEMORY_HOST);
            qptr += num_entries;
            hypre_TMemcpy(qptr, hypre_CommBlockRemBoxes(comm_block),
                          hypre_Box, num_entries,
                          memory_location_mpi, HYPRE_MEMORY_HOST);

            /* Shift dptr to the next block */
            dptr += hypre_CommPrefixSize(num_entries, num_values);

            hypre_TFree(hypre_CommBlockRemBoxnums(comm_block), HYPRE_MEMORY_HOST);
            hypre_TFree(hypre_CommBlockRemBoxes(comm_block), HYPRE_MEMORY_HOST);
            hypre_TFree(hypre_CommBlockRemOrders(comm_block), HYPRE_MEMORY_HOST);
         }
      }
   }

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   j = 0;
   for (i = 0 ; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      hypre_MPI_Irecv(recv_buffers_mpi[i],
                      recv_bufsizes[i] * sizeof(HYPRE_Complex),
                      hypre_MPI_BYTE, hypre_CommTypeProc(comm_type),
                      tag, comm, &requests[j++]);
   }
   hypre_TFree(recv_bufsizes, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      hypre_MPI_Isend(send_buffers_mpi[i],
                      send_bufsizes[i] * sizeof(HYPRE_Complex),
                      hypre_MPI_BYTE, hypre_CommTypeProc(comm_type),
                      tag, comm, &requests[j++]);

      /* Reset first communication flag (send type) */
      hypre_CommTypeFirstComm(comm_type) = 0;
   }
   hypre_TFree(send_bufsizes, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------
    * exchange local data
    *--------------------------------------------------------------------*/

   hypre_ExchangeLocalData(comm_pkg, send_data, recv_data, action);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_TAlloc(hypre_CommHandle, 1, HYPRE_MEMORY_HOST);
   sdata       = hypre_TAlloc(HYPRE_Complex*, num_blocks, HYPRE_MEMORY_HOST);
   rdata       = hypre_TAlloc(HYPRE_Complex*, num_blocks, HYPRE_MEMORY_HOST);
   for (b = 0; b < num_blocks; b++)
   {
      sdata[b] = send_data[b];
      rdata[b] = recv_data[b];
   }

   hypre_CommHandleCommPkg(comm_handle)        = comm_pkg;
   hypre_CommHandleSendData(comm_handle)       = sdata;
   hypre_CommHandleRecvData(comm_handle)       = rdata;
   hypre_CommHandleNumRequests(comm_handle)    = num_requests;
   hypre_CommHandleRequests(comm_handle)       = requests;
   hypre_CommHandleStatus(comm_handle)         = status;
   hypre_CommHandleSendBuffers(comm_handle)    = send_buffers;
   hypre_CommHandleRecvBuffers(comm_handle)    = recv_buffers;
   hypre_CommHandleSendBuffersMPI(comm_handle) = send_buffers_mpi;
   hypre_CommHandleRecvBuffersMPI(comm_handle) = recv_buffers_mpi;
   hypre_CommHandleAction(comm_handle)         = action;

   *comm_handle_ptr = comm_handle;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Finalize a communication exchange. This routine blocks until all
 * of the communication requests are completed.
 *
 * The communication requests are completed, and the receive buffer is
 * manually unpacked.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructCommunicationFinalize( hypre_CommHandle *comm_handle )
{
   hypre_CommPkg        *comm_pkg            = hypre_CommHandleCommPkg(comm_handle);
   HYPRE_Complex       **recv_data           = hypre_CommHandleRecvData(comm_handle);
   HYPRE_Complex       **send_buffers        = hypre_CommHandleSendBuffers(comm_handle);
   HYPRE_Complex       **recv_buffers        = hypre_CommHandleRecvBuffers(comm_handle);
   HYPRE_Complex       **send_buffers_mpi    = hypre_CommHandleSendBuffersMPI(comm_handle);
   HYPRE_Complex       **recv_buffers_mpi    = hypre_CommHandleRecvBuffersMPI(comm_handle);
   HYPRE_Int             action              = hypre_CommHandleAction(comm_handle);

   HYPRE_Int             num_sends           = hypre_CommPkgNumSends(comm_pkg);
   HYPRE_Int             num_recvs           = hypre_CommPkgNumRecvs(comm_pkg);
   HYPRE_Int             num_blocks          = hypre_CommPkgNumBlocks(comm_pkg);
   HYPRE_Int             recv_buf_size       = hypre_CommPkgRecvBufsize(comm_pkg);
   hypre_Index          *recv_strides        = hypre_CommPkgRecvStrides(comm_pkg);
   hypre_BoxArray      **recv_data_spaces    = hypre_CommPkgRecvDataSpaces(comm_pkg);
   HYPRE_Int           **recv_data_offsets   = hypre_CommPkgRecvDataOffsets(comm_pkg);
   HYPRE_MemoryLocation  memory_location     = hypre_CommPkgMemoryLocation(comm_pkg);
   HYPRE_MemoryLocation  memory_location_mpi;

   hypre_CommType       *comm_type;
   hypre_CommBlock      *comm_block;
   hypre_CommEntry      *comm_entry;
   HYPRE_Int             num_values, num_entries, offset;

   HYPRE_Int            *length_array;
   HYPRE_Int            *stride_array, unitst_array[HYPRE_MAXDIM + 1];
   HYPRE_Int            *imap;

   HYPRE_Complex        *kptr, *lptr;
   HYPRE_Complex        *dptr;
   HYPRE_Int            *qptr;

   HYPRE_Int            *boxnums;
   hypre_Box            *boxes;
   HYPRE_Int            *orders;

   HYPRE_Int             i, j, b, d, ll, ndim;

   /* Set memory location for MPI buffers */
   memory_location_mpi = hypre_GetGpuAwareMPI() ? memory_location : HYPRE_MEMORY_HOST;

   /*--------------------------------------------------------------------
    * finish communications
    *--------------------------------------------------------------------*/

   if (hypre_CommHandleNumRequests(comm_handle))
   {
      hypre_MPI_Waitall(hypre_CommHandleNumRequests(comm_handle),
                        hypre_CommHandleRequests(comm_handle),
                        hypre_CommHandleStatus(comm_handle));
   }

   /*--------------------------------------------------------------------
    * unpack receive buffer data
    *--------------------------------------------------------------------*/

   /* This can happen only with GPU runs without GPU-aware MPI */
   if ((recv_buffers != recv_buffers_mpi) && (num_recvs > 0))
   {
      hypre_TMemcpy(recv_buffers[0], recv_buffers_mpi[0],
                    HYPRE_Complex, recv_buf_size,
                    memory_location, memory_location_mpi);
   }

   /* Unpack prefix information */
   for (i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);

      /* We use recv_buffers_mpi to possibly avoid a redundant D2H transfer */
      dptr = (HYPRE_Complex *) recv_buffers_mpi[i];

      if (hypre_CommTypeFirstComm(comm_type))
      {
         for (b = 0; b < num_blocks; b++)
         {
            comm_block = hypre_CommTypeBlock(comm_type, b);
            num_values = hypre_CommBlockNumValues(comm_block);

            qptr = (HYPRE_Int *) dptr;

            /* Set boxnums and boxes from MPI recv buffer */
            if (hypre_GetActualMemLocation(memory_location_mpi) != hypre_MEMORY_DEVICE)
            {
               num_entries = *qptr;
               qptr ++;
               orders = qptr;
               qptr += num_entries * num_values;
               boxnums = qptr;
               qptr += num_entries;
               boxes = (hypre_Box *) qptr;
            }
            else
            {
               hypre_TMemcpy(&num_entries, qptr, HYPRE_Int, 1,
                             HYPRE_MEMORY_HOST, memory_location_mpi);
               qptr ++;

               orders = hypre_TAlloc(HYPRE_Int, num_entries * num_values, HYPRE_MEMORY_HOST);
               hypre_TMemcpy(orders, qptr, HYPRE_Int, num_entries * num_values,
                             HYPRE_MEMORY_HOST, memory_location_mpi);
               qptr += num_entries * num_values;

               boxnums = hypre_TAlloc(HYPRE_Int, num_entries, HYPRE_MEMORY_HOST);
               hypre_TMemcpy(boxnums, qptr, HYPRE_Int, num_entries,
                             HYPRE_MEMORY_HOST, memory_location_mpi);
               qptr += num_entries;

               boxes = hypre_TAlloc(hypre_Box, num_entries, HYPRE_MEMORY_HOST);
               hypre_TMemcpy(boxes, qptr, hypre_Box, num_entries,
                             HYPRE_MEMORY_HOST, memory_location_mpi);
            }

            hypre_CommBlockNumEntries(comm_block) = num_entries;
            hypre_CommBlockEntries(comm_block) =
               hypre_TAlloc(hypre_CommEntry, num_entries, HYPRE_MEMORY_HOST);
            hypre_CommBlockIMaps(comm_block) =
               hypre_TAlloc(HYPRE_Int, (num_entries * num_values), HYPRE_MEMORY_HOST);
            hypre_CommBlockSetEntries(comm_block, boxnums, boxes, orders,
                                      recv_strides[b],
                                      recv_data_spaces[b],
                                      recv_data_offsets[b]);

            /* Shift dptr to the next block */
            dptr += hypre_CommPrefixSize(num_entries, num_values);

            /* Free work arrays */
            if (hypre_GetActualMemLocation(memory_location_mpi) == hypre_MEMORY_DEVICE)
            {
               hypre_TFree(boxes, HYPRE_MEMORY_HOST);
               hypre_TFree(orders, HYPRE_MEMORY_HOST);
               hypre_TFree(boxnums, HYPRE_MEMORY_HOST);
            }
         }
      }
   }

   /* Unpack RecvType entries */
   for (i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);

      dptr = (HYPRE_Complex *) recv_buffers[i];
      if (hypre_CommTypeFirstComm(comm_type))
      {
         for (b = 0; b < num_blocks; b++)
         {
            comm_block  = hypre_CommTypeBlock(comm_type, b);
            num_values  = hypre_CommBlockNumValues(comm_block);
            num_entries = hypre_CommBlockNumEntries(comm_block);

            /* Shift dptr to the next block */
            dptr += hypre_CommPrefixSize(num_entries, num_values);
         }

         /* Reset first communication flag (recv type) */
         hypre_CommTypeFirstComm(comm_type) = 0;
      }

      for (b = 0; b < num_blocks; b++)
      {
         comm_block  = hypre_CommTypeBlock(comm_type, b);
         num_entries = hypre_CommBlockNumEntries(comm_block);

         for (j = 0; j < num_entries; j++)
         {
            comm_entry   = hypre_CommBlockEntry(comm_block, j);
            ndim         = hypre_CommEntryNDim(comm_entry);
            length_array = hypre_CommEntryLengthArray(comm_entry);
            stride_array = hypre_CommEntryStrideArray(comm_entry);
            imap         = hypre_CommEntryIMap(comm_entry);
            offset       = hypre_CommEntryOffset(comm_entry);

            unitst_array[0] = 1;
            for (d = 0; d < ndim; d++)
            {
               unitst_array[d + 1] = unitst_array[d] * length_array[d];
            }

            lptr = recv_data[b] + offset;
            for (ll = 0; ll < length_array[ndim]; ll++)
            {
               kptr = lptr + imap[ll] * stride_array[ndim];

#define DEVICE_VAR is_device_ptr(kptr,dptr)
               hypre_BasicBoxLoop2Begin(ndim, length_array,
                                        stride_array, ki,
                                        unitst_array, di);
               {
                  kptr[ki] = (action > 0) ? kptr[ki] + dptr[di] : dptr[di];
               }
               hypre_BoxLoop2End(ki, di);
#undef DEVICE_VAR

               dptr += unitst_array[ndim];
            }
         }
      }
   }

   /*--------------------------------------------------------------------
    * Free up communication handle
    *--------------------------------------------------------------------*/

   if (send_buffers_mpi != send_buffers)
   {
      hypre_TFree(send_buffers_mpi[0], memory_location_mpi);
      hypre_TFree(send_buffers_mpi, HYPRE_MEMORY_HOST);
   }
   if (recv_buffers_mpi != recv_buffers)
   {
      hypre_TFree(recv_buffers_mpi[0], memory_location_mpi);
      hypre_TFree(recv_buffers_mpi, HYPRE_MEMORY_HOST);
   }
   if (num_sends > 0)
   {
      hypre_StructCommunicationReleaseBuffer(send_buffers[0], memory_location);
   }
   if (num_recvs > 0)
   {
      hypre_StructCommunicationReleaseBuffer(recv_buffers[0], memory_location);
   }
   hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_CommHandleSendData(comm_handle), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_CommHandleRecvData(comm_handle), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_CommHandleRequests(comm_handle), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_CommHandleStatus(comm_handle), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Execute local data exchanges.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ExchangeLocalData( hypre_CommPkg   *comm_pkg,
                         HYPRE_Complex  **send_data,
                         HYPRE_Complex  **recv_data,
                         HYPRE_Int        action )
{
   HYPRE_Int        num_blocks = hypre_CommPkgNumBlocks(comm_pkg);

   hypre_CommType  *copy_fr_type;
   hypre_CommType  *copy_to_type;
   hypre_CommBlock *copy_fr_block;
   hypre_CommBlock *copy_to_block;
   hypre_CommEntry *copy_fr_entry;
   hypre_CommEntry *copy_to_entry;

   HYPRE_Complex   *fr_dp, *fr_dpl;
   HYPRE_Int       *fr_stride_array;
   HYPRE_Int       *fr_imap;
   HYPRE_Complex   *to_dp, *to_dpl;
   HYPRE_Int       *to_stride_array;
   HYPRE_Int       *to_imap;

   HYPRE_Int       *length_array;
   HYPRE_Int        i, b, ll, ndim;

   /*--------------------------------------------------------------------
    * copy local data
    *--------------------------------------------------------------------*/

   copy_fr_type = hypre_CommPkgCopyFromType(comm_pkg);
   copy_to_type = hypre_CommPkgCopyToType(comm_pkg);

   for (b = 0; b < num_blocks; b++)
   {
      copy_fr_block = hypre_CommTypeBlock(copy_fr_type, b);
      copy_to_block = hypre_CommTypeBlock(copy_to_type, b);

      for (i = 0; i < hypre_CommBlockNumEntries(copy_fr_block); i++)
      {
         copy_fr_entry = hypre_CommBlockEntry(copy_fr_block, i);
         copy_to_entry = hypre_CommBlockEntry(copy_to_block, i);

         fr_dp = send_data[b] + hypre_CommEntryOffset(copy_fr_entry);
         to_dp = recv_data[b] + hypre_CommEntryOffset(copy_to_entry);

         /* copy data only when necessary */
         if (to_dp != fr_dp)
         {
            ndim = hypre_CommEntryNDim(copy_fr_entry);
            length_array = hypre_CommEntryLengthArray(copy_fr_entry);

            fr_stride_array = hypre_CommEntryStrideArray(copy_fr_entry);
            to_stride_array = hypre_CommEntryStrideArray(copy_to_entry);
            fr_imap = hypre_CommEntryIMap(copy_fr_entry);
            to_imap = hypre_CommEntryIMap(copy_to_entry);

            for (ll = 0; ll < length_array[ndim]; ll++)
            {
               if (fr_imap[ll] > -1)
               {
                  fr_dpl = fr_dp + (fr_imap[ll]) * fr_stride_array[ndim];
                  to_dpl = to_dp + (to_imap[ll]) * to_stride_array[ndim];

#define DEVICE_VAR is_device_ptr(to_dpl,fr_dpl)
                  hypre_BasicBoxLoop2Begin(ndim, length_array,
                                           fr_stride_array, fi,
                                           to_stride_array, ti);
                  {
                     if (action > 0)
                     {
                        /* add the data to existing values in memory */
                        to_dpl[ti] += fr_dpl[fi];
                     }
                     else
                     {
                        /* copy the data over existing values in memory */
                        to_dpl[ti] = fr_dpl[fi];
                     }
                  }
                  hypre_BoxLoop2End(fi, ti);
#undef DEVICE_VAR
               }
            }
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgDestroy( hypre_CommPkg *comm_pkg )
{
   hypre_CommType    *comm_type;
   hypre_CommBlock   *comm_block;
   HYPRE_Int          i, b, num_comms, num_blocks;
   hypre_Index       *recv_strides;
   hypre_BoxArray   **recv_data_spaces;
   HYPRE_Int        **recv_data_offsets;
   HYPRE_Int         *boxes_match;

   if (comm_pkg)
   {
      num_blocks        = hypre_CommPkgNumBlocks(comm_pkg);
      recv_strides      = hypre_CommPkgRecvStrides(comm_pkg);
      recv_data_spaces  = hypre_CommPkgRecvDataSpaces(comm_pkg);
      recv_data_offsets = hypre_CommPkgRecvDataOffsets(comm_pkg);
      boxes_match       = hypre_CommPkgBoxesMatch(comm_pkg);

      /* Send */
      num_comms = hypre_CommPkgNumSends(comm_pkg);
      for (i = 0; i < num_comms; i++)
      {
         comm_type = hypre_CommPkgSendType(comm_pkg, i);
         for (b = 0; b < num_blocks; b++)
         {
            comm_block = hypre_CommTypeBlock(comm_type, b);
            hypre_TFree(hypre_CommBlockEntries(comm_block), HYPRE_MEMORY_HOST);
            hypre_TFree(hypre_CommBlockIMaps(comm_block), HYPRE_MEMORY_HOST);
            hypre_TFree(hypre_CommBlockRemBoxnums(comm_block), HYPRE_MEMORY_HOST);
            hypre_TFree(hypre_CommBlockRemBoxes(comm_block), HYPRE_MEMORY_HOST);
            hypre_TFree(hypre_CommBlockRemOrders(comm_block), HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_CommTypeBlocks(comm_type), HYPRE_MEMORY_HOST);
      }

      /* Recv */
      num_comms = hypre_CommPkgNumRecvs(comm_pkg);
      for (i = 0; i < num_comms; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);

         /* This is only set up if a communication is done */
         if (!hypre_CommTypeFirstComm(comm_type))
         {
            for (b = 0; b < num_blocks; b++)
            {
               comm_block = hypre_CommTypeBlock(comm_type, b);
               hypre_TFree(hypre_CommBlockEntries(comm_block), HYPRE_MEMORY_HOST);
               hypre_TFree(hypre_CommBlockIMaps(comm_block), HYPRE_MEMORY_HOST);
            }
         }
         hypre_TFree(hypre_CommTypeBlocks(comm_type), HYPRE_MEMORY_HOST);
      }

      /* CopyFrom */
      comm_type = hypre_CommPkgCopyFromType(comm_pkg);
      for (b = 0; b < num_blocks; b++)
      {
         comm_block = hypre_CommTypeBlock(comm_type, b);
         hypre_TFree(hypre_CommBlockEntries(comm_block), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_CommBlockIMaps(comm_block), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_CommBlockRemBoxnums(comm_block), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_CommBlockRemBoxes(comm_block), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_CommBlockRemOrders(comm_block), HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_CommTypeBlocks(comm_type), HYPRE_MEMORY_HOST);
      hypre_TFree(comm_type, HYPRE_MEMORY_HOST);

      /* CopyTo */
      comm_type = hypre_CommPkgCopyToType(comm_pkg);
      for (b = 0; b < num_blocks; b++)
      {
         comm_block = hypre_CommTypeBlock(comm_type, b);
         hypre_TFree(hypre_CommBlockEntries(comm_block), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_CommBlockIMaps(comm_block), HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_CommTypeBlocks(comm_type), HYPRE_MEMORY_HOST);
      hypre_TFree(comm_type, HYPRE_MEMORY_HOST);

      for (b = 0; b < num_blocks; b++)
      {
         hypre_BoxArrayDestroy(recv_data_spaces[b]);
         hypre_TFree(recv_data_offsets[b], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(recv_strides, HYPRE_MEMORY_HOST);
      hypre_TFree(recv_data_spaces, HYPRE_MEMORY_HOST);
      hypre_TFree(recv_data_offsets, HYPRE_MEMORY_HOST);
      hypre_TFree(boxes_match, HYPRE_MEMORY_HOST);

      hypre_TFree(comm_pkg, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}
