/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
#endif

/* this computes a (large enough) size (in doubles) for the message prefix */
#define hypre_CommPrefixSize(ne)                                        \
   ( (((1+ne)*sizeof(HYPRE_Int) + ne*sizeof(hypre_Box))/sizeof(HYPRE_Complex)) + 1 )

/*--------------------------------------------------------------------------
 * Create a communication package.  A grid-based description of a communication
 * exchange is passed in.  This description is then compiled into an
 * intermediate processor-based description of the communication.  The
 * intermediate processor-based description is used directly to pack and unpack
 * buffers during the communications.
 *
 * The 'orders' argument is dimension 'num_transforms' x 'num_values' and should
 * have a one-to-one correspondence with the transform data in 'comm_info'.
 *
 * If 'reverse' is > 0, then the meaning of send/recv is reversed
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgCreate( hypre_CommInfo   *comm_info,
                     hypre_BoxArray   *send_data_space,
                     hypre_BoxArray   *recv_data_space,
                     HYPRE_Int         num_values,
                     HYPRE_Int       **orders,
                     HYPRE_Int         reverse,
                     MPI_Comm          comm,
                     hypre_CommPkg   **comm_pkg_ptr )
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
   HYPRE_Int           **cp_orders;

   hypre_CommPkg        *comm_pkg;
   hypre_CommType       *comm_types;
   hypre_CommType       *comm_type;
   hypre_CommEntryType  *ct_entries;
   HYPRE_Int            *ct_rem_boxnums;
   hypre_Box            *ct_rem_boxes;
   HYPRE_Int            *comm_boxes_p, *comm_boxes_i, *comm_boxes_j;
   HYPRE_Int             num_boxes, num_entries, num_comms, comm_bufsize;

   hypre_BoxArray       *box_array;
   hypre_Box            *box;
   hypre_BoxArray       *rbox_array;
   hypre_Box            *rbox;
   hypre_Box            *data_box;
   HYPRE_Int            *data_offsets;
   HYPRE_Int             data_offset;
   hypre_IndexRef        send_coord, send_dir;
   HYPRE_Int            *send_order;

   HYPRE_Int             i, j, k, p, m, size, p_old, my_proc;

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

   hypre_MPI_Comm_rank(comm, &my_proc );

   /*------------------------------------------------------
    * Set up various entries in CommPkg
    *------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1, HYPRE_MEMORY_HOST);

   hypre_CommPkgComm(comm_pkg)      = comm;
   hypre_CommPkgFirstComm(comm_pkg) = 1;
   hypre_CommPkgNDim(comm_pkg)      = ndim;
   hypre_CommPkgNumValues(comm_pkg) = num_values;
   hypre_CommPkgNumOrders(comm_pkg) = 0;
   hypre_CommPkgOrders(comm_pkg)    = NULL;
   if ( (send_transforms != NULL) && (orders != NULL) )
   {
      hypre_CommPkgNumOrders(comm_pkg) = num_transforms;
      cp_orders = hypre_TAlloc(HYPRE_Int *, num_transforms, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_transforms; i++)
      {
         cp_orders[i] = hypre_TAlloc(HYPRE_Int, num_values, HYPRE_MEMORY_HOST);
         for (j = 0; j < num_values; j++)
         {
            cp_orders[i][j] = orders[i][j];
         }
      }
      hypre_CommPkgOrders(comm_pkg) = cp_orders;
   }
   hypre_CopyIndex(send_stride, hypre_CommPkgSendStride(comm_pkg));
   hypre_CopyIndex(recv_stride, hypre_CommPkgRecvStride(comm_pkg));

   /* set identity transform and send_coord/dir/order if needed below */
   hypre_CommPkgIdentityOrder(comm_pkg) = hypre_TAlloc(HYPRE_Int, num_values, HYPRE_MEMORY_HOST);
   send_coord = hypre_CommPkgIdentityCoord(comm_pkg);
   send_dir   = hypre_CommPkgIdentityDir(comm_pkg);
   send_order = hypre_CommPkgIdentityOrder(comm_pkg);
   for (i = 0; i < ndim; i++)
   {
      hypre_IndexD(send_coord, i) = i;
      hypre_IndexD(send_dir, i) = 1;
   }
   for (i = 0; i < num_values; i++)
   {
      send_order[i] = i;
   }

   /*------------------------------------------------------
    * Set up send CommType information
    *------------------------------------------------------*/

   /* set data_offsets and compute num_boxes, num_entries */
   data_offsets = hypre_TAlloc(HYPRE_Int, hypre_BoxArraySize(send_data_space), HYPRE_MEMORY_HOST);
   data_offset = 0;
   num_boxes = 0;
   num_entries = 0;
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
         hypre_ForBoxI(j, box_array)
         {
            box = hypre_BoxArrayBox(box_array, j);
            if (hypre_BoxVolume(box) != 0)
            {
               num_entries++;
            }
         }
      }
   }

   /* set up comm_boxes_[pij] */
   comm_boxes_p = hypre_TAlloc(HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_i = hypre_TAlloc(HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_j = hypre_TAlloc(HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   num_boxes = 0;
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

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = hypre_CTAlloc(hypre_CommType, (num_boxes + 1), HYPRE_MEMORY_HOST);
   ct_entries = hypre_TAlloc(hypre_CommEntryType, num_entries, HYPRE_MEMORY_HOST);
   ct_rem_boxnums = hypre_TAlloc(HYPRE_Int, num_entries, HYPRE_MEMORY_HOST);
   ct_rem_boxes = hypre_TAlloc(hypre_Box, num_entries, HYPRE_MEMORY_HOST);
   hypre_CommPkgEntries(comm_pkg)    = ct_entries;
   hypre_CommPkgRemBoxnums(comm_pkg) = ct_rem_boxnums;
   hypre_CommPkgRemBoxes(comm_pkg)   = ct_rem_boxes;

   p_old = -1;
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
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms + 1];
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }
            hypre_CommTypeProc(comm_type)       = p;
            hypre_CommTypeBufsize(comm_type)    = 0;
            hypre_CommTypeNumEntries(comm_type) = 0;
            hypre_CommTypeEntries(comm_type)    = ct_entries;
            hypre_CommTypeRemBoxnums(comm_type) = ct_rem_boxnums;
            hypre_CommTypeRemBoxes(comm_type)   = ct_rem_boxes;
            p_old = p;
         }

         k = hypre_CommTypeNumEntries(comm_type);
         hypre_BoxGetStrideVolume(box, send_stride, &size);
         hypre_CommTypeBufsize(comm_type) += (size * num_values);
         comm_bufsize                     += (size * num_values);
         rbox_array = hypre_BoxArrayArrayBoxArray(send_rboxes, i);
         data_box = hypre_BoxArrayBox(send_data_space, i);
         if (send_transforms != NULL)
         {
            send_coord = coords[send_transforms[i][j]];
            send_dir   = dirs[send_transforms[i][j]];
            if (orders != NULL)
            {
               send_order = cp_orders[send_transforms[i][j]];
            }
         }
         hypre_CommTypeSetEntry(box, send_stride, send_coord, send_dir,
                                send_order, data_box, data_offsets[i],
                                hypre_CommTypeEntry(comm_type, k));
         hypre_CommTypeRemBoxnum(comm_type, k) = send_rboxnums[i][j];
         hypre_CopyBox(hypre_BoxArrayBox(rbox_array, j),
                       hypre_CommTypeRemBox(comm_type, k));
         hypre_CommTypeNumEntries(comm_type) ++;
         ct_entries     ++;
         ct_rem_boxnums ++;
         ct_rem_boxes   ++;
      }
   }

   /* add space for prefix info */
   for (m = 1; m < (num_comms + 1); m++)
   {
      comm_type = &comm_types[m];
      k = hypre_CommTypeNumEntries(comm_type);
      size = hypre_CommPrefixSize(k);
      hypre_CommTypeBufsize(comm_type) += size;
      comm_bufsize                     += size;
   }

   /* set send info in comm_pkg */
   comm_types = hypre_TReAlloc(comm_types, hypre_CommType, (num_comms + 1), HYPRE_MEMORY_HOST);
   hypre_CommPkgSendBufsize(comm_pkg)  = comm_bufsize;
   hypre_CommPkgNumSends(comm_pkg)     = num_comms;
   hypre_CommPkgSendTypes(comm_pkg)    = &comm_types[1];
   hypre_CommPkgCopyFromType(comm_pkg) = &comm_types[0];

   /* free up data_offsets */
   hypre_TFree(data_offsets, HYPRE_MEMORY_HOST);

   /*------------------------------------------------------
    * Set up recv CommType information
    *------------------------------------------------------*/

   /* set data_offsets and compute num_boxes */
   data_offsets = hypre_TAlloc(HYPRE_Int, hypre_BoxArraySize(recv_data_space), HYPRE_MEMORY_HOST);
   data_offset = 0;
   num_boxes = 0;
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
   hypre_CommPkgRecvDataOffsets(comm_pkg) = data_offsets;
   hypre_CommPkgRecvDataSpace(comm_pkg) = hypre_BoxArrayDuplicate(recv_data_space);

   /* set up comm_boxes_[pij] */
   comm_boxes_p = hypre_TReAlloc(comm_boxes_p, HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_i = hypre_TReAlloc(comm_boxes_i, HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   comm_boxes_j = hypre_TReAlloc(comm_boxes_j, HYPRE_Int, num_boxes, HYPRE_MEMORY_HOST);
   num_boxes = 0;
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

   p_old = -1;
   num_comms = 0;
   comm_bufsize = 0;
   comm_type = &comm_types[0];
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
               num_comms++;
            }
            else
            {
               comm_type = &comm_types[0];
            }
            hypre_CommTypeProc(comm_type)       = p;
            hypre_CommTypeBufsize(comm_type)    = 0;
            hypre_CommTypeNumEntries(comm_type) = 0;
            p_old = p;
         }

         k = hypre_CommTypeNumEntries(comm_type);
         hypre_BoxGetStrideVolume(box, recv_stride, &size);
         hypre_CommTypeBufsize(comm_type) += (size * num_values);
         comm_bufsize                     += (size * num_values);
         hypre_CommTypeNumEntries(comm_type) ++;
      }
   }

   /* add space for prefix info */
   for (m = 1; m < (num_comms + 1); m++)
   {
      comm_type = &comm_types[m];
      k = hypre_CommTypeNumEntries(comm_type);
      size = hypre_CommPrefixSize(k);
      hypre_CommTypeBufsize(comm_type) += size;
      comm_bufsize                     += size;
   }

   /* set recv info in comm_pkg */
   comm_types = hypre_TReAlloc(comm_types, hypre_CommType, (num_comms + 1), HYPRE_MEMORY_HOST);
   hypre_CommPkgRecvBufsize(comm_pkg) = comm_bufsize;
   hypre_CommPkgNumRecvs(comm_pkg)    = num_comms;
   hypre_CommPkgRecvTypes(comm_pkg)   = &comm_types[1];
   hypre_CommPkgCopyToType(comm_pkg)  = &comm_types[0];

   /* if CommInfo send/recv boxes don't match, compute a max bufsize */
   if ( !hypre_CommInfoBoxesMatch(comm_info) )
   {
      hypre_CommPkgRecvBufsize(comm_pkg) = 0;
      for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);

         /* subtract off old (incorrect) prefix size */
         num_entries = hypre_CommTypeNumEntries(comm_type);
         hypre_CommTypeBufsize(comm_type) -= hypre_CommPrefixSize(num_entries);

         /* set num_entries to number of grid points and add new prefix size */
         num_entries = hypre_CommTypeBufsize(comm_type);
         hypre_CommTypeNumEntries(comm_type) = num_entries;
         size = hypre_CommPrefixSize(num_entries);
         hypre_CommTypeBufsize(comm_type) += size;
         hypre_CommPkgRecvBufsize(comm_pkg) += hypre_CommTypeBufsize(comm_type);
      }
   }

   hypre_CommPkgSendBufsizeFirstComm(comm_pkg) = hypre_CommPkgSendBufsize(comm_pkg);
   hypre_CommPkgRecvBufsizeFirstComm(comm_pkg) = hypre_CommPkgRecvBufsize(comm_pkg);

   /*------------------------------------------------------
    * Debugging stuff - ONLY WORKS FOR 3D
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
      hypre_CommEntryType  *comm_entry;
      HYPRE_Int             offset, dim;
      HYPRE_Int            *length;
      HYPRE_Int            *stride;

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
            offset = hypre_CommEntryTypeOffset(comm_entry);
            dim    = hypre_CommEntryTypeDim(comm_entry);
            length = hypre_CommEntryTypeLengthArray(comm_entry);
            stride = hypre_CommEntryTypeStrideArray(comm_entry);
            hypre_fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                          i, offset, dim,
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
         offset = hypre_CommEntryTypeOffset(comm_entry);
         dim    = hypre_CommEntryTypeDim(comm_entry);
         length = hypre_CommEntryTypeLengthArray(comm_entry);
         stride = hypre_CommEntryTypeStrideArray(comm_entry);
         hypre_fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                       i, offset, dim,
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
 * Note that this routine assumes an identity coordinate transform
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommTypeSetEntries( hypre_CommType  *comm_type,
                          HYPRE_Int       *boxnums,
                          hypre_Box       *boxes,
                          hypre_Index      stride,
                          hypre_Index      coord,
                          hypre_Index      dir,
                          HYPRE_Int       *order,
                          hypre_BoxArray  *data_space,
                          HYPRE_Int       *data_offsets )
{
   HYPRE_Int             num_entries = hypre_CommTypeNumEntries(comm_type);
   hypre_CommEntryType  *entries     = hypre_CommTypeEntries(comm_type);
   hypre_Box            *box;
   hypre_Box            *data_box;
   HYPRE_Int             i, j;

   for (j = 0; j < num_entries; j++)
   {
      i = boxnums[j];
      box = &boxes[j];
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_CommTypeSetEntry(box, stride, coord, dir, order,
                             data_box, data_offsets[i], &entries[j]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommTypeSetEntry( hypre_Box           *box,
                        hypre_Index          stride,
                        hypre_Index          coord,
                        hypre_Index          dir,
                        HYPRE_Int           *order,
                        hypre_Box           *data_box,
                        HYPRE_Int            data_box_offset,
                        hypre_CommEntryType *comm_entry )
{
   HYPRE_Int     dim, ndim = hypre_BoxNDim(box);
   HYPRE_Int     offset;
   HYPRE_Int    *length_array, tmp_length_array[HYPRE_MAXDIM];
   HYPRE_Int    *stride_array, tmp_stride_array[HYPRE_MAXDIM];
   hypre_Index   size;
   HYPRE_Int     i, j;

   length_array = hypre_CommEntryTypeLengthArray(comm_entry);
   stride_array = hypre_CommEntryTypeStrideArray(comm_entry);

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
   while (i < dim)
   {
      if (length_array[i] == 1)
      {
         for (j = i; j < (dim - 1); j++)
         {
            length_array[j] = length_array[j + 1];
            stride_array[j] = stride_array[j + 1];
         }
         length_array[dim - 1] = 1;
         stride_array[dim - 1] = 1;
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

   /* if every len was 1 we need to fix to communicate at least one */
   if (!dim)
   {
      dim = 1;
   }

   hypre_CommEntryTypeOffset(comm_entry) = offset;
   hypre_CommEntryTypeDim(comm_entry) = dim;
   hypre_CommEntryTypeOrder(comm_entry) = order;

   return hypre_error_flag;
}

HYPRE_Complex *
hypre_StructCommunicationGetBuffer(HYPRE_MemoryLocation memory_location,
                                   HYPRE_Int            size)
{
   HYPRE_Complex *ptr;

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      if (size > hypre_HandleStructCommSendBufferSize(hypre_handle()))
      {
         HYPRE_Int new_size = 5 * size;
         hypre_HandleStructCommSendBufferSize(hypre_handle()) = new_size;
         hypre_TFree(hypre_HandleStructCommSendBuffer(hypre_handle()), memory_location);
         hypre_HandleStructCommSendBuffer(hypre_handle()) = hypre_CTAlloc(HYPRE_Complex, new_size,
                                                                          memory_location);
      }

      ptr = hypre_HandleStructCommSendBuffer(hypre_handle());
   }
   else
#endif
   {
      ptr = hypre_CTAlloc(HYPRE_Complex, size, memory_location);
   }

   return ptr;
}

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
hypre_InitializeCommunication( hypre_CommPkg        *comm_pkg,
                               HYPRE_Complex        *send_data,
                               HYPRE_Complex        *recv_data,
                               HYPRE_Int             action,
                               HYPRE_Int             tag,
                               hypre_CommHandle    **comm_handle_ptr )
{
   hypre_CommHandle    *comm_handle;

   HYPRE_Int            ndim       = hypre_CommPkgNDim(comm_pkg);
   HYPRE_Int            num_values = hypre_CommPkgNumValues(comm_pkg);
   HYPRE_Int            num_sends  = hypre_CommPkgNumSends(comm_pkg);
   HYPRE_Int            num_recvs  = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm       = hypre_CommPkgComm(comm_pkg);

   HYPRE_Int            num_requests;
   hypre_MPI_Request   *requests;
   hypre_MPI_Status    *status;

   HYPRE_Complex      **send_buffers;
   HYPRE_Complex      **recv_buffers;
   HYPRE_Complex      **send_buffers_mpi;
   HYPRE_Complex      **recv_buffers_mpi;

   hypre_CommType      *comm_type, *from_type, *to_type;
   hypre_CommEntryType *comm_entry;
   HYPRE_Int            num_entries;

   HYPRE_Int           *length_array;
   HYPRE_Int           *stride_array, unitst_array[HYPRE_MAXDIM + 1];
   HYPRE_Int           *order;

   HYPRE_Complex       *dptr, *kptr, *lptr;
   HYPRE_Int           *qptr;

   HYPRE_Int            i, j, d, ll;
   HYPRE_Int            size;

   HYPRE_MemoryLocation memory_location     = hypre_HandleMemoryLocation(hypre_handle());
   HYPRE_MemoryLocation memory_location_mpi = memory_location;

   /*--------------------------------------------------------------------
    * allocate requests and status
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(hypre_MPI_Request, num_requests, HYPRE_MEMORY_HOST);
   status = hypre_CTAlloc(hypre_MPI_Status, num_requests, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------
    * allocate buffers
    *--------------------------------------------------------------------*/

   /* allocate send buffers */
   send_buffers = hypre_TAlloc(HYPRE_Complex *, num_sends, HYPRE_MEMORY_HOST);
   if (num_sends > 0)
   {
      size = hypre_CommPkgSendBufsize(comm_pkg);
      send_buffers[0] = hypre_StructCommunicationGetBuffer(memory_location, size);
      for (i = 1; i < num_sends; i++)
      {
         comm_type = hypre_CommPkgSendType(comm_pkg, i - 1);
         size = hypre_CommTypeBufsize(comm_type);
         send_buffers[i] = send_buffers[i - 1] + size;
      }
   }

   /* allocate recv buffers */
   recv_buffers = hypre_TAlloc(HYPRE_Complex *, num_recvs, HYPRE_MEMORY_HOST);
   if (num_recvs > 0)
   {
      size = hypre_CommPkgRecvBufsize(comm_pkg);
      recv_buffers[0] = hypre_StructCommunicationGetBuffer(memory_location, size);
      for (i = 1; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i - 1);
         size = hypre_CommTypeBufsize(comm_type);
         recv_buffers[i] = recv_buffers[i - 1] + size;
      }
   }

   /*--------------------------------------------------------------------
    * pack send buffers
    *--------------------------------------------------------------------*/

   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      num_entries = hypre_CommTypeNumEntries(comm_type);

      dptr = (HYPRE_Complex *) send_buffers[i];
      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         dptr += hypre_CommPrefixSize(num_entries);
      }

      for (j = 0; j < num_entries; j++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, j);
         length_array = hypre_CommEntryTypeLengthArray(comm_entry);
         stride_array = hypre_CommEntryTypeStrideArray(comm_entry);
         order = hypre_CommEntryTypeOrder(comm_entry);
         unitst_array[0] = 1;
         for (d = 1; d <= ndim; d++)
         {
            unitst_array[d] = unitst_array[d - 1] * length_array[d - 1];
         }

         lptr = send_data + hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < num_values; ll++)
         {
            if (order[ll] > -1)
            {
               kptr = lptr + order[ll] * stride_array[ndim];

#define DEVICE_VAR is_device_ptr(dptr,kptr)
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

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      if (hypre_GetGpuAwareMPI())
      {
#if defined(HYPRE_USING_GPU)
         hypre_ForceSyncComputeStream(hypre_handle());
#endif
         send_buffers_mpi = send_buffers;
         recv_buffers_mpi = recv_buffers;
      }
      else
      {
         memory_location_mpi = HYPRE_MEMORY_HOST;

         send_buffers_mpi = hypre_TAlloc(HYPRE_Complex *, num_sends, HYPRE_MEMORY_HOST);
         if (num_sends > 0)
         {
            size = hypre_CommPkgSendBufsize(comm_pkg);
            send_buffers_mpi[0] = hypre_CTAlloc(HYPRE_Complex, size, memory_location_mpi);
            for (i = 1; i < num_sends; i++)
            {
               send_buffers_mpi[i] = send_buffers_mpi[i - 1] + (send_buffers[i] - send_buffers[i - 1]);
            }
            hypre_TMemcpy(send_buffers_mpi[0], send_buffers[0], HYPRE_Complex, size, HYPRE_MEMORY_HOST,
                          memory_location);
         }

         recv_buffers_mpi = hypre_TAlloc(HYPRE_Complex *, num_recvs, HYPRE_MEMORY_HOST);
         if (num_recvs > 0)
         {
            size = hypre_CommPkgRecvBufsize(comm_pkg);
            recv_buffers_mpi[0] = hypre_CTAlloc(HYPRE_Complex, size, memory_location_mpi);
            for (i = 1; i < num_recvs; i++)
            {
               recv_buffers_mpi[i] = recv_buffers_mpi[i - 1] + (recv_buffers[i] - recv_buffers[i - 1]);
            }
         }
      }
   }
   else
#endif
   {
      send_buffers_mpi = send_buffers;
      recv_buffers_mpi = recv_buffers;
   }

   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      num_entries = hypre_CommTypeNumEntries(comm_type);

      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         qptr = (HYPRE_Int *) send_buffers_mpi[i];
         hypre_TMemcpy(qptr, &num_entries,
                       HYPRE_Int, 1, memory_location_mpi, HYPRE_MEMORY_HOST);
         qptr ++;
         hypre_TMemcpy(qptr, hypre_CommTypeRemBoxnums(comm_type),
                       HYPRE_Int, num_entries, memory_location_mpi, HYPRE_MEMORY_HOST);
         qptr += num_entries;
         hypre_TMemcpy(qptr, hypre_CommTypeRemBoxes(comm_type),
                       hypre_Box, num_entries, memory_location_mpi, HYPRE_MEMORY_HOST);
         hypre_CommTypeRemBoxnums(comm_type) = NULL;
         hypre_CommTypeRemBoxes(comm_type) = NULL;
      }
   }

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   j = 0;
   for (i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      hypre_MPI_Irecv(recv_buffers_mpi[i],
                      hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex),
                      hypre_MPI_BYTE, hypre_CommTypeProc(comm_type),
                      tag, comm, &requests[j++]);
      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
         hypre_CommTypeBufsize(comm_type)   -= size;
         hypre_CommPkgRecvBufsize(comm_pkg) -= size;
      }
   }

   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      hypre_MPI_Isend(send_buffers_mpi[i],
                      hypre_CommTypeBufsize(comm_type)*sizeof(HYPRE_Complex),
                      hypre_MPI_BYTE, hypre_CommTypeProc(comm_type),
                      tag, comm, &requests[j++]);
      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
         hypre_CommTypeBufsize(comm_type)   -= size;
         hypre_CommPkgSendBufsize(comm_pkg) -= size;
      }
   }

   /*--------------------------------------------------------------------
    * set up CopyToType and exchange local data
    *--------------------------------------------------------------------*/

   if ( hypre_CommPkgFirstComm(comm_pkg) )
   {
      from_type = hypre_CommPkgCopyFromType(comm_pkg);
      to_type   = hypre_CommPkgCopyToType(comm_pkg);
      num_entries = hypre_CommTypeNumEntries(from_type);
      hypre_CommTypeNumEntries(to_type) = num_entries;
      hypre_CommTypeEntries(to_type) =
         hypre_TAlloc(hypre_CommEntryType, num_entries, HYPRE_MEMORY_HOST);
      hypre_CommTypeSetEntries(to_type,
                               hypre_CommTypeRemBoxnums(from_type),
                               hypre_CommTypeRemBoxes(from_type),
                               hypre_CommPkgRecvStride(comm_pkg),
                               hypre_CommPkgIdentityCoord(comm_pkg),
                               hypre_CommPkgIdentityDir(comm_pkg),
                               hypre_CommPkgIdentityOrder(comm_pkg),
                               hypre_CommPkgRecvDataSpace(comm_pkg),
                               hypre_CommPkgRecvDataOffsets(comm_pkg));
      hypre_TFree(hypre_CommPkgRemBoxnums(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommPkgRemBoxes(comm_pkg), HYPRE_MEMORY_HOST);
   }

   hypre_ExchangeLocalData(comm_pkg, send_data, recv_data, action);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_TAlloc(hypre_CommHandle, 1, HYPRE_MEMORY_HOST);

   hypre_CommHandleCommPkg(comm_handle)        = comm_pkg;
   hypre_CommHandleSendData(comm_handle)       = send_data;
   hypre_CommHandleRecvData(comm_handle)       = recv_data;
   hypre_CommHandleNumRequests(comm_handle)    = num_requests;
   hypre_CommHandleRequests(comm_handle)       = requests;
   hypre_CommHandleStatus(comm_handle)         = status;
   hypre_CommHandleSendBuffers(comm_handle)    = send_buffers;
   hypre_CommHandleRecvBuffers(comm_handle)    = recv_buffers;
   hypre_CommHandleAction(comm_handle)         = action;
   hypre_CommHandleSendBuffersMPI(comm_handle) = send_buffers_mpi;
   hypre_CommHandleRecvBuffersMPI(comm_handle) = recv_buffers_mpi;

   *comm_handle_ptr = comm_handle;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Finalize a communication exchange.  This routine blocks until all
 * of the communication requests are completed.
 *
 * The communication requests are completed, and the receive buffer is
 * manually unpacked.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   hypre_CommPkg       *comm_pkg         = hypre_CommHandleCommPkg(comm_handle);
   HYPRE_Complex      **send_buffers     = hypre_CommHandleSendBuffers(comm_handle);
   HYPRE_Complex      **recv_buffers     = hypre_CommHandleRecvBuffers(comm_handle);
   HYPRE_Complex      **send_buffers_mpi = hypre_CommHandleSendBuffersMPI(comm_handle);
   HYPRE_Complex      **recv_buffers_mpi = hypre_CommHandleRecvBuffersMPI(comm_handle);
   HYPRE_Int            action           = hypre_CommHandleAction(comm_handle);

   HYPRE_Int            ndim         = hypre_CommPkgNDim(comm_pkg);
   HYPRE_Int            num_values   = hypre_CommPkgNumValues(comm_pkg);
   HYPRE_Int            num_sends    = hypre_CommPkgNumSends(comm_pkg);
   HYPRE_Int            num_recvs    = hypre_CommPkgNumRecvs(comm_pkg);

   hypre_CommType      *comm_type;
   hypre_CommEntryType *comm_entry;
   HYPRE_Int            num_entries;

   HYPRE_Int           *length_array;
   HYPRE_Int           *stride_array, unitst_array[HYPRE_MAXDIM + 1];

   HYPRE_Complex       *kptr, *lptr;
   HYPRE_Complex       *dptr;
   HYPRE_Int           *qptr;

   HYPRE_Int           *boxnums;
   hypre_Box           *boxes;

   HYPRE_Int            i, j, d, ll;

   HYPRE_MemoryLocation memory_location     = hypre_HandleMemoryLocation(hypre_handle());
   HYPRE_MemoryLocation memory_location_mpi = memory_location;

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   if (!hypre_GetGpuAwareMPI())
   {
      memory_location_mpi = HYPRE_MEMORY_HOST;
   }
#endif

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
    * if FirstComm, unpack prefix information and set 'num_entries' and
    * 'entries' for RecvType
    *--------------------------------------------------------------------*/

   if ( hypre_CommPkgFirstComm(comm_pkg) )
   {
      hypre_CommEntryType *ct_entries;

      num_entries = 0;
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);

         qptr = (HYPRE_Int *) recv_buffers_mpi[i];

         hypre_TMemcpy(&hypre_CommTypeNumEntries(comm_type), qptr,
                       HYPRE_Int, 1, HYPRE_MEMORY_HOST, memory_location_mpi);

         num_entries += hypre_CommTypeNumEntries(comm_type);
      }

      /* allocate CommType entries 'ct_entries' */
      ct_entries = hypre_TAlloc(hypre_CommEntryType, num_entries, HYPRE_MEMORY_HOST);

      /* unpack prefix information and set RecvType entries */
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);
         hypre_CommTypeEntries(comm_type) = ct_entries;
         ct_entries += hypre_CommTypeNumEntries(comm_type);

         qptr = (HYPRE_Int *) recv_buffers_mpi[i];
         //num_entries = *qptr;
         num_entries = hypre_CommTypeNumEntries(comm_type);
         qptr ++;
         boxnums = qptr;
         qptr += num_entries;
         boxes = (hypre_Box *) qptr;
         //TODO boxnums
         hypre_CommTypeSetEntries(comm_type, boxnums, boxes,
                                  hypre_CommPkgRecvStride(comm_pkg),
                                  hypre_CommPkgIdentityCoord(comm_pkg),
                                  hypre_CommPkgIdentityDir(comm_pkg),
                                  hypre_CommPkgIdentityOrder(comm_pkg),
                                  hypre_CommPkgRecvDataSpace(comm_pkg),
                                  hypre_CommPkgRecvDataOffsets(comm_pkg));
      }
   }

   /*--------------------------------------------------------------------
    * unpack receive buffer data
    *--------------------------------------------------------------------*/

   /* Note: hypre_CommPkgRecvBufsize is different in the first comm */
   if (recv_buffers != recv_buffers_mpi)
   {
      if (num_recvs > 0)
      {
         HYPRE_Int recv_buf_size;

         recv_buf_size = hypre_CommPkgFirstComm(comm_pkg) ? hypre_CommPkgRecvBufsizeFirstComm(comm_pkg) :
                         hypre_CommPkgRecvBufsize(comm_pkg);

         hypre_TMemcpy(recv_buffers[0], recv_buffers_mpi[0], HYPRE_Complex, recv_buf_size,
                       memory_location, memory_location_mpi);
      }
   }

   for (i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      num_entries = hypre_CommTypeNumEntries(comm_type);

      dptr = (HYPRE_Complex *) recv_buffers[i];

      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         dptr += hypre_CommPrefixSize(num_entries);
      }

      for (j = 0; j < num_entries; j++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, j);
         length_array = hypre_CommEntryTypeLengthArray(comm_entry);
         stride_array = hypre_CommEntryTypeStrideArray(comm_entry);
         unitst_array[0] = 1;
         for (d = 1; d <= ndim; d++)
         {
            unitst_array[d] = unitst_array[d - 1] * length_array[d - 1];
         }

         lptr = hypre_CommHandleRecvData(comm_handle) +
                hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < num_values; ll++)
         {
            kptr = lptr + ll * stride_array[ndim];

#define DEVICE_VAR is_device_ptr(kptr,dptr)
            hypre_BasicBoxLoop2Begin(ndim, length_array,
                                     stride_array, ki,
                                     unitst_array, di);
            {
               if (action > 0)
               {
                  kptr[ki] += dptr[di];
               }
               else
               {
                  kptr[ki] = dptr[di];
               }
            }
            hypre_BoxLoop2End(ki, di);
#undef DEVICE_VAR

            dptr += unitst_array[ndim];
         }
      }
   }

   /*--------------------------------------------------------------------
    * turn off first communication indicator
    *--------------------------------------------------------------------*/

   hypre_CommPkgFirstComm(comm_pkg) = 0;

   /*--------------------------------------------------------------------
    * Free up communication handle
    *--------------------------------------------------------------------*/

   hypre_TFree(hypre_CommHandleRequests(comm_handle), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_CommHandleStatus(comm_handle), HYPRE_MEMORY_HOST);
   if (num_sends > 0)
   {
      hypre_StructCommunicationReleaseBuffer(send_buffers[0], memory_location);
   }
   if (num_recvs > 0)
   {
      hypre_StructCommunicationReleaseBuffer(recv_buffers[0], memory_location);
   }

   hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);

   if (send_buffers != send_buffers_mpi)
   {
      hypre_TFree(send_buffers_mpi[0], memory_location_mpi);
      hypre_TFree(send_buffers_mpi, HYPRE_MEMORY_HOST);
   }
   if (recv_buffers != recv_buffers_mpi)
   {
      hypre_TFree(recv_buffers_mpi[0], memory_location_mpi);
      hypre_TFree(recv_buffers_mpi, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Execute local data exchanges.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ExchangeLocalData( hypre_CommPkg *comm_pkg,
                         HYPRE_Complex *send_data,
                         HYPRE_Complex *recv_data,
                         HYPRE_Int      action )
{
   HYPRE_Int            ndim       = hypre_CommPkgNDim(comm_pkg);
   HYPRE_Int            num_values = hypre_CommPkgNumValues(comm_pkg);
   hypre_CommType      *copy_fr_type;
   hypre_CommType      *copy_to_type;
   hypre_CommEntryType *copy_fr_entry;
   hypre_CommEntryType *copy_to_entry;

   HYPRE_Complex       *fr_dp;
   HYPRE_Int           *fr_stride_array;
   HYPRE_Complex       *to_dp;
   HYPRE_Int           *to_stride_array;
   HYPRE_Complex       *fr_dpl, *to_dpl;

   HYPRE_Int           *length_array;
   HYPRE_Int            i, ll;

   HYPRE_Int           *order;

   /*--------------------------------------------------------------------
    * copy local data
    *--------------------------------------------------------------------*/

   copy_fr_type = hypre_CommPkgCopyFromType(comm_pkg);
   copy_to_type = hypre_CommPkgCopyToType(comm_pkg);

   for (i = 0; i < hypre_CommTypeNumEntries(copy_fr_type); i++)
   {
      copy_fr_entry = hypre_CommTypeEntry(copy_fr_type, i);
      copy_to_entry = hypre_CommTypeEntry(copy_to_type, i);

      fr_dp = send_data + hypre_CommEntryTypeOffset(copy_fr_entry);
      to_dp = recv_data + hypre_CommEntryTypeOffset(copy_to_entry);

      /* copy data only when necessary */
      if (to_dp != fr_dp)
      {
         length_array = hypre_CommEntryTypeLengthArray(copy_fr_entry);

         fr_stride_array = hypre_CommEntryTypeStrideArray(copy_fr_entry);
         to_stride_array = hypre_CommEntryTypeStrideArray(copy_to_entry);
         order = hypre_CommEntryTypeOrder(copy_fr_entry);

         for (ll = 0; ll < num_values; ll++)
         {
            if (order[ll] > -1)
            {
               fr_dpl = fr_dp + (order[ll]) * fr_stride_array[ndim];
               to_dpl = to_dp + (      ll ) * to_stride_array[ndim];

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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CommPkgDestroy( hypre_CommPkg *comm_pkg )
{
   hypre_CommType  *comm_type;
   HYPRE_Int      **orders;
   HYPRE_Int        i;

   if (comm_pkg)
   {
      /* note that entries are allocated in two stages for To/Recv */
      if (hypre_CommPkgNumRecvs(comm_pkg) > 0)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, 0);
         hypre_TFree(hypre_CommTypeEntries(comm_type), HYPRE_MEMORY_HOST);
      }
      comm_type = hypre_CommPkgCopyToType(comm_pkg);
      hypre_TFree(hypre_CommTypeEntries(comm_type), HYPRE_MEMORY_HOST);
      hypre_TFree(comm_type, HYPRE_MEMORY_HOST);

      comm_type = hypre_CommPkgCopyFromType(comm_pkg);
      hypre_TFree(comm_type, HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_CommPkgEntries(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommPkgRemBoxnums(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_CommPkgRemBoxes(comm_pkg), HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_CommPkgRecvDataOffsets(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_BoxArrayDestroy(hypre_CommPkgRecvDataSpace(comm_pkg));

      orders = hypre_CommPkgOrders(comm_pkg);
      for (i = 0; i < hypre_CommPkgNumOrders(comm_pkg); i++)
      {
         hypre_TFree(orders[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(orders, HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_CommPkgIdentityOrder(comm_pkg), HYPRE_MEMORY_HOST);

      hypre_TFree(comm_pkg, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}
