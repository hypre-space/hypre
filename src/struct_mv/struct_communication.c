/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.21 $
 ***********************************************************************EHEADER*/

#include "headers.h"

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
#endif

/* this computes a (large enough) size (in doubles) for the message prefix */
#define hypre_CommPrefixSize(ne)                                        \
   ( (((1+ne)*sizeof(HYPRE_Int) + ne*sizeof(hypre_Box))/sizeof(double)) + 1 )

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
   hypre_BoxArrayArray  *send_boxes;
   hypre_BoxArrayArray  *recv_boxes;
   hypre_IndexRef        send_stride;
   hypre_IndexRef        recv_stride;
   HYPRE_Int           **send_processes;
   HYPRE_Int           **recv_processes;
   HYPRE_Int           **send_rboxnums;
   hypre_BoxArrayArray  *send_rboxes;

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
      send_transforms = hypre_CommInfoSendTransforms(comm_info); /* may be NULL */
   }
   num_transforms = hypre_CommInfoNumTransforms(comm_info);
   coords         = hypre_CommInfoCoords(comm_info); /* may be NULL */
   dirs           = hypre_CommInfoDirs(comm_info);   /* may be NULL */

   hypre_MPI_Comm_rank(comm, &my_proc );

   /*------------------------------------------------------
    * Set up various entries in CommPkg
    *------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1);

   hypre_CommPkgComm(comm_pkg)      = comm;
   hypre_CommPkgFirstComm(comm_pkg) = 1;
   hypre_CommPkgNumValues(comm_pkg) = num_values;
   hypre_CommPkgNumOrders(comm_pkg) = 0;
   hypre_CommPkgOrders(comm_pkg)    = NULL;
   if ( (send_transforms != NULL) && (orders != NULL) )
   {
      hypre_CommPkgNumOrders(comm_pkg) = num_transforms;
      cp_orders = hypre_TAlloc(HYPRE_Int *, num_transforms);
      for (i = 0; i < num_transforms; i++)
      {
         cp_orders[i] = hypre_TAlloc(HYPRE_Int, num_values);
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
   hypre_CommPkgIdentityOrder(comm_pkg) = hypre_TAlloc(HYPRE_Int, num_values);
   send_coord = hypre_CommPkgIdentityCoord(comm_pkg);
   send_dir   = hypre_CommPkgIdentityDir(comm_pkg);
   send_order = hypre_CommPkgIdentityOrder(comm_pkg);
   for (i = 0; i < 3; i++)
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
   data_offsets = hypre_TAlloc(HYPRE_Int, hypre_BoxArraySize(send_data_space));
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
   comm_boxes_p = hypre_TAlloc(HYPRE_Int, num_boxes);
   comm_boxes_i = hypre_TAlloc(HYPRE_Int, num_boxes);
   comm_boxes_j = hypre_TAlloc(HYPRE_Int, num_boxes);
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
   hypre_qsort3i(comm_boxes_p, comm_boxes_i, comm_boxes_j, 0, num_boxes-1);

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = hypre_CTAlloc(hypre_CommType, (num_boxes + 1));
   ct_entries = hypre_TAlloc(hypre_CommEntryType, num_entries);
   ct_rem_boxnums = hypre_TAlloc(HYPRE_Int, num_entries);
   ct_rem_boxes = hypre_TAlloc(hypre_Box, num_entries);
   hypre_CommPkgEntries(comm_pkg)    = ct_entries;
   hypre_CommPkgRemBoxnums(comm_pkg) = ct_rem_boxnums;
   hypre_CommPkgRemBoxes(comm_pkg)   = ct_rem_boxes;

   p_old = -1;
   num_comms = 0;
   comm_bufsize = 0;
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
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms+1];
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
         hypre_CommTypeBufsize(comm_type) += (size*num_values);
         comm_bufsize                     += (size*num_values);
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
   comm_types = hypre_TReAlloc(comm_types, hypre_CommType, (num_comms + 1));
   hypre_CommPkgSendBufsize(comm_pkg)  = comm_bufsize;
   hypre_CommPkgNumSends(comm_pkg)     = num_comms;
   hypre_CommPkgSendTypes(comm_pkg)    = &comm_types[1];
   hypre_CommPkgCopyFromType(comm_pkg) = &comm_types[0];

   /* free up data_offsets */
   hypre_TFree(data_offsets);

   /*------------------------------------------------------
    * Set up recv CommType information
    *------------------------------------------------------*/

   /* set data_offsets and compute num_boxes */
   data_offsets = hypre_TAlloc(HYPRE_Int, hypre_BoxArraySize(recv_data_space));
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
   comm_boxes_p = hypre_TReAlloc(comm_boxes_p, HYPRE_Int, num_boxes);
   comm_boxes_i = hypre_TReAlloc(comm_boxes_i, HYPRE_Int, num_boxes);
   comm_boxes_j = hypre_TReAlloc(comm_boxes_j, HYPRE_Int, num_boxes);
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
   hypre_qsort3i(comm_boxes_p, comm_boxes_i, comm_boxes_j, 0, num_boxes-1);

   /* compute comm_types */

   /* make sure there is at least 1 comm_type allocated */
   comm_types = hypre_CTAlloc(hypre_CommType, (num_boxes + 1));

   p_old = -1;
   num_comms = 0;
   comm_bufsize = 0;
   for (m = 0; m < num_boxes; m++)
   {
      i = comm_boxes_i[m];
      j = comm_boxes_j[m];
      box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      box = hypre_BoxArrayBox(box_array, j);

      if (hypre_BoxVolume(box) != 0)
      {
         p = comm_boxes_p[m];

         /* start a new comm_type */
         if (p != p_old)
         {
            if (p != my_proc)
            {
               comm_type = &comm_types[num_comms+1];
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
         hypre_CommTypeBufsize(comm_type) += (size*num_values);
         comm_bufsize                     += (size*num_values);
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
   comm_types = hypre_TReAlloc(comm_types, hypre_CommType, (num_comms + 1));
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

   /*------------------------------------------------------
    * Debugging stuff
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
                    hypre_IndexX(send_stride),
                    hypre_IndexY(send_stride),
                    hypre_IndexZ(send_stride));
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
                          hypre_BoxIMinX(box),
                          hypre_BoxIMinY(box),
                          hypre_BoxIMinZ(box),
                          hypre_BoxIMaxX(box),
                          hypre_BoxIMaxY(box),
                          hypre_BoxIMaxZ(box));
            hypre_fprintf(file, "(%d,%d): %d,%d\n",
                          i, j, send_processes[i][j], send_rboxnums[i][j]);
         }
      }

      hypre_fprintf(file, "\n\n============================\n\n");
      hypre_fprintf(file, "RECV boxes:\n\n");

      hypre_fprintf(file, "Stride = (%d,%d,%d)\n",
                    hypre_IndexX(recv_stride),
                    hypre_IndexY(recv_stride),
                    hypre_IndexZ(recv_stride));
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
                          hypre_BoxIMinX(box),
                          hypre_BoxIMinY(box),
                          hypre_BoxIMinZ(box),
                          hypre_BoxIMaxX(box),
                          hypre_BoxIMaxY(box),
                          hypre_BoxIMaxZ(box));
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

   hypre_TFree(comm_boxes_p);
   hypre_TFree(comm_boxes_i);
   hypre_TFree(comm_boxes_j);

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
   HYPRE_Int     offset;
   HYPRE_Int     dim;
   HYPRE_Int    *length_array, tmp_length_array[3];
   HYPRE_Int    *stride_array, tmp_stride_array[3];
   hypre_Index   size;
   HYPRE_Int     i, j;

   length_array = hypre_CommEntryTypeLengthArray(comm_entry);
   stride_array = hypre_CommEntryTypeStrideArray(comm_entry);

   /* initialize offset */
   offset = data_box_offset + hypre_BoxIndexRank(data_box, hypre_BoxIMin(box));

   /* initialize length_array and stride_array */
   hypre_BoxGetStrideSize(box, stride, size);
   for (i = 0; i < 3; i++)
   {
      length_array[i] = hypre_IndexD(size, i);
      stride_array[i] = hypre_IndexD(stride, i);
      for (j = 0; j < i; j++)
      {
         stride_array[i] *= hypre_BoxSizeD(data_box, j);
      }
   }
   stride_array[3] = hypre_BoxVolume(data_box);

   /* make adjustments for dir */
   for (i = 0; i < 3; i++)
   {
      if (dir[i] < 0)
      {
         offset += (length_array[i] - 1)*stride_array[i];
         stride_array[i] = -stride_array[i];
      }
   }

   /* make adjustments for coord */
   for (i = 0; i < 3; i++)
   {
      tmp_length_array[i] = length_array[i];
      tmp_stride_array[i] = stride_array[i];
   }
   for (i = 0; i < 3; i++)
   {
      j = coord[i];
      length_array[j] = tmp_length_array[i];
      stride_array[j] = tmp_stride_array[i];
   }

   /* eliminate dimensions with length_array = 1 */
   dim = 3;
   i = 0;
   while (i < dim)
   {
      if(length_array[i] == 1)
      {
         for(j = i; j < (dim - 1); j++)
         {
            length_array[j] = length_array[j+1];
            stride_array[j] = stride_array[j+1];
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
   for (i = (dim-1); i > 0; i--)
   {
      for (j = 0; j < i; j++)
      {
         if (length_array[j] < length_array[j+1])
         {
            i_tmp             = length_array[j];
            length_array[j]   = length_array[j+1];
            length_array[j+1] = i_tmp;

            i_tmp             = stride_array[j];
            stride_array[j]   = stride_array[j+1];
            stride_array[j+1] = i_tmp;
         }
      }
   }
#endif

   /* if every len was 1 we need to fix to communicate at least one */
   if(!dim)
   {
      dim = 1;
   }

   hypre_CommEntryTypeOffset(comm_entry) = offset;
   hypre_CommEntryTypeDim(comm_entry) = dim;
   hypre_CommEntryTypeOrder(comm_entry) = order;
 
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
hypre_InitializeCommunication( hypre_CommPkg     *comm_pkg,
                               double            *send_data,
                               double            *recv_data,
                               HYPRE_Int          action,
                               HYPRE_Int          tag,
                               hypre_CommHandle **comm_handle_ptr )
{
   hypre_CommHandle    *comm_handle;
                     
   HYPRE_Int            num_values = hypre_CommPkgNumValues(comm_pkg);
   HYPRE_Int            num_sends  = hypre_CommPkgNumSends(comm_pkg);
   HYPRE_Int            num_recvs  = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm       = hypre_CommPkgComm(comm_pkg);
                     
   HYPRE_Int            num_requests;
   hypre_MPI_Request         *requests;
   hypre_MPI_Status          *status;
   double             **send_buffers;
   double             **recv_buffers;

   hypre_CommType      *comm_type, *from_type, *to_type;
   hypre_CommEntryType *comm_entry;
   HYPRE_Int            num_entries;

   HYPRE_Int           *length_array, length;
   HYPRE_Int           *stride_array, stride;
   HYPRE_Int           *order;

   double              *dptr, *iptr, *jptr, *kptr, *lptr;
   HYPRE_Int           *qptr;

   HYPRE_Int            i, j, ii, jj, kk, ll;
   HYPRE_Int            size;
                      
   /*--------------------------------------------------------------------
    * allocate requests and status
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(hypre_MPI_Request, num_requests);
   status   = hypre_CTAlloc(hypre_MPI_Status, num_requests);

   /*--------------------------------------------------------------------
    * allocate buffers
    *--------------------------------------------------------------------*/

   /* allocate send buffers */
   send_buffers = hypre_TAlloc(double *, num_sends);
   if (num_sends > 0)
   {
      size = hypre_CommPkgSendBufsize(comm_pkg);
      send_buffers[0] = hypre_SharedTAlloc(double, size);
      for (i = 1; i < num_sends; i++)
      {
         comm_type = hypre_CommPkgSendType(comm_pkg, i-1);
         size = hypre_CommTypeBufsize(comm_type);
         send_buffers[i] = send_buffers[i-1] + size;
      }
   }

   /* allocate recv buffers */
   recv_buffers = hypre_TAlloc(double *, num_recvs);
   if (num_recvs > 0)
   {
      size = hypre_CommPkgRecvBufsize(comm_pkg);
      recv_buffers[0] = hypre_SharedTAlloc(double, size);
      for (i = 1; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i-1);
         size = hypre_CommTypeBufsize(comm_type);
         recv_buffers[i] = recv_buffers[i-1] + size;
      }
   }

   /*--------------------------------------------------------------------
    * pack send buffers
    *--------------------------------------------------------------------*/

   for (i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      num_entries = hypre_CommTypeNumEntries(comm_type);

      dptr = (double *) send_buffers[i];

      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         qptr = (HYPRE_Int *) send_buffers[i];
         *qptr = num_entries;
         qptr ++;
         memcpy(qptr, hypre_CommTypeRemBoxnums(comm_type),
                num_entries*sizeof(HYPRE_Int));
         qptr += num_entries;
         memcpy(qptr, hypre_CommTypeRemBoxes(comm_type),
                num_entries*sizeof(hypre_Box));

         hypre_CommTypeRemBoxnums(comm_type) = NULL;
         hypre_CommTypeRemBoxes(comm_type)   = NULL;

         dptr += hypre_CommPrefixSize(num_entries);
      }

      for (j = 0; j < num_entries; j++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, j);
         length_array = hypre_CommEntryTypeLengthArray(comm_entry);
         stride_array = hypre_CommEntryTypeStrideArray(comm_entry);
         order = hypre_CommEntryTypeOrder(comm_entry);

         lptr = send_data + hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < num_values; ll++)
         {
            if (order[ll] > -1)
            {
               kptr = lptr + order[ll]*stride_array[3];
               for (kk = 0; kk < length_array[2]; kk++)
               {
                  jptr = kptr;
                  for (jj = 0; jj < length_array[1]; jj++)
                  {
                     if (stride_array[0] == 1)
                     {
                        memcpy(dptr, jptr, length_array[0]*sizeof(double));
                     }
                     else
                     {
                        iptr = jptr;
                        length = length_array[0];
                        stride = stride_array[0];
                        for (ii = 0; ii < length; ii++)
                        {
                           dptr[ii] = *iptr;
                           iptr += stride;
                        }
                     }
                     dptr += length_array[0];
                     jptr += stride_array[1];
                  }
                  kptr += stride_array[2];
               }
            }
            else
            {
               size = length_array[0]*length_array[1]*length_array[2];
               memset(dptr, 0, size*sizeof(double));
               dptr += size;
            }

         }
      }
   }

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   j = 0;
   for(i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      hypre_MPI_Irecv(recv_buffers[i],
                      hypre_CommTypeBufsize(comm_type)*sizeof(double), hypre_MPI_BYTE,
                      hypre_CommTypeProc(comm_type), tag, comm, &requests[j++]);
      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
         hypre_CommTypeBufsize(comm_type)   -= size;
         hypre_CommPkgRecvBufsize(comm_pkg) -= size;
      }
   }

   for(i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      hypre_MPI_Isend(send_buffers[i],
                      hypre_CommTypeBufsize(comm_type)*sizeof(double), hypre_MPI_BYTE,
                      hypre_CommTypeProc(comm_type), tag, comm, &requests[j++]);
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
         hypre_TAlloc(hypre_CommEntryType, num_entries);
      hypre_CommTypeSetEntries(to_type,
                               hypre_CommTypeRemBoxnums(from_type),
                               hypre_CommTypeRemBoxes(from_type),
                               hypre_CommPkgRecvStride(comm_pkg),
                               hypre_CommPkgIdentityCoord(comm_pkg),
                               hypre_CommPkgIdentityDir(comm_pkg),
                               hypre_CommPkgIdentityOrder(comm_pkg),
                               hypre_CommPkgRecvDataSpace(comm_pkg),
                               hypre_CommPkgRecvDataOffsets(comm_pkg));
      hypre_TFree(hypre_CommPkgRemBoxnums(comm_pkg));
      hypre_TFree(hypre_CommPkgRemBoxes(comm_pkg));
   }

   hypre_ExchangeLocalData(comm_pkg, send_data, recv_data, action);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_TAlloc(hypre_CommHandle, 1);

   hypre_CommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_CommHandleSendData(comm_handle)    = send_data;
   hypre_CommHandleRecvData(comm_handle)    = recv_data;
   hypre_CommHandleNumRequests(comm_handle) = num_requests;
   hypre_CommHandleRequests(comm_handle)    = requests;
   hypre_CommHandleStatus(comm_handle)      = status;
   hypre_CommHandleSendBuffers(comm_handle) = send_buffers;
   hypre_CommHandleRecvBuffers(comm_handle) = recv_buffers;
   hypre_CommHandleAction(comm_handle)      = action;

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
   hypre_CommPkg       *comm_pkg     = hypre_CommHandleCommPkg(comm_handle);
   double             **send_buffers = hypre_CommHandleSendBuffers(comm_handle);
   double             **recv_buffers = hypre_CommHandleRecvBuffers(comm_handle);
   HYPRE_Int            action       = hypre_CommHandleAction(comm_handle);
                      
   HYPRE_Int            num_values   = hypre_CommPkgNumValues(comm_pkg);
   HYPRE_Int            num_sends    = hypre_CommPkgNumSends(comm_pkg);
   HYPRE_Int            num_recvs    = hypre_CommPkgNumRecvs(comm_pkg);

   hypre_CommType      *comm_type;
   hypre_CommEntryType *comm_entry;
   HYPRE_Int            num_entries;

   HYPRE_Int           *length_array, length;
   HYPRE_Int           *stride_array, stride;

   double              *iptr, *jptr, *kptr, *lptr;
   double              *dptr;
   HYPRE_Int           *qptr;

   HYPRE_Int           *boxnums;
   hypre_Box           *boxes;

   HYPRE_Int            i, j, ii, jj, kk, ll;

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
      hypre_CommEntryType  *ct_entries;

      num_entries = 0;
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);

         qptr = (HYPRE_Int *) recv_buffers[i];
         hypre_CommTypeNumEntries(comm_type) = *qptr;
         num_entries += hypre_CommTypeNumEntries(comm_type);
      }

      /* allocate CommType entries 'ct_entries' */
      ct_entries = hypre_TAlloc(hypre_CommEntryType, num_entries);

      /* unpack prefix information and set RecvType entries */
      for (i = 0; i < num_recvs; i++)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, i);
         hypre_CommTypeEntries(comm_type) = ct_entries;
         ct_entries += hypre_CommTypeNumEntries(comm_type);

         qptr = (HYPRE_Int *) recv_buffers[i];
         num_entries = *qptr;
         qptr ++;
         boxnums = qptr;
         qptr += num_entries;
         boxes = (hypre_Box *) qptr;
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

   for (i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      num_entries = hypre_CommTypeNumEntries(comm_type);

      dptr = (double *) recv_buffers[i];
      if ( hypre_CommPkgFirstComm(comm_pkg) )
      {
         dptr += hypre_CommPrefixSize(num_entries);
      }

      for (j = 0; j < num_entries; j++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, j);
         length_array = hypre_CommEntryTypeLengthArray(comm_entry);
         stride_array = hypre_CommEntryTypeStrideArray(comm_entry);

         lptr = hypre_CommHandleRecvData(comm_handle) +
            hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < num_values; ll++)
         {
            kptr = lptr;
            for (kk = 0; kk < length_array[2]; kk++)
            {
               jptr = kptr;
               for (jj = 0; jj < length_array[1]; jj++)
               {
                  if (action > 0)
                  {
                     /* add the data to existing values in memory */
                     iptr = jptr;
                     length = length_array[0];
                     stride = stride_array[0];
                     for (ii = 0; ii < length; ii++)
                     {
                        *iptr += dptr[ii];
                        iptr += stride;
                     }
                  }
                  else
                  {
                     /* copy the data over existing values in memory */
                     if (stride_array[0] == 1)
                     {
                        memcpy(jptr, dptr, length_array[0]*sizeof(double));
                     }
                     else
                     {
                        iptr = jptr;
                        length = length_array[0];
                        stride = stride_array[0];
                        for (ii = 0; ii < length; ii++)
                        {
                           *iptr = dptr[ii];
                           iptr += stride;
                        }
                     }
                  }
                  dptr += length_array[0];
                  jptr += stride_array[1];
               }
               kptr += stride_array[2];
            }
            lptr += stride_array[3];
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

   hypre_TFree(hypre_CommHandleRequests(comm_handle));
   hypre_TFree(hypre_CommHandleStatus(comm_handle));
   if (num_sends > 0)
   {
      hypre_SharedTFree(send_buffers[0]);
   }
   if (num_recvs > 0)
   {
      hypre_SharedTFree(recv_buffers[0]);
   }
   hypre_TFree(send_buffers);
   hypre_TFree(recv_buffers);
   hypre_TFree(comm_handle);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Execute local data exchanges.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ExchangeLocalData( hypre_CommPkg *comm_pkg,
                         double        *send_data,
                         double        *recv_data,
                         HYPRE_Int      action )
{
   HYPRE_Int            num_values = hypre_CommPkgNumValues(comm_pkg);
   hypre_CommType      *copy_from_type;
   hypre_CommType      *copy_to_type;
   hypre_CommEntryType *copy_from_entry;
   hypre_CommEntryType *copy_to_entry;

   double              *from_dp;
   HYPRE_Int           *from_stride_array;
   HYPRE_Int            from_i;
   double              *to_dp;
   HYPRE_Int           *to_stride_array;
   HYPRE_Int            to_i;
                      
   HYPRE_Int           *length_array;
   HYPRE_Int            i0, i1, i2, i3;

   HYPRE_Int           *order;

   HYPRE_Int            i;

   /*--------------------------------------------------------------------
    * copy local data
    *--------------------------------------------------------------------*/

   copy_from_type = hypre_CommPkgCopyFromType(comm_pkg);
   copy_to_type   = hypre_CommPkgCopyToType(comm_pkg);

   for (i = 0; i < hypre_CommTypeNumEntries(copy_from_type); i++)
   {
      copy_from_entry = hypre_CommTypeEntry(copy_from_type, i);
      copy_to_entry   = hypre_CommTypeEntry(copy_to_type, i);

      from_dp = send_data + hypre_CommEntryTypeOffset(copy_from_entry);
      to_dp   = recv_data + hypre_CommEntryTypeOffset(copy_to_entry);

      /* copy data only when necessary */
      if (to_dp != from_dp)
      {
         length_array = hypre_CommEntryTypeLengthArray(copy_from_entry);

         from_stride_array = hypre_CommEntryTypeStrideArray(copy_from_entry);
         to_stride_array = hypre_CommEntryTypeStrideArray(copy_to_entry);
         order = hypre_CommEntryTypeOrder(copy_from_entry);

         for (i3 = 0; i3 < num_values; i3++)
         {
            if (order[i3] > -1)
            {
               for (i2 = 0; i2 < length_array[2]; i2++)
               {
                  for (i1 = 0; i1 < length_array[1]; i1++)
                  {
                     from_i = (order[i3]*from_stride_array[3] +
                               i2*from_stride_array[2] +
                               i1*from_stride_array[1]  );
                     to_i = (i3*to_stride_array[3] +
                             i2*to_stride_array[2] +
                             i1*to_stride_array[1]  );
                     if (action > 0)
                     {
                        /* add the data to existing values in memory */
                        for (i0 = 0; i0 < length_array[0]; i0++)
                        {
                           to_dp[to_i] += from_dp[from_i];
                        
                           from_i += from_stride_array[0];
                           to_i += to_stride_array[0];
                        }
                     }
                     else
                     {
                        /* copy the data over existing values in memory */
                        for (i0 = 0; i0 < length_array[0]; i0++)
                        {
                           to_dp[to_i] = from_dp[from_i];
                        
                           from_i += from_stride_array[0];
                           to_i += to_stride_array[0];
                        }
                     }
                  }
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
   hypre_CommType  *comm_type;
   HYPRE_Int      **orders;
   HYPRE_Int        i;

   if (comm_pkg)
   {
      /* note that entries are allocated in two stages for To/Recv */
      if (hypre_CommPkgNumRecvs(comm_pkg) > 0)
      {
         comm_type = hypre_CommPkgRecvType(comm_pkg, 0);
         hypre_TFree(hypre_CommTypeEntries(comm_type));
      }
      comm_type = hypre_CommPkgCopyToType(comm_pkg);
      hypre_TFree(hypre_CommTypeEntries(comm_type));
      hypre_TFree(comm_type);

      comm_type = hypre_CommPkgCopyFromType(comm_pkg);
      hypre_TFree(comm_type);

      hypre_TFree(hypre_CommPkgEntries(comm_pkg));
      hypre_TFree(hypre_CommPkgRemBoxnums(comm_pkg));
      hypre_TFree(hypre_CommPkgRemBoxes(comm_pkg));

      hypre_TFree(hypre_CommPkgRecvDataOffsets(comm_pkg));
      hypre_BoxArrayDestroy(hypre_CommPkgRecvDataSpace(comm_pkg));

      orders = hypre_CommPkgOrders(comm_pkg);
      for (i = 0; i < hypre_CommPkgNumOrders(comm_pkg); i++)
      {
         hypre_TFree(orders[i]);
      }
      hypre_TFree(orders);

      hypre_TFree(hypre_CommPkgIdentityOrder(comm_pkg));

      hypre_TFree(comm_pkg);
   }

   return hypre_error_flag;
}
