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
 * $Revision: 2.13 $
 ***********************************************************************EHEADER*/



#include "headers.h"

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
#endif

/* this computes a (large enough) size (in doubles) for the message prefix */
#define hypre_CommPrefixSize(ne) \
( ((ne*sizeof(int) + ne*sizeof(hypre_Box))/sizeof(double)) + 1 )

/*--------------------------------------------------------------------------
 * Create a communication package.  A grid-based description of a
 * communication exchange is passed in.  This description is then
 * compiled into an intermediate processor-based description of the
 * communication.  The intermediate processor-based description is
 * used directly to pack and unpack buffers during the communications.
 *
 * Note: The input boxes and processes are destroyed.
 *--------------------------------------------------------------------------*/

int
hypre_CommPkgCreate( hypre_CommInfo   *comm_info,
                     hypre_BoxArray   *send_data_space,
                     hypre_BoxArray   *recv_data_space,
                     int               num_values,
                     MPI_Comm          comm,
                     hypre_CommPkg   **comm_pkg_ptr )
{
   int                   ierr = 0;

   hypre_BoxArrayArray  *send_boxes;
   hypre_BoxArrayArray  *recv_boxes;
   hypre_IndexRef        send_stride;
   hypre_IndexRef        recv_stride;
   int                 **send_processes;
   int                 **recv_processes;
   int                 **send_rboxnums;
   hypre_BoxArrayArray  *send_rboxes;

   hypre_CommPkg        *comm_pkg;
   hypre_CommType       *comm_types;
   hypre_CommType       *comm_type;
   hypre_CommEntryType  *ct_entries;
   int                  *ct_loc_boxnums;
   int                  *ct_rem_boxnums;
   hypre_Box            *ct_loc_boxes;
   hypre_Box            *ct_rem_boxes;
   int                  *p_comm_types;
   int                   num_comms, num_entries, comm_bufsize;

   hypre_BoxArray       *box_array;
   hypre_Box            *box;
   hypre_BoxArray       *rbox_array;
   hypre_Box            *rbox;
   hypre_Box            *data_box;
   int                  *data_offsets;
   int                   data_offset;
                        
   int                   i, j, k, p, m, size;
   int                   num_procs, my_proc;
                        
   /*------------------------------------------------------
    *------------------------------------------------------*/

   send_boxes     = hypre_CommInfoSendBoxes(comm_info);
   recv_boxes     = hypre_CommInfoRecvBoxes(comm_info);
   send_stride    = hypre_CommInfoSendStride(comm_info);
   recv_stride    = hypre_CommInfoRecvStride(comm_info);
   send_processes = hypre_CommInfoSendProcesses(comm_info);
   recv_processes = hypre_CommInfoRecvProcesses(comm_info);
   send_rboxnums  = hypre_CommInfoSendRBoxnums(comm_info);
   send_rboxes    = hypre_CommInfoSendRBoxes(comm_info);

   MPI_Comm_size(comm, &num_procs );
   MPI_Comm_rank(comm, &my_proc );

   /*------------------------------------------------------
    * Set up various entries in CommPkg
    *------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1);

   hypre_CommPkgComm(comm_pkg)       = comm;
   hypre_CommPkgFirstSend(comm_pkg)  = 1;
   hypre_CommPkgFirstRecv(comm_pkg)  = 1;
   hypre_CommPkgNumValues(comm_pkg)  = num_values;
   hypre_CopyIndex(send_stride, hypre_CommPkgSendStride(comm_pkg));
   hypre_CopyIndex(recv_stride, hypre_CommPkgRecvStride(comm_pkg));

   /*------------------------------------------------------
    * Set up send CommType information
    *------------------------------------------------------*/

   p_comm_types = hypre_CTAlloc(int, num_procs);

   /* set send_data_offsets and send_data_space */
   data_offsets = hypre_TAlloc(int, hypre_BoxArraySize(send_data_space));
   data_offset = 0;
   hypre_ForBoxI(i, send_data_space)
      {
         data_offsets[i] = data_offset;
         data_box = hypre_BoxArrayBox(send_data_space, i);
         data_offset += hypre_BoxVolume(data_box) * num_values;
      }

   /* compute num_comms and use -p_comm_types to compute num_entries */

   num_comms = 0;
   num_entries = 0;
   hypre_ForBoxArrayI(i, send_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);

         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               p = send_processes[i][j];

               if (hypre_BoxVolume(box) != 0)
               {
                  p_comm_types[p]--;
                  num_entries++;
                  if ((p_comm_types[p] == -1) && (p != my_proc))
                  {
                     num_comms++;
                  }
               }
            }
      }

   /* compute comm_types */

   comm_types = hypre_CTAlloc(hypre_CommType, (num_comms + 1));
   ct_entries = hypre_TAlloc(hypre_CommEntryType, num_entries);
   ct_loc_boxnums = hypre_TAlloc(int, num_entries);
   ct_rem_boxnums = hypre_TAlloc(int, num_entries);
   ct_loc_boxes = hypre_TAlloc(hypre_Box, num_entries);
   ct_rem_boxes = hypre_TAlloc(hypre_Box, num_entries);

   /* initialize local copy type */
   comm_type = &comm_types[0];
   k = -p_comm_types[my_proc];
   p_comm_types[my_proc] = 0;
   hypre_CommTypeProc(comm_type)       = my_proc;
   hypre_CommTypeNumEntries(comm_type) = 0;
   hypre_CommTypeEntries(comm_type)    = ct_entries;
   hypre_CommTypeLocBoxnums(comm_type) = ct_loc_boxnums;
   hypre_CommTypeRemBoxnums(comm_type) = ct_rem_boxnums;
   hypre_CommTypeLocBoxes(comm_type)   = ct_loc_boxes;
   hypre_CommTypeRemBoxes(comm_type)   = ct_rem_boxes;
   ct_entries     += k;
   ct_loc_boxnums += k;
   ct_rem_boxnums += k;
   ct_loc_boxes   += k;
   ct_rem_boxes   += k;

   m = 1;
   comm_bufsize = 0;
   hypre_ForBoxArrayI(i, send_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
         rbox_array = hypre_BoxArrayArrayBoxArray(send_rboxes, i);

         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               rbox = hypre_BoxArrayBox(rbox_array, j);
               p = send_processes[i][j];

               if (hypre_BoxVolume(box) != 0)
               {
                  /* initialize comm type for process p */
                  if (p_comm_types[p] < 0)
                  {
                     comm_type = &comm_types[m];
                     k = -p_comm_types[p];
                     p_comm_types[p] = m;
                     hypre_CommTypeProc(comm_type) = p;
                     size = hypre_CommPrefixSize(k);
                     hypre_CommTypeBufsize(comm_type) = size;
                     comm_bufsize                    += size;
                     hypre_CommTypeNumEntries(comm_type) = 0;
                     hypre_CommTypeEntries(comm_type)    = ct_entries;
                     hypre_CommTypeLocBoxnums(comm_type) = ct_loc_boxnums;
                     hypre_CommTypeRemBoxnums(comm_type) = ct_rem_boxnums;
                     hypre_CommTypeLocBoxes(comm_type)   = ct_loc_boxes;
                     hypre_CommTypeRemBoxes(comm_type)   = ct_rem_boxes;
                     ct_entries     += k;
                     ct_loc_boxnums += k;
                     ct_rem_boxnums += k;
                     ct_loc_boxes   += k;
                     ct_rem_boxes   += k;
                     m++;
                  }

                  comm_type = &comm_types[p_comm_types[p]];
                  k = hypre_CommTypeNumEntries(comm_type);
                  hypre_BoxGetStrideVolume(box, send_stride, &size);
                  hypre_CommTypeBufsize(comm_type) += (size*num_values);
                  comm_bufsize                     += (size*num_values);
                  hypre_CommTypeLocBoxnum(comm_type, k) = i;
                  hypre_CommTypeRemBoxnum(comm_type, k) = send_rboxnums[i][j];
                  hypre_CopyBox(box, hypre_CommTypeLocBox(comm_type, k));
                  hypre_CopyBox(rbox, hypre_CommTypeRemBox(comm_type, k));
                  hypre_CommTypeNumEntries(comm_type) ++;
               }
            }
      }

   /* Loop over comm_types and set entries */
   for (m = 0; m < (num_comms + 1); m++)
   {
      comm_type = &comm_types[m];
      hypre_CommTypeSetEntries(comm_type,
                               hypre_CommTypeLocBoxnums(comm_type),
                               hypre_CommTypeLocBoxes(comm_type),
                               hypre_CommPkgSendStride(comm_pkg),
                               hypre_CommPkgNumValues(comm_pkg),
                               send_data_space, data_offsets );
      if (m > 0)
      {
         hypre_CommTypeLocBoxnums(comm_type) = NULL;
         hypre_CommTypeLocBoxes(comm_type)   = NULL;
      }

      /* reset p_comm_types to zero */
      p_comm_types[hypre_CommTypeProc(comm_type)] = 0;
   }

   /* set send info in comm_pkg */
   hypre_CommPkgSendBufsize(comm_pkg)  = comm_bufsize;
   hypre_CommPkgNumSends(comm_pkg)     = num_comms;
   hypre_CommPkgSendTypes(comm_pkg)    = &comm_types[1];
   hypre_CommPkgCopyFromType(comm_pkg) = &comm_types[0];

   /* free up data_offsets */
   hypre_TFree(data_offsets);

   /* free up local box info */
   comm_type = hypre_CommPkgCopyFromType(comm_pkg);
   hypre_TFree(hypre_CommTypeLocBoxnums(comm_type));
   hypre_TFree(hypre_CommTypeLocBoxes(comm_type));

   /*------------------------------------------------------
    * Set up recv CommType information
    *------------------------------------------------------*/

   /* set data_offsets and recv_data_space */
   data_offsets = hypre_TAlloc(int, hypre_BoxArraySize(recv_data_space));
   data_offset = 0;
   hypre_ForBoxI(i, recv_data_space)
      {
         data_offsets[i] = data_offset;
         data_box = hypre_BoxArrayBox(recv_data_space, i);
         data_offset += hypre_BoxVolume(data_box) * num_values;
      }
   hypre_CommPkgRecvDataOffsets(comm_pkg) = data_offsets;
   hypre_CommPkgRecvDataSpace(comm_pkg) =
      hypre_BoxArrayDuplicate(recv_data_space);

   /* compute num_comms and use -p_comm_types to compute num_entries */

   num_comms = 0;
   num_entries = 0;
   hypre_ForBoxArrayI(i, recv_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);

         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               p = recv_processes[i][j];

               if (hypre_BoxVolume(box) != 0)
               {
                  p_comm_types[p]--;
                  num_entries++;
                  if ((p_comm_types[p] == -1) && (p != my_proc))
                  {
                     num_comms++;
                  }
               }
            }
      }

   /* compute comm_types */

   comm_types = hypre_CTAlloc(hypre_CommType, (num_comms + 1));
   ct_entries = hypre_TAlloc(hypre_CommEntryType, num_entries);

   /* initialize local copy type */
   comm_type = &comm_types[0];
   k = -p_comm_types[my_proc];
   p_comm_types[my_proc] = 0;
   hypre_CommTypeProc(comm_type)       = my_proc;
   hypre_CommTypeNumEntries(comm_type) = 0;
   hypre_CommTypeEntries(comm_type)    = ct_entries;
   ct_entries += k;

   m = 1;
   comm_bufsize = 0;
   hypre_ForBoxArrayI(i, recv_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);

         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               p = recv_processes[i][j];

               if (hypre_BoxVolume(box) != 0)
               {
                  /* initialize comm type for process p */
                  if (p_comm_types[p] < 0)
                  {
                     comm_type = &comm_types[m];
                     k = -p_comm_types[p];
                     p_comm_types[p] = m;
                     hypre_CommTypeProc(comm_type) = p;
                     size = hypre_CommPrefixSize(k);
                     hypre_CommTypeBufsize(comm_type) = size;
                     comm_bufsize                    += size;
                     hypre_CommTypeNumEntries(comm_type) = 0;
                     hypre_CommTypeEntries(comm_type)    = ct_entries;
                     ct_entries += k;
                     m++;
                  }

                  comm_type = &comm_types[p_comm_types[p]];
                  hypre_BoxGetStrideVolume(box, recv_stride, &size);
                  hypre_CommTypeBufsize(comm_type) += (size*num_values);
                  comm_bufsize                     += (size*num_values);
                  hypre_CommTypeNumEntries(comm_type) ++;
               }
            }
      }

   /* set copy type entries */
   comm_type = hypre_CommPkgCopyFromType(comm_pkg);
   hypre_CommTypeSetEntries(&comm_types[0],
                            hypre_CommTypeRemBoxnums(comm_type),
                            hypre_CommTypeRemBoxes(comm_type),
                            hypre_CommPkgRecvStride(comm_pkg),
                            hypre_CommPkgNumValues(comm_pkg),
                            hypre_CommPkgRecvDataSpace(comm_pkg),
                            hypre_CommPkgRecvDataOffsets(comm_pkg) );

   /* set recv info in comm_pkg */
   hypre_CommPkgRecvBufsize(comm_pkg) = comm_bufsize;
   hypre_CommPkgNumRecvs(comm_pkg)    = num_comms;
   hypre_CommPkgRecvTypes(comm_pkg)   = &comm_types[1];
   hypre_CommPkgCopyToType(comm_pkg)  = &comm_types[0];

   /*------------------------------------------------------
    * Debugging stuff
    *------------------------------------------------------*/

#if DEBUG
{
   MPI_Comm_rank(MPI_COMM_WORLD, &my_proc);

   sprintf(filename, "zcommboxes.%05d", my_proc);

   if ((file = fopen(filename, "a")) == NULL)
   {
      printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   fprintf(file, "\n\n============================\n\n");
   fprintf(file, "SEND boxes:\n\n");

   fprintf(file, "Stride = (%d,%d,%d)\n",
           hypre_IndexX(send_stride),
           hypre_IndexY(send_stride),
           hypre_IndexZ(send_stride));
   fprintf(file, "BoxArrayArraySize = %d\n",
           hypre_BoxArrayArraySize(send_boxes));
   hypre_ForBoxArrayI(i, send_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);

         fprintf(file, "BoxArraySize = %d\n", hypre_BoxArraySize(box_array));
         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               fprintf(file, "(%d,%d): (%d,%d,%d) x (%d,%d,%d)\n",
                       i, j,
                       hypre_BoxIMinX(box),
                       hypre_BoxIMinY(box),
                       hypre_BoxIMinZ(box),
                       hypre_BoxIMaxX(box),
                       hypre_BoxIMaxY(box),
                       hypre_BoxIMaxZ(box));
               fprintf(file, "(%d,%d): %d,%d\n",
                       i, j, send_processes[i][j], send_rboxnums[i][j]);
            }
      }

   fprintf(file, "\n\n============================\n\n");
   fprintf(file, "RECV boxes:\n\n");

   fprintf(file, "Stride = (%d,%d,%d)\n",
           hypre_IndexX(recv_stride),
           hypre_IndexY(recv_stride),
           hypre_IndexZ(recv_stride));
   fprintf(file, "BoxArrayArraySize = %d\n",
           hypre_BoxArrayArraySize(recv_boxes));
   hypre_ForBoxArrayI(i, recv_boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);

         fprintf(file, "BoxArraySize = %d\n", hypre_BoxArraySize(box_array));
         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               fprintf(file, "(%d,%d): (%d,%d,%d) x (%d,%d,%d)\n",
                       i, j,
                       hypre_BoxIMinX(box),
                       hypre_BoxIMinY(box),
                       hypre_BoxIMinZ(box),
                       hypre_BoxIMaxX(box),
                       hypre_BoxIMaxY(box),
                       hypre_BoxIMaxZ(box));
               fprintf(file, "(%d,%d): %d\n",
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
   int                   offset, dim;
   int                  *length;
   int                  *stride;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_proc);

   sprintf(filename, "zcommentries.%05d", my_proc);

   if ((file = fopen(filename, "a")) == NULL)
   {
      printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   fprintf(file, "\n\n============================\n\n");
   fprintf(file, "SEND entries:\n\n");

   fprintf(file, "num_sends = %d\n", hypre_CommPkgNumSends(comm_pkg));

   comm_types = hypre_CommPkgCopyFromType(comm_pkg);
   for (m = 0; m < (hypre_CommPkgNumSends(comm_pkg) + 1); m++)
   {
      comm_type = &comm_types[m];
      fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
      fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
      for (i = 0; i < hypre_CommTypeNumEntries(comm_type); i++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, i);
         offset = hypre_CommEntryTypeOffset(comm_entry);
         dim    = hypre_CommEntryTypeDim(comm_entry);
         length = hypre_CommEntryTypeLengthArray(comm_entry);
         stride = hypre_CommEntryTypeStrideArray(comm_entry);
         fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
                 i, offset, dim,
                 length[0], length[1], length[2], length[3],
                 stride[0], stride[1], stride[2], stride[3]);
      }
   }

   fprintf(file, "\n\n============================\n\n");
   fprintf(file, "RECV entries:\n\n");

   fprintf(file, "num_recvs = %d\n", hypre_CommPkgNumRecvs(comm_pkg));

   comm_types = hypre_CommPkgCopyToType(comm_pkg);

   comm_type = &comm_types[0];
   fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
   fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
   for (i = 0; i < hypre_CommTypeNumEntries(comm_type); i++)
   {
      comm_entry = hypre_CommTypeEntry(comm_type, i);
      offset = hypre_CommEntryTypeOffset(comm_entry);
      dim    = hypre_CommEntryTypeDim(comm_entry);
      length = hypre_CommEntryTypeLengthArray(comm_entry);
      stride = hypre_CommEntryTypeStrideArray(comm_entry);
      fprintf(file, "%d: %d,%d,(%d,%d,%d,%d),(%d,%d,%d,%d)\n",
              i, offset, dim,
              length[0], length[1], length[2], length[3],
              stride[0], stride[1], stride[2], stride[3]);
   }

   for (m = 1; m < (hypre_CommPkgNumRecvs(comm_pkg) + 1); m++)
   {
      comm_type = &comm_types[m];
      fprintf(file, "process     = %d\n", hypre_CommTypeProc(comm_type));
      fprintf(file, "num_entries = %d\n", hypre_CommTypeNumEntries(comm_type));
   }

   fflush(file);
   fclose(file);
}
#endif

   /*------------------------------------------------------
    * Clean up
    *------------------------------------------------------*/

   hypre_CommInfoDestroy(comm_info);
   hypre_TFree(p_comm_types);

   *comm_pkg_ptr = comm_pkg;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_CommTypeSetEntries( hypre_CommType  *comm_type,
                          int             *boxnums,
                          hypre_Box       *boxes,
                          hypre_Index      stride,
                          int              num_values,
                          hypre_BoxArray  *data_space,
                          int             *data_offsets )
{
   int         ierr = 0;

   int                   num_entries = hypre_CommTypeNumEntries(comm_type);
   hypre_CommEntryType  *entries     = hypre_CommTypeEntries(comm_type);
   hypre_Box            *box;
   hypre_Box            *data_box;
   int                   i, j;
                
   for (j = 0; j < num_entries; j++)
   {
      i = boxnums[j];
      box = &boxes[j];
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_CommTypeSetEntry(box, stride, data_box, num_values, data_offsets[i],
                             &entries[j]);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_CommTypeSetEntry( hypre_Box           *box,
                        hypre_Index          stride,
                        hypre_Box           *data_box,
                        int                  num_values,
                        int                  data_box_offset,
                        hypre_CommEntryType *comm_entry )
{
   int           ierr = 0;
   int          *length_array;
   int          *stride_array;
   hypre_Index   size;
   int           i, j, dim;

   /*------------------------------------------------------
    * Set offset
    *------------------------------------------------------*/

   hypre_CommEntryTypeOffset(comm_entry) =
      data_box_offset + hypre_BoxIndexRank(data_box, hypre_BoxIMin(box));

   /*------------------------------------------------------
    * Set length_array, stride_array, and dim
    *------------------------------------------------------*/

   length_array = hypre_CommEntryTypeLengthArray(comm_entry);
   stride_array = hypre_CommEntryTypeStrideArray(comm_entry);
 
   /* initialize length_array */
   hypre_BoxGetStrideSize(box, stride, size);
   for (i = 0; i < 3; i++)
   {
      length_array[i] = hypre_IndexD(size, i);
   }
   length_array[3] = num_values;

   /* initialize stride_array */
   for (i = 0; i < 3; i++)
   {
      stride_array[i] = hypre_IndexD(stride, i);
      for (j = 0; j < i; j++)
      {
         stride_array[i] *= hypre_BoxSizeD(data_box, j);
      }
   }
   stride_array[3] = hypre_BoxVolume(data_box);

   /* eliminate dimensions with length_array = 1 */
   dim = 4;
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

   hypre_CommEntryTypeDim(comm_entry) = dim;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * Initialize a non-blocking communication exchange.
 *
 * The communication buffers are created, the send buffer is manually
 * packed, and the communication requests are posted.
 *--------------------------------------------------------------------------*/

int
hypre_InitializeCommunication( hypre_CommPkg     *comm_pkg,
                               double            *send_data,
                               double            *recv_data,
                               hypre_CommHandle **comm_handle_ptr )
{
   int                  ierr = 0;
                     
   hypre_CommHandle    *comm_handle;
                     
   int                  num_sends = hypre_CommPkgNumSends(comm_pkg);
   int                  num_recvs = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm      = hypre_CommPkgComm(comm_pkg);
                     
   int                  num_requests;
   MPI_Request         *requests;
   MPI_Status          *status;
   double             **send_buffers;
   double             **recv_buffers;

   hypre_CommType      *comm_type;
   hypre_CommEntryType *comm_entry;
   int                  num_entries;

   int                 *length_array, length;
   int                 *stride_array, stride;

   double              *dptr, *iptr, *jptr, *kptr, *lptr;
   int                 *qptr;

   int                  i, j, ii, jj, kk, ll;
   int                  size;
                      
   /*--------------------------------------------------------------------
    * allocate requests and status
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status   = hypre_CTAlloc(MPI_Status, num_requests);

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

      if ( hypre_CommPkgFirstSend(comm_pkg) )
      {
         qptr = (int *) send_buffers[i];

         memcpy(qptr, hypre_CommTypeRemBoxnums(comm_type),
                num_entries*sizeof(int));
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

         lptr = send_data + hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < length_array[3]; ll++)
         {
            kptr = lptr;
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
            lptr += stride_array[3];
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
      MPI_Irecv(recv_buffers[i],
                hypre_CommTypeBufsize(comm_type)*sizeof(double), MPI_BYTE,
                hypre_CommTypeProc(comm_type), 0, comm, &requests[j++]);
      if ( hypre_CommPkgFirstRecv(comm_pkg) )
      {
         size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
         hypre_CommTypeBufsize(comm_type)   -= size;
         hypre_CommPkgRecvBufsize(comm_pkg) -= size;
      }
   }

   for(i = 0; i < num_sends; i++)
   {
      comm_type = hypre_CommPkgSendType(comm_pkg, i);
      MPI_Isend(send_buffers[i],
                hypre_CommTypeBufsize(comm_type)*sizeof(double), MPI_BYTE,
                hypre_CommTypeProc(comm_type), 0, comm, &requests[j++]);
      if ( hypre_CommPkgFirstSend(comm_pkg) )
      {
         size = hypre_CommPrefixSize(hypre_CommTypeNumEntries(comm_type));
         hypre_CommTypeBufsize(comm_type)   -= size;
         hypre_CommPkgSendBufsize(comm_pkg) -= size;
      }
   }

   hypre_ExchangeLocalData(comm_pkg, send_data, recv_data);

   /*--------------------------------------------------------------------
    * free up remote boxnums and turn off first_send indicator
    *--------------------------------------------------------------------*/

   if ( hypre_CommPkgFirstSend(comm_pkg) )
   {
      comm_type = hypre_CommPkgCopyFromType(comm_pkg);
      hypre_TFree(hypre_CommTypeRemBoxnums(comm_type));
      hypre_TFree(hypre_CommTypeRemBoxes(comm_type));

      hypre_CommPkgFirstSend(comm_pkg) = 0;
   }

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

   *comm_handle_ptr = comm_handle;

   return ierr;
}

/*--------------------------------------------------------------------------
 * Finalize a communication exchange.  This routine blocks until all
 * of the communication requests are completed.
 *
 * The communication requests are completed, and the receive buffer is
 * manually unpacked.
 *--------------------------------------------------------------------------*/

int
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   
   int              ierr = 0;

   hypre_CommPkg   *comm_pkg     = hypre_CommHandleCommPkg(comm_handle);
   double         **send_buffers = hypre_CommHandleSendBuffers(comm_handle);
   double         **recv_buffers = hypre_CommHandleRecvBuffers(comm_handle);
   int              num_sends    = hypre_CommPkgNumSends(comm_pkg);
   int              num_recvs    = hypre_CommPkgNumRecvs(comm_pkg);

   hypre_CommType      *comm_type;
   hypre_CommEntryType *comm_entry;
   int                  num_entries;

   int                 *length_array, length;
   int                 *stride_array, stride;

   double              *iptr, *jptr, *kptr, *lptr;
   double              *dptr;
   int                 *qptr;

   int                 *boxnums;
   hypre_Box           *boxes;

   int                  i, j, ii, jj, kk, ll;

   /*--------------------------------------------------------------------
    * finish communications
    *--------------------------------------------------------------------*/

   if (hypre_CommHandleNumRequests(comm_handle))
   {
      MPI_Waitall(hypre_CommHandleNumRequests(comm_handle),
                  hypre_CommHandleRequests(comm_handle),
                  hypre_CommHandleStatus(comm_handle));
   }

   /*--------------------------------------------------------------------
    * unpack recv buffers
    *--------------------------------------------------------------------*/

   for (i = 0; i < num_recvs; i++)
   {
      comm_type = hypre_CommPkgRecvType(comm_pkg, i);
      num_entries = hypre_CommTypeNumEntries(comm_type);

      dptr = (double *) recv_buffers[i];

      if ( hypre_CommPkgFirstRecv(comm_pkg) )
      {
         qptr = (int *) recv_buffers[i];
         boxnums = (int *) qptr;
         qptr += num_entries;
         boxes   = (hypre_Box *) qptr;

         hypre_CommTypeSetEntries(comm_type, boxnums, boxes,
                                  hypre_CommPkgRecvStride(comm_pkg),
                                  hypre_CommPkgNumValues(comm_pkg),
                                  hypre_CommPkgRecvDataSpace(comm_pkg),
                                  hypre_CommPkgRecvDataOffsets(comm_pkg) );

         dptr += hypre_CommPrefixSize(num_entries);
      }

      for (j = 0; j < num_entries; j++)
      {
         comm_entry = hypre_CommTypeEntry(comm_type, j);
         length_array = hypre_CommEntryTypeLengthArray(comm_entry);
         stride_array = hypre_CommEntryTypeStrideArray(comm_entry);

         lptr = hypre_CommHandleRecvData(comm_handle) +
            hypre_CommEntryTypeOffset(comm_entry);
         for (ll = 0; ll < length_array[3]; ll++)
         {
            kptr = lptr;
            for (kk = 0; kk < length_array[2]; kk++)
            {
               jptr = kptr;
               for (jj = 0; jj < length_array[1]; jj++)
               {
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
    * turn off first_recv indicator
    *--------------------------------------------------------------------*/

   hypre_CommPkgFirstRecv(comm_pkg) = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * Execute local data exchanges.
 *--------------------------------------------------------------------------*/

int
hypre_ExchangeLocalData( hypre_CommPkg *comm_pkg,
                         double        *send_data,
                         double        *recv_data )
{
   hypre_CommType      *copy_from_type;
   hypre_CommType      *copy_to_type;
   hypre_CommEntryType *copy_from_entry;
   hypre_CommEntryType *copy_to_entry;

   double              *from_dp;
   int                 *from_stride_array;
   int                  from_i;
   double              *to_dp;
   int                 *to_stride_array;
   int                  to_i;
                      
   int                 *length_array;
   int                  i0, i1, i2, i3;

   int                  i;
   int                  ierr = 0;

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

         for (i3 = 0; i3 < length_array[3]; i3++)
         {
            for (i2 = 0; i2 < length_array[2]; i2++)
            {
               for (i1 = 0; i1 < length_array[1]; i1++)
               {
                  from_i = (i3*from_stride_array[3] +
                            i2*from_stride_array[2] +
                            i1*from_stride_array[1]  );
                  to_i = (i3*to_stride_array[3] +
                          i2*to_stride_array[2] +
                          i1*to_stride_array[1]  );
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

   return ( ierr );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_CommPkgDestroy( hypre_CommPkg *comm_pkg )
{
   int ierr = 0;
   hypre_CommType  *comm_types;

   if (comm_pkg)
   {
      comm_types = hypre_CommPkgCopyToType(comm_pkg);
      hypre_TFree(hypre_CommTypeEntries(comm_types));
      hypre_TFree(hypre_CommTypeLocBoxnums(comm_types));
      hypre_TFree(hypre_CommTypeRemBoxnums(comm_types));
      hypre_TFree(hypre_CommTypeLocBoxes(comm_types));
      hypre_TFree(hypre_CommTypeRemBoxes(comm_types));
      hypre_TFree(comm_types);

      comm_types = hypre_CommPkgCopyFromType(comm_pkg);
      hypre_TFree(hypre_CommTypeEntries(comm_types));
      hypre_TFree(hypre_CommTypeLocBoxnums(comm_types));
      hypre_TFree(hypre_CommTypeRemBoxnums(comm_types));
      hypre_TFree(hypre_CommTypeLocBoxes(comm_types));
      hypre_TFree(hypre_CommTypeRemBoxes(comm_types));
      hypre_TFree(comm_types);

      hypre_TFree(hypre_CommPkgRecvDataOffsets(comm_pkg));
      hypre_BoxArrayDestroy(hypre_CommPkgRecvDataSpace(comm_pkg));

      hypre_TFree(comm_pkg);
   }

   return ierr;
}
