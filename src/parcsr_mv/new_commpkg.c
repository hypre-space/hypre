/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*----------------------------------------------------
 * Communication package that uses an assumed partition
 *  AHB 6/04
 *-----------------------------------------------------*/

#include "_hypre_parcsr_mv.h"

/* some debugging tools*/
#define mydebug 0

/*==========================================================================*/

HYPRE_Int
hypre_PrintCommpkg(hypre_ParCSRMatrix *A, const char *file_name)
{
   HYPRE_Int  num_components, num_sends, num_recvs;

   HYPRE_Int *recv_vec_starts, *recv_procs;
   HYPRE_Int *send_map_starts, *send_map_elements, *send_procs;

   HYPRE_Int  i;
   HYPRE_Int  my_id;
   MPI_Comm   comm;
   hypre_ParCSRCommPkg *comm_pkg;

   char   new_file[80];
   FILE *fp;

   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   comm = hypre_ParCSRCommPkgComm(comm_pkg);
   num_components = hypre_ParCSRCommPkgNumComponents(comm_pkg);
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   send_map_elements = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   hypre_MPI_Comm_rank(comm, &my_id);

   hypre_sprintf(new_file, "%s.%d", file_name, my_id);

   fp = fopen(new_file, "w");
   hypre_fprintf(fp, "num_components = %d\n", num_components);
   hypre_fprintf(fp, "num_recvs = %d\n", num_recvs);
   for (i = 0; i < num_recvs; i++)
   {
      hypre_fprintf(fp, "recv_proc [start, end] = %d [%d, %d] \n", recv_procs[i], recv_vec_starts[i],
                    recv_vec_starts[i + 1] - 1);
   }

   hypre_fprintf(fp, "num_sends = %d\n", num_sends);
   for (i = 0; i < num_sends; i++)
   {
      hypre_fprintf(fp, "send_proc [start, end] = %d [%d, %d] \n", send_procs[i], send_map_starts[i],
                    send_map_starts[i + 1] - 1);
   }

   for (i = 0; i < send_map_starts[num_sends]; i++)
   {
      hypre_fprintf(fp, "send_map_elements (%d) = %d\n", i, send_map_elements[i]);
   }

   fclose(fp);

   return hypre_error_flag;
}

/*------------------------------------------------------------------------------
 * hypre_ParCSRCommPkgCreateApart_core
 *
 * This does the work for  hypre_ParCSRCommPkgCreateApart - we have to split it
 * off so that it can also be used for block matrices.
 *------------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRCommPkgCreateApart_core(
   /* input args: */
   MPI_Comm   comm,
   HYPRE_BigInt *col_map_off_d,
   HYPRE_BigInt  first_col_diag,
   HYPRE_Int  num_cols_off_d,
   HYPRE_BigInt  global_num_cols,
   /* pointers to output args: */
   HYPRE_Int  *p_num_recvs,
   HYPRE_Int **p_recv_procs,
   HYPRE_Int **p_recv_vec_starts,
   HYPRE_Int  *p_num_sends,
   HYPRE_Int **p_send_procs,
   HYPRE_Int **p_send_map_starts,
   HYPRE_Int **p_send_map_elements,
   /* additional input assumed part */
   hypre_IJAssumedPart *apart)
{
   HYPRE_Int        num_procs, myid;
   HYPRE_Int        j, i;
   HYPRE_BigInt     range_start, range_end;

   HYPRE_BigInt     big_size;
   HYPRE_Int        size;
   HYPRE_Int        count;

   HYPRE_Int        num_recvs, *recv_procs = NULL, *recv_vec_starts = NULL;
   HYPRE_Int        tmp_id, prev_id;

   HYPRE_Int        num_sends;

   HYPRE_Int        ex_num_contacts, *ex_contact_procs = NULL, *ex_contact_vec_starts = NULL;
   HYPRE_BigInt     *ex_contact_buf = NULL;

   HYPRE_Int        num_ranges;
   HYPRE_BigInt     upper_bound;


   HYPRE_BigInt     *response_buf = NULL;
   HYPRE_Int        *response_buf_starts = NULL;

   HYPRE_Int        max_response_size;

   hypre_DataExchangeResponse        response_obj1, response_obj2;
   hypre_ProcListElements            send_proc_obj;

#if mydebug
   HYPRE_Int tmp_int, index;
#endif

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );


#if mydebug

   hypre_printf("myid = %i, my assumed local range: [%i, %i]\n", myid,
                apart->row_start, apart->row_end);

   for (i = 0; i < apart.length; i++)
   {
      hypre_printf("myid = %d, proc %d owns assumed partition range = [%d, %d]\n",
                   myid, apart->proc_list[i], apart->row_start_list[i],
                   apart->row_end_list[i]);
   }

   hypre_printf("myid = %d, length of apart = %d\n", myid, apart->length);

#endif

   /*-----------------------------------------------------------
    *  Everyone knows where their assumed range is located
    * (because of the assumed partition object (apart).
    *  For the comm. package, each proc must know it's receive
    *  procs (who it will receive data from and how much data)
    *  and its send procs
    *  (who it will send data to) and the indices of the elements
    *  to be sent.  This is based on the non-zero
    *  entries in its rows. Each proc should know this from the user.
    *-----------------------------------------------------------*/

   /*------------------------------------------------------------
    *  First, get the receive processors
    *  each par_csr matrix will have a certain number of columns
    *  (num_cols_off_d) given in col_map_offd[] for which it needs
    *  data from another processor.
    *------------------------------------------------------------*/

   /*calculate the assumed receive processors*/

   /* need to populate num_recvs, *recv_procs, and *recv_vec_starts
      (correlates to starts in col_map_off_d for recv_procs) for
      the comm. package*/


   /*create contact information*/

   ex_num_contacts = 0;

   /*estimate the storage needed*/
   if (num_cols_off_d > 0 && (apart->row_end - apart->row_start) > 0  )
   {
      big_size = col_map_off_d[num_cols_off_d - 1] - col_map_off_d[0];

      size = (HYPRE_Int)(big_size / (apart->row_end - apart->row_start)) + 2;
   }
   else
   {
      size = 0;
   }

   /*we will contact each with a range of cols that we need*/
   /* it is ok to contact yourself - because then there doesn't
      need to be separate code */

   ex_contact_procs = hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);
   ex_contact_vec_starts =  hypre_CTAlloc(HYPRE_Int,  size + 1, HYPRE_MEMORY_HOST);
   ex_contact_buf =  hypre_CTAlloc(HYPRE_BigInt,  size * 2, HYPRE_MEMORY_HOST);

   range_end = -1;
   for (i = 0; i < num_cols_off_d; i++)
   {
      if (col_map_off_d[i] > range_end)
      {
         hypre_GetAssumedPartitionProcFromRow(comm, col_map_off_d[i],
                                              0, global_num_cols, &tmp_id);

         if (ex_num_contacts == size) /*need more space? */
         {
            size += 20;
            ex_contact_procs = hypre_TReAlloc(ex_contact_procs, HYPRE_Int, size, HYPRE_MEMORY_HOST);
            ex_contact_vec_starts = hypre_TReAlloc(ex_contact_vec_starts,  HYPRE_Int,  size + 1,
                                                   HYPRE_MEMORY_HOST);
            ex_contact_buf = hypre_TReAlloc(ex_contact_buf,  HYPRE_BigInt,  size * 2,
                                            HYPRE_MEMORY_HOST);
         }

         /* end of prev. range */
         if (ex_num_contacts > 0)
         {
            ex_contact_buf[ex_num_contacts * 2 - 1] = col_map_off_d[i - 1];
         }

         /*start new range*/
         ex_contact_procs[ex_num_contacts] = tmp_id;
         ex_contact_vec_starts[ex_num_contacts] = ex_num_contacts * 2;
         ex_contact_buf[ex_num_contacts * 2] =  col_map_off_d[i];

         ex_num_contacts++;

         hypre_GetAssumedPartitionRowRange(comm, tmp_id, 0, global_num_cols,
                                           &range_start, &range_end);
      }
   }

   /*finish the starts*/
   ex_contact_vec_starts[ex_num_contacts] =  ex_num_contacts * 2;

   /*finish the last range*/
   if (ex_num_contacts > 0)
   {
      ex_contact_buf[ex_num_contacts * 2 - 1] = col_map_off_d[num_cols_off_d - 1];
   }

   /*don't allocate space for responses */

   /*create response object*/
   response_obj1.fill_response = hypre_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/
   response_obj1.data2 = NULL;

   max_response_size = 6;  /* 6 means we can fit 3 ranges*/

   hypre_DataExchangeList(ex_num_contacts, ex_contact_procs,
                          ex_contact_buf, ex_contact_vec_starts, sizeof(HYPRE_BigInt),
                          sizeof(HYPRE_BigInt), &response_obj1, max_response_size, 1,
                          comm, (void**) &response_buf, &response_buf_starts);

   /*now create recv_procs[] and recv_vec_starts[] and num_recvs
     from the complete data in response_buf - this array contains
     a proc_id followed by an upper bound for the range.  */

   /*initialize */
   num_recvs = 0;
   size  = ex_num_contacts + 20; /* num of recv procs should be roughly similar size
                                 to number of contacts  - add a buffer of 20*/

   recv_procs = hypre_CTAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);
   recv_vec_starts =  hypre_CTAlloc(HYPRE_Int,  size + 1, HYPRE_MEMORY_HOST);
   recv_vec_starts[0] = 0;

   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];
   num_ranges = num_ranges / 2;

   prev_id = -1;
   j = 0;
   count = 0;

   /* loop through ranges */
   for (i = 0; i < num_ranges; i++)
   {
      upper_bound = response_buf[i * 2 + 1];
      count = 0;
      /* loop through off_d entries - counting how many are in the range */
      while (j < num_cols_off_d && col_map_off_d[j] <= upper_bound)
      {
         j++;
         count++;
      }
      if (count > 0)
      {
         /*add the range if the proc id != myid*/
         tmp_id = response_buf[i * 2];
         if (tmp_id != myid)
         {
            if (tmp_id != prev_id) /*increment the number of recvs */
            {
               /*check size of recv buffers*/
               if (num_recvs == size)
               {
                  size += 20;
                  recv_procs = hypre_TReAlloc(recv_procs, HYPRE_Int,  size, HYPRE_MEMORY_HOST);
                  recv_vec_starts = hypre_TReAlloc(recv_vec_starts, HYPRE_Int,
                                                   size + 1, HYPRE_MEMORY_HOST);
               }

               recv_vec_starts[num_recvs + 1] = j; /*the new start is at this element*/
               recv_procs[num_recvs] =  tmp_id; /*add the new processor*/
               num_recvs++;
            }
            else
            {
               /*same processor - just change the vec starts*/
               recv_vec_starts[num_recvs] = j; /*the new start is at this element*/
            }
         }
         prev_id = tmp_id;
      }
   }

#if mydebug
   for (i = 0; i < num_recvs; i++)
   {
      hypre_printf("myid = %d, recv proc = %d, vec_starts = [%d : %d]\n",
                   myid, recv_procs[i], recv_vec_starts[i], recv_vec_starts[i + 1] - 1);
   }
#endif

   /*------------------------------------------------------------
    *  determine the send processors
    *  each processor contacts its recv procs to let them
    *  know they are a send processor
    *-------------------------------------------------------------*/

   /* the contact information is the recv_processor infomation - so
      nothing more to do to generate contact info*/

   /* the response we expect is just a confirmation*/
   hypre_TFree(response_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf_starts, HYPRE_MEMORY_HOST);
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/
   /*estimate for inital storage allocation that we send to as many procs
     as we recv from + pad by 5*/
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_recvs + 5;
   send_proc_obj.id = hypre_CTAlloc(HYPRE_Int,  send_proc_obj.storage_length, HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts = hypre_CTAlloc(HYPRE_Int,  send_proc_obj.storage_length + 1,
                                            HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = num_cols_off_d;
   send_proc_obj.elements = hypre_CTAlloc(HYPRE_BigInt,  send_proc_obj.element_storage_length,
                                          HYPRE_MEMORY_HOST);

   response_obj2.fill_response = hypre_FillResponseIJDetermineSendProcs;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj; /*this is where we keep info from contacts*/

   max_response_size = 0;

   hypre_DataExchangeList(num_recvs, recv_procs,
                          col_map_off_d, recv_vec_starts, sizeof(HYPRE_BigInt),
                          sizeof(HYPRE_BigInt), &response_obj2, max_response_size, 2,
                          comm,  (void **) &response_buf, &response_buf_starts);

   num_sends = send_proc_obj.length;

   /*send procs are in send_proc_object.id */
   /*send proc starts are in send_proc_obj.vec_starts */

#if mydebug
   hypre_printf("myid = %d, num_sends = %d\n", myid, num_sends);
   for (i = 0; i < num_sends; i++)
   {
      tmp_int = send_proc_obj.vec_starts[i + 1] - send_proc_obj.vec_starts[i];
      index = send_proc_obj.vec_starts[i];
      for (j = 0; j < tmp_int; j++)
      {
         hypre_printf("myid = %d, send proc = %d, send element = %d\n", myid,
                      send_proc_obj.id[i], send_proc_obj.elements[index + j]);
      }
   }
#endif

   /*-----------------------------------------------------------
    *  We need to sort the send procs and send elements (to produce
    *  the same result as with the standard comm package)
    *   11/07/05
    *-----------------------------------------------------------*/

   {

      HYPRE_Int *orig_order;
      HYPRE_Int *orig_send_map_starts;
      HYPRE_BigInt *orig_send_elements;
      HYPRE_Int  ct, sz, pos;

      orig_order = hypre_CTAlloc(HYPRE_Int,  num_sends, HYPRE_MEMORY_HOST);
      orig_send_map_starts = hypre_CTAlloc(HYPRE_Int,  num_sends + 1, HYPRE_MEMORY_HOST);
      orig_send_elements = hypre_CTAlloc(HYPRE_BigInt,  send_proc_obj.vec_starts[num_sends],
                                         HYPRE_MEMORY_HOST);

      orig_send_map_starts[0] = 0;
      /* copy send map starts and elements */
      for (i = 0; i < num_sends; i++)
      {
         orig_order[i] = i;
         orig_send_map_starts[i + 1] = send_proc_obj.vec_starts[i + 1];
      }
      for (i = 0; i < send_proc_obj.vec_starts[num_sends]; i++)
      {
         orig_send_elements[i] = send_proc_obj.elements[i];
      }
      /* sort processor ids - keep track of original order */
      hypre_qsort2i( send_proc_obj.id, orig_order, 0, num_sends - 1 );

      /* now rearrange vec starts and send elements to correspond to proc ids */
      ct = 0;
      for (i = 0; i < num_sends; i++)
      {
         pos = orig_order[i];
         sz = orig_send_map_starts[pos + 1] - orig_send_map_starts[pos];
         send_proc_obj.vec_starts[i + 1] =  ct + sz;
         for (j = 0; j < sz; j++)
         {
            send_proc_obj.elements[ct + j] = orig_send_elements[orig_send_map_starts[pos] + j];
         }
         ct += sz;
      }
      /* clean up */
      hypre_TFree(orig_order, HYPRE_MEMORY_HOST);
      hypre_TFree(orig_send_elements, HYPRE_MEMORY_HOST);
      hypre_TFree(orig_send_map_starts, HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------
    *  Return output info for setting up the comm package
    *-----------------------------------------------------------*/

   if (!num_recvs)
   {
      hypre_TFree(recv_procs, HYPRE_MEMORY_HOST);
      recv_procs = NULL;
   }
   if (!num_sends)
   {
      hypre_TFree(send_proc_obj.id, HYPRE_MEMORY_HOST);
      send_proc_obj.id = NULL;
   }

   *p_num_recvs = num_recvs;
   *p_recv_procs = recv_procs;
   *p_recv_vec_starts = recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_proc_obj.id;
   *p_send_map_starts = send_proc_obj.vec_starts;

   /*send map elements have global index - need local instead*/
   /*need to fix this !!! */

   if (num_sends)
   {
      HYPRE_Int *tmp_elements = hypre_CTAlloc(HYPRE_Int, send_proc_obj.vec_starts[num_sends],
                                              HYPRE_MEMORY_HOST);
      for (i = 0; i < send_proc_obj.vec_starts[num_sends]; i++)
      {
         //send_proc_obj.elements[i] -= first_col_diag;
         tmp_elements[i] = (HYPRE_Int)(send_proc_obj.elements[i] - first_col_diag);
      }
      *p_send_map_elements =  tmp_elements;
      hypre_TFree(send_proc_obj.elements, HYPRE_MEMORY_HOST);
      send_proc_obj.elements = NULL;

   }
   else
   {
      hypre_TFree(send_proc_obj.elements, HYPRE_MEMORY_HOST);
      send_proc_obj.elements = NULL;
      *p_send_map_elements =  NULL;
   }

   //*p_send_map_elements =  send_proc_obj.elements;

   /*-----------------------------------------------------------
    *  Clean up
    *-----------------------------------------------------------*/

   hypre_TFree(ex_contact_procs, HYPRE_MEMORY_HOST);
   hypre_TFree(ex_contact_vec_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(ex_contact_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf_starts, HYPRE_MEMORY_HOST);

   /* don't free send_proc_obj.id,send_proc_obj.vec_starts,send_proc_obj.elements;
      recv_procs, recv_vec_starts.  These are aliased to the comm package and
      will be destroyed there */

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommPkgCreateApart
 * this is an alternate way of constructing the comm package
 * compare with hypre_ParCSRCommPkgCreate() in par_csr_communication.c
 * which should be more scalable
 *-------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRCommPkgCreateApart
(
   /* input args: */
   MPI_Comm   comm,
   HYPRE_BigInt *col_map_off_d,
   HYPRE_BigInt  first_col_diag,
   HYPRE_Int  num_cols_off_d,
   HYPRE_BigInt  global_num_cols,
   hypre_IJAssumedPart *apart,
   /* output */
   hypre_ParCSRCommPkg *comm_pkg
)
{
   HYPRE_Int  num_sends, *send_procs, *send_map_starts;
   HYPRE_Int  num_recvs, *recv_procs, *recv_vec_starts;
   HYPRE_Int *send_map_elmts;

   /*-----------------------------------------------------------
    * get commpkg info information
    *----------------------------------------------------------*/

   hypre_ParCSRCommPkgCreateApart_core( comm, col_map_off_d, first_col_diag,
                                        num_cols_off_d, global_num_cols,
                                        &num_recvs, &recv_procs, &recv_vec_starts,
                                        &num_sends, &send_procs, &send_map_starts,
                                        &send_map_elmts, apart);

   /* Fill the communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_NewCommPkgDestroy
 *
 * Destroy the comm package
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_NewCommPkgDestroy( hypre_ParCSRMatrix *parcsr_A )
{
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);

   /*even if num_sends and num_recvs  = 0, storage may have been allocated */

   if (hypre_ParCSRCommPkgSendProcs(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendProcs(comm_pkg), HYPRE_MEMORY_HOST);
   }
   if (hypre_ParCSRCommPkgSendMapElmts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendMapElmts(comm_pkg), HYPRE_MEMORY_HOST);
   }
   if (hypre_ParCSRCommPkgSendMapStarts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg), HYPRE_MEMORY_HOST);
   }
   if (hypre_ParCSRCommPkgRecvProcs(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgRecvProcs(comm_pkg), HYPRE_MEMORY_HOST);
   }
   if (hypre_ParCSRCommPkgRecvVecStarts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg), HYPRE_MEMORY_HOST);
   }

   hypre_TFree(comm_pkg, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixCommPkg(parcsr_A) = NULL;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_RangeFillResponseIJDetermineRecvProcs
 *
 * Fill response function for determining the recv. processors
 * data exchange
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_RangeFillResponseIJDetermineRecvProcs( void      *p_recv_contact_buf,
                                             HYPRE_Int  contact_size,
                                             HYPRE_Int  contact_proc,
                                             void      *ro,
                                             MPI_Comm   comm,
                                             void     **p_send_response_buf,
                                             HYPRE_Int *response_message_size )
{
   HYPRE_UNUSED_VAR(contact_size);
   HYPRE_UNUSED_VAR(contact_proc);
   HYPRE_UNUSED_VAR(p_send_response_buf);

   HYPRE_Int    myid, tmp_id;
   HYPRE_BigInt row_end;
   HYPRE_Int    j;
   HYPRE_Int    index, size;
   HYPRE_BigInt row_val;

   HYPRE_BigInt   *send_response_buf = (HYPRE_BigInt *) *p_send_response_buf;
   HYPRE_BigInt   *recv_contact_buf = (HYPRE_BigInt * ) p_recv_contact_buf;


   hypre_DataExchangeResponse  *response_obj = (hypre_DataExchangeResponse*)ro;
   hypre_IJAssumedPart               *part = (hypre_IJAssumedPart*)response_obj->data1;

   HYPRE_Int overhead = response_obj->send_response_overhead;

   /*-------------------------------------------------------------------
    * we are getting a range of off_d entries - need to see if we own them
    * or how many ranges to send back  - send back
    * with format [proc_id end_row  proc_id #end_row  proc_id #end_row etc...].
    *----------------------------------------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid);

   /* populate send_response_buf */

   index = 0; /*count entries in send_response_buf*/

   j = 0; /*marks which partition of the assumed partition we are in */
   row_val = recv_contact_buf[0]; /*beginning of range*/
   row_end = part->row_end_list[part->sort_index[j]];
   tmp_id  = part->proc_list[part->sort_index[j]];

   /*check storage in send_buf for adding the ranges */
   size = 2 * (part->length);

   if (response_obj->send_response_storage < size)
   {

      response_obj->send_response_storage =  hypre_max(size, 20);
      send_response_buf = hypre_TReAlloc(send_response_buf, HYPRE_BigInt,
                                         response_obj->send_response_storage + overhead,
                                         HYPRE_MEMORY_HOST);
      *p_send_response_buf = send_response_buf;    /* needed when using ReAlloc */
   }

   while (row_val > row_end) /*which partition to start in */
   {
      j++;
      row_end = part->row_end_list[part->sort_index[j]];
      tmp_id = part->proc_list[part->sort_index[j]];
   }

   /*add this range*/
   send_response_buf[index++] = (HYPRE_BigInt)tmp_id;
   send_response_buf[index++] = row_end;

   j++; /*increase j to look in next partition */

   /*any more?  - now compare with end of range value*/
   row_val = recv_contact_buf[1]; /*end of range*/
   while (j < part->length && row_val > row_end )
   {
      row_end = part->row_end_list[part->sort_index[j]];
      tmp_id = part->proc_list[part->sort_index[j]];

      send_response_buf[index++] = (HYPRE_BigInt) tmp_id;
      send_response_buf[index++] = row_end;

      j++;
   }

   *response_message_size = index;
   *p_send_response_buf = send_response_buf;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_FillResponseIJDetermineSendProcs
 *
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FillResponseIJDetermineSendProcs(void       *p_recv_contact_buf,
                                       HYPRE_Int   contact_size,
                                       HYPRE_Int   contact_proc,
                                       void       *ro,
                                       MPI_Comm    comm,
                                       void      **p_send_response_buf,
                                       HYPRE_Int  *response_message_size )
{
   HYPRE_UNUSED_VAR(p_send_response_buf);

   HYPRE_Int    myid;
   HYPRE_Int    i, index, count, elength;

   HYPRE_BigInt *recv_contact_buf = (HYPRE_BigInt * ) p_recv_contact_buf;

   hypre_DataExchangeResponse  *response_obj = (hypre_DataExchangeResponse*)ro;

   hypre_ProcListElements      *send_proc_obj = (hypre_ProcListElements*)response_obj->data2;


   hypre_MPI_Comm_rank(comm, &myid );

   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length += 20; /*add space for 20 more processors*/
      send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id, HYPRE_Int,
                                         send_proc_obj->storage_length, HYPRE_MEMORY_HOST);
      send_proc_obj->vec_starts = hypre_TReAlloc(send_proc_obj->vec_starts, HYPRE_Int,
                                                 send_proc_obj->storage_length + 1,
                                                 HYPRE_MEMORY_HOST);
   }

   /*initialize*/
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the number of elements*/

   /*send proc*/
   send_proc_obj->id[count] = contact_proc;

   /*do we need more storage for the elements?*/
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = hypre_max(contact_size, 50);
      elength += index;
      send_proc_obj->elements = hypre_TReAlloc(send_proc_obj->elements,
                                               HYPRE_BigInt, elength, HYPRE_MEMORY_HOST);
      send_proc_obj->element_storage_length = elength;
   }
   /*populate send_proc_obj*/
   for (i = 0; i < contact_size; i++)
   {
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count + 1] = index;
   send_proc_obj->length++;

   /*output - no message to return (confirmation) */
   *response_message_size = 0;

   return hypre_error_flag;
}
