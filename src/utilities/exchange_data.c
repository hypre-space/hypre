/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* see exchange_data.README for additional information */
/* AHB 6/04 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"

/*---------------------------------------------------
 * hypre_CreateBinaryTree
 *
 * Get the processors position in the binary tree (i.e.,
 * its children and parent processor ids)
 *----------------------------------------------------*/

HYPRE_Int
hypre_CreateBinaryTree(HYPRE_Int          myid,
                       HYPRE_Int          num_procs,
                       hypre_BinaryTree **tree_ptr)
{
   hypre_BinaryTree *tree;
   HYPRE_Int  i, proc, size = 0;
   HYPRE_Int  *tmp_child_id;
   HYPRE_Int  num = 0, parent = 0;

   tree = hypre_CTAlloc(hypre_BinaryTree, 1, HYPRE_MEMORY_HOST);

   /* initialize */
   proc = myid;

   /*how many children can a processor have?*/
   for (i = 1; i < num_procs; i *= 2)
   {
      size++;
   }

   /* allocate space */
   tmp_child_id = hypre_TAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);

   /* find children and parent */
   for (i = 1; i < num_procs; i *= 2)
   {
      if ( (proc % 2) == 0)
      {
         if ( (myid + i) < num_procs )
         {
            tmp_child_id[num] = myid + i;
            num++;
         }
         proc /= 2;
      }
      else
      {
         parent = myid - i;
         break;
      }
   }

   hypre_BinaryTreeParentId(tree) = parent;
   hypre_BinaryTreeNumChild(tree) = num;
   hypre_BinaryTreeChildIds(tree) = tmp_child_id;

   *tree_ptr = tree;

   return hypre_error_flag;
}

/*---------------------------------------------------
 * hypre_DestroyBinaryTree()
 *
 * Destroy storage created by hypre_CreateBinaryTree
 *----------------------------------------------------*/

HYPRE_Int
hypre_DestroyBinaryTree(hypre_BinaryTree *tree)
{
   if (tree)
   {
      hypre_TFree(hypre_BinaryTreeChildIds(tree), HYPRE_MEMORY_HOST);
      hypre_TFree(tree, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*---------------------------------------------------
 * hypre_DataExchangeList()
 *
 * This function is for sending a list of messages ("contacts" to
 * a list of processors.  The receiving processors
 * do not know how many messages they are getting. The
 * sending process expects a "response" (either a confirmation or
 * some sort of data back from the receiving processor).
 *----------------------------------------------------*/

/* should change to where the buffers for sending and receiving are voids
   instead of ints - then cast accordingly */

HYPRE_Int
hypre_DataExchangeList(HYPRE_Int num_contacts,
                       HYPRE_Int *contact_proc_list,
                       void *contact_send_buf,
                       HYPRE_Int *contact_send_buf_starts,
                       HYPRE_Int contact_obj_size,
                       HYPRE_Int response_obj_size,
                       hypre_DataExchangeResponse *response_obj,
                       HYPRE_Int max_response_size,
                       HYPRE_Int rnum,
                       MPI_Comm comm,
                       void **p_response_recv_buf,
                       HYPRE_Int **p_response_recv_buf_starts)
{
   /*-------------------------------------------
    *  parameters:
    *
    *    num_contacts              = how many procs to contact
    *    contact_proc_list         = list of processors to contact
    *    contact_send_buf          = array of data to send
    *    contact_send_buf_starts   = index for contact_send_buf corresponding to
    *                                contact_proc_list
    *    contact_obj_size          = sizeof() one obj in contact list

    *    response_obj_size          = sizeof() one obj in response_recv_buf
    *    response_obj              = this will give us the function we need to
    *                                fill the reponse as well as
    *                                any data we might need to accomplish that
    *    max_response_size         = max size of a single response expected (do NOT
    *                                need to be an absolute upper bound)
    *    rnum                      = two consequentive exchanges should have different
    *                                rnums. Alternate rnum = 1
    *                                and rnum=2  - these flags will be even (so odd
    *                                numbered tags could be used in calling code)
    *    p_response_recv_buf       = where to receive the reponses - will be allocated
    *                                in this function
    *    p_response_recv_buf_starts  = index of p_response_buf corresponding to
    *                                contact_buf_list - will be allocated here

    *-------------------------------------------*/

   HYPRE_Int  num_procs, myid;
   HYPRE_Int  i;
   HYPRE_Int  terminate, responses_complete;
   HYPRE_Int  children_complete;
   HYPRE_Int  contact_flag;
   HYPRE_Int  proc;
   HYPRE_Int  contact_size;

   HYPRE_Int  size, post_size, copy_size;
   HYPRE_Int  total_size, count;

   void *start_ptr = NULL, *index_ptr = NULL;
   HYPRE_Int  *int_ptr = NULL;

   void *response_recv_buf = NULL;
   void *send_response_buf = NULL;

   HYPRE_Int  *response_recv_buf_starts = NULL;
   void *initial_recv_buf = NULL;

   void *recv_contact_buf = NULL;
   HYPRE_Int  recv_contact_buf_size = 0;

   HYPRE_Int  response_message_size = 0;

   HYPRE_Int  overhead;

   HYPRE_Int  max_response_size_bytes;

   HYPRE_Int  max_response_total_bytes;

   void **post_array = NULL;  /*this must be set to null or realloc will crash */
   HYPRE_Int  post_array_storage = 0;
   HYPRE_Int  post_array_size = 0;
   HYPRE_Int   num_post_recvs = 0;

   void **contact_ptrs = NULL, **response_ptrs = NULL, **post_ptrs = NULL;

   hypre_BinaryTree *tree = NULL;

   hypre_MPI_Request *response_requests = NULL, *contact_requests = NULL;
   hypre_MPI_Status  *response_statuses = NULL, *contact_statuses = NULL;

   hypre_MPI_Request  *post_send_requests = NULL, *post_recv_requests = NULL;
   hypre_MPI_Status   *post_send_statuses = NULL, *post_recv_statuses = NULL;

   hypre_MPI_Request *term_requests = NULL, term_request1, request_parent;
   hypre_MPI_Status  *term_statuses = NULL, term_status1, status_parent;
   hypre_MPI_Status  status, fill_status;

   const HYPRE_Int contact_tag = 1000 * rnum;
   const HYPRE_Int response_tag = 1002 * rnum;
   const HYPRE_Int term_tag =  1004 * rnum;
   const HYPRE_Int post_tag = 1006 * rnum;

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /* ---------initializations ----------------*/

   /* if the response_obj_size or contact_obj_size is 0, set to sizeof(HYPRE_Int) */
   if (!response_obj_size) { response_obj_size = sizeof(HYPRE_Int); }
   if (!contact_obj_size) { contact_obj_size = sizeof(HYPRE_Int); }

   max_response_size_bytes = max_response_size * response_obj_size;


   /* pre-allocate the max space for responding to contacts */
   overhead = (HYPRE_Int)hypre_ceil((HYPRE_Real) sizeof(HYPRE_Int) /
                                    response_obj_size); /*for appending an integer*/

   max_response_total_bytes = (max_response_size + overhead) * response_obj_size;

   response_obj->send_response_overhead = overhead;
   response_obj->send_response_storage = max_response_size;

   /*send_response_buf = hypre_TAlloc(char, max_response_total_bytes);*/
   send_response_buf = hypre_CTAlloc(char, (max_response_size + overhead) * response_obj_size,
                                     HYPRE_MEMORY_HOST);

   /*allocate space for inital recv array for the responses - give each processor
     size max_response_size */

   initial_recv_buf = hypre_TAlloc(char, max_response_total_bytes * num_contacts, HYPRE_MEMORY_HOST);
   response_recv_buf_starts =   hypre_CTAlloc(HYPRE_Int,  num_contacts + 1, HYPRE_MEMORY_HOST);

   contact_ptrs = hypre_TAlloc( void *,  num_contacts, HYPRE_MEMORY_HOST);
   response_ptrs = hypre_TAlloc(void *,  num_contacts, HYPRE_MEMORY_HOST);

   /*-------------SEND CONTACTS AND POST RECVS FOR RESPONSES---*/

   for (i = 0; i <= num_contacts; i++)
   {
      response_recv_buf_starts[i] = i * (max_response_size + overhead);
   }

   /* Send "contact" messages to the list of processors and
      pre-post receives to wait for their response*/

   responses_complete = 1;
   if (num_contacts > 0)
   {
      responses_complete = 0;
      response_requests = hypre_CTAlloc(hypre_MPI_Request,  num_contacts, HYPRE_MEMORY_HOST);
      response_statuses = hypre_CTAlloc(hypre_MPI_Status,  num_contacts, HYPRE_MEMORY_HOST);
      contact_requests = hypre_CTAlloc(hypre_MPI_Request,  num_contacts, HYPRE_MEMORY_HOST);
      contact_statuses = hypre_CTAlloc(hypre_MPI_Status,  num_contacts, HYPRE_MEMORY_HOST);

      /* post receives - could be confirmation or data*/
      /* the size to post is max_response_total_bytes*/

      for (i = 0; i < num_contacts; i++)
      {
         /* response_ptrs[i] =  initial_recv_buf + i*max_response_total_bytes ; */
         response_ptrs[i] = (void *)((char *) initial_recv_buf +
                                     i * max_response_total_bytes) ;

         hypre_MPI_Irecv(response_ptrs[i], max_response_total_bytes,
                         hypre_MPI_BYTE, contact_proc_list[i],
                         response_tag, comm, &response_requests[i]);
      }

      /* send out contact messages */
      start_ptr = contact_send_buf;
      for (i = 0; i < num_contacts; i++)
      {
         contact_ptrs[i] = start_ptr;
         size =  contact_send_buf_starts[i + 1] - contact_send_buf_starts[i]  ;
         hypre_MPI_Isend(contact_ptrs[i], size * contact_obj_size,
                         hypre_MPI_BYTE, contact_proc_list[i],
                         contact_tag, comm, &contact_requests[i]);
         /*  start_ptr += (size*contact_obj_size); */
         start_ptr = (void *) ((char *) start_ptr  + (size * contact_obj_size));
      }
   }

   /*------------BINARY TREE-----------------------*/

   /*Now let's find out our binary tree information and
     initialize for the termination check sweep */
   terminate = 1; /*indicates whether we can stop probing for contact */
   children_complete = 1;/*indicates whether we have recv. term messages
                           from our children*/

   if (num_procs > 1)
   {
      hypre_CreateBinaryTree(myid, num_procs, &tree);

      /* we will get a message from all of our children when they
         have received responses for all of their contacts.
         So post receives now */

      term_requests = hypre_CTAlloc(hypre_MPI_Request, tree -> num_child, HYPRE_MEMORY_HOST);
      term_statuses = hypre_CTAlloc(hypre_MPI_Status, tree -> num_child, HYPRE_MEMORY_HOST);

      for (i = 0; i < tree -> num_child; i++)
      {
         hypre_MPI_Irecv(NULL, 0, HYPRE_MPI_INT, (tree -> child_id)[i], term_tag, comm,
                         &term_requests[i]);
      }

      terminate = 0;
      children_complete = 0;
   }
   else if (num_procs == 1 && num_contacts > 0) /* added 11/08 */
   {
      terminate = 0;
   }

   /*---------PROBE LOOP-----------------------------------------*/

   /*Look for incoming contact messages - don't know how many I will get!*/

   while (!terminate)
   {
      /* did I receive any contact messages? */
      hypre_MPI_Iprobe(hypre_MPI_ANY_SOURCE, contact_tag, comm,
                       &contact_flag, &status);

      while (contact_flag)
      {
         /* received contacts - from who and what do we do ?*/
         proc = status.hypre_MPI_SOURCE;
         hypre_MPI_Get_count(&status, hypre_MPI_BYTE, &contact_size);

         contact_size = contact_size / contact_obj_size;

         /*---------------FILL RESPONSE ------------------------*/

         /*first receive the contact buffer - then call a function
           to determine how to populate the send buffer for the reponse*/

         /* do we have enough space to recv it? */
         if (contact_size > recv_contact_buf_size)
         {
            recv_contact_buf = hypre_TReAlloc((char*)recv_contact_buf,
                                              char, contact_obj_size * contact_size, HYPRE_MEMORY_HOST);
            recv_contact_buf_size = contact_size;
         }

         /* this must be blocking - can't fill recv without the buffer*/
         hypre_MPI_Recv(recv_contact_buf, contact_size * contact_obj_size,
                        hypre_MPI_BYTE, proc, contact_tag, comm, &fill_status);

         response_obj->fill_response(recv_contact_buf, contact_size, proc,
                                     response_obj, comm, &send_response_buf,
                                     &response_message_size );

         /* we need to append the size of the send obj */
         /* first we copy out any part that may be needed to send later so we don't overwrite */
         post_size = response_message_size - max_response_size;
         if (post_size > 0) /*we will need to send the extra information later */
         {
            /*hypre_printf("myid = %d, post_size = %d\n", myid, post_size);*/

            if (post_array_size == post_array_storage)

            {
               /* allocate room for more posts  - add 20*/
               post_array_storage += 20;
               post_array = hypre_TReAlloc(post_array,  void *,  post_array_storage, HYPRE_MEMORY_HOST);
               post_send_requests =
                  hypre_TReAlloc(post_send_requests,  hypre_MPI_Request,
                                 post_array_storage, HYPRE_MEMORY_HOST);
            }
            /* allocate space for the data this post only*/
            /* this should not happen often (unless a poor max_size has been chosen)
               - so we will allocate space for the data as needed */
            size = post_size * response_obj_size;
            post_array[post_array_size] =  hypre_TAlloc(char, size, HYPRE_MEMORY_HOST);
            /* index_ptr =  send_response_buf + max_response_size_bytes */;
            index_ptr = (void *) ((char *) send_response_buf +
                                  max_response_size_bytes);

            hypre_TMemcpy(post_array[post_array_size], index_ptr, char,  size, HYPRE_MEMORY_HOST,
                          HYPRE_MEMORY_HOST);

            /*now post any part of the message that is too long with a non-blocking
              send and a different tag */

            hypre_MPI_Isend(post_array[post_array_size], size,
                            hypre_MPI_BYTE, proc, post_tag,
                            /*hypre_MPI_COMM_WORLD, */
                            comm,
                            &post_send_requests[post_array_size]);

            post_array_size++;
         }

         /*now append the size information into the overhead storage */
         /* index_ptr =  send_response_buf + max_response_size_bytes; */
         index_ptr = (void *) ((char *) send_response_buf +
                               max_response_size_bytes);

         hypre_TMemcpy(index_ptr,  &response_message_size, HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                       HYPRE_MEMORY_HOST);

         /*send the block of data that includes the overhead */
         /* this is a blocking send - the recv has already been posted */
         hypre_MPI_Send(send_response_buf, max_response_total_bytes,
                        hypre_MPI_BYTE, proc, response_tag, comm);

         /*--------------------------------------------------------------*/

         /* look for any more contact messages*/
         hypre_MPI_Iprobe(hypre_MPI_ANY_SOURCE, contact_tag, comm,
                          &contact_flag, &status);
      }

      /* no more contact messages waiting - either
         (1) check to see if we have received all of our response messages
         (2) participate in termination (check for messages from children)
         (3) participate in termination sweep (check for message from parent) */

      if (!responses_complete)
      {
         hypre_MPI_Testall(num_contacts, response_requests, &responses_complete,
                           response_statuses);
         if (responses_complete && num_procs == 1) { terminate = 1; } /*added 11/08 */

      }
      else if (!children_complete) /* have all of our children received all of their
                                     response messages?*/
      {
         hypre_MPI_Testall(tree -> num_child, term_requests, &children_complete,
                           term_statuses);

         /* if we have gotten term messages from all of our children, send a term
            message to our parent.  Then post a receive to hear back from parent */
         if (children_complete & (myid > 0)) /*root does not have a parent*/
         {
            hypre_MPI_Isend(NULL, 0, HYPRE_MPI_INT, tree -> parent_id, term_tag,
                            comm, &request_parent);

            hypre_MPI_Irecv(NULL, 0, HYPRE_MPI_INT, tree -> parent_id, term_tag,
                            comm, &term_request1);
         }
      }
      else /*have we gotten a term message from our parent? */
      {
         if (myid == 0) /* root doesn't have a parent */
         {
            terminate = 1;
         }
         else
         {
            hypre_MPI_Test(&term_request1, &terminate, &term_status1);
         }
         if (terminate) /*tell children to terminate */
         {
            if (myid > 0 ) { hypre_MPI_Wait(&request_parent, &status_parent); }

            for (i = 0; i < tree -> num_child; i++)
            {
               /*a blocking send  - recv has been posted already*/
               hypre_MPI_Send(NULL, 0, HYPRE_MPI_INT, (tree -> child_id)[i],
                              term_tag, comm);
            }
         }
      }
   }

   /* end of (!terminate) loop */

   /* ----some clean up before post-processing ----*/
   if (recv_contact_buf_size > 0)
   {
      hypre_TFree(recv_contact_buf, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(send_response_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(contact_ptrs, HYPRE_MEMORY_HOST);
   hypre_TFree(response_ptrs, HYPRE_MEMORY_HOST);

   /*-----------------POST PROCESSING------------------------------*/

   /* more data to receive? */
   /* move to recv buffer and update response_recv_buf_starts */

   total_size = 0;  /*total number of items in response buffer */
   num_post_recvs = 0; /*num of post processing recvs to post */
   start_ptr = initial_recv_buf;
   response_recv_buf_starts[0] = 0; /*already allocated above */

   /*an extra loop to determine sizes.  This is better than reallocating
     the array that will be used in posting the irecvs */
   for (i = 0; i < num_contacts; i++)
   {
      int_ptr = (HYPRE_Int *) ((char *) start_ptr + max_response_size_bytes); /*the overhead HYPRE_Int*/

      response_message_size =  *int_ptr;
      response_recv_buf_starts[i + 1] =
         response_recv_buf_starts[i] + response_message_size;
      total_size +=  response_message_size;
      if (max_response_size < response_message_size) { num_post_recvs++; }
      /* start_ptr += max_response_total_bytes; */
      start_ptr = (void *) ((char *) start_ptr + max_response_total_bytes);
   }

   post_recv_requests = hypre_TAlloc(hypre_MPI_Request,  num_post_recvs, HYPRE_MEMORY_HOST);
   post_recv_statuses = hypre_TAlloc(hypre_MPI_Status,  num_post_recvs, HYPRE_MEMORY_HOST);
   post_ptrs = hypre_TAlloc(void *,  num_post_recvs, HYPRE_MEMORY_HOST);

   /*second loop to post any recvs and set up recv_response_buf */
   response_recv_buf = hypre_TAlloc(char, total_size * response_obj_size, HYPRE_MEMORY_HOST);
   index_ptr = response_recv_buf;
   start_ptr = initial_recv_buf;
   count = 0;

   for (i = 0; i < num_contacts; i++)
   {
      response_message_size =
         response_recv_buf_starts[i + 1] - response_recv_buf_starts[i];
      copy_size = hypre_min(response_message_size, max_response_size);

      hypre_TMemcpy(index_ptr,  start_ptr,  char, copy_size * response_obj_size, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_HOST);
      /* index_ptr += copy_size*response_obj_size; */
      index_ptr = (void *) ((char *) index_ptr + copy_size * response_obj_size);

      if (max_response_size < response_message_size)
      {
         size = (response_message_size - max_response_size) * response_obj_size;
         post_ptrs[count] = index_ptr;
         hypre_MPI_Irecv(post_ptrs[count], size, hypre_MPI_BYTE,
                         contact_proc_list[i], post_tag,
                         comm, &post_recv_requests[count]);
         count++;
         /* index_ptr+=size;*/
         index_ptr =  (void *) ((char *) index_ptr + size);
      }

      /* start_ptr += max_response_total_bytes; */
      start_ptr = (void *) ((char *) start_ptr + max_response_total_bytes);
   }

   post_send_statuses = hypre_TAlloc(hypre_MPI_Status,  post_array_size, HYPRE_MEMORY_HOST);

   /*--------------CLEAN UP------------------- */

   hypre_TFree(initial_recv_buf, HYPRE_MEMORY_HOST);

   if (num_contacts > 0 )
   {
      /*these should be done */
      hypre_MPI_Waitall(num_contacts, contact_requests, contact_statuses);

      hypre_TFree(response_requests, HYPRE_MEMORY_HOST);
      hypre_TFree(response_statuses, HYPRE_MEMORY_HOST);
      hypre_TFree(contact_requests, HYPRE_MEMORY_HOST);
      hypre_TFree(contact_statuses, HYPRE_MEMORY_HOST);
   }

   /* clean up from the post processing - the arrays, requests, etc. */

   if (num_post_recvs)
   {
      hypre_MPI_Waitall(num_post_recvs, post_recv_requests, post_recv_statuses);
      hypre_TFree(post_recv_requests, HYPRE_MEMORY_HOST);
      hypre_TFree(post_recv_statuses, HYPRE_MEMORY_HOST);
      hypre_TFree(post_ptrs, HYPRE_MEMORY_HOST);
   }

   if (post_array_size)
   {
      hypre_MPI_Waitall(post_array_size, post_send_requests, post_send_statuses);

      hypre_TFree(post_send_requests, HYPRE_MEMORY_HOST);
      hypre_TFree(post_send_statuses, HYPRE_MEMORY_HOST);

      for (i = 0; i < post_array_size; i++)
      {
         hypre_TFree(post_array[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(post_array, HYPRE_MEMORY_HOST);
   }

   if (num_procs > 1)
   {
      hypre_TFree(term_requests, HYPRE_MEMORY_HOST);
      hypre_TFree(term_statuses, HYPRE_MEMORY_HOST);

      hypre_DestroyBinaryTree(tree);
   }

   /* output  */
   *p_response_recv_buf = response_recv_buf;
   *p_response_recv_buf_starts = response_recv_buf_starts;

   return hypre_error_flag;
}
