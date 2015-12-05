/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/


#ifndef hypre_EXCHANGE_DATA_HEADER
#define hypre_EXCHANGE_DATA_HEADER

#define hypre_BinaryTreeParentId(tree)      (tree->parent_id)
#define hypre_BinaryTreeNumChild(tree)      (tree->num_child)
#define hypre_BinaryTreeChildIds(tree)      (tree->child_id)
#define hypre_BinaryTreeChildId(tree, i)    (tree->child_id[i])


typedef struct
{
   HYPRE_Int                   parent_id;
   HYPRE_Int                   num_child;
   HYPRE_Int		        *child_id;
} hypre_BinaryTree;



/* In the fill_response() function the user needs to set the recv__buf
   and the response_message_size.  Memory of size send_response_storage has been
   alllocated for the send_buf (in exchange_data) - if more is needed, then
   realloc and adjust
   the send_response_storage.  The realloc amount should be storage+overhead. 
   If the response is an empty "confirmation" message, then set
   response_message_size =0 (and do not modify the send_buf) */


typedef struct
{
   HYPRE_Int    (*fill_response)(void* recv_buf, HYPRE_Int contact_size, 
                           HYPRE_Int contact_proc, void* response_obj, 
                           MPI_Comm comm, void** response_buf, 
                           HYPRE_Int* response_message_size);
   HYPRE_Int     send_response_overhead; /*set by exchange data */
   HYPRE_Int     send_response_storage;  /*storage allocated for send_response_buf*/
   void    *data1;                 /*data fields user may want to access in fill_response */
   void    *data2;
   
} hypre_DataExchangeResponse;


HYPRE_Int hypre_CreateBinaryTree(HYPRE_Int, HYPRE_Int, hypre_BinaryTree*);
HYPRE_Int hypre_DestroyBinaryTree(hypre_BinaryTree*);


HYPRE_Int hypre_DataExchangeList(HYPRE_Int num_contacts, 
		     HYPRE_Int *contact_proc_list, void *contact_send_buf, 
		     HYPRE_Int *contact_send_buf_starts, HYPRE_Int contact_obj_size, 
                     HYPRE_Int response_obj_size,
		     hypre_DataExchangeResponse *response_obj, HYPRE_Int max_response_size, 
                     HYPRE_Int rnum, MPI_Comm comm,  void **p_response_recv_buf, 
                     HYPRE_Int **p_response_recv_buf_starts);


#endif /* end of header */
