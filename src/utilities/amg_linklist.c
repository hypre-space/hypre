/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/***************************************************************************
 *
 *    Routines for linked list for boomerAMG
 *
 ****************************************************************************/

#include "_hypre_utilities.h"

#define hypre_LIST_HEAD -1
#define hypre_LIST_TAIL -2


/**************************************************************
 *
 * dispose_elt(): dispose of memory space used by the element
 *                pointed to by element_ptr and return it to
 *                the memory pool.
 *
 **************************************************************/
void hypre_dispose_elt ( hypre_LinkList element_ptr )
{
   hypre_TFree(element_ptr, HYPRE_MEMORY_HOST);
}



/*****************************************************************
 *
 * remove_point:   removes a point from the lists
 *
 ****************************************************************/
void
hypre_remove_point(hypre_LinkList   *LoL_head_ptr,
                   hypre_LinkList         *LoL_tail_ptr,
                   HYPRE_Int               measure,
                   HYPRE_Int               index,
                   HYPRE_Int              *lists,
                   HYPRE_Int              *where)

{
   hypre_LinkList  LoL_head = *LoL_head_ptr;
   hypre_LinkList  LoL_tail = *LoL_tail_ptr;
   hypre_LinkList  list_ptr;

   list_ptr =  LoL_head;

   do
   {
      if (measure == list_ptr->data)
      {

         /* point to be removed is only point on list,
            which must be destroyed */
         if (list_ptr->head == index && list_ptr->tail == index)
         {
            /* removing only list, so num_left better be 0! */
            if (list_ptr == LoL_head && list_ptr == LoL_tail)
            {
               LoL_head = NULL;
               LoL_tail = NULL;
               hypre_dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else if (LoL_head == list_ptr) /*removing 1st (max_measure) list */
            {
               list_ptr -> next_elt -> prev_elt = NULL;
               LoL_head = list_ptr->next_elt;
               hypre_dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else if (LoL_tail == list_ptr)     /* removing last list */
            {
               list_ptr -> prev_elt -> next_elt = NULL;
               LoL_tail = list_ptr->prev_elt;
               hypre_dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else
            {
               list_ptr -> next_elt -> prev_elt = list_ptr -> prev_elt;
               list_ptr -> prev_elt -> next_elt = list_ptr -> next_elt;
               hypre_dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
         }
         else if (list_ptr->head == index)      /* index is head of list */
         {
            list_ptr->head = lists[index];
            where[lists[index]] = hypre_LIST_HEAD;
            return;
         }
         else if (list_ptr->tail == index)      /* index is tail of list */
         {
            list_ptr->tail = where[index];
            lists[where[index]] = hypre_LIST_TAIL;
            return;
         }
         else                              /* index is in middle of list */
         {
            lists[where[index]] = lists[index];
            where[lists[index]] = where[index];
            return;
         }
      }
      list_ptr = list_ptr -> next_elt;
   }
   while (list_ptr != NULL);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "No such list!\n");

   return ;
}

/*****************************************************************
 *
 * hypre_create_elt() : Create an element using Item for its data field
 *
 *****************************************************************/
hypre_LinkList hypre_create_elt( HYPRE_Int Item )
{
   hypre_LinkList   new_elt_ptr;

   /* Allocate memory space for the new node.
    * return with error if no space available
    */
   if ( (new_elt_ptr = hypre_TAlloc(hypre_ListElement, 1, HYPRE_MEMORY_HOST)) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "\n create_elt: malloc failed \n\n");
   }
   else
      /* new_elt_ptr = hypre_CTAlloc(hypre_LinkList, 1); */
   {
      new_elt_ptr -> data = Item;
      new_elt_ptr -> next_elt = NULL;
      new_elt_ptr -> prev_elt = NULL;
      new_elt_ptr -> head = hypre_LIST_TAIL;
      new_elt_ptr -> tail = hypre_LIST_HEAD;
   }

   return (new_elt_ptr);
}

/*****************************************************************
 *
 * enter_on_lists  places point in new list
 *
 ****************************************************************/
void
hypre_enter_on_lists(hypre_LinkList   *LoL_head_ptr,
                     hypre_LinkList   *LoL_tail_ptr,
                     HYPRE_Int         measure,
                     HYPRE_Int         index,
                     HYPRE_Int        *lists,
                     HYPRE_Int        *where)
{
   hypre_LinkList   LoL_head = *LoL_head_ptr;
   hypre_LinkList   LoL_tail = *LoL_tail_ptr;

   hypre_LinkList   list_ptr;
   hypre_LinkList   new_ptr;

   HYPRE_Int         old_tail;

   list_ptr =  LoL_head;

   if (LoL_head == NULL)   /* no lists exist yet */
   {
      new_ptr = hypre_create_elt(measure);
      new_ptr->head = index;
      new_ptr->tail = index;
      lists[index] = hypre_LIST_TAIL;
      where[index] = hypre_LIST_HEAD;
      LoL_head = new_ptr;
      LoL_tail = new_ptr;

      *LoL_head_ptr = LoL_head;
      *LoL_tail_ptr = LoL_tail;
      return;
   }
   else
   {
      do
      {
         if (measure > list_ptr->data)
         {
            new_ptr = hypre_create_elt(measure);
            new_ptr->head = index;
            new_ptr->tail = index;
            lists[index] = hypre_LIST_TAIL;
            where[index] = hypre_LIST_HEAD;

            if ( list_ptr->prev_elt != NULL)
            {
               new_ptr->prev_elt            = list_ptr->prev_elt;
               list_ptr->prev_elt->next_elt = new_ptr;
               list_ptr->prev_elt           = new_ptr;
               new_ptr->next_elt            = list_ptr;
            }
            else
            {
               new_ptr->next_elt  = list_ptr;
               list_ptr->prev_elt = new_ptr;
               new_ptr->prev_elt  = NULL;
               LoL_head = new_ptr;
            }

            *LoL_head_ptr = LoL_head;
            *LoL_tail_ptr = LoL_tail;
            return;
         }
         else if (measure == list_ptr->data)
         {
            old_tail = list_ptr->tail;
            lists[old_tail] = index;
            where[index] = old_tail;
            lists[index] = hypre_LIST_TAIL;
            list_ptr->tail = index;
            return;
         }

         list_ptr = list_ptr->next_elt;
      }
      while (list_ptr != NULL);

      new_ptr = hypre_create_elt(measure);
      new_ptr->head = index;
      new_ptr->tail = index;
      lists[index] = hypre_LIST_TAIL;
      where[index] = hypre_LIST_HEAD;
      LoL_tail->next_elt = new_ptr;
      new_ptr->prev_elt = LoL_tail;
      new_ptr->next_elt = NULL;
      LoL_tail = new_ptr;

      *LoL_head_ptr = LoL_head;
      *LoL_tail_ptr = LoL_tail;

      return;
   }
}

