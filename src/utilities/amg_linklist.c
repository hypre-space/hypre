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



/***************************************************************************
 *
 *    Routines for linked list for boomerAMG
 *
 ****************************************************************************/

#include "_hypre_utilities.h"

#define LIST_HEAD -1
#define LIST_TAIL -2


/**************************************************************
 *
 * dispose_elt(): dispose of memory space used by the element
 *                pointed to by element_ptr.  Use the 'free()'
 *                system call to return it to the free memory 
 *                pool.
 *
 **************************************************************/
void dispose_elt ( hypre_LinkList element_ptr )
{
   free( element_ptr );
}



/*****************************************************************
 * 
 * remove_point:   removes a point from the lists
 *
 ****************************************************************/
void 
remove_point(hypre_LinkList   *LoL_head_ptr, 
             hypre_LinkList   *LoL_tail_ptr, 
             HYPRE_Int                 measure,
             HYPRE_Int                 index, 
             HYPRE_Int                *lists, 
             HYPRE_Int                *where)

{
   hypre_LinkList   LoL_head = *LoL_head_ptr;
   hypre_LinkList   LoL_tail = *LoL_tail_ptr;
   hypre_LinkList   list_ptr;

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
               dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else if (LoL_head == list_ptr) /*removing 1st (max_measure) list */
            {
               list_ptr -> next_elt -> prev_elt = NULL;
               LoL_head = list_ptr->next_elt;
               dispose_elt(list_ptr);
               
               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else if (LoL_tail == list_ptr)     /* removing last list */
            {
               list_ptr -> prev_elt -> next_elt = NULL;
               LoL_tail = list_ptr->prev_elt;
               dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else
            {
               list_ptr -> next_elt -> prev_elt = list_ptr -> prev_elt;
               list_ptr -> prev_elt -> next_elt = list_ptr -> next_elt;
               dispose_elt(list_ptr);
               
               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
         }
         else if (list_ptr->head == index)      /* index is head of list */
         {
            list_ptr->head = lists[index];
            where[lists[index]] = LIST_HEAD;
            return;
         }
         else if (list_ptr->tail == index)      /* index is tail of list */
         {
            list_ptr->tail = where[index];
            lists[where[index]] = LIST_TAIL;
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
   } while (list_ptr != NULL);
   
   hypre_printf("No such list!\n");
   return;
}

/*****************************************************************
 *
 * create_elt() : Create an element using Item for its data field
 *
 *****************************************************************/
hypre_LinkList create_elt( HYPRE_Int Item )
{
    hypre_LinkList   new_elt_ptr;
 
    /* Allocate memory space for the new node. 
     * return with error if no space available
     */

    if ( (new_elt_ptr = (hypre_LinkList) malloc (sizeof(hypre_ListElement))) == NULL)
    {
       hypre_printf("\n create_elt: malloc failed \n\n");
    }
    else 

       /*   new_elt_ptr = hypre_CTAlloc(hypre_LinkList, 1); */

    {
       new_elt_ptr -> data = Item;
       new_elt_ptr -> next_elt = NULL;
       new_elt_ptr -> prev_elt = NULL;
       new_elt_ptr -> head = LIST_TAIL;
       new_elt_ptr -> tail = LIST_HEAD;
    }

    return (new_elt_ptr);
}

/*****************************************************************
 * 
 * enter_on_lists   places point in new list
 *
 ****************************************************************/
void 
enter_on_lists(hypre_LinkList   *LoL_head_ptr, 
               hypre_LinkList   *LoL_tail_ptr, 
               HYPRE_Int                 measure,
               HYPRE_Int                 index, 
               HYPRE_Int                *lists, 
               HYPRE_Int                *where)
{
   hypre_LinkList   LoL_head = *LoL_head_ptr;
   hypre_LinkList   LoL_tail = *LoL_tail_ptr;

   hypre_LinkList   list_ptr;
   hypre_LinkList   new_ptr;

   HYPRE_Int         old_tail;

   list_ptr =  LoL_head;

   if (LoL_head == NULL)   /* no lists exist yet */
   {
      new_ptr = create_elt(measure);
      new_ptr->head = index;
      new_ptr->tail = index;
      lists[index] = LIST_TAIL;
      where[index] = LIST_HEAD; 
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
         new_ptr = create_elt(measure);
         new_ptr->head = index;
         new_ptr->tail = index;
         lists[index] = LIST_TAIL;
         where[index] = LIST_HEAD;

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
         lists[index] = LIST_TAIL;
         list_ptr->tail = index;
         return;
      }
      
      list_ptr = list_ptr->next_elt;
   } while (list_ptr != NULL);

   new_ptr = create_elt(measure);   
   new_ptr->head = index;
   new_ptr->tail = index;
   lists[index] = LIST_TAIL;
   where[index] = LIST_HEAD;
   LoL_tail->next_elt = new_ptr;
   new_ptr->prev_elt = LoL_tail;
   new_ptr->next_elt = NULL;
   LoL_tail = new_ptr;

   *LoL_head_ptr = LoL_head;
   *LoL_tail_ptr = LoL_tail;
   return;
 }
}


