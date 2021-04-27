/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"

/*==========================================================================
 * Member functions: hypre_BoxBTNode
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_BoxBTNodeCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTNodeCreate( HYPRE_Int         ndim,
                       hypre_BoxBTNode **btnode_ptr )
{
   hypre_BoxBTNode *btnode;
   HYPRE_Int        d;

   btnode = hypre_TAlloc(hypre_BoxBTNode, 1, HYPRE_MEMORY_HOST);

   hypre_BoxBTNodeNumIndices(btnode) = 0;
   hypre_BoxBTNodeBox(btnode)        = hypre_BoxCreate(ndim);
   hypre_BoxBTNodeLeft(btnode)       = NULL;
   hypre_BoxBTNodeRight(btnode)      = NULL;
   for (d = 0; d < HYPRE_MAXDIM; d++)
   {
      hypre_BoxBTNodeIndices(btnode, d) = NULL;
   }

   /* Set pointer */
   *btnode_ptr = btnode;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTNodeSetIndices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTNodeSetIndices( hypre_BoxBTNode * btnode,
                           HYPRE_Int         num_indices,
                           HYPRE_Int       **indices )
{
   HYPRE_Int  ndim = hypre_BoxBTNodeNDim(btnode);
   HYPRE_Int  d, i;

   hypre_BoxBTNodeNumIndices(btnode) = num_indices;
   for (d = 0; d < ndim; d++)
   {
      hypre_BoxBTNodeIndices(btnode, d) = hypre_TAlloc(HYPRE_Int, num_indices, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_indices; i++)
      {
         hypre_BoxBTNodeIndex(btnode, d, i) = indices[d][i];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTNodeInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTNodeInitialize( hypre_BoxBTNode  *btnode,
                           HYPRE_Int         num_indices,
                           HYPRE_Int       **indices,
                           hypre_Box        *box )
{
   HYPRE_Int  ndim = hypre_BoxNDim(box);
   HYPRE_Int  d;

   hypre_CopyBox(box, hypre_BoxBTNodeBox(btnode));
   hypre_BoxBTNodeNumIndices(btnode) = num_indices;
   for (d = 0; d < ndim; d++)
   {
      hypre_BoxBTNodeIndices(btnode, d) = indices[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTNodeDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTNodeDestroy( hypre_BoxBTNode *btnode )
{
   HYPRE_Int         d;

   if (btnode)
   {
      for (d = 0; d < HYPRE_MAXDIM; d++)
      {
         hypre_TFree(hypre_BoxBTNodeIndices(btnode, d), HYPRE_MEMORY_HOST);
      }

      hypre_BoxDestroy(hypre_BoxBTNodeBox(btnode));
      //hypre_assert(hypre_BoxBTNodeLeft(btnode) == NULL);
      //hypre_assert(hypre_BoxBTNodeRight(btnode) == NULL);
      hypre_TFree(btnode, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*==========================================================================
 * Member functions: hypre_BoxBinTree
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_BoxBinTreeCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBinTreeCreate( HYPRE_Int          ndim,
                        hypre_BoxBinTree **boxbt_ptr )
{
   hypre_BoxBinTree  *boxbt;
   hypre_BoxBTNode   *btroot;

   boxbt = hypre_TAlloc(hypre_BoxBinTree, 1, HYPRE_MEMORY_HOST);

   hypre_BoxBTNodeCreate(ndim, &btroot);
   hypre_BoxBinTreeRoot(boxbt) = btroot;

   /* Set pointer */
   *boxbt_ptr = boxbt;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBinTreeInitialize
 *
 * Note: indices_in is copied to indices, which will be freed when
 *       calling hypre_BoxBinTreeDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBinTreeInitialize( hypre_BoxBinTree   *boxbt,
                            HYPRE_Int           num_indices,
                            HYPRE_Int         **indices_in,
                            hypre_Box          *box )
{
   hypre_BoxBTNode   *btroot;
   HYPRE_Int         *indices[HYPRE_MAXDIM];
   HYPRE_Int          ndim = hypre_BoxNDim(box);

   HYPRE_Int          d, i;

   for (d = 0; d < ndim; d++)
   {
      indices[d] = hypre_CTAlloc(HYPRE_Int, num_indices, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_indices; i++)
      {
         indices[d][i] = indices_in[d][i];
      }
   }

   btroot = hypre_BoxBinTreeRoot(boxbt);
   hypre_BoxBTNodeInitialize(btroot, num_indices, indices, box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBinTreeDestroy
 *
 * Destroy a binary tree of boxes via iterative post-order tree traversal.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBinTreeDestroy( hypre_BoxBinTree *boxbt )
{
   hypre_BoxBTNode   *btwork;
   hypre_BoxBTNode   *btnode;
   hypre_BoxBTNode   *lnode;
   hypre_BoxBTNode   *rnode;
   hypre_BoxBTStack  *btstack;

   if (boxbt)
   {
      /* Allocate memory for stack */
      hypre_BoxBTStackCreate(&btstack);
      hypre_BoxBTStackInitialize(16, btstack);

      /* Perform iterative post-order traversal */
      btnode = hypre_BoxBinTreeRoot(boxbt);
      while ((btnode != NULL) || (hypre_BoxBTStackSize(btstack) > 0))
      {
         /* Add nodes to stack */
         while (btnode)
         {
            lnode = hypre_BoxBTNodeLeft(btnode);
            rnode = hypre_BoxBTNodeRight(btnode);

            if (rnode)
            {
               hypre_BoxBTStackInsert(rnode, btstack);
            }
            hypre_BoxBTStackInsert(btnode, btstack);

            /* Move to left-most node */
            btnode = lnode;
         }

         /* Retrieve node from stack */
         hypre_BoxBTStackDelete(btstack, &btnode);
         rnode = hypre_BoxBTNodeRight(btnode);

         if (rnode &&
             hypre_BoxBTStackSize(btstack) &&
             hypre_BoxBTStackNodePeek(btstack) == rnode)
         {
            /* Swap the two last nodes of stack */
            hypre_BoxBTStackDelete(btstack, &btwork);
            hypre_BoxBTStackInsert(btnode, btstack);
            btnode = rnode;
         }
         else
         {
            /* Free data associated with node */
            hypre_BoxBTNodeDestroy(btnode);
            btnode = NULL;
         }
      }

      /* Free memory */
      hypre_BoxBTStackDestroy(btstack);
      hypre_TFree(boxbt, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*==========================================================================
 * Member functions: hypre_BoxBTStack
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_BoxBTStackCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTStackCreate( hypre_BoxBTStack  **btstack_ptr )
{
   hypre_BoxBTStack  *btstack;

   btstack = hypre_TAlloc(hypre_BoxBTStack, 1, HYPRE_MEMORY_HOST);

   hypre_BoxBTStackNodes(btstack)    = NULL;
   hypre_BoxBTStackCapacity(btstack) = 0;
   hypre_BoxBTStackSize(btstack)     = 0;

   /* Set pointer */
   *btstack_ptr = btstack;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTStackInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTStackInitialize( HYPRE_Int           capacity,
                            hypre_BoxBTStack   *btstack )
{
   HYPRE_Int  i;

   hypre_BoxBTStackCapacity(btstack) = capacity;
   hypre_BoxBTStackNodes(btstack) = hypre_TAlloc(hypre_BoxBTNode *, capacity, HYPRE_MEMORY_HOST);
   for (i = 0; i < capacity; i++)
   {
      hypre_BoxBTStackNode(btstack, i) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTStackDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTStackDestroy( hypre_BoxBTStack *btstack )
{
   hypre_BoxBTNode  **nodes = hypre_BoxBTStackNodes(btstack);

   hypre_TFree(nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(btstack, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTStackInsert
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTStackInsert( hypre_BoxBTNode    *btnode,
                        hypre_BoxBTStack   *btstack )
{
   HYPRE_Int   capacity = hypre_BoxBTStackCapacity(btstack);
   HYPRE_Int   size     = hypre_BoxBTStackSize(btstack);

   /* Double the capacity if limit is reached */
   if (size == capacity)
   {
      capacity = hypre_max(1, 2*capacity);
      if (capacity < 0)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Integer overflow! Using capacity=INT_MAX");
         capacity = HYPRE_INT_MAX;
      }
      hypre_BoxBTStackCapacity(btstack) = capacity;
      hypre_BoxBTStackNodes(btstack) = hypre_TReAlloc(hypre_BoxBTStackNodes(btstack),
                                                      hypre_BoxBTNode *, capacity,
                                                      HYPRE_MEMORY_HOST);
   }

   hypre_BoxBTStackSize(btstack)++;
   hypre_BoxBTStackNodePeek(btstack) = btnode;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTStackDelete
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTStackDelete( hypre_BoxBTStack    *btstack,
                        hypre_BoxBTNode    **btnode_ptr )
{
   HYPRE_Int size = hypre_BoxBTStackSize(btstack);

   if (size > 0)
   {
      *btnode_ptr = hypre_BoxBTStackNodePeek(btstack);
      hypre_BoxBTStackNodePeek(btstack) = NULL;
      hypre_BoxBTStackSize(btstack)--;
   }
   else
   {
       hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Empty stack!");
      *btnode_ptr = NULL;
   }

   return hypre_error_flag;
}

/*==========================================================================
 * Member functions: hypre_BoxBTQueue
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_BoxBTQueueCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTQueueCreate( hypre_BoxBTQueue  **btqueue_ptr )
{
   hypre_BoxBTQueue  *btqueue;

   btqueue = hypre_TAlloc(hypre_BoxBTQueue, 1, HYPRE_MEMORY_HOST);

   hypre_BoxBTQueueHead(btqueue)     = 0;
   hypre_BoxBTQueueTail(btqueue)     = 0;
   hypre_BoxBTQueueSize(btqueue)     = 0;
   hypre_BoxBTQueueCapacity(btqueue) = 0;
   hypre_BoxBTQueueNodes(btqueue)    = NULL;

   /* Set pointer */
   *btqueue_ptr = btqueue;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTQueueInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTQueueInitialize( HYPRE_Int           capacity,
                            hypre_BoxBTQueue   *btqueue )
{
   HYPRE_Int  i;

   hypre_BoxBTQueueCapacity(btqueue) = capacity;
   hypre_BoxBTQueueNodes(btqueue) = hypre_TAlloc(hypre_BoxBTNode *, capacity, HYPRE_MEMORY_HOST);
   for (i = 0; i < capacity; i++)
   {
      hypre_BoxBTQueueNode(btqueue, i) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTQueueDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTQueueDestroy( hypre_BoxBTQueue *btqueue )
{
   hypre_BoxBTNode  **nodes = hypre_BoxBTQueueNodes(btqueue);

   hypre_TFree(nodes, HYPRE_MEMORY_HOST);
   hypre_TFree(btqueue, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTStackInsert
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTQueueInsert( hypre_BoxBTNode    *btnode,
                        hypre_BoxBTQueue   *btqueue )
{
   HYPRE_Int   capacity = hypre_BoxBTQueueCapacity(btqueue);
   HYPRE_Int   size     = hypre_BoxBTQueueSize(btqueue);
   HYPRE_Int   *tail    = &hypre_BoxBTQueueTail(btqueue);

   HYPRE_Int   i, offset;

   /* Double the capacity if limit is reached */
   if (capacity == size)
   {
      offset   = capacity;
      capacity += hypre_max(*tail, capacity);
      if (capacity < 0)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Integer overflow! Using capacity=INT_MAX");
         capacity = HYPRE_INT_MAX;
      }
      hypre_BoxBTQueueCapacity(btqueue) = capacity;
      hypre_BoxBTQueueNodes(btqueue) = hypre_TReAlloc(hypre_BoxBTQueueNodes(btqueue),
                                                      hypre_BoxBTNode *, capacity,
                                                      HYPRE_MEMORY_HOST);

      /* Reorganize items in the queue */
      for (i = 0; i < *tail; i++)
      {
         hypre_BoxBTQueueNode(btqueue, offset + i) = hypre_BoxBTQueueNode(btqueue, i);
         hypre_BoxBTQueueNode(btqueue, i) = NULL;
      }
      for (i = *tail + offset; i < capacity; i++)
      {
         hypre_BoxBTQueueNode(btqueue, i) = NULL;
      }
      hypre_BoxBTQueueTail(btqueue) += offset;
   }

   hypre_BoxBTQueueNode(btqueue, *tail) = btnode;
   if (*tail == (capacity - 1))
   {
      hypre_BoxBTQueueTail(btqueue) = 0;
   }
   else
   {
      hypre_BoxBTQueueTail(btqueue)++;
   }
   hypre_BoxBTQueueSize(btqueue)++;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoxBTStackDelete
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxBTQueueDelete( hypre_BoxBTQueue    *btqueue,
                        hypre_BoxBTNode    **btnode_ptr )
{
   HYPRE_Int   capacity = hypre_BoxBTQueueCapacity(btqueue);
   HYPRE_Int   size     = hypre_BoxBTQueueSize(btqueue);
   HYPRE_Int   head     = hypre_BoxBTQueueHead(btqueue);

   if (size)
   {
      *btnode_ptr = hypre_BoxBTQueueNode(btqueue, head);
      hypre_BoxBTQueueNode(btqueue, head) = NULL;
      if (head == (capacity - 1))
      {
         hypre_BoxBTQueueHead(btqueue) = 0;
      }
      else
      {
         hypre_BoxBTQueueHead(btqueue)++;
      }
      hypre_BoxBTQueueSize(btqueue)--;
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Empty queue!");
      *btnode_ptr = NULL;
   }

   return hypre_error_flag;
}
