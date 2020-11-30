/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for different box data structures
 *
 * TODO: Move BoxArray here
 *
 *****************************************************************************/

#ifndef hypre_BOX_DS_HEADER
#define hypre_BOX_DS_HEADER

/*--------------------------------------------------------------------------
 * hypre_BoxBTNode:
 *
 * Box binary tree node
 *--------------------------------------------------------------------------*/
typedef struct hypre_BoxBTNode_struct
{
   HYPRE_Int     num_indices;
   HYPRE_Int    *indices[HYPRE_MAXDIM];
   hypre_Box    *box;
   struct hypre_BoxBTNode_struct  *left;
   struct hypre_BoxBTNode_struct  *right;
} hypre_BoxBTNode;

#define hypre_BoxBTNodeNDim(btnode)        ((btnode) -> box -> ndim)
#define hypre_BoxBTNodeNumIndices(btnode)  ((btnode) -> num_indices)
#define hypre_BoxBTNodeIndices(btnode, d)  ((btnode) -> indices[d])
#define hypre_BoxBTNodeIndex(btnode, d, i) ((btnode) -> indices[d][i])
#define hypre_BoxBTNodeBox(btnode)         ((btnode) -> box)
#define hypre_BoxBTNodeLeft(btnode)        ((btnode) -> left)
#define hypre_BoxBTNodeRight(btnode)       ((btnode) -> right)

/*--------------------------------------------------------------------------
 * hypre_BoxBinTree:
 *
 * Box binary tree
 *--------------------------------------------------------------------------*/
typedef struct hypre_BoxBinTree_struct
{
   hypre_BoxBTNode   *btroot;
} hypre_BoxBinTree;

#define hypre_BoxBinTreeRoot(boxbt)        ((boxbt) -> btroot)

/*--------------------------------------------------------------------------
 * hypre_BoxBTStack:
 *
 * Stack of box binary tree nodes
 *--------------------------------------------------------------------------*/
typedef struct hypre_BoxBTStack_struct
{
   HYPRE_Int          size;
   HYPRE_Int          capacity;
   hypre_BoxBTNode  **nodes;
} hypre_BoxBTStack;

#define hypre_BoxBTStackSize(btstack)      ((btstack) -> size)
#define hypre_BoxBTStackCapacity(btstack)  ((btstack) -> capacity)
#define hypre_BoxBTStackNodes(btstack)     ((btstack) -> nodes)
#define hypre_BoxBTStackNode(btstack, i)   ((btstack) -> nodes[i])
#define hypre_BoxBTStackNodePeek(btstack)  ((btstack) -> nodes[(btstack) -> size - 1])

/*--------------------------------------------------------------------------
 * hypre_BoxBTQueue:
 *
 * Queue of box binary tree nodes
 *--------------------------------------------------------------------------*/
typedef struct hypre_BoxBTQueue_struct
{
   HYPRE_Int          head;
   HYPRE_Int          tail;
   HYPRE_Int          size;
   HYPRE_Int          capacity;
   hypre_BoxBTNode  **nodes;
} hypre_BoxBTQueue;

#define hypre_BoxBTQueueHead(btqueue)      ((btqueue) -> head)
#define hypre_BoxBTQueueTail(btqueue)      ((btqueue) -> tail)
#define hypre_BoxBTQueueSize(btqueue)      ((btqueue) -> size)
#define hypre_BoxBTQueueCapacity(btqueue)  ((btqueue) -> capacity)
#define hypre_BoxBTQueueNodes(btqueue)     ((btqueue) -> nodes)
#define hypre_BoxBTQueueNode(btqueue, i)   ((btqueue) -> nodes[i])

#endif
