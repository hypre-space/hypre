/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_UTILITIES_H
#define HYPRE_UTILITIES_H

/* iterator class to generate iterate index in [first, last)
 * NOTE: last is "past-the-end"
 * direction ==  1, [first, first + 1, .... , last - 1]
 * direction == -1, [last - 1, last -2, ... , first]
 *
 * iter = hypre_IteratorCreate(0, 5, 1)
 * for ( i = hypre_IteratorBegin(iter); i != hypre_IteratorEnd(iter); i = hypre_IteratorStep(iter, i) )
 * { // i = 0, 1, 2, 3, 4 }
 *
 * iter = hypre_IteratorCreate(0, 5, -1)
 * for ( i = hypre_IteratorBegin(iter); i != hypre_IteratorEnd(iter); i = hypre_IteratorStep(iter, i) )
 * { // i = 4, 3, 2, 1, 0 }
 */
typedef struct
{
   HYPRE_Int first;       /* the first index */
   HYPRE_Int last;        /* the past-the-last index */
   HYPRE_Int direction;   /* 1: ++; -1: -- */
} hypre_Iterator;

#define hypre_IteratorFirst(iter)      ((iter) -> first)
#define hypre_IteratorLast(iter)       ((iter) -> last)
#define hypre_IteratorDirection(iter)  ((iter) -> direction)

/* constructor */
static inline hypre_Iterator *
hypre_IteratorCreate( HYPRE_Int first,
                      HYPRE_Int last,
                      HYPRE_Int direction )
{
   hypre_Iterator *iter = hypre_TAlloc(hypre_Iterator, 1, HYPRE_MEMORY_HOST);
   hypre_IteratorFirst(iter) = first;
   hypre_IteratorLast(iter) = last;
   hypre_IteratorDirection(iter) = direction;

   return iter;
}

/* destructor */
static inline HYPRE_Int
hypre_IteratorDestroy(hypre_Iterator *iter)
{
   hypre_TFree(iter, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* member functions */
static inline HYPRE_Int
hypre_IteratorBegin(hypre_Iterator *iter)
{
   return hypre_IteratorDirection(iter) > 0 ? hypre_IteratorFirst(iter) : hypre_IteratorLast(iter) - 1;
}

static inline HYPRE_Int
hypre_IteratorEnd(hypre_Iterator *iter)
{
   return hypre_IteratorDirection(iter) > 0 ? hypre_IteratorLast(iter) : hypre_IteratorFirst(iter) - 1;
}

static inline HYPRE_Int
hypre_IteratorStep(hypre_Iterator *iter)
{
   return hypre_IteratorDirection(iter) > 0 ? 1 : -1;
}

#endif

