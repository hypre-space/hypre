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




#include "SortedSet_dh.h"
#include "shellSort_dh.h"
#include "Mem_dh.h"


#undef __FUNC__
#define __FUNC__ "SortedSet_dhCreate"
void SortedSet_dhCreate(SortedSet_dh *ss, HYPRE_Int size)
{
  START_FUNC_DH
  struct _sortedset_dh* tmp = (struct _sortedset_dh*)MALLOC_DH(sizeof(struct _sortedset_dh)); CHECK_V_ERROR;
  *ss= tmp;

  tmp->n = size;
  tmp->list = (HYPRE_Int*)MALLOC_DH(size*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  tmp->count = 0;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "SortedSet_dhDestroy"
void SortedSet_dhDestroy(SortedSet_dh ss)
{
  START_FUNC_DH
  if (ss->list != NULL) { FREE_DH(ss->list); CHECK_V_ERROR; }
  FREE_DH(ss); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedSet_dhInsert"
void SortedSet_dhInsert(SortedSet_dh ss, HYPRE_Int idx)
{
  START_FUNC_DH
  bool isInserted = false;
  HYPRE_Int ct = ss->count;
  HYPRE_Int *list = ss->list;
  HYPRE_Int i, n = ss->n;

  /* determine if item was already inserted */
  for (i=0; i<ct; ++i) {
    if (list[i] == idx) {
      isInserted = true;
      break;
    }
  }

  /* is we need to insert the item, first check for overflow
     and reallocate if necessary, then append the index to the
     end of the list.
  */
  if (! isInserted) {
    if (ct == n) {
      HYPRE_Int *tmp = (HYPRE_Int*)MALLOC_DH(n*2*sizeof(HYPRE_Int)); CHECK_V_ERROR;
      memcpy(tmp, list, n*sizeof(HYPRE_Int));
      FREE_DH(list); CHECK_V_ERROR;
      list = ss->list = tmp;
      ss->n *= 2;
    }

    list[ct] = idx;
    ss->count += 1;
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedSet_dhGetList"
void SortedSet_dhGetList(SortedSet_dh ss, HYPRE_Int **list, HYPRE_Int *count)
{
  START_FUNC_DH
  shellSort_int(ss->count, ss->list);
  *list = ss->list;
  *count = ss->count;
  END_FUNC_DH
}

