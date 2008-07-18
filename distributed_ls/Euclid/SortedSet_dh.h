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




#ifndef SORTED_SET_DH
#define SORTED_SET_DH

#include "euclid_common.h"

struct _sortedset_dh {
  int n;   /* max items that can be stored */
  int *list;  /* list of inserted elements */
  int count;  /* the number of elements in the list */
};

extern void SortedSet_dhCreate(SortedSet_dh *ss, int initialSize);
extern void SortedSet_dhDestroy(SortedSet_dh ss);
extern void SortedSet_dhInsert(SortedSet_dh ss, int idx);
extern void SortedSet_dhGetList(SortedSet_dh ss, int **list, int *count);


#endif
