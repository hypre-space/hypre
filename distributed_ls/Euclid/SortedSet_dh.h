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
