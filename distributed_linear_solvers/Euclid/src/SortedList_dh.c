#include "SortedList_dh.h"
#include "Mem_dh.h"
#include "Hash_dh.h"


struct _sortedList_dh {
  int m;
  int row;        /* local number of row being factored */
  int beg_row;    /* global number of first locally owned row */
  int count;       /* number of items entered in the list */
  int *o2n_local;
  Hash_dh o2n_external;

  SRecord *list;  /* the sorted list */
  int alloc;      /* allocated length of list */
  int getLower;   /* index used for returning lower tri elts */
  int get;        /* index of returning all elts; */
};

static void lengthen_list_private(SortedList_dh sList);


#undef __FUNC__
#define __FUNC__ "SortedList_dhCreate"
void SortedList_dhCreate(SortedList_dh *sList)
{
  START_FUNC_DH
  struct _sortedList_dh* tmp = (struct _sortedList_dh*)MALLOC_DH(
                                 sizeof(struct _sortedList_dh)); CHECK_V_ERROR;
  *sList = tmp;
  tmp->m = 0;
  tmp->row = -1;
  tmp->beg_row = 0;
  tmp->count = 0;
  tmp->o2n_external = NULL;
  tmp->o2n_local = NULL;

  tmp->get = 0;
  tmp->getLower = 0;
  tmp->alloc = 0;
  tmp->list = NULL;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhDestroy"
void SortedList_dhDestroy(SortedList_dh sList)
{
  START_FUNC_DH
  if (sList->o2n_local != NULL) { FREE_DH(sList->o2n_local); CHECK_V_ERROR; }
  if (sList->list != NULL) { FREE_DH(sList->list); CHECK_V_ERROR; }
  FREE_DH(sList); CHECK_V_ERROR;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhInit"
void SortedList_dhInit(SortedList_dh sList, int m, int beg_row, 
                              int *n2o_local, Hash_dh o2n_external)
{
  START_FUNC_DH
  int i, *o2n_local;

  sList->o2n_local = o2n_local = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) o2n_local[n2o_local[i]] = i;

  sList->m = m;
  sList->beg_row = beg_row;
  sList->o2n_external = o2n_external;

  /* heuristic: "m" should be a good number of nodes */
  sList->list = (SRecord*)MALLOC_DH(m*sizeof(SRecord)); 
  sList->alloc = m;
  sList->list[0].col = INT_MAX;
  sList->list[0].next = 0;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhReset"
void SortedList_dhReset(SortedList_dh sList, int row)
{
  START_FUNC_DH
  sList->row = row;
  sList->count = 0;
  sList->get = 0;
  sList->getLower = 0;
  sList->list[0].next = 0;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhReadCount"
int SortedList_dhReadCount(SortedList_dh sList)
{
  START_FUNC_DH
  END_FUNC_VAL(sList->count)
}

#undef __FUNC__
#define __FUNC__ "SortedList_dhResetGetSmallest"
void SortedList_dhResetGetSmallest(SortedList_dh sList)
{
  START_FUNC_DH
  sList->getLower = 0;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "SortedList_dhGetSmallest"
SRecord * SortedList_dhGetSmallest(SortedList_dh sList)
{
  START_FUNC_DH
  SRecord *node = NULL;
  SRecord *list = sList->list;
  int get = sList->get;

  get = list[get].next;

  if (list[get].col < INT_MAX) {
    node = &(list[get]);
    sList->get = get;
  }
  END_FUNC_VAL(node)
}

#undef __FUNC__
#define __FUNC__ "SortedList_dhGetSmallestLowerTri"
SRecord * SortedList_dhGetSmallestLowerTri(SortedList_dh sList)
{
  START_FUNC_DH
  SRecord *node = NULL;
  SRecord *list = sList->list;
  int getLower = sList->getLower;
  int globalRow = sList->row + sList->beg_row;

  getLower = list[getLower].next;

  if (list[getLower].col < globalRow) {
    node = &(list[getLower]);
    sList->getLower = getLower;
  }
  END_FUNC_VAL(node)
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhPermuteAndInsert"
void SortedList_dhPermuteAndInsert(SortedList_dh sList, SRecord *sr)
{
  START_FUNC_DH
  int col = sr->col;
  int beg_row = sList->beg_row, end_row = beg_row + sList->m;

  if (col >= beg_row && col < end_row) {
    col -= beg_row;
    col = sList->o2n_local[col] + beg_row;
  } else {
    if (sList->o2n_external != NULL) {
      HashData *r;
      r = Hash_dhLookup(sList->o2n_external, col); CHECK_V_ERROR;
      if (r == NULL) {
        col = -1;
        sprintf(msgBuf_dh, "lookup failed for external node= %i", col);
        SET_V_ERROR(msgBuf_dh);
      } 
    } else {
      col = -1;
      sprintf(msgBuf_dh, "Hash table is null; can't permute external node= %i", col+1);
      SET_INFO(msgBuf_dh);
    }
  }

  if (col != -1) {
    sr->col = col;
    SortedList_dhInsert(sList, sr); CHECK_V_ERROR;
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhInsertOrUpdate"
void SortedList_dhInsertOrUpdate(SortedList_dh sList, SRecord *sr)
{
  START_FUNC_DH
  SRecord *node = SortedList_dhFind(sList, sr); CHECK_V_ERROR;

  if (node == NULL) {
    SortedList_dhInsert(sList, sr); CHECK_V_ERROR;
  } else {
    node->level = MIN(sr->level, node->level);
  }
  END_FUNC_DH
}


/* note: this does NOT check to see if item was already inserted! */
#undef __FUNC__
#define __FUNC__ "SortedList_dhInsert"
void SortedList_dhInsert(SortedList_dh sList, SRecord *sr)
{
  START_FUNC_DH
  int prev, next;
  int ct, col = sr->col;
  SRecord *list = sList->list;

  /* lengthen list if out of space */
  sList->count += 1;
  ct = sList->count;
  if (ct == sList->alloc) {
    lengthen_list_private(sList); CHECK_V_ERROR;
  }

  /* add new node to end of list */
  list[ct].col = col;
  list[ct].level = sr->level;
  list[ct].val = sr->val;

  /* splice new node into list */
  prev = 0;
  next = list[0].next;
  while (col > list[next].col) {
    prev = next;
    next = list[next].next;
  }
  list[prev].next = ct;
  list[ct].next = next;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "SortedList_dhFind"
SRecord * SortedList_dhFind(SortedList_dh sList, SRecord *sr)
{
  START_FUNC_DH
  int i, count = sList->count;
  int c = sr->col;
  SRecord *s = sList->list;
  SRecord *node = NULL;

  /* no need to traverse list in sorted order */
  for (i=1; i<=count; ++i) {
    if (s[i].col == c) {
      node = &(s[i]);
      break;
    }
  }
  END_FUNC_VAL(node)
}

#undef __FUNC__
#define __FUNC__ "lengthen_list_private"
void lengthen_list_private(SortedList_dh sList)
{
  START_FUNC_DH
  SRecord *tmp = sList->list;
  sList->list = (SRecord*)MALLOC_DH(2*sList->alloc * sizeof(SRecord));
  memcpy(sList->list, tmp, sList->alloc * sizeof(SRecord));
  sList->alloc *= 2;
  SET_INFO("doubling size of sList->list");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "SortedList_dhPrint"
void SortedList_dhPrint(SortedList_dh sList, FILE *fp)
{
  START_FUNC_DH
  SRecord *list = sList->list;
  int get;

  get = list[0].next;

  fprintf(fp, "\ncontents of sorted linked list (count= %i):\n", sList->count);
  while (list[get].col < INT_MAX) {
    fprintf(fp, "col= %i  level= %i  val= %g\n", 
                      list[get].col+1, list[get].level, list[get].val);
    get = list[get].next;
  }
  fprintf(fp, "\n");
  fflush(fp);
  END_FUNC_DH
}

