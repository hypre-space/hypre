#include "Hash_dh.h"
#include "Parser_dh.h"
#include "Mem_dh.h"

static void Hash_dhInit_private(Hash_dh h, int s);

#define CUR_MARK_INIT  -1


struct _hash_node_private {
  int      key;
  int      mark;
  HashData data;
};


#undef __FUNC__
#define __FUNC__ "Hash_dhCreate"
void Hash_dhCreate(Hash_dh *h, int size)
{
  START_FUNC_DH
  struct _hash_dh* tmp = (struct _hash_dh*)MALLOC_DH(
                                             sizeof(struct _hash_dh)); CHECK_V_ERROR;
  *h = tmp;
  tmp->size = 0;
  tmp->count = 0;
  tmp->curMark = CUR_MARK_INIT + 1;
  tmp->data = NULL;

  Hash_dhInit_private(*h,size); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Hash_dhDestroy"
void Hash_dhDestroy(Hash_dh h)
{
  START_FUNC_DH
  if (h->data != NULL) { FREE_DH(h->data); CHECK_V_ERROR; }
  FREE_DH(h); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Hash_dhReset"
void Hash_dhReset(Hash_dh h)
{
  START_FUNC_DH
  h->count = 0;
  h->curMark += 1;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Hash_dhInit_private"
void Hash_dhInit_private(Hash_dh h, int s)
{
  START_FUNC_DH
  int i;
  int size = 16;
  HashRecord *data;

  /* want table size to be a power of 2: */
  while (size < s) size *= 2;
  /* rule-of-thumb: ensure there's some padding */
  if ( (size-s) < (.1 * size) ) { size *= 2.0; }
  h->size = size;

/*
  sprintf(msgBuf_dh, "requested size = %i; allocated size = %i", s, size); 
  SET_INFO(msgBuf_dh);
*/

  /* allocate and zero the hash table */
  data = h->data = (HashRecord*)MALLOC_DH(size*sizeof(HashRecord)); CHECK_V_ERROR;
  for (i=0; i<size; ++i) {
    data[i].key = -1;
    data[i].mark = -1;
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Hash_dhLookup"
HashData * Hash_dhLookup(Hash_dh h, int key)
{
  START_FUNC_DH
  int i, start;
  int curMark = h->curMark;
  int size = h->size;
  HashData *retval = NULL;
  HashRecord *data = h->data;

  HASH_1(key, size, &start)

  for (i=0; i<size; ++i) {
    int tmp, idx;
    HASH_2(key, size, &tmp)
    idx = (start + i*tmp) % size;
    if (data[idx].mark != curMark) {
      break;  /* key wasn't found */
    } else {
      if (data[idx].key == key) {
        retval = &(data[idx].data);
        break;
      }
    } 
  }
  END_FUNC_VAL(retval)
}


/* 
  TODO: (1) check for already-inserted  (done?)
        (2) rehash, if table grows too large
*/
#undef __FUNC__
#define __FUNC__ "Hash_dhInsert"
void Hash_dhInsert(Hash_dh h, int key, HashData *dataIN)
{
  START_FUNC_DH
  int i, start, size = h->size;
  int curMark = h->curMark;
  HashRecord *data;

  data = h->data;

  /* check for overflow */
  h->count += 1;
  if (h->count == h->size) {
    SET_V_ERROR("hash table overflow; rehash need implementing!");
  }

  HASH_1(key, size, &start)

  for (i=0; i<size; ++i) {
    int tmp, idx;
    HASH_2(key, size, &tmp)

    idx = (start + i*tmp) % size;
    if (data[idx].mark < curMark) {
      data[idx].key = key;
      data[idx].mark = curMark;
      memcpy(&(data[idx].data), dataIN, sizeof(HashData));
      break;
    }
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Hash_dhPrint"
void Hash_dhPrint(Hash_dh h, FILE *fp)
{
  START_FUNC_DH
  int i, size = h->size;
  int curMark = h->curMark;
  HashRecord *data = h->data;


  fprintf(fp, "\n--------------------------- hash table \n");
  for (i=0; i<size; ++i) {
    if (data[i].mark == curMark) {
      fprintf(fp, "key = %2i;  iData = %3i;  fData = %g\n",
                  data[i].key, data[i].data.iData, data[i].data.fData);
    }
  }
  fprintf(fp, "\n");
  END_FUNC_DH
}
