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




#include "Hash_i_dh.h"
#include "Parser_dh.h"
#include "Mem_dh.h"

#define DEFAULT_TABLE_SIZE 16

static void rehash_private(Hash_i_dh h);


/*--------------------------------------------------------------
 * hash functions (double hashing is used)
 *--------------------------------------------------------------*/
#define HASH_1(k,size,idxOut)    \
         {  *idxOut = k % size;  }

#define HASH_2(k,size,idxOut)      \
          {  \
            HYPRE_Int r = k % (size-13); \
            r = (r % 2) ? r : r+1; \
            *idxOut = r;           \
          }


/*--------------------------------------------------------------
 * class structure
 *--------------------------------------------------------------*/
typedef struct _hash_i_node_private Hash_i_Record;

struct _hash_i_node_private {
  HYPRE_Int  key;
  HYPRE_Int  mark;
  HYPRE_Int  data;
};


struct _hash_i_dh {
  HYPRE_Int         size;   /* total slots in table */
  HYPRE_Int         count;  /* number of items inserted in table */
  HYPRE_Int         curMark;/* used by Reset */
  Hash_i_Record *data;
};


/*--------------------------------------------------------------
 * class methods follow
 *--------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "Hash_i_dhCreate"
void Hash_i_dhCreate(Hash_i_dh *h, HYPRE_Int sizeIN)
{
  START_FUNC_DH
  HYPRE_Int i, size;
  Hash_i_Record *tmp2;
  struct _hash_i_dh* tmp;

  size = DEFAULT_TABLE_SIZE;
  if (sizeIN == -1) { 
    sizeIN = size = DEFAULT_TABLE_SIZE; 
  }
  tmp = (struct _hash_i_dh*)MALLOC_DH( sizeof(struct _hash_i_dh)); CHECK_V_ERROR;
  *h = tmp;
  tmp->size = 0;
  tmp->count = 0;
  tmp->curMark = 0;
  tmp->data = NULL;

  /*
     determine initial hash table size.  If this is too small,
     it will be dynamically enlarged as needed by Hash_i_dhInsert()
     See "double hashing," p. 255, "Algorithms," Cormen, et. al.
   */
  while (size < sizeIN) size *= 2;  /* want table size to be a power of 2: */
  /* rule-of-thumb: ensure there's at least 10% padding */
  if ( (size-sizeIN) < (.1 * size) ) { size *= 2.0; }
  tmp->size = size;


  /* allocate and zero the hash table */
  tmp2 = tmp->data = (Hash_i_Record*)MALLOC_DH(size*sizeof(Hash_i_Record)); CHECK_V_ERROR;
  for (i=0; i<size; ++i) {
    tmp2[i].key = -1;
    tmp2[i].mark = -1;
    /* "tmp2[i].data" needn't be initialized */
  }

  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Hash_i_dhDestroy"
void Hash_i_dhDestroy(Hash_i_dh h)
{
  START_FUNC_DH
  if (h->data != NULL) { FREE_DH(h->data); CHECK_V_ERROR; }
  FREE_DH(h); CHECK_V_ERROR;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Hash_i_dhReset"
void Hash_i_dhReset(Hash_i_dh h)
{
  START_FUNC_DH
  h->count = 0;
  h->curMark += 1;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Hash_i_dhLookup"
HYPRE_Int Hash_i_dhLookup(Hash_i_dh h, HYPRE_Int key)
{
  START_FUNC_DH
  HYPRE_Int idx, inc, i, start;
  HYPRE_Int curMark = h->curMark;
  HYPRE_Int size = h->size;
  HYPRE_Int retval = -1;
  Hash_i_Record *data = h->data;

  HASH_1(key, size, &start)
  HASH_2(key, size, &inc)

/*hypre_printf("Hash_i_dhLookup:: key: %i  tableSize: %i start: %i  inc: %i\n", key, size, start, inc);
*/

  for (i=0; i<size; ++i) {
    idx = (start + i*inc) % size;

/* hypre_printf("   idx= %i\n", idx); */

    if (data[idx].mark != curMark) {
      break;  /* key wasn't found */
    } else {
      if (data[idx].key == key) {
        retval = data[idx].data;
        break;
      }
    } 
  }
  END_FUNC_VAL(retval)
}


#undef __FUNC__
#define __FUNC__ "Hash_i_dhInsert"
void Hash_i_dhInsert(Hash_i_dh h, HYPRE_Int key, HYPRE_Int dataIN)
{
  START_FUNC_DH
  HYPRE_Int i, idx, inc, start, size;
  HYPRE_Int curMark = h->curMark;
  Hash_i_Record *data;
  bool success = false;

  if (dataIN < 0) {
    hypre_sprintf(msgBuf_dh, "data = %i must be >= 0", dataIN);
    SET_V_ERROR(msgBuf_dh);
  }

  /* enlarge table if necessary */
  if (h->count >= 0.9 * h->size) {
    rehash_private(h); CHECK_V_ERROR;
  }

  size = h->size;
  data = h->data;
  h->count += 1;    /* for this insertion */

  HASH_1(key, size, &start)
  HASH_2(key, size, &inc)



/*hypre_printf("Hash_i_dhInsert::  tableSize= %i  start= %i  inc= %i\n", size, start, inc);
*/
  for (i=0; i<size; ++i) {
    idx = (start + i*inc) % size;

/* hypre_printf("   idx= %i\n", idx);
*/

    /* check for previous insertion */
    if (data[idx].mark == curMark  &&  data[idx].key == key) {
      hypre_sprintf(msgBuf_dh, "key,data= <%i, %i> already inserted", key, dataIN);
      SET_V_ERROR(msgBuf_dh);
    }

    if (data[idx].mark < curMark) {
      data[idx].key = key;
      data[idx].mark = curMark;
      data[idx].data = dataIN;
      success = true;
      break;
    }
  }

  if (! success) {  /* should be impossible to be here, I think . . . */
    hypre_sprintf(msgBuf_dh, "Failed to insert key= %i, data= %i", key, dataIN);
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "rehash_private"
void rehash_private(Hash_i_dh h)
{
  START_FUNC_DH
  HYPRE_Int i, 
      old_size = h->size, 
      new_size = old_size*2,
      oldCurMark = h->curMark;
  Hash_i_Record *oldData = h->data, 
                 *newData;

  hypre_sprintf(msgBuf_dh, "rehashing; old_size= %i, new_size= %i", old_size, new_size);
  SET_INFO(msgBuf_dh);

  /* allocate new data table, and install it in the Hash_i_dh object;
     essentially, we reinitialize the hash object.
   */
  newData = (Hash_i_Record*)MALLOC_DH(new_size*sizeof(Hash_i_Record)); CHECK_V_ERROR;
  for (i=0; i<new_size; ++i) {
    newData[i].key = -1;
    newData[i].mark = -1;
  }
  h->size = new_size;
  h->data = newData;
  h->count = 0;
  h->curMark = 0;

  for (i=h->count; i<new_size; ++i) { 
    newData[i].key = -1;
    newData[i].mark = -1;
  }

  /* insert <key, data> pairs from old table to new table;
     wouldn't have been called) it's simplest to sweep through
     the old table.
   */
  for (i=0; i<old_size; ++i) {
    if (oldData[i].mark == oldCurMark) {
      Hash_i_dhInsert(h, oldData[i].key, oldData[i].data); CHECK_V_ERROR;
    }
  }
   
  FREE_DH(oldData); CHECK_V_ERROR;
  END_FUNC_DH
}
