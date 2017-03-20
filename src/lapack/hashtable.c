/* Hash tables (fixed size)
 * 15-122 Principles of Imperative Computation, Fall 2010 
 * Frank Pfenning
 */

#include <stdbool.h>
#include <stdlib.h>
#include "xalloc.h"
#include "contracts.h"
#include "hashtable.h"

/* Interface type definitions */
/* see hashtable.h */

/* Chains, implemented as linked lists */
typedef struct chain* chain;

/* alpha = n/m = num_elems/size */
struct table {
  int size;			/* m */
  int num_elems;		/* n */
  chain* array;			/* \length(array) == size */
  ht_key (*elem_key)(ht_elem e); /* extracting keys from elements */
  bool (*equal)(ht_key k1, ht_key k2); /* comparing keys */
  int (*hash)(ht_key k, int m);	       /* hashing keys */
};

struct list {
  ht_elem data;
  struct list* next;
};
typedef struct list* list;
/* linked lists may be NULL (= end of list) */
/* we do not check for circularity */

void list_free(list p, void (*elem_free)(ht_elem e)) {
  list q;
  while (p != NULL) {
    if (p->data != NULL && elem_free != NULL)
      /* free element, if such a function is supplied */
      (*elem_free)(p->data);
    q = p->next;
    free(p);
    p = q;
  }
}

/* chains */

chain chain_new ();
ht_elem chain_insert(table H, chain C, ht_elem e);
ht_elem chain_search(table H, chain C, ht_key k);
void chain_free(chain C, void (*elem_free)(ht_elem e));

struct chain {
  list list;
};

/* valid chains are not null */
bool is_chain(chain C) {
  return C != NULL;
}

chain chain_new()
{
  chain C = xmalloc(sizeof(struct chain));
  C->list = NULL;
  ENSURES(is_chain(C));
  return C;
}

/* chain_find(p, k) returns list element whose
 * data field has key k, or NULL if none exists
 */
list chain_find(table H, chain C, ht_key k)
{ REQUIRES(is_chain(C));
  list p = C->list;
  while (p != NULL) {
    if ((*H->equal)(k, (*H->elem_key)(p->data)))
      return p;
    p = p->next;
  }
  return NULL;
}

ht_elem chain_insert(table H, chain C, ht_elem e)
{ REQUIRES(is_chain(C) && e != NULL);
  list p = chain_find(H, C, (*H->elem_key)(e));
  if (p == NULL) {
    /* insert new element at the beginning */
    list new_item = xmalloc(sizeof(struct list));
    new_item->data = e;
    new_item->next = C->list;
    C->list = new_item;
    ENSURES(is_chain(C));
    return NULL;		/* did not overwrite entry */
  } else {
    /* overwrite existing entry with given key */
    ht_elem old_e = p->data;
    p->data = e;
    ENSURES(is_chain(C));
    return old_e;		/* return old entry */
  }
}

ht_elem chain_search(table H, chain C, ht_key k)
{ REQUIRES(is_chain(C));
  list p = chain_find(H, C, k);
  if (p == NULL) return NULL;
  else return p->data;
}

void chain_free(chain C, void (*elem_free)(ht_elem e))
{ REQUIRES(is_chain(C));
  list_free(C->list, elem_free);
  free(C);
}

/* Hash table interface */
/* see hashtable.h */

/* Hash table implementation */

/* is_h_chain(C, h, m) - all of chain C's keys are equal to h */
/* keys should also be pairwise distinct, but we do not check that */
/* table size is m */
bool is_h_chain (table H, chain C, int h, int m)
{ REQUIRES(0 <= h && h < m);
  if (C == NULL) return false;
  list p = C->list;
  while (p != NULL) {
    if (p->data == NULL) return false;
    if ((*H->hash)((*H->elem_key)(p->data),m) != h)
      return false;
    p = p->next;
  }
  return true;
}

bool is_table(table H)
//@requires H != NULL && H->size == \length(H->array);
{
  int i; int m;
  /* array elements may be NULL or chains */
  if (H == NULL) return false;
  m = H->size;
  for (i = 0; i < m; i++) {
    chain C = H->array[i];
    if (!(C == NULL || is_h_chain(H, C, i, m))) return false;
  }
  return true;
}

table table_new(int init_size,
		 ht_key (*elem_key)(ht_elem e),
		 bool (*equal)(ht_key k1, ht_key k2),
		 int (*hash)(ht_key k, int m))
{ REQUIRES(init_size > 1);
  chain* A = xcalloc(init_size, sizeof(chain));
  table H = xmalloc(sizeof(struct table));
  H->size = init_size;
  H->num_elems = 0;
  H->array = A;			/* all initialized to NULL; */
  H->elem_key = elem_key;
  H->equal = equal;
  H->hash = hash;
  ENSURES(is_table(H));
  return H;
}

ht_elem table_insert(table H, ht_elem e) 
{ REQUIRES(is_table(H));
  ht_elem old_e;
  ht_key k = (*H->elem_key)(e);
  int h = (*H->hash)(k, H->size);
  if (H->array[h] == NULL)
    H->array[h] = chain_new();
  old_e = chain_insert(H, H->array[h], e);
  if (old_e != NULL) return old_e;
  H->num_elems++;
  ENSURES(is_table(H));
  ENSURES(table_search(H, (*H->elem_key)(e)) == e); /* pointer equality */
  return NULL;
}

ht_elem table_search(table H, ht_key k)
{ REQUIRES(is_table(H));
  int h = (*H->hash)(k, H->size);
  if (H->array[h] == NULL) return NULL;
  ht_elem e = chain_search(H, H->array[h], k);
  ENSURES(e == NULL || (*H->equal)((*H->elem_key)(e), k));
  return e;
}

void table_free(table H, void (*elem_free)(ht_elem e))
{ REQUIRES(is_table(H));
  int i;
  for (i = 0; i < H->size; i++) {
    chain C = H->array[i];
    if (C != NULL) chain_free(C, elem_free);
  }
  free(H->array);
  free(H);
}
