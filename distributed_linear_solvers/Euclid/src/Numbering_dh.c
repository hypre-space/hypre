#include "Numbering_dh.h"
#include "Mat_dh.h"
#include "Hash_dh.h"
#include "Mem_dh.h"
#include "shellSort_dh.h"


#undef __FUNC__
#define __FUNC__ "Numbering_dhCreate"
void Numbering_dhCreate(Numbering_dh *numb)
{
  START_FUNC_DH
  struct _numbering_dh* tmp = (struct _numbering_dh*)MALLOC_DH(sizeof(struct _numbering_dh)); CHECK_V_ERROR;
  *numb = tmp;

  tmp->size = 0;
  tmp->first = 0;
  tmp->num_loc = 0;
  tmp->num_ext = 0;
  tmp->global_to_local = NULL;
  tmp->local_to_global = NULL;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Numbering_dhDestroy"
void Numbering_dhDestroy(Numbering_dh numb)
{
  START_FUNC_DH
  if (numb->global_to_local != NULL) { 
    Hash_dhDestroy(numb->global_to_local); CHECK_V_ERROR; 
  }
  if (numb->local_to_global != NULL) { 
    FREE_DH(numb->local_to_global); CHECK_V_ERROR;
  }
  FREE_DH(numb); CHECK_V_ERROR;
  END_FUNC_DH
}


/*
The internal indices are numbered 0 to nlocal-1 so they do not 
need to be sorted.  The external indices are sorted so that 
the indices from a given processor are stored contiguously.
Then in the matvec, no reordering of the data is needed.
*/

#undef __FUNC__
#define __FUNC__ "Numbering_dhSetup"
void Numbering_dhSetup(Numbering_dh numb, Mat_dh mat)
{
  START_FUNC_DH
  int row, i, len, *ind;
  int num_external = 0;
  int m = mat->m, size;
  Hash_dh global_to_local;
  HashData data, *dataPtr;
  int first = mat->beg_row;
  int last  = first+m;
  int *local_to_global;

  /* this stops purify from complaining, when the hash table is printed! */
  data.fData = 0.0;

  numb->first = first;
  numb->num_loc = m;

  /* Allocate space for look-up tables */
  size = m*2;  /* heuristic; assume number of external indices is
                  not greater that the number of local indices.
                */
  numb->size = size;
  Hash_dhCreate(&(numb->global_to_local), m); CHECK_V_ERROR;
  global_to_local = numb->global_to_local;
  numb->local_to_global = local_to_global = 
          (int*)MALLOC_DH(size*sizeof(int)); CHECK_V_ERROR;
  
  /* Set up the local part of local_to_global */
  for (i=0; i<m; i++) local_to_global[i] = first + i;

  /* Fill local_to_global array */
  for (row=0; row<m; row++) {
    len = mat->rp[row+1] - mat->rp[row];
    ind = mat->cval + mat->rp[row];

    for (i=0; i<len; i++) {
      int index = ind[i];

      /* Only interested in external indices */
      if (index < first || index >= last) {

        /* if index hasn't been previously inserted, do so now. */
        dataPtr = Hash_dhLookup(global_to_local, ind[i]);
        if (dataPtr == NULL) {
          /* check for data overflow.  should probably
           * realloc global_to_local and local_to_global . . . todo
           */
          if (m+num_external >= size) {
            int newSize = size*1.5;  /* heuristic */
            int *tmp = (int*)MALLOC_DH(newSize*sizeof(int)); CHECK_V_ERROR;
            memcpy(tmp, local_to_global, size*sizeof(size));
            FREE_DH(local_to_global); CHECK_V_ERROR;
            size = numb->size = newSize;
            numb->local_to_global = local_to_global = tmp;
            SET_INFO("reallocated local_to_global[]");
          }

          /* add external (non-local) index to maps */
          data.iData = num_external;
          Hash_dhInsert(global_to_local, index, &data); CHECK_V_ERROR;
          local_to_global[m+num_external] = index;
          num_external++;
        }
      }
    }
  }
  numb->num_ext = num_external;

  /* Sort the indices */
  shellSort_int(num_external, &(local_to_global[m]));

  /* Redo the global_to_local table for the sorted indices */
  Hash_dhReset(global_to_local); CHECK_V_ERROR;
  for (i=0; i<num_external; i++) {
    data.iData = i+m;
    Hash_dhInsert(global_to_local, local_to_global[i+m], &data); CHECK_V_ERROR;
  }

  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Numbering_dhGlobalToLocal"
void Numbering_dhGlobalToLocal(Numbering_dh numb, int len, 
                                      int *global, int *local)
{
  START_FUNC_DH
  int i;
  int first = numb->first;
  int last = first + numb->num_loc;
  HashData *dataPtr;
  Hash_dh  global_to_local = numb->global_to_local;

  for (i=0; i<len; i++) {
    if (global[i] >= first && global[i] < last) {
      local[i] = global[i] - first;
    } else {
      dataPtr = Hash_dhLookup(global_to_local, global[i]);
      if (dataPtr == NULL) {
        sprintf(msgBuf_dh, "global index %i not found in map\n", global[i]);
        SET_V_ERROR(msgBuf_dh);
      } else {
        local[i] = dataPtr->iData;
      }
    } 
  }
  END_FUNC_DH
}
