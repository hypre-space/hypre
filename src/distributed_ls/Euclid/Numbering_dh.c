/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"
/* #include "Numbering_dh.h" */
/* #include "Mat_dh.h" */
/* #include "Hash_i_dh.h" */
/* #include "Mem_dh.h" */
/* #include "shellSort_dh.h" */
/* #include "Parser_dh.h" */

#undef __FUNC__
#define __FUNC__ "Numbering_dhCreate"
void Numbering_dhCreate(Numbering_dh *numb)
{
  START_FUNC_DH
  struct _numbering_dh* tmp = (struct _numbering_dh*)MALLOC_DH(sizeof(struct _numbering_dh)); CHECK_V_ERROR;
  *numb = tmp;

  tmp->size = 0;
  tmp->first = 0;
  tmp->m = 0;
  tmp->num_ext = 0;
  tmp->num_extLo = 0;
  tmp->num_extHi = 0;
  tmp->idx_ext = NULL;
  tmp->idx_extLo = NULL;
  tmp->idx_extHi = NULL;
  tmp->idx_ext = NULL;
  tmp->debug = Parser_dhHasSwitch(parser_dh, "-debug_Numbering");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Numbering_dhDestroy"
void Numbering_dhDestroy(Numbering_dh numb)
{
  START_FUNC_DH
  if (numb->global_to_local != NULL) { 
    Hash_i_dhDestroy(numb->global_to_local); CHECK_V_ERROR; 
  }
  if (numb->idx_ext != NULL) { 
    FREE_DH(numb->idx_ext); CHECK_V_ERROR;
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
  HYPRE_Int       i, len, *cval = mat->cval;
  HYPRE_Int       num_ext, num_extLo, num_extHi;
  HYPRE_Int       m = mat->m, size;
  Hash_i_dh global_to_local_hash;
  HYPRE_Int       first = mat->beg_row, last  = first+m;
  HYPRE_Int       *idx_ext;
  HYPRE_Int       data;
/*  HYPRE_Int       debug = false; */

/*   if (logFile != NULL && numb->debug) debug = true; */

  numb->first = first;
  numb->m = m;

  /* Allocate space for look-up tables */

  /* initial guess: there are at most 'm' external indices */
  numb->size = size = m;
  Hash_i_dhCreate(&(numb->global_to_local), m); CHECK_V_ERROR;

  global_to_local_hash = numb->global_to_local;
  idx_ext = numb->idx_ext = (HYPRE_Int*)MALLOC_DH(size*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  
  /* find all external indices; at the end of this block, 
     idx_ext[] will contain an unsorted list of external indices.
   */
  len = mat->rp[m];
  num_ext = num_extLo = num_extHi = 0;
  for (i=0; i<len; i++) {       /* for each nonzero "index" in the matrix */
    HYPRE_Int index = cval[i];

    /* Only interested in external indices */
    if (index < first || index >= last) {

      /* if index hasn't been previously inserted, do so now. */
      data = Hash_i_dhLookup(global_to_local_hash, cval[i]); CHECK_V_ERROR;

      if (data == -1) {  /* index hasn't been inserted, so do so now  */

        /* reallocate idx_ext array if we're out of
           space.  The global_to_local hash table may also need
           to be enlarged, but the hash object will take care of that.
         */
        /* RL : why ``m+num_ext'' instead of ``num_ext+1'' ??? */
        if (m+num_ext >= size) {
          HYPRE_Int newSize = hypre_max(m+num_ext+1, size*1.5);  /* heuristic */
          HYPRE_Int *tmp = (HYPRE_Int*)MALLOC_DH(newSize*sizeof(HYPRE_Int)); CHECK_V_ERROR;
          hypre_TMemcpy(tmp,  idx_ext, size, size, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
          FREE_DH(idx_ext); CHECK_V_ERROR;
          size = numb->size = newSize;
          numb->idx_ext = idx_ext = tmp;
          SET_INFO("reallocated ext_idx[]");
        }

        /* insert external index */
        Hash_i_dhInsert(global_to_local_hash, index, num_ext); CHECK_V_ERROR;
        idx_ext[num_ext] = index;

        num_ext++;
        if (index < first) { num_extLo++; }
        else               { num_extHi++; }
      }
    }
  }

  numb->num_ext = num_ext;
  numb->num_extLo = num_extLo;
  numb->num_extHi = num_extHi;
  numb->idx_extLo = idx_ext;
  numb->idx_extHi = idx_ext + num_extLo;

  /* sort the list of external indices, then redo the hash
     table; the table is used to convert external indices
     in Numbering_dhGlobalToLocal()
  */
  shellSort_int(num_ext, idx_ext);

  Hash_i_dhReset(global_to_local_hash); CHECK_V_ERROR;
  for (i=0; i<num_ext; i++) {
    Hash_i_dhInsert(global_to_local_hash, idx_ext[i], i+m); CHECK_V_ERROR;
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Numbering_dhGlobalToLocal"
void Numbering_dhGlobalToLocal(Numbering_dh numb, HYPRE_Int len, 
                                      HYPRE_Int *global, HYPRE_Int *local)
{
  START_FUNC_DH
  HYPRE_Int i;
  HYPRE_Int first = numb->first;
  HYPRE_Int last = first + numb->m;
  HYPRE_Int data;
  Hash_i_dh  global_to_local = numb->global_to_local;

  for (i=0; i<len; i++) {
    HYPRE_Int idxGlobal = global[i];
    if (idxGlobal >= first && idxGlobal < last) {
      local[i] = idxGlobal - first;
       /* note: for matvec setup, numb->num_extLo = 0. */
    } else {
      data = Hash_i_dhLookup(global_to_local, idxGlobal); CHECK_V_ERROR;
      if (data == -1) {
        hypre_sprintf(msgBuf_dh, "global index %i not found in map\n", idxGlobal);
        SET_V_ERROR(msgBuf_dh);
      } else {
        local[i] = data;
      }
    } 
  }
  END_FUNC_DH
}

