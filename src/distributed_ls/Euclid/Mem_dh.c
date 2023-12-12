/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"
/* #include "Parser_dh.h" */
/* #include "Mem_dh.h" */

/* TODO: error checking is not complete; memRecord_dh  need to
         be done in Mem_dhMalloc() and Mem_dhFree()l
*/


  /* a memRecord_dh is pre and post-pended to every
   * piece of memory obtained by calling MALLOC_DH
   */
typedef struct {
    HYPRE_Real size;
    HYPRE_Real cookie;
} memRecord_dh;

struct _mem_dh {
  HYPRE_Real maxMem;        /* max allocated at any point in time */
  HYPRE_Real curMem;        /* total currently allocated */
  HYPRE_Real totalMem;      /* total cumulative malloced */
  HYPRE_Real mallocCount;  /* number of times mem_dh->malloc has been called. */
  HYPRE_Real freeCount;    /* number of times mem_dh->free has been called. */
};


#undef __FUNC__
#define __FUNC__ "Mem_dhCreate"
void Mem_dhCreate(Mem_dh *m)
{
  START_FUNC_DH
  struct _mem_dh *tmp = (struct _mem_dh*)PRIVATE_MALLOC(sizeof(struct _mem_dh)); CHECK_V_ERROR;
  *m = tmp;
  tmp->maxMem = 0.0;
  tmp->curMem = 0.0;
  tmp->totalMem = 0.0;
  tmp->mallocCount = 0.0;
  tmp->freeCount = 0.0;
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Mem_dhDestroy"
void Mem_dhDestroy(Mem_dh m)
{
  START_FUNC_DH
  if (Parser_dhHasSwitch(parser_dh, "-eu_mem")) {
    Mem_dhPrint(m, stdout, false); CHECK_V_ERROR;
  }

  PRIVATE_FREE(m);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "Mem_dhMalloc"
void* Mem_dhMalloc(Mem_dh m, size_t size)
{
  START_FUNC_DH_2
  void *retval;
  memRecord_dh *tmp;
  size_t s = size + 2*sizeof(memRecord_dh);
  void *address;

  address = PRIVATE_MALLOC(s);

  if (address == NULL) {
    hypre_sprintf(msgBuf_dh, "PRIVATE_MALLOC failed; totalMem = %g; requested additional = %i", m->totalMem, (HYPRE_Int)s);
    SET_ERROR(NULL, msgBuf_dh);
  }

  retval = (char*)address + sizeof(memRecord_dh);

  /* we prepend and postpend a private record to the
   * requested chunk of memory; this permits tracking the
   * sizes of freed memory, along with other rudimentary
   * error checking.  This is modeled after the PETSc code.
   */
  tmp = (memRecord_dh*)address;
  tmp->size = (HYPRE_Real) s;

  m->mallocCount += 1;
  m->totalMem += (HYPRE_Real)s;
  m->curMem += (HYPRE_Real)s;
  m->maxMem = MAX(m->maxMem, m->curMem);

  END_FUNC_VAL_2( retval )
}


#undef __FUNC__
#define __FUNC__ "Mem_dhFree"
void Mem_dhFree(Mem_dh m, void *ptr)
{
  HYPRE_UNUSED_VAR(m);

  START_FUNC_DH_2
  HYPRE_Real size;
  char *tmp = (char*)ptr;
  memRecord_dh *rec;
  tmp -= sizeof(memRecord_dh);
  rec = (memRecord_dh*)tmp;
  size = rec->size;

  mem_dh->curMem -= size;
  mem_dh->freeCount += 1;

  PRIVATE_FREE(tmp);
  END_FUNC_DH_2
}


#undef __FUNC__
#define __FUNC__ "Mem_dhPrint"
void  Mem_dhPrint(Mem_dh m, FILE* fp, bool allPrint)
{
  START_FUNC_DH_2
  if (fp == NULL) SET_V_ERROR("fp == NULL");
  if (myid_dh == 0 || allPrint) {
    HYPRE_Real tmp;
    hypre_fprintf(fp, "---------------------- Euclid memory report (start)\n");
    hypre_fprintf(fp, "malloc calls = %g\n", m->mallocCount);
    hypre_fprintf(fp, "free   calls = %g\n", m->freeCount);
    hypre_fprintf(fp, "curMem          = %g Mbytes (should be zero)\n",
                                                   m->curMem/1000000);
    tmp = m->totalMem / 1000000;
    hypre_fprintf(fp, "total allocated = %g Mbytes\n", tmp);
    hypre_fprintf(fp, "max malloc      = %g Mbytes (max allocated at any point in time)\n", m->maxMem/1000000);
    hypre_fprintf(fp, "\n");
    hypre_fprintf(fp, "---------------------- Euclid memory report (end)\n");
  }
  END_FUNC_DH_2
}
