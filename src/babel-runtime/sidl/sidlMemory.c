/*
 * File:        sidlMemory.c
 * Copyright:   (c) 2004 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.7 $
 * Date:        $Date: 2007/09/27 19:35:42 $
 * Description: Replacement memory allocation functions
 *
 */

#include "sidlMemory.h"

#if ! HAVE_MALLOC
#ifdef __cplusplus
extern "C"
#endif
void *malloc(size_t);

void *rpl_malloc(size_t n)
{
  return (n == 0) ? malloc(1) : malloc(n);
}
#endif

#if ! HAVE_REALLOC
#ifdef __cplusplus
extern "C"
#endif
void *realloc(void *, size_t);

void *rpl_realloc(void *ptr, size_t n)
{
  if (ptr) {
    if (n) return realloc(ptr, n);
    free(ptr);
    return NULL;
  }
  else {
    if (n) return malloc(n);
    return malloc(1);
  }
}

#endif


