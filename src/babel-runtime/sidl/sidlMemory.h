/*
 * File:        memory.h
 * Copyright:   (c) 2004 The Regents of the University of California
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Replacement memory allocation functions
 *
 */

#ifndef included_sidlMemory_h
#define included_sidlMemory_h
#include "babel_config.h"
#ifdef __cplusplus
extern "C" {
#endif
#ifdef malloc
#undef malloc
#endif
#ifdef realloc
#undef realloc
#endif
#include <sys/types.h>

#if ! HAVE_MALLOC
void *rpl_malloc(size_t n);
#endif

#if ! HAVE_REALLOC
void *rpl_realloc(void *ptr, size_t n);
#endif

#ifdef __cplusplus
}
#endif
#endif /*  included_sidlMemory_h */
