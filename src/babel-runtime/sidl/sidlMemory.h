/*
 * File:        memory.h
 * Copyright:   (c) 2004 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.6 $
 * Date:        $Date: 2006/08/29 22:29:48 $
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
