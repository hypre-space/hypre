/*
 * File:        sidlArray.c
 * Copyright:   (c) 2004 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.6 $
 * Date:        $Date: 2007/09/27 19:35:42 $
 * Description: Generic array data type functions
 *
 * Copyright (c) 2004, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include "sidlArray.h"
#include <stddef.h>
#ifdef SIDL_DEBUG_REFCOUNT
#include "sidlOps.h"
#include <stdio.h>
#include <stdlib.h>

struct sidl_Array_list_t {
  struct sidl_Array_list_t *d_next;
  struct sidl__array       *d_array;
};

static struct sidl_Array_list_t *s_array_list = NULL;

const char *safeArrayType(struct sidl__array *array) {
  const static char * const s_array_types[] = {
    "illegal",
    "bool",
    "char",
    "dcomplex",
    "double",
    "fcomplex",
    "float",
    "int",
    "long",
    "opaque",
    "string",
    "interface"
  };
  int32_t arrayType = (array && (array->d_dimen <= 7) && (array->d_dimen > 0) && (array->d_refcount >= 0) && array->d_lower && array->d_upper && array->d_stride && array->d_vtable && array->d_vtable->d_arraytype)
    ? sidl__array_type(array) : 0;
  if ((arrayType < 0) ||
      (arrayType >= sizeof(s_array_types)/sizeof(char*))) {
    arrayType = 0;
  }
  return s_array_types[arrayType];
}

static void
sidl_report_arrays(void *ignored)
{
  struct sidl_Array_list_t *ptr = s_array_list;
  if (ptr) {
    do {
      fprintf(stderr, "babel: leaked array %p reference count %d (type %s)\n",
              ptr->d_array, 
              ptr->d_array ? ptr->d_array->d_refcount : -1,
              safeArrayType(ptr->d_array));

      ptr = ptr->d_next;
    } while (ptr);
  }
  else {
    fprintf(stderr, "babel: no arrays leaked\n");
  }
  while (s_array_list) {
    ptr = s_array_list->d_next;
    free((void *)s_array_list);
    s_array_list = ptr;
  }
}

static void
sidl_initialize_array_list(void)
{
  static int s_not_initialized = 1;
  if (s_not_initialized) {
    s_not_initialized = 0;
    sidl_atexit(sidl_report_arrays, NULL);
  }
}

void
sidl__array_add(struct sidl__array *array) {
  sidl_initialize_array_list();
  if (array) {
    struct sidl_Array_list_t *ptr = 
      malloc(sizeof(struct sidl_Array_list_t));
    ptr->d_next = s_array_list;
    ptr->d_array = array;
    s_array_list = ptr;
    fprintf(stderr, 
            "babel: create array %p initial count %d (type %s)\n",
            array, array->d_refcount, safeArrayType(array));
  }
}

void 
sidl__array_remove(struct sidl__array * const array)
{
  sidl_initialize_array_list();
  if (array) {
    struct sidl_Array_list_t *prev, *ptr;
    if (s_array_list && (s_array_list->d_array == array)) {
      ptr = s_array_list->d_next;
      free((void *)s_array_list);
      s_array_list = ptr;
    }
    else {
      prev = s_array_list;
      ptr = (prev ? prev->d_next : NULL);
      while (ptr) {
        if (ptr->d_array == array) {
          prev->d_next = ptr->d_next;
          free((void *)ptr);
          return;
        }
        prev = ptr;
        ptr = ptr->d_next;
      }
      fprintf(stderr, "babel: array data type invariant failure %p\n", array);
    }
  }
}
#else
void
sidl__array_add(struct sidl__array * const array)
{
}

void 
sidl__array_remove(struct sidl__array * const array)
{
}
#endif /* SIDL_DEBUG_REFCOUNT */


void
sidl__array_addRef(struct sidl__array* array)
{
  if (array) ++(array->d_refcount);
#ifdef SIDL_DEBUG_REFCOUNT
  fprintf(stderr, "babel: array addRef %p new count %d (type %s)\n",
          array, array ? array->d_refcount : -1, safeArrayType(array));
#endif
}

void
sidl__array_deleteRef(struct sidl__array *array)
{
  register const int selfDestruct = array && !(--(array->d_refcount));
#ifdef SIDL_DEBUG_REFCOUNT
  fprintf(stderr, "babel: array deleteRef %p new count %d (type %s)\n",
          array, array ? array->d_refcount : -1, safeArrayType(array));
#endif
  if (selfDestruct) {
    (*(array->d_vtable->d_destroy))(array);
  }
}

struct sidl__array *
sidl__array_smartCopy(struct sidl__array *array)
{
  return array ? (array->d_vtable->d_smartcopy)(array) : NULL;
}

int32_t
sidl__array_dimen(const struct sidl__array *array)
{
  return array ? sidlArrayDim(array) : 0;
}

int32_t
sidl__array_lower(const struct sidl__array *array, int32_t ind)
{
  return (array && (ind >= 0) && (ind < sidlArrayDim(array))) 
   ? sidlLower(array, ind) : 0;
}

int32_t
sidl__array_upper(const struct sidl__array *array, int32_t ind)
{
  return (array && (ind >= 0) && (ind < sidlArrayDim(array))) 
   ? sidlUpper(array, ind) : -1;
}

int32_t
sidl__array_length(const struct sidl__array *array, int32_t ind)
{
  return (array && (ind >= 0) && (ind < sidlArrayDim(array))) 
    ? sidlLength(array, ind) : 0;
}

int32_t
sidl__array_stride(const struct sidl__array *array, int32_t ind)
{
  return (array && (ind >= 0) && (ind < sidlArrayDim(array))) 
    ? sidlStride(array, ind) : -1;
}

int32_t
sidl__array_type(const struct sidl__array*array)
{
  return (array)
    ? (array->d_vtable->d_arraytype)() : 0;
}

/**
 * Return a true value iff the array is a contiguous column-major ordered
 * array.  A NULL array argument causes 0 to be returned.
 */
sidl_bool
sidl__array_isColumnOrder(const struct sidl__array* array)
{
  if (!array) return FALSE;
  else {
    register int32_t i;
    register int32_t size;
    register const int32_t dimen = sidlArrayDim(array);
    for(i = 0, size = 1; i < dimen ; ++i) {
      if (array->d_stride[i] != size) return FALSE;
      size *= (1 + array->d_upper[i] - array->d_lower[i]);
    }
    return TRUE;
  }
}

/**
 * Return a true value iff the array is a contiguous row-major ordered
 * array.  A NULL array argument causes 0 to be returned.
 */
sidl_bool
sidl__array_isRowOrder(const struct sidl__array* array)
{
  if (!array) return FALSE;
  else {
    register int32_t i = sidlArrayDim(array) - 1;
    register int32_t size;
    for(size = 1; i >= 0 ; --i) {
      if (array->d_stride[i] != size) return FALSE;
      size *= (1 + array->d_upper[i] - array->d_lower[i]);
    }
    return TRUE;
  }
}
