/*
 * File:        sidlArray.c
 * Copyright:   (c) 2004 The Regents of the University of California
 * Revision:    @(#) $Revision$
 * Date:        $Date$
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
#include <stdio.h>


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
  int32_t arrayType = array ? sidl__array_type(array) : 0;
  if ((arrayType < 0) ||
      (arrayType >= sizeof(s_array_types)/sizeof(char*))) {
    arrayType = 0;
  }
  return s_array_types[arrayType];
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
