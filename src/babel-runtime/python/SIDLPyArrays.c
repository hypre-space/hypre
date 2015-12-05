/*
 * File:        SIDLPyArrays.c
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name: V1-9-0b $
 * Revision:    @(#) $Revision: 1.4 $
 * Date:        $Date: 2003/04/07 21:44:24 $
 * Description: Runtime support routines to convert SIDL arrays to/from Python
 *
 * This file provides functions to convert SIDL arrays to Python with or
 * without borrowing data and functions to convert Python arrays to SIDL
 * with or without borrowing. When borrowing data is not possible, data
 * is copied.
 * Copyright (c) 2000-2001, The Regents of the University of Calfornia.
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

#define SIDLPyArrays_MODULE 1
#include "SIDLPyArrays.h"

#include "SIDLArray.h"
#include "SIDL_bool_IOR.h"
#include "SIDL_char_IOR.h"
#include "SIDL_dcomplex_IOR.h"
#include "SIDL_double_IOR.h"
#include "SIDL_fcomplex_IOR.h"
#include "SIDL_float_IOR.h"
#include "SIDL_int_IOR.h"
#include "SIDL_long_IOR.h"
#include "SIDL_opaque_IOR.h"
#include "SIDL_string_IOR.h"
#include "SIDL_BaseInterface_IOR.h"

/* include Numerical Python header file */
#include "Numeric/arrayobject.h"
#include <stdlib.h>
#include <string.h>

/**
 * Convert the array description information from the Numerical Python array
 * into a form that SIDL likes.  SIDL needs a vector of lower and upper
 * index bounds, and a vector holding the stride.
 */
SIDL_array__extract_python_info_RETURN
SIDL_array__extract_python_info SIDL_array__extract_python_info_PROTO
{
  *lower = *upper = *stride = NULL;
  if (PyArray_Check(pya)) {
    const PyArrayObject *array = (PyArrayObject *)pya;
    if (array->nd >= 0) {
      /* a zero dimensional Python array is like a scalar */
      const int32_t dimen = array->nd ? array->nd : 1;
      int32_t i;
      *dimension = dimen;
      *lower = malloc(sizeof(int32_t) * dimen);
      *upper = malloc(sizeof(int32_t) * dimen);
      *stride = malloc(sizeof(int32_t) * dimen);
      **lower = **upper = 0;
      **stride = 1;
      for(i = 0; i < array->nd; ++i) {
        (*lower)[i] = 0;
        (*upper)[i] = array->dimensions[i] - 1;
        (*stride)[i] = array->strides[i]/array->descr->elsize;
        if ((array->strides[i] % array->descr->elsize) != 0) {
          goto error;
        }
      }
      return 1;
    }
  }
 error:
  if (*lower)  free(*lower);
  if (*upper)  free(*upper);
  if (*stride) free(*stride);
  *lower = *upper = *stride = NULL;
  return 0;
}

/**
 * Return a true value iff the array is contiguous and in column-major order
 * (i.e. like FORTRAN).
 *
 *   stride    the stride for each dimension in bytes.
 *   numelem   the number of elements in each dimension.
 *   dim       the number of dimensions.
 *   elsize    the element size in bytes.
 */
static int
columnMajorOrder(const int     stride[],
                 const int32_t numelem[],
                 const int32_t dim,
                 const int32_t elsize)
{
  int32_t dimStride = elsize, i;
  for(i = 0; i < dim; ++i) {
    if (stride[i] != dimStride) return FALSE;
    dimStride *= numelem[i];
  }
  return TRUE;
}

/**
 * Return a true value iff the array is contiguous and in row-major order
 * (i.e. like C). 
 *
 *   stride    the stride for each dimension in bytes.
 *   numelem   the number of elements in each dimension.
 *   dim       the number of dimensions.
 *   elsize    the element size in bytes.
 */
static int
rowMajorOrder(const int     stride[],
              const int32_t numelem[],
              const int32_t dim,
              const int32_t elsize)
{
  int dimStride = elsize, i;
  for(i = dim - 1; i >= 0; --i) {
    if (stride[i] != dimStride) return FALSE;
    dimStride *= numelem[i];
  }
  return TRUE;
}

/**
 * Return the number elements in the array times elsize.  If elsize is the
 * element size in bytes, this will return the size of the array in bytes.
 * If elsize is 1, this returns the number of elements in the array.
 *
 *   numelem     the number of elements in each dimension. Array of dim
 *               integers.
 *   dim         the number of dimensions.
 *   elsize      to get the size of the array in bytes, pass in the
 *               size of an element in bytes. To get the number of
 *               elements, pass in 1.
 */
static size_t
arraySize(const int32_t numelem[],
          const int32_t dim,
          const int32_t elsize)
{
  size_t result = 0;
  if (dim > 0) {
    int i;
    result = elsize;
    for(i = 0; i < dim; ++i) {
      result *= numelem[i];
    }
  }
  return result;
}

/**
 * A parameterized function to copy one strided array into another.
 */
#define copyData_impl(name, type) \
static void copy_ ## name ## _data(char * restrict dest, \
                                   const int dstride[], /* not int32_t */ \
                                   const char * restrict src, \
                                   const int sstride[], /* not int32_t */ \
                                   const int32_t numelem[], \
                                   const int32_t dim) \
{ \
  size_t num = arraySize(numelem, dim, sizeof(type)); \
  if (num > 0) { \
    if ((columnMajorOrder(sstride, numelem, dim, sizeof(type)) && \
         columnMajorOrder(dstride, numelem, dim, sizeof(type))) || \
        (rowMajorOrder(sstride, numelem, dim, sizeof(type)) && \
         rowMajorOrder(dstride, numelem, dim, sizeof(type)))) { \
      memcpy(dest, src, num); \
    } \
    else { \
      int32_t * restrict ind = malloc(sizeof(int32_t)*dim), i; \
      memset(ind, 0, sizeof(int32_t)*dim); \
      num /= sizeof(type); \
      while (num--) { \
        *((type * restrict)dest) = *((const type * restrict)src); \
        for (i = 0; i < dim ; ++i) { \
          dest += dstride[i]; \
          src += sstride[i]; \
          if (++(ind[i]) < numelem[i]) { \
            break; \
          } \
          else { \
            ind[i] = 0; \
            dest -= (numelem[i]*dstride[i]); \
            src -= (numelem[i]*sstride[i]); \
          } \
        } \
      } \
      free(ind); \
    } \
  } \
}

copyData_impl(bool, int)
copyData_impl(char, char)
copyData_impl(int, int32_t)
#if (SIZEOF_SHORT == 8) || (SIZEOF_INT == 8) || (SIZEOF_LONG == 8)
copyData_impl(long, int64_t)
#endif
copyData_impl(float, float)
copyData_impl(double, double)
copyData_impl(fcomplex, struct SIDL_fcomplex)
copyData_impl(dcomplex, struct SIDL_dcomplex)

/*
 * Parameterized function implementation to borrow a particular type
 * array.
 */
#define SIDL_borrow_impl(stype,dtype,pytype) \
{\
  if (PyArray_Check(obj)) {\
    PyArrayObject *pya = (PyArrayObject *)obj;\
    if (pytype == pya->descr->type_num) {\
      int32_t dimension, *lower, *upper, *stride;\
      if (SIDL_array__extract_python_info((PyObject *)pya, &dimension, \
                                          &lower, &upper, &stride)) {\
        *array = SIDL_ ## stype ## __array_borrow((dtype *)pya->data,\
                                         dimension, lower, upper, stride);\
        free(lower);\
        free(upper);\
        free(stride);\
        return (*array != NULL);\
      }\
    }\
  }\
  return SIDL_ ## stype ## __clone_python_array_row(obj, array);\
}


SIDL_bool__borrow_python_array_RETURN
SIDL_bool__borrow_python_array SIDL_bool__borrow_python_array_PROTO
SIDL_borrow_impl(bool, SIDL_bool, PyArray_INT)

SIDL_char__borrow_python_array_RETURN
SIDL_char__borrow_python_array SIDL_char__borrow_python_array_PROTO
SIDL_borrow_impl(char, char, PyArray_CHAR)

SIDL_dcomplex__borrow_python_array_RETURN
SIDL_dcomplex__borrow_python_array SIDL_dcomplex__borrow_python_array_PROTO
SIDL_borrow_impl(dcomplex, struct SIDL_dcomplex, PyArray_CDOUBLE)

SIDL_double__borrow_python_array_RETURN
SIDL_double__borrow_python_array SIDL_double__borrow_python_array_PROTO
SIDL_borrow_impl(double, double, PyArray_DOUBLE)

SIDL_fcomplex__borrow_python_array_RETURN
SIDL_fcomplex__borrow_python_array SIDL_fcomplex__borrow_python_array_PROTO
SIDL_borrow_impl(fcomplex, struct SIDL_fcomplex, PyArray_CFLOAT)

SIDL_float__borrow_python_array_RETURN
SIDL_float__borrow_python_array SIDL_float__borrow_python_array_PROTO
SIDL_borrow_impl(float, float, PyArray_FLOAT)

#if SIZEOF_SHORT == 4
SIDL_int__borrow_python_array_RETURN
SIDL_int__borrow_python_array SIDL_int__borrow_python_array_PROTO
SIDL_borrow_impl(int, int32_t, PyArray_SHORT)
#else
#if SIZEOF_INT == 4
SIDL_int__borrow_python_array_RETURN
SIDL_int__borrow_python_array SIDL_int__borrow_python_array_PROTO
SIDL_borrow_impl(int, int32_t, PyArray_INT)
#else
#if SIZEOF_LONG == 4
SIDL_int__borrow_python_array_RETURN
SIDL_int__borrow_python_array SIDL_int__borrow_python_array_PROTO
SIDL_borrow_impl(int, int32_t, PyArray_LONG)
#else
#error No 32-bit integer available.
#endif
#endif
#endif

#if SIZEOF_SHORT == 8
SIDL_long__borrow_python_array_RETURN
SIDL_long__borrow_python_array SIDL_long__borrow_python_array_PROTO
SIDL_borrow_impl(long, int64_t, PyArray_SHORT)
#else
#if SIZEOF_INT == 8
SIDL_long__borrow_python_array_RETURN
SIDL_long__borrow_python_array SIDL_long__borrow_python_array_PROTO
SIDL_borrow_impl(long, int64_t, PyArray_INT)
#else
#if SIZEOF_LONG == 8
SIDL_long__borrow_python_array_RETURN
SIDL_long__borrow_python_array SIDL_long__borrow_python_array_PROTO
SIDL_borrow_impl(long, int64_t, PyArray_LONG)
#else
     /* none of Numeric Python types is 64 bits */
SIDL_long__borrow_python_array_RETURN
SIDL_long__borrow_python_array SIDL_long__borrow_python_array_PROTO
{
  return SIDL_long__clone_python_array_row(obj,array);
}
#endif
#endif
#endif

SIDL_string__borrow_python_array_RETURN
SIDL_string__borrow_python_array SIDL_string__borrow_python_array_PROTO
{
  return SIDL_string__clone_python_array_row(obj,array);
}

SIDL_opaque__borrow_python_array_RETURN
SIDL_opaque__borrow_python_array SIDL_opaque__borrow_python_array_PROTO
{
  return SIDL_opaque__clone_python_array_row(obj,array);
}

/*
 * Parameterized implementation of a clone function.
 */
#define ClonePython_impl(sidlname, sidltype, pyarraytype, order) \
{ \
  int result = 0; \
  PyArrayObject *pya = \
    (PyArrayObject *)PyArray_FromObject(obj, pyarraytype, 0, 0); \
  *array = 0; \
  if (pya) { \
    int32_t dimen, *lower, *upper, *stride; \
    if (SIDL_array__extract_python_info((PyObject *)pya, &dimen, \
                                        &lower, &upper, &stride)) { \
      *array = \
         SIDL_ ## sidlname ## __array_create ## order(dimen, lower, upper); \
      if (*array) { \
        int *bytestride = malloc(sizeof(int)*dimen), i; \
        int32_t *numelem = malloc(sizeof(int32_t)*dimen); \
        for(i = 0; i < dimen; ++i){ \
          numelem[i] = 1 + upper[i] - lower[i]; \
          bytestride[i] = sizeof(sidltype)* \
             SIDLStride(*array, i); \
        } \
        copy_ ## sidlname ## _data((char *)((*array)->d_firstElement),  \
                       bytestride, \
                       pya->data, \
                       pya->strides, \
                       numelem, \
                       dimen); \
        free(numelem); \
        free(bytestride); \
        result = 1; \
      } \
      free(stride); \
      free(upper); \
      free(lower); \
    } \
    Py_DECREF(pya); \
  } \
  return result; \
}

SIDL_bool__clone_python_array_column_RETURN
SIDL_bool__clone_python_array_column SIDL_bool__clone_python_array_column_PROTO
ClonePython_impl(bool, SIDL_bool, PyArray_INT, Col)

SIDL_char__clone_python_array_column_RETURN
SIDL_char__clone_python_array_column SIDL_char__clone_python_array_column_PROTO
ClonePython_impl(char, char, PyArray_CHAR, Col)

SIDL_dcomplex__clone_python_array_column_RETURN
SIDL_dcomplex__clone_python_array_column SIDL_dcomplex__clone_python_array_column_PROTO
ClonePython_impl(dcomplex, struct SIDL_dcomplex, PyArray_CDOUBLE, Col)

SIDL_double__clone_python_array_column_RETURN
SIDL_double__clone_python_array_column SIDL_double__clone_python_array_column_PROTO
ClonePython_impl(double, double, PyArray_DOUBLE, Col)

SIDL_fcomplex__clone_python_array_column_RETURN
SIDL_fcomplex__clone_python_array_column SIDL_fcomplex__clone_python_array_column_PROTO
ClonePython_impl(fcomplex, struct SIDL_fcomplex, PyArray_CFLOAT, Col)

SIDL_float__clone_python_array_column_RETURN
SIDL_float__clone_python_array_column SIDL_float__clone_python_array_column_PROTO
ClonePython_impl(float, float, PyArray_FLOAT, Col)

#if SIZEOF_SHORT == 4
SIDL_int__clone_python_array_column_RETURN
SIDL_int__clone_python_array_column SIDL_int__clone_python_array_column_PROTO
ClonePython_impl(int, int32_t, PyArray_SHORT, Col)
#else
#if SIZEOF_INT == 4
SIDL_int__clone_python_array_column_RETURN
SIDL_int__clone_python_array_column SIDL_int__clone_python_array_column_PROTO
ClonePython_impl(int, int32_t, PyArray_INT, Col)
#else
#if SIZEOF_LONG == 4
SIDL_int__clone_python_array_column_RETURN
SIDL_int__clone_python_array_column SIDL_int__clone_python_array_column_PROTO
ClonePython_impl(int, int32_t, PyArray_LONG, Col)
#else
#error No 32-bit integer available.
#endif
#endif
#endif

#if SIZEOF_SHORT == 8
SIDL_long__clone_python_array_column_RETURN
SIDL_long__clone_python_array_column SIDL_long__clone_python_array_column_PROTO
ClonePython_impl(long, int64_t, PyArray_SHORT, Col)
#else
#if SIZEOF_INT == 8
SIDL_long__clone_python_array_column_RETURN
SIDL_long__clone_python_array_column SIDL_long__clone_python_array_column_PROTO
ClonePython_impl(long, int64_t, PyArray_INT, Col)
#else
#if SIZEOF_LONG == 8
SIDL_long__clone_python_array_column_RETURN
SIDL_long__clone_python_array_column SIDL_long__clone_python_array_column_PROTO
ClonePython_impl(long, int64_t, PyArray_LONG, Col)
#else
static PyArrayObject *
toNumericArray(PyObject *obj)
{
  PyArrayObject *result = NULL;
  if (obj) {
    if (PyArray_Check(obj)) {
      result = (PyArrayObject *)obj;
      Py_INCREF(obj);
    }
    else {
      result = (PyArrayObject *)
        PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
    }
  }
  return result;
}

#define long_copy(srctype,convert) \
for(i = 0 ; i < size ; ++i ) { \
  *dest = (int64_t)convert(*((const srctype * restrict)src)); \
  for (j = 0 ;  j < dimen; ++j ) { \
    dest += stride[j]; \
    src  += srcstride[j]; \
    if ((++(ind[j])) < numelem[j]) { \
      break; \
    } \
    else { \
      ind[j] = 0; \
      dest -= (numelem[j]*stride[j]); \
      src  -= (numelem[j]*srcstride[j]); \
    } \
  } \
}

static int64_t
pythonToLong(const PyObject *obj)
{
  int64_t result;
  if (obj) {
    if (PyInt_Check(obj)) result = PyInt_AsLong((PyObject *)obj);
    else {
      PyObject *lnum = PyNumber_Long((PyObject *)obj);
      if (lnum) {
#if HAVE_LONG_LONG
        result = PyLong_AsLongLong(lnum);
#else
        result = PyLong_AsLong(lnum);
#endif
        Py_DECREF(lnum);
      }
      else {
        lnum = PyNumber_Int((PyObject *)obj);
        if (lnum) {
          result = PyInt_AsLong(lnum);
          Py_DECREF(lnum);
        }
        else {
          result = 0;
        }
      }
    }
  }
  else {
    result = 0;
  }
  return result;
}


static void
clone_long_python(PyArrayObject *pya,
                  struct SIDL_long__array *array,
                  const int32_t lower[],
                  const int32_t upper[],
                  const int32_t stride[])
{
  const int size = PyArray_Size((PyObject *)pya);
  const int32_t dimen = pya->nd;
  const int32_t sdim = SIDL_long__array_dimen(array);
  int i, j;
  int64_t * restrict dest = SIDL_long__array_first(array);
  const char * restrict src = pya->data;
  const int * restrict srcstride = pya->strides;
  int32_t * restrict ind =  malloc(sizeof(int32_t)*sdim);
  int32_t * restrict numelem = malloc(sizeof(int32_t)*sdim);
  if (ind && numelem) {
    memset(ind, 0, sizeof(int32_t)*sdim);
    for(i = 0; i < sdim; ++i ){
      numelem[i] = SIDLUpper(array,i) - SIDLLower(array,i) + 1;
    }
    switch(pya->descr->type_num) {
    case PyArray_CHAR:
      long_copy(char,);
      break;
    case PyArray_UBYTE:
      long_copy(unsigned char,);
      break;
    case PyArray_SBYTE:
      long_copy(signed char,);
      break;
    case PyArray_SHORT:
      long_copy(short,);
      break;
    case PyArray_INT:
      long_copy(int,);
      break;
    case PyArray_LONG:
      long_copy(long,);
      break;
    case PyArray_FLOAT:
      long_copy(float,);
      break;
    case PyArray_DOUBLE:
      long_copy(double,);
      break;
    case PyArray_OBJECT:
      long_copy(PyObject *,pythonToLong);
      break;
    }
  }
  if (ind)      free(ind);
  if (numelem)  free(numelem);
}

SIDL_long__clone_python_array_column_RETURN
SIDL_long__clone_python_array_column SIDL_long__clone_python_array_column_PROTO
{
  int result = 0;
  PyArrayObject *pya = toNumericArray(obj);
  *array = NULL;
  if (pya) {
    int32_t dimen;
    int32_t *lower, *upper, *stride;
    if (SIDL_array__extract_python_info((PyObject *)pya, &dimen,
                                        &lower, &upper, &stride)) {
      *array =
         SIDL_long__array_createCol(dimen, lower, upper);
      clone_long_python(pya, *array, lower, upper, stride);
      free(lower);
      free(upper);
      free(stride);
      result = 1;
    }
    Py_DECREF((PyObject *)pya);
  }
  return result;
}
#endif
#endif
#endif


SIDL_bool__clone_python_array_row_RETURN
SIDL_bool__clone_python_array_row SIDL_bool__clone_python_array_row_PROTO
ClonePython_impl(bool, SIDL_bool, PyArray_INT, Row)

SIDL_char__clone_python_array_row_RETURN
SIDL_char__clone_python_array_row SIDL_char__clone_python_array_row_PROTO
ClonePython_impl(char, char, PyArray_CHAR, Row)

SIDL_dcomplex__clone_python_array_row_RETURN
SIDL_dcomplex__clone_python_array_row SIDL_dcomplex__clone_python_array_row_PROTO
ClonePython_impl(dcomplex, struct SIDL_dcomplex, PyArray_CDOUBLE, Row)

SIDL_double__clone_python_array_row_RETURN
SIDL_double__clone_python_array_row SIDL_double__clone_python_array_row_PROTO
ClonePython_impl(double, double, PyArray_DOUBLE, Row)

SIDL_fcomplex__clone_python_array_row_RETURN
SIDL_fcomplex__clone_python_array_row SIDL_fcomplex__clone_python_array_row_PROTO
ClonePython_impl(fcomplex, struct SIDL_fcomplex, PyArray_CFLOAT, Row)

SIDL_float__clone_python_array_row_RETURN
SIDL_float__clone_python_array_row SIDL_float__clone_python_array_row_PROTO
ClonePython_impl(float, float, PyArray_FLOAT, Row)

SIDL_int__clone_python_array_row_RETURN
SIDL_int__clone_python_array_row SIDL_int__clone_python_array_row_PROTO
ClonePython_impl(int, int, PyArray_INT, Row)

#if SIZEOF_SHORT == 8
SIDL_long__clone_python_array_row_RETURN
SIDL_long__clone_python_array_row SIDL_long__clone_python_array_row_PROTO
ClonePython_impl(long, int64_t, PyArray_SHORT, Row)
#else
#if SIZEOF_INT == 8
SIDL_long__clone_python_array_row_RETURN
SIDL_long__clone_python_array_row SIDL_long__clone_python_array_row_PROTO
ClonePython_impl(long, int64_t, PyArray_INT, Row)
#else
#if SIZEOF_LONG == 8
SIDL_long__clone_python_array_row_RETURN
SIDL_long__clone_python_array_row SIDL_long__clone_python_array_row_PROTO
ClonePython_impl(long, int64_t, PyArray_LONG, Row)
#else
SIDL_long__clone_python_array_row_RETURN
SIDL_long__clone_python_array_row SIDL_long__clone_python_array_row_PROTO
{
  int result = 0;
  PyArrayObject *pya = toNumericArray(obj);
  *array = NULL;
  if (pya) {
    int32_t dimen, *lower, *upper, *stride;
    if (SIDL_array__extract_python_info((PyObject *)pya, &dimen,
                                        &lower, &upper, &stride)) {
      *array =
         SIDL_long__array_createRow(dimen, lower, upper);
      clone_long_python(pya, *array, lower, upper, stride);
      result = 1;
      free(lower);
      free(upper);
      free(stride);
    }
    Py_DECREF((PyObject *)pya);
  }
  return result;
}
#endif
#endif
#endif


SIDL_array__convert_python_RETURN
SIDL_array__convert_python SIDL_array__convert_python_PROTO
{
  int result = FALSE;
  if (PyArray_Check(pya_src)) {
    int i, j;
    const int size = PyArray_Size(pya_src);
    const PyArrayObject *pya = (PyArrayObject *)pya_src;
    int32_t *ind = malloc(sizeof(int32_t) * dimen);
    char *pydata = pya->data;
    memset(ind, 0, sizeof(int32_t) * dimen);
    if (size == 1) {              /* handle zero dimensional PyArray */
      if (!((*setfunc)(sidl_dest, ind, *(PyObject **)pydata))) {
        result = TRUE;
      }
    }
    else {
      for(i = 0; i < size; ++i){
        if ((*setfunc)(sidl_dest, ind, *(PyObject **)pydata)) {
          goto error;
        }
        for(j = 0; j < dimen; ++j) {
          pydata += pya->strides[j];
          if (++(ind[j]) < pya->dimensions[j]) {
            break;
          }
          else {
            ind[j] = 0;
            pydata -= (pya->dimensions[j]*pya->strides[j]);
          }
        }
      }
      result = TRUE;
    }
  error:
    free(ind);
  }
  return result;
}

int CopyOpaquePointer(void *array,
                      const int32_t *ind,
                      PyObject *data)
{
  if (PyCObject_Check(data)) {
    SIDL_opaque__array_set((struct SIDL_opaque__array*)array,
                           ind,
                           PyCObject_AsVoidPtr(data));
    return FALSE;
  }
  return TRUE;
}

int CopyStringPointer(void *array,
                      const int32_t *ind,
                      PyObject *data)
{
  PyObject *str = PyObject_Str(data);
  if (str) {
    SIDL_string__array_set((struct SIDL_string__array*)array, ind,
                           PyString_AsString(str));
    Py_DECREF(str);
    return FALSE;
  }
  return TRUE;
}

SIDL_opaque__clone_python_array_column_RETURN
SIDL_opaque__clone_python_array_column SIDL_opaque__clone_python_array_column_PROTO
{
  int result = 0;
  PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
  *array = NULL;
  if (pya) {
    if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
      int dimen, *lower, *upper, *stride;
      if (SIDL_array__extract_python_info(pya, &dimen, &lower, &upper,
                                          &stride)) {
        *array = SIDL_opaque__array_createCol(dimen, lower, upper);
        free(stride);
        free(upper);
        free(lower);
        result = SIDL_array__convert_python(pya, dimen, *array,
                                            CopyOpaquePointer);
        if (*array && !result) {
          SIDL_opaque__array_deleteRef(*array);
          *array = NULL;
        }
      }
    }
    Py_DECREF(pya);
  }
  return result;
}

SIDL_string__clone_python_array_column_RETURN
SIDL_string__clone_python_array_column SIDL_string__clone_python_array_column_PROTO
{
  int result = 0;
  PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
  *array = NULL;
  if (pya) {
    if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
      int dimen, *lower, *upper, *stride;
      if (SIDL_array__extract_python_info(pya, &dimen, &lower, &upper,
                                          &stride)) {
        *array = SIDL_string__array_createCol(dimen, lower, upper);
        free(stride);
        free(upper);
        free(lower);
        result = SIDL_array__convert_python(pya, dimen, *array, 
                                            CopyStringPointer);
        if (*array && !result) {
          SIDL_string__array_deleteRef(*array);
          *array = NULL;
        }
      }
    }
    Py_DECREF(pya);
  }
  return result;
}


SIDL_opaque__clone_python_array_row_RETURN
SIDL_opaque__clone_python_array_row SIDL_opaque__clone_python_array_row_PROTO
{
  int result = 0;
  PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
  *array = NULL;
  if (pya) {
    if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
      int dimen, *lower, *upper, *stride;
      if (SIDL_array__extract_python_info(pya, &dimen, &lower, &upper,
                                          &stride)) {
        *array = SIDL_opaque__array_createRow(dimen, lower, upper);
        free(stride);
        free(upper);
        free(lower);
        result = SIDL_array__convert_python(pya, dimen, *array,
                                            CopyOpaquePointer);
        if (*array && !result) {
          SIDL_opaque__array_deleteRef(*array);
          *array = NULL;
        }
      }
    }
    Py_DECREF(pya);
  }
  return result;
}

SIDL_string__clone_python_array_row_RETURN
SIDL_string__clone_python_array_row SIDL_string__clone_python_array_row_PROTO
{
  int result = 0;
  PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
  *array = NULL;
  if (pya) {
    if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
      int dimen, *lower, *upper, *stride;
      if (SIDL_array__extract_python_info(pya, &dimen, &lower, &upper,
                                          &stride)) {
        *array = SIDL_string__array_createRow(dimen, lower, upper);
        free(stride);
        free(upper);
        free(lower);
        result = SIDL_array__convert_python(pya, dimen, *array, 
                                            CopyStringPointer);
        if (*array && !result) {
          SIDL_string__array_deleteRef(*array);
          *array = NULL;
        }
      }
    }
    Py_DECREF(pya);
  }
  return result;
}


#define CloneSIDL_impl(sidlname, sidltype, pyarraytype) \
{ \
  PyArrayObject *pya = NULL; \
  if (array) { \
    const int dimen = SIDLArrayDim(array); \
    int i; \
    int32_t *numelem = malloc(sizeof(int32_t)*dimen); \
    int *pynumelem = malloc(sizeof(int)*dimen); \
    int *bytestride = malloc(sizeof(int)*dimen); \
    for(i = 0; i < dimen; ++i){ \
      numelem[i] = 1 + SIDLUpper(array,i) - SIDLLower(array,i); \
      pynumelem[i] = (int)numelem[i]; \
      bytestride[i] = sizeof(sidltype)* \
        SIDLStride(array, i); \
    } \
    pya = (PyArrayObject *)PyArray_FromDims(dimen, pynumelem, pyarraytype); \
    if (pya) { \
      copy_ ## sidlname ## _data(pya->data, \
                     pya->strides, \
                     (char *)((array)->d_firstElement),  \
                     bytestride, \
                     numelem, \
                     dimen); \
    } \
    free(pynumelem); \
    free(numelem); \
    free(bytestride); \
  } \
  return (PyObject *)pya; \
}

SIDL_bool__python_clone_array_RETURN
SIDL_bool__python_clone_array SIDL_bool__python_clone_array_PROTO
CloneSIDL_impl(bool, SIDL_bool, PyArray_INT)

SIDL_char__python_clone_array_RETURN
SIDL_char__python_clone_array SIDL_char__python_clone_array_PROTO
CloneSIDL_impl(char, char, PyArray_CHAR)

SIDL_dcomplex__python_clone_array_RETURN
SIDL_dcomplex__python_clone_array SIDL_dcomplex__python_clone_array_PROTO
CloneSIDL_impl(dcomplex, struct SIDL_dcomplex, PyArray_CDOUBLE)

SIDL_double__python_clone_array_RETURN
SIDL_double__python_clone_array SIDL_double__python_clone_array_PROTO
CloneSIDL_impl(double, double, PyArray_DOUBLE)

SIDL_fcomplex__python_clone_array_RETURN
SIDL_fcomplex__python_clone_array SIDL_fcomplex__python_clone_array_PROTO
CloneSIDL_impl(fcomplex, struct SIDL_fcomplex, PyArray_CFLOAT)

SIDL_float__python_clone_array_RETURN
SIDL_float__python_clone_array SIDL_float__python_clone_array_PROTO
CloneSIDL_impl(float, float, PyArray_FLOAT)

#if SIZEOF_SHORT == 4
SIDL_int__python_clone_array_RETURN
SIDL_int__python_clone_array SIDL_int__python_clone_array_PROTO
CloneSIDL_impl(int, int32_t, PyArray_SHORT)
#else
#if SIZEOF_INT == 4
SIDL_int__python_clone_array_RETURN
SIDL_int__python_clone_array SIDL_int__python_clone_array_PROTO
CloneSIDL_impl(int, int32_t, PyArray_INT)
#else
#if SIZEOF_LONG == 4
SIDL_int__python_clone_array_RETURN
SIDL_int__python_clone_array SIDL_int__python_clone_array_PROTO
CloneSIDL_impl(int, int32_t, PyArray_LONG)
#else
#error No 32-bit integer available.
#endif
#endif
#endif

#if SIZEOF_SHORT == 8
SIDL_long__python_clone_array_RETURN
SIDL_long__python_clone_array SIDL_long__python_clone_array_PROTO
CloneSIDL_impl(long, int64_t, PyArray_SHORT)
#else
#if SIZEOF_INT == 8
SIDL_long__python_clone_array_RETURN
SIDL_long__python_clone_array SIDL_long__python_clone_array_PROTO
CloneSIDL_impl(long, int64_t, PyArray_INT)
#else
#if SIZEOF_LONG == 8
SIDL_long__python_clone_array_RETURN
SIDL_long__python_clone_array SIDL_long__python_clone_array_PROTO
CloneSIDL_impl(long, int64_t, PyArray_LONG)
#else
static int
getAndConvertLong(void *array,
                  const int32_t *ind,
                  PyObject **dest)
{
  int64_t val =
    SIDL_long__array_get((struct SIDL_long__array *)array, ind);
  *dest = PyLong_FromLongLong(val);
  return FALSE;
}

SIDL_long__python_clone_array_RETURN
SIDL_long__python_clone_array SIDL_long__python_clone_array_PROTO
{
  PyObject *pya = NULL;
  if (array) {
    const int dimen = SIDL_long__array_dimen(array);
    int i;
    int32_t *lower = malloc(sizeof(int32_t) * dimen);
    int32_t *upper = malloc(sizeof(int32_t) * dimen);
    int32_t *numelem = malloc(sizeof(int32_t) * dimen);
    int *pynumelem = malloc(sizeof(int) * dimen);
    for(i = 0 ; i < dimen; ++i ){
      lower[i] = SIDL_long__array_lower(array, i);
      upper[i] = SIDL_long__array_upper(array, i);
      numelem[i] = 1 + upper[i] - lower[i];
      pynumelem[i] = numelem[i];
    }
    pya = PyArray_FromDims(dimen, pynumelem, PyArray_OBJECT);
    if (pya) {
      if (!SIDL_array__convert_sidl(pya, dimen, lower, upper,
                                    numelem, array, getAndConvertLong)) {
        Py_DECREF(pya);
        pya = NULL;
      }
    }
    free(pynumelem);
    free(numelem);
    free(upper);
    free(lower);
  }
  return pya;
}
#endif
#endif
#endif

SIDL_array__convert_sidl_RETURN
SIDL_array__convert_sidl SIDL_array__convert_sidl_PROTO
{
  if (PyArray_Check(pya_dest)) {
    PyArrayObject *pya = (PyArrayObject *)pya_dest;
    size_t size = arraySize(numelem, dimen, 1);
    int i;
    char *dest = pya->data;
    while (size--) {
      if ((*getfunc)(sidl_src, lower, (PyObject **)dest))
        return FALSE;
      for(i = 0; i < dimen; ++i) {
        dest += pya->strides[i];
        if (++(lower[i]) <= upper[i]) {
          break;
        }
        else {
          dest -= pya->strides[i]*numelem[i];
          lower[i] -= numelem[i];
        }
      }
    }
    return TRUE;
  }
  return FALSE;
}

static int
getAndConvertString(void *array, const int32_t *ind, PyObject **dest)
{
  char *str = 
    SIDL_string__array_get((struct SIDL_string__array *)array, ind);
  if (str) {
    *dest = PyString_FromString(str); 
    free(str);
  }
  else {
    Py_INCREF(Py_None);
    *dest = Py_None;
  }
  return FALSE;
}

SIDL_string__python_clone_array_RETURN
SIDL_string__python_clone_array SIDL_string__python_clone_array_PROTO
{
  PyObject *pya = NULL;
  if (array) {
    const int dimen = SIDL_string__array_dimen(array);
    int i;
    int32_t *lower = malloc(sizeof(int32_t) * dimen);
    int32_t *upper = malloc(sizeof(int32_t) * dimen);
    int32_t *numelem = malloc(sizeof(int32_t) * dimen);
    int *pynumelem = malloc(sizeof(int) * dimen);
    for(i = 0; i < dimen; ++i) {
      lower[i] = SIDL_string__array_lower(array, i);
      upper[i] = SIDL_string__array_upper(array, i);
      numelem[i] = 1 + upper[i] - lower[i];
      pynumelem[i] = (int)numelem[i];
    }
    pya = PyArray_FromDims(dimen, pynumelem, PyArray_OBJECT);
    if (pya) {
      if (!SIDL_array__convert_sidl(pya, dimen, lower, upper, 
                                    numelem, array, getAndConvertString)) {
        Py_DECREF(pya);
        pya = NULL;
      }
    }
    free(pynumelem);
    free(numelem);
    free(upper);
    free(lower);
  }
  return pya;
}

static int
getAndConvertOpaque(void *array, const int32_t *ind, PyObject **dest)
{
  void *vptr =
    SIDL_opaque__array_get((struct SIDL_opaque__array *)array, ind);
  *dest = PyCObject_FromVoidPtr(vptr, NULL);
  return FALSE;
}

SIDL_opaque__python_clone_array_RETURN
SIDL_opaque__python_clone_array SIDL_opaque__python_clone_array_PROTO
{
  PyObject *pya = NULL;
  if (array) {
    const int dimen = SIDL_opaque__array_dimen(array);
    int i;
    int32_t *lower = malloc(sizeof(int32_t) * dimen);
    int32_t *upper = malloc(sizeof(int32_t) * dimen);
    int32_t *numelem = malloc(sizeof(int32_t) * dimen);
    int *pynumelem = malloc(sizeof(int) * dimen);
    for(i = 0; i < dimen; ++i) {
      lower[i] = SIDL_opaque__array_lower(array, i);
      upper[i] = SIDL_opaque__array_upper(array, i);
      numelem[i] = 1 + upper[i] - lower[i];
      pynumelem[i] = (int)numelem[i];
    }
    pya = PyArray_FromDims(dimen, pynumelem, PyArray_OBJECT);
    if (pya) {
      if (!SIDL_array__convert_sidl(pya, dimen, lower, upper,
                                    numelem, array, getAndConvertOpaque)) {
        Py_DECREF(pya);
        pya = NULL;
      }
    }
    free(numelem);
    free(upper);
    free(lower);
  }
  return pya;
}

SIDL_bool__python_borrow_array_RETURN
SIDL_bool__python_borrow_array SIDL_bool__python_borrow_array_PROTO
{ return SIDL_bool__python_clone_array(array); }

SIDL_char__python_borrow_array_RETURN
SIDL_char__python_borrow_array SIDL_char__python_borrow_array_PROTO
{ return SIDL_char__python_clone_array(array); }

SIDL_dcomplex__python_borrow_array_RETURN
SIDL_dcomplex__python_borrow_array SIDL_dcomplex__python_borrow_array_PROTO
{ return SIDL_dcomplex__python_clone_array(array); }

SIDL_double__python_borrow_array_RETURN
SIDL_double__python_borrow_array SIDL_double__python_borrow_array_PROTO
{ return SIDL_double__python_clone_array(array); }

SIDL_fcomplex__python_borrow_array_RETURN
SIDL_fcomplex__python_borrow_array SIDL_fcomplex__python_borrow_array_PROTO
{ return SIDL_fcomplex__python_clone_array(array); }

SIDL_float__python_borrow_array_RETURN
SIDL_float__python_borrow_array SIDL_float__python_borrow_array_PROTO
{ return SIDL_float__python_clone_array(array); }

SIDL_int__python_borrow_array_RETURN
SIDL_int__python_borrow_array SIDL_int__python_borrow_array_PROTO
{ return SIDL_int__python_clone_array(array); }

SIDL_long__python_borrow_array_RETURN
SIDL_long__python_borrow_array SIDL_long__python_borrow_array_PROTO
{ return SIDL_long__python_clone_array(array); }

SIDL_opaque__python_borrow_array_RETURN
SIDL_opaque__python_borrow_array SIDL_opaque__python_borrow_array_PROTO
{ return SIDL_opaque__python_clone_array(array); }

SIDL_string__python_borrow_array_RETURN
SIDL_string__python_borrow_array SIDL_string__python_borrow_array_PROTO
{ return SIDL_string__python_clone_array(array); }

#ifdef __cplusplus
extern "C" void initSIDLPyArrays(void);
#else
extern void initSIDLPyArrays(void);
#endif

static struct PyMethodDef spa_methods[] = {
  /* this module exports no methods */
  { NULL, NULL }
};

void
initSIDLPyArrays(void) 
{
  PyObject *module, *dict, *c_api;
  static void *spa_api[SIDLPyArrays_API_pointers];
  module = Py_InitModule("SIDLPyArrays", spa_methods);
  import_array();
  dict = PyModule_GetDict(module);
  spa_api[SIDL_bool__borrow_python_array_NUM] =
    (void *)SIDL_bool__borrow_python_array;
  spa_api[SIDL_bool__clone_python_array_column_NUM] =
    (void *)SIDL_bool__clone_python_array_column;
  spa_api[SIDL_bool__clone_python_array_row_NUM] =
    (void *)SIDL_bool__clone_python_array_row;
  spa_api[SIDL_bool__python_borrow_array_NUM] =
    (void *)SIDL_bool__python_borrow_array;
  spa_api[SIDL_bool__python_clone_array_NUM] =
    (void *)SIDL_bool__python_clone_array;
  spa_api[SIDL_bool__python_deleteRef_array_NUM] =
    (void *)SIDL_bool__array_deleteRef;
  spa_api[SIDL_char__borrow_python_array_NUM] =
    (void *)SIDL_char__borrow_python_array;
  spa_api[SIDL_char__clone_python_array_column_NUM] =
    (void *)SIDL_char__clone_python_array_column;
  spa_api[SIDL_char__clone_python_array_row_NUM] =
    (void *)SIDL_char__clone_python_array_row;
  spa_api[SIDL_char__python_borrow_array_NUM] =
    (void *)SIDL_char__python_borrow_array;
  spa_api[SIDL_char__python_clone_array_NUM] =
    (void *)SIDL_char__python_clone_array;
  spa_api[SIDL_char__python_deleteRef_array_NUM] =
    (void *)SIDL_char__array_deleteRef;
  spa_api[SIDL_dcomplex__borrow_python_array_NUM] =
    (void *)SIDL_dcomplex__borrow_python_array;
  spa_api[SIDL_dcomplex__clone_python_array_column_NUM] =
    (void *)SIDL_dcomplex__clone_python_array_column;
  spa_api[SIDL_dcomplex__clone_python_array_row_NUM] =
    (void *)SIDL_dcomplex__clone_python_array_row;
  spa_api[SIDL_dcomplex__python_borrow_array_NUM] =
    (void *)SIDL_dcomplex__python_borrow_array;
  spa_api[SIDL_dcomplex__python_clone_array_NUM] =
    (void *)SIDL_dcomplex__python_clone_array;
  spa_api[SIDL_dcomplex__python_deleteRef_array_NUM] =
    (void *)SIDL_dcomplex__array_deleteRef;
  spa_api[SIDL_double__borrow_python_array_NUM] =
    (void *)SIDL_double__borrow_python_array;
  spa_api[SIDL_double__clone_python_array_column_NUM] =
    (void *)SIDL_double__clone_python_array_column;
  spa_api[SIDL_double__clone_python_array_row_NUM] =
    (void *)SIDL_double__clone_python_array_row;
  spa_api[SIDL_double__python_borrow_array_NUM] =
    (void *)SIDL_double__python_borrow_array;
  spa_api[SIDL_double__python_clone_array_NUM] =
    (void *)SIDL_double__python_clone_array;
  spa_api[SIDL_double__python_deleteRef_array_NUM] =
    (void *)SIDL_double__array_deleteRef;
  spa_api[SIDL_fcomplex__borrow_python_array_NUM] =
    (void *)SIDL_fcomplex__borrow_python_array;
  spa_api[SIDL_fcomplex__clone_python_array_column_NUM] =
    (void *)SIDL_fcomplex__clone_python_array_column;
  spa_api[SIDL_fcomplex__clone_python_array_row_NUM] =
    (void *)SIDL_fcomplex__clone_python_array_row;
  spa_api[SIDL_fcomplex__python_borrow_array_NUM] =
    (void *)SIDL_fcomplex__python_borrow_array;
  spa_api[SIDL_fcomplex__python_clone_array_NUM] =
    (void *)SIDL_fcomplex__python_clone_array;
  spa_api[SIDL_fcomplex__python_deleteRef_array_NUM] =
    (void *)SIDL_fcomplex__array_deleteRef;
  spa_api[SIDL_float__borrow_python_array_NUM] =
    (void *)SIDL_float__borrow_python_array;
  spa_api[SIDL_float__clone_python_array_column_NUM] =
    (void *)SIDL_float__clone_python_array_column;
  spa_api[SIDL_float__clone_python_array_row_NUM] =
    (void *)SIDL_float__clone_python_array_row;
  spa_api[SIDL_float__python_borrow_array_NUM] =
    (void *)SIDL_float__python_borrow_array;
  spa_api[SIDL_float__python_clone_array_NUM] =
    (void *)SIDL_float__python_clone_array;
  spa_api[SIDL_float__python_deleteRef_array_NUM] =
    (void *)SIDL_float__array_deleteRef;
  spa_api[SIDL_int__borrow_python_array_NUM] =
    (void *)SIDL_int__borrow_python_array;
  spa_api[SIDL_int__clone_python_array_column_NUM] =
    (void *)SIDL_int__clone_python_array_column;
  spa_api[SIDL_int__clone_python_array_row_NUM] =
    (void *)SIDL_int__clone_python_array_row;
  spa_api[SIDL_int__python_borrow_array_NUM] =
    (void *)SIDL_int__python_borrow_array;
  spa_api[SIDL_int__python_clone_array_NUM] =
    (void *)SIDL_int__python_clone_array;
  spa_api[SIDL_int__python_deleteRef_array_NUM] =
    (void *)SIDL_int__array_deleteRef;
  spa_api[SIDL_long__borrow_python_array_NUM] =
    (void *)SIDL_long__borrow_python_array;
  spa_api[SIDL_long__clone_python_array_column_NUM] =
    (void *)SIDL_long__clone_python_array_column;
  spa_api[SIDL_long__clone_python_array_row_NUM] =
    (void *)SIDL_long__clone_python_array_row;
  spa_api[SIDL_long__python_borrow_array_NUM] =
    (void *)SIDL_long__python_borrow_array;
  spa_api[SIDL_long__python_clone_array_NUM] =
    (void *)SIDL_long__python_clone_array;
  spa_api[SIDL_long__python_deleteRef_array_NUM] =
    (void *)SIDL_long__array_deleteRef;
  spa_api[SIDL_opaque__borrow_python_array_NUM] =
    (void *)SIDL_opaque__borrow_python_array;
  spa_api[SIDL_opaque__clone_python_array_column_NUM] =
    (void *)SIDL_opaque__clone_python_array_column;
  spa_api[SIDL_opaque__clone_python_array_row_NUM] =
    (void *)SIDL_opaque__clone_python_array_row;
  spa_api[SIDL_opaque__python_borrow_array_NUM] =
    (void *)SIDL_opaque__python_borrow_array;
  spa_api[SIDL_opaque__python_clone_array_NUM] =
    (void *)SIDL_opaque__python_clone_array;
  spa_api[SIDL_opaque__python_deleteRef_array_NUM] =
    (void *)SIDL_opaque__array_deleteRef;
  spa_api[SIDL_string__borrow_python_array_NUM] =
    (void *)SIDL_string__borrow_python_array;
  spa_api[SIDL_string__clone_python_array_column_NUM] =
    (void *)SIDL_string__clone_python_array_column;
  spa_api[SIDL_string__clone_python_array_row_NUM] =
    (void *)SIDL_string__clone_python_array_row;
  spa_api[SIDL_string__python_borrow_array_NUM] =
    (void *)SIDL_string__python_borrow_array;
  spa_api[SIDL_string__python_clone_array_NUM] =
    (void *)SIDL_string__python_clone_array;
  spa_api[SIDL_string__python_deleteRef_array_NUM] =
    (void *)SIDL_string__array_deleteRef;
  spa_api[SIDL_array__convert_python_NUM] =
    (void *)SIDL_array__convert_python;
  spa_api[SIDL_array__convert_sidl_NUM] =
    (void *)SIDL_array__convert_sidl;
  spa_api[SIDL_array__extract_python_info_NUM] =
    (void *)SIDL_array__extract_python_info;
  c_api = PyCObject_FromVoidPtr((void *)spa_api, NULL);
  if (c_api) {
    PyDict_SetItemString(dict, "_C_API", c_api);
    Py_DECREF(c_api);
  }
  if (PyErr_Occurred()) {
    Py_FatalError("Can't initialize module SIDLPyArrays.");
  }
}
