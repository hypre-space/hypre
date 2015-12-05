/*
 * File:        sidlPyArrays.c
 * Copyright:   (c) 2001 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.6 $
 * Date:        $Date: 2007/09/27 19:35:21 $
 * Description: Runtime support routines to convert sidl arrays to/from Python
 *
 * This file provides functions to convert sidl arrays to Python with or
 * without borrowing data and functions to convert Python arrays to sidl
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

#define sidlPyArrays_MODULE 1
#include "sidlPyArrays.h"

#include "sidlObjA.h"
#include "sidlArray.h"
#include "sidl_bool_IOR.h"
#include "sidl_char_IOR.h"
#include "sidl_dcomplex_IOR.h"
#include "sidl_double_IOR.h"
#include "sidl_fcomplex_IOR.h"
#include "sidl_float_IOR.h"
#include "sidl_int_IOR.h"
#include "sidl_long_IOR.h"
#include "sidl_opaque_IOR.h"
#include "sidl_string_IOR.h"
#include "sidl_interface_IOR.h"
#include "sidl_BaseInterface_IOR.h"

/* include Numerical Python header file */
#include "Numeric/arrayobject.h"
#include <stdlib.h>
#include <string.h>

/* Python specializations of the basic array types */

struct sidl__array * 
sidl_python_smartCp(struct sidl__array *array) {
  sidl__array_addRef(array); 
  return array; 
}

#define specialized_array(sidlType, cType)                              \
struct sidl_python_## sidlType ##_array {                               \
  struct sidl_## sidlType ##__array  d_array;                           \
  PyObject                  *d_numarray;                                \
};                                                                      \
                                                                        \
void sidl_python_## sidlType ##_destroy(struct sidl__array *array)      \
{                                                                       \
  if (array) {                                                          \
    struct sidl_python_## sidlType ##_array *parray =                   \
      (struct sidl_python_## sidlType ##_array *)array;                 \
    Py_XDECREF(parray->d_numarray);                                     \
    sidl__array_remove(array);                                          \
    free((void *)array);                                                \
  }                                                                     \
}                                                                       \
                                                                        \
static int32_t                                                          \
sidl_python_## sidlType ##_type(void) {                                 \
  return sidl_## sidlType ##_array;                                     \
}                                                                       \
                                                                        \
static const struct sidl__array_vtable                                  \
s_## sidlType ##_vtable = {                                             \
  sidl_python_## sidlType ##_destroy,                                   \
  sidl_python_smartCp,                                                  \
  sidl_python_## sidlType ##_type                                       \
};                                                                      \
                                                                        \
static struct sidl_## sidlType ##__array *                              \
sidl_python_## sidlType ##_create(const int32_t dimen,                  \
                          const int32_t lower[],                        \
                          const int32_t upper[],                        \
                          const int32_t stride[],                       \
                          cType         *first,                         \
                          PyObject      *pyobj) {                       \
  static const size_t arraySize =                                       \
    sizeof(struct sidl_python_## sidlType ##_array) +                   \
    (sizeof(int32_t) -                                                  \
     (sizeof(struct sidl_python_## sidlType ##_array)                   \
      % sizeof(int32_t))) % sizeof(int32_t);                            \
  struct sidl_python_## sidlType ##_array *result =                     \
    (struct sidl_python_## sidlType ##_array *)malloc                   \
    (arraySize + 3 * sizeof(int32_t) * dimen);                          \
  if (result) {                                                         \
    result->d_array.d_metadata.d_vtable = &s_## sidlType ##_vtable;     \
    result->d_array.d_metadata.d_dimen = dimen;                         \
    result->d_array.d_metadata.d_refcount = 1;                          \
    result->d_array.d_metadata.d_lower = (int32_t *)                    \
      ((char *)result + arraySize);                                     \
    result->d_array.d_metadata.d_upper =                                \
      result->d_array.d_metadata.d_lower + dimen;                       \
    result->d_array.d_metadata.d_stride =                               \
      result->d_array.d_metadata.d_upper + dimen;                       \
    memcpy(result->d_array.d_metadata.d_lower,                          \
           lower,sizeof(int32_t)*dimen);                                \
    memcpy(result->d_array.d_metadata.d_upper,                          \
           upper,sizeof(int32_t)*dimen);                                \
    memcpy(result->d_array.d_metadata.d_stride,                         \
           stride,sizeof(int32_t)*dimen);                               \
    result->d_array.d_firstElement = first;                             \
    sidl__array_add((struct sidl__array*)&(result->d_array));           \
    Py_XINCREF(pyobj);                                                  \
    result->d_numarray = pyobj;                                         \
  }                                                                     \
  return (struct sidl_## sidlType ##__array *)result;                   \
}

specialized_array(bool,sidl_bool)
specialized_array(char,char)
specialized_array(dcomplex,struct sidl_dcomplex)
specialized_array(double,double)
specialized_array(fcomplex,struct sidl_fcomplex)
specialized_array(float,float)
specialized_array(int,int32_t)
specialized_array(long,int64_t)

typedef struct {
  PyObject_HEAD
  struct sidl__array *d_array;
} SIDLArrayObject;

static void
sao_dealloc(SIDLArrayObject* self)
{
  if (self->d_array) sidl__array_deleteRef(self->d_array);
  self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
sao_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  SIDLArrayObject* self;
  self = (SIDLArrayObject *)type->tp_alloc(type, 0);
  if (self)  self->d_array = NULL;
  return (PyObject *)self;
}

static PyTypeObject sidlPyArrayType = {
  PyObject_HEAD_INIT(NULL)
  0,                            /* ob_size*/
  "sidlPyArrays.SIDLArrayWrapper", /* tp_name */
  sizeof(SIDLArrayObject),      /* tp_basicsize */
  0,                            /* tp_itemsize */
  (destructor)sao_dealloc,      /* tp_dealloc*/
  0,                            /* tp_print*/
  0,                            /* tp_getattr*/
  0,                            /* tp_setattr*/
  0,                            /* tp_compare*/
  0,                            /* tp_repr*/
  0,                            /* tp_as_number*/
  0,                            /* tp_as_sequence*/
  0,                            /* tp_as_mapping*/
  0,                            /* tp_hash */
  0,                            /* tp_call*/
  0,                            /* tp_str*/
  0,                            /* tp_getattro*/
  0,                            /* tp_setattro*/
  0,                            /* tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,           /* tp_flags*/
  "Python type to wrap up SIDL arrays. Not useful to end users.",              /* tp_doc */
  0,                            /* tp_traverse */
  0,                            /* tp_clear */
  0,                            /* tp_richcompare */
  0,                            /* tp_weaklistoffset */
  0,                            /* tp_iter */
  0,                            /* tp_iternext */
  0,                            /* tp_methods */
  0,                            /* tp_members */
  0,                            /* tp_getset */
  0,                            /* tp_base */
  0,                            /* tp_dict */
  0,                            /* tp_descr_get */
  0,                            /* tp_descr_set */
  0,                            /* tp_dictoffset */
  0,                            /* tp_init */
  0,                            /* tp_alloc */
  sao_new,                      /* tp_new */
};

#define sao_Check(op) ((op)->ob_type == &sidlPyArrayType)

/**
 * Convert the array description information from the Numerical Python array
 * into a form that sidl likes.  sidl needs a vector of lower and upper
 * index bounds, and a vector holding the stride.
 */
sidl_array__extract_python_info_RETURN
sidl_array__extract_python_info sidl_array__extract_python_info_PROTO
{
  if (PyArray_Check(pya)) {
    const PyArrayObject *array = (PyArrayObject *)pya;
    if (array->nd >= 0) {
      /* a zero dimensional Python array is like a scalar */
      const int32_t dimen = array->nd ? array->nd : 1;
      int32_t i;
      if (dimen > SIDL_MAX_ARRAY_DIMENSION) return 0;
      *dimension = dimen;
      lower[0] = upper[0] = 0;
      stride[0] = 1;
      for(i = 0; i < array->nd; ++i) {
        lower[i] = 0;
        upper[i] = array->dimensions[i] - 1;
        stride[i] = array->strides[i]/array->descr->elsize;
        if ((array->strides[i] % array->descr->elsize) != 0) {
          return 0;
        }
      }
      return 1;
    }
  }
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
static void copy_ ## name ## _data(char * /* restrict */ dest, \
                                   const int dstride[], /* not int32_t */ \
                                   const char * /* restrict */ src, \
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
      int32_t * /* restrict */ ind = malloc(sizeof(int32_t)*dim), i; \
      memset(ind, 0, sizeof(int32_t)*dim); \
      num /= sizeof(type); \
      while (num--) { \
        *((type * /* restrict */)dest) = *((const type * /* restrict */)src); \
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
copyData_impl(fcomplex, struct sidl_fcomplex)
copyData_impl(dcomplex, struct sidl_dcomplex)

static 
struct sidl__array *
alreadySIDLArray(PyArrayObject *pya,
                 const size_t   dataSize,
                 const int      sidlArrayType,
                 const int      orderEnum)
{
  struct sidl__array *result = NULL;
  if (pya->base && sao_Check(pya->base) &&
      (sidl__array_type(((SIDLArrayObject *)(pya->base))->d_array) == sidlArrayType)) {
    result = ((SIDLArrayObject *)(pya->base))->d_array;
    {
      const int32_t dimen = sidlArrayDim(result);
      int i;
      if ((dimen != pya->nd) ||
          ((sidl_column_major_order == orderEnum) &&
           !sidl__array_isColumnOrder(result)) ||
          ((sidl_row_major_order == orderEnum) &&
           !sidl__array_isRowOrder(result)))
        return NULL;
      /* check strides and extents */
      for(i = 0; i < dimen; ++i) {
        if (!((sidlStride(result, i)*dataSize == pya->strides[i]) &&
              (sidlLength(result, i) == pya->dimensions[i]))) return NULL;
      }
      return sidl__array_smartCopy(result);
    }
  }
  return result;
}

/*
 * Parameterized function implementation to borrow a particular type
 * array.
 */
#define sidl_borrow_impl(stype,dtype,pytype) \
{\
  if (PyArray_Check(obj)) {\
    PyArrayObject *pya = (PyArrayObject *)obj;\
    if ((*array = (struct sidl_ ## stype ## __array *) \
         alreadySIDLArray(pya, sizeof(dtype), sidl_ ## stype ## _array, sidl_general_order)))\
      return 1;\
    if (pytype == pya->descr->type_num) { \
      int32_t dimension,\
         lower[SIDL_MAX_ARRAY_DIMENSION],\
         upper[SIDL_MAX_ARRAY_DIMENSION],\
         stride[SIDL_MAX_ARRAY_DIMENSION];\
      if (sidl_array__extract_python_info((PyObject *)pya, &dimension, \
                                          lower, upper, stride)) {\
        *array = sidl_python_ ## stype ## _create\
           (dimension, lower, upper, stride, (dtype *)pya->data, \
            (PyObject *)pya);\
        return (*array != NULL);\
      }\
    }\
  }\
  return sidl_ ## stype ## __clone_python_array(obj, array);\
}


sidl_bool__borrow_python_array_RETURN
sidl_bool__borrow_python_array sidl_bool__borrow_python_array_PROTO
sidl_borrow_impl(bool, sidl_bool, PyArray_INT)

sidl_char__borrow_python_array_RETURN
sidl_char__borrow_python_array sidl_char__borrow_python_array_PROTO
{
  if (PyArray_Check(obj)) {
    PyArrayObject *pya = (PyArrayObject *)obj;
    if ((*array = (struct sidl_char__array *)
         alreadySIDLArray(pya, sizeof(char), 
                          sidl_char_array,
                          sidl_general_order))) return 1;
    if ((PyArray_CHAR == pya->descr->type_num) ||
        (PyArray_UBYTE == pya->descr->type_num) ||
        (PyArray_SBYTE == pya->descr->type_num)) {
      int32_t dimension, 
        lower[SIDL_MAX_ARRAY_DIMENSION], 
        upper[SIDL_MAX_ARRAY_DIMENSION], 
        stride[SIDL_MAX_ARRAY_DIMENSION];
      if (sidl_array__extract_python_info((PyObject *)pya, &dimension, 
                                          lower, upper, stride)) {
        *array = sidl_python_char_create(dimension, lower, upper, stride,
                                         (char *)pya->data, (PyObject*)pya);
        return (*array != NULL);
      }
    }
  }
  return sidl_char__clone_python_array(obj, array);
}

sidl_dcomplex__borrow_python_array_RETURN
sidl_dcomplex__borrow_python_array sidl_dcomplex__borrow_python_array_PROTO
sidl_borrow_impl(dcomplex, struct sidl_dcomplex, PyArray_CDOUBLE)

sidl_double__borrow_python_array_RETURN
sidl_double__borrow_python_array sidl_double__borrow_python_array_PROTO
sidl_borrow_impl(double, double, PyArray_DOUBLE)

sidl_fcomplex__borrow_python_array_RETURN
sidl_fcomplex__borrow_python_array sidl_fcomplex__borrow_python_array_PROTO
sidl_borrow_impl(fcomplex, struct sidl_fcomplex, PyArray_CFLOAT)

sidl_float__borrow_python_array_RETURN
sidl_float__borrow_python_array sidl_float__borrow_python_array_PROTO
sidl_borrow_impl(float, float, PyArray_FLOAT)

sidl_int__borrow_python_array_RETURN
sidl_int__borrow_python_array sidl_int__borrow_python_array_PROTO
{
  if (PyArray_Check(obj)) {
    PyArrayObject *pya = (PyArrayObject *)obj;
    if ((*array = (struct sidl_int__array *)
         alreadySIDLArray(pya, sizeof(int32_t), 
                          sidl_int_array, sidl_general_order))) return 1;
    if (0
#if SIZEOF_SHORT == 4
        || (PyArray_SHORT == pya->descr->type_num)
#ifdef PyArray_UNSIGNED_TYPES
        || (PyArray_USHORT == pya->descr->type_num)
#endif
#endif
#if SIZEOF_INT == 4
        || (PyArray_INT == pya->descr->type_num)
#ifdef PyArray_UNSIGNED_TYPES
        || (PyArray_UINT == pya->descr->type_num)
#endif
#endif
#if SIZEOF_LONG == 4
        || (PyArray_LONG == pya->descr->type_num)
#endif
        ) {
      int32_t dimension, lower[SIDL_MAX_ARRAY_DIMENSION],
        upper[SIDL_MAX_ARRAY_DIMENSION],
        stride[SIDL_MAX_ARRAY_DIMENSION];
      if (sidl_array__extract_python_info((PyObject *)pya, &dimension, 
                                          lower, upper, stride)) {
        *array = sidl_python_int_create(dimension, lower, upper, stride,
                                        (int32_t *)pya->data, (PyObject*)pya);
        return (*array != NULL);
      }
    }
  }
  return sidl_int__clone_python_array(obj, array);
}

sidl_long__borrow_python_array_RETURN
sidl_long__borrow_python_array sidl_long__borrow_python_array_PROTO
{
  if (PyArray_Check(obj)) {
    PyArrayObject *pya = (PyArrayObject *)obj;
    if ((*array = (struct sidl_long__array *)
         alreadySIDLArray(pya, sizeof(int64_t), 
                          sidl_long_array, sidl_general_order))) return 1;
    if (0
#if SIZEOF_SHORT == 8
        || (PyArray_SHORT == pya->descr->type_num)
#ifdef PyArray_UNSIGNED_TYPES
        || (PyArray_USHORT == pya->descr->type_num)
#endif
#endif
#if SIZEOF_INT == 8
        || (PyArray_INT == pya->descr->type_num)
#ifdef PyArray_UNSIGNED_TYPES
        || (PyArray_UINT == pya->descr->type_num)
#endif
#endif
#if SIZEOF_LONG == 8
        || (PyArray_LONG == pya->descr->type_num)
#endif
        ) {
      int32_t dimension, lower[SIDL_MAX_ARRAY_DIMENSION],
        upper[SIDL_MAX_ARRAY_DIMENSION],
        stride[SIDL_MAX_ARRAY_DIMENSION];
      if (sidl_array__extract_python_info((PyObject *)pya, &dimension, 
                                          lower, upper, stride)) {
        *array = sidl_python_long_create(dimension, lower, upper, stride,
                                         (int64_t *)pya->data, (PyObject*)pya);
        return (*array != NULL);
      }
    }
  }
  return sidl_long__clone_python_array(obj, array);
}

sidl_string__borrow_python_array_RETURN
sidl_string__borrow_python_array sidl_string__borrow_python_array_PROTO
{
  return sidl_string__clone_python_array(obj,array);
}

sidl_opaque__borrow_python_array_RETURN
sidl_opaque__borrow_python_array sidl_opaque__borrow_python_array_PROTO
{
  return sidl_opaque__clone_python_array(obj,array);
}

/*
 * Parameterized implementation of a clone function.
 */
#define ClonePython_impl(sidlname, sidltype, pyarraytype, order, orderenum) \
{ \
  int result = 0; \
  *array = 0; \
  if (obj == Py_None) { \
    result = 1; \
  } \
  else { \
    PyArrayObject *pya = \
      (PyArrayObject *)PyArray_FromObject(obj, pyarraytype, 0, 0); \
    if (pya) { \
      int32_t dimen,\
          lower[SIDL_MAX_ARRAY_DIMENSION],\
          upper[SIDL_MAX_ARRAY_DIMENSION],\
          stride[SIDL_MAX_ARRAY_DIMENSION]; \
      if ((*array = (struct sidl_ ## sidlname ## __array *) \
           alreadySIDLArray(pya, sizeof(sidltype), \
           sidl_ ## sidlname ## _array, orderenum))) { \
        Py_DECREF(pya); \
        return 1;\
      }\
      if (sidl_array__extract_python_info((PyObject *)pya, &dimen, \
                                          lower, upper, stride)) { \
        *array = \
           sidl_ ## sidlname ## __array_create ## order(dimen, lower, upper); \
        if (*array) { \
          int *bytestride = malloc(sizeof(int)*dimen), i; \
          int32_t *numelem = malloc(sizeof(int32_t)*dimen); \
          for(i = 0; i < dimen; ++i){ \
            numelem[i] = 1 + upper[i] - lower[i]; \
            bytestride[i] = sizeof(sidltype)* \
               sidlStride(*array, i); \
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
      } \
      Py_DECREF(pya); \
    } \
  } \
  return result; \
}

sidl_bool__clone_python_array_column_RETURN
sidl_bool__clone_python_array_column sidl_bool__clone_python_array_column_PROTO
ClonePython_impl(bool, sidl_bool, PyArray_INT, Col, sidl_column_major_order)

sidl_char__clone_python_array_column_RETURN
sidl_char__clone_python_array_column sidl_char__clone_python_array_column_PROTO
ClonePython_impl(char, char, PyArray_CHAR, Col, sidl_column_major_order)

sidl_dcomplex__clone_python_array_column_RETURN
sidl_dcomplex__clone_python_array_column sidl_dcomplex__clone_python_array_column_PROTO
ClonePython_impl(dcomplex, struct sidl_dcomplex, PyArray_CDOUBLE, Col, sidl_column_major_order)

sidl_double__clone_python_array_column_RETURN
sidl_double__clone_python_array_column sidl_double__clone_python_array_column_PROTO
ClonePython_impl(double, double, PyArray_DOUBLE, Col, sidl_column_major_order)

sidl_fcomplex__clone_python_array_column_RETURN
sidl_fcomplex__clone_python_array_column sidl_fcomplex__clone_python_array_column_PROTO
ClonePython_impl(fcomplex, struct sidl_fcomplex, PyArray_CFLOAT, Col, sidl_column_major_order)

sidl_float__clone_python_array_column_RETURN
sidl_float__clone_python_array_column sidl_float__clone_python_array_column_PROTO
ClonePython_impl(float, float, PyArray_FLOAT, Col, sidl_column_major_order)

#if SIZEOF_SHORT == 4
sidl_int__clone_python_array_column_RETURN
sidl_int__clone_python_array_column sidl_int__clone_python_array_column_PROTO
ClonePython_impl(int, int32_t, PyArray_SHORT, Col, sidl_column_major_order)
#else
#if SIZEOF_INT == 4
sidl_int__clone_python_array_column_RETURN
sidl_int__clone_python_array_column sidl_int__clone_python_array_column_PROTO
ClonePython_impl(int, int32_t, PyArray_INT, Col, sidl_column_major_order)
#else
#if SIZEOF_LONG == 4
sidl_int__clone_python_array_column_RETURN
sidl_int__clone_python_array_column sidl_int__clone_python_array_column_PROTO
ClonePython_impl(int, int32_t, PyArray_LONG, Col, sidl_column_major_order)
#else
#error No 32-bit integer available.
#endif
#endif
#endif

#if SIZEOF_SHORT == 8
sidl_long__clone_python_array_column_RETURN
sidl_long__clone_python_array_column sidl_long__clone_python_array_column_PROTO
ClonePython_impl(long, int64_t, PyArray_SHORT, Col, sidl_column_major_order)
#else
#if SIZEOF_INT == 8
sidl_long__clone_python_array_column_RETURN
sidl_long__clone_python_array_column sidl_long__clone_python_array_column_PROTO
ClonePython_impl(long, int64_t, PyArray_INT, Col, sidl_column_major_order)
#else
#if SIZEOF_LONG == 8
sidl_long__clone_python_array_column_RETURN
sidl_long__clone_python_array_column sidl_long__clone_python_array_column_PROTO
ClonePython_impl(long, int64_t, PyArray_LONG, Col, sidl_column_major_order)
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
  *dest = (int64_t)convert(*((const srctype * /* restrict */)src)); \
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
#if defined(HAVE_LONG_LONG)
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
                  struct sidl_long__array *array,
                  const int32_t lower[],
                  const int32_t upper[],
                  const int32_t stride[])
{
  const int size = PyArray_Size((PyObject *)pya);
  const int32_t dimen = pya->nd;
  const int32_t sdim = sidl_long__array_dimen(array);
  int i, j;
  int64_t * /* restrict */ dest = sidl_long__array_first(array);
  const char * /* restrict */ src = pya->data;
  const int * /* restrict */ srcstride = pya->strides;
  int32_t * /* restrict */ ind =  malloc(sizeof(int32_t)*sdim);
  int32_t * /* restrict */ numelem = malloc(sizeof(int32_t)*sdim);
  if (ind && numelem) {
    memset(ind, 0, sizeof(int32_t)*sdim);
    for(i = 0; i < sdim; ++i ){
      numelem[i] = sidlLength(array,i);
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

sidl_long__clone_python_array_column_RETURN
sidl_long__clone_python_array_column sidl_long__clone_python_array_column_PROTO
{
  int result = 0;
  *array = NULL;
  if (obj == Py_None) {
    result = 1;
  }
  else {
    PyArrayObject *pya = toNumericArray(obj);
    if (pya) {
      int32_t dimen;
      int32_t lower[SIDL_MAX_ARRAY_DIMENSION],
        upper[SIDL_MAX_ARRAY_DIMENSION],
        stride[SIDL_MAX_ARRAY_DIMENSION];
      if (sidl_array__extract_python_info((PyObject *)pya, &dimen,
                                          lower, upper, stride)) {
        *array =
          sidl_long__array_createCol(dimen, lower, upper);
        clone_long_python(pya, *array, lower, upper, stride);
        result = 1;
      }
      Py_DECREF((PyObject *)pya);
    }
  }
  return result;
}
#endif
#endif
#endif


sidl_bool__clone_python_array_row_RETURN
sidl_bool__clone_python_array_row sidl_bool__clone_python_array_row_PROTO
ClonePython_impl(bool, sidl_bool, PyArray_INT, Row, sidl_row_major_order)

sidl_char__clone_python_array_row_RETURN
sidl_char__clone_python_array_row sidl_char__clone_python_array_row_PROTO
ClonePython_impl(char, char, PyArray_CHAR, Row, sidl_row_major_order)

sidl_dcomplex__clone_python_array_row_RETURN
sidl_dcomplex__clone_python_array_row sidl_dcomplex__clone_python_array_row_PROTO
ClonePython_impl(dcomplex, struct sidl_dcomplex, PyArray_CDOUBLE, Row, sidl_row_major_order)

sidl_double__clone_python_array_row_RETURN
sidl_double__clone_python_array_row sidl_double__clone_python_array_row_PROTO
ClonePython_impl(double, double, PyArray_DOUBLE, Row, sidl_row_major_order)

sidl_fcomplex__clone_python_array_row_RETURN
sidl_fcomplex__clone_python_array_row sidl_fcomplex__clone_python_array_row_PROTO
ClonePython_impl(fcomplex, struct sidl_fcomplex, PyArray_CFLOAT, Row, sidl_row_major_order)

sidl_float__clone_python_array_row_RETURN
sidl_float__clone_python_array_row sidl_float__clone_python_array_row_PROTO
ClonePython_impl(float, float, PyArray_FLOAT, Row, sidl_row_major_order)

sidl_int__clone_python_array_row_RETURN
sidl_int__clone_python_array_row sidl_int__clone_python_array_row_PROTO
ClonePython_impl(int, int, PyArray_INT, Row, sidl_row_major_order)

#if SIZEOF_SHORT == 8
sidl_long__clone_python_array_row_RETURN
sidl_long__clone_python_array_row sidl_long__clone_python_array_row_PROTO
ClonePython_impl(long, int64_t, PyArray_SHORT, Row, sidl_row_major_order)
#else
#if SIZEOF_INT == 8
sidl_long__clone_python_array_row_RETURN
sidl_long__clone_python_array_row sidl_long__clone_python_array_row_PROTO
ClonePython_impl(long, int64_t, PyArray_INT, Row, sidl_row_major_order)
#else
#if SIZEOF_LONG == 8
sidl_long__clone_python_array_row_RETURN
sidl_long__clone_python_array_row sidl_long__clone_python_array_row_PROTO
ClonePython_impl(long, int64_t, PyArray_LONG, Row, sidl_row_major_order)
#else
sidl_long__clone_python_array_row_RETURN
sidl_long__clone_python_array_row sidl_long__clone_python_array_row_PROTO
{
  int result = 0;
  *array = NULL;
  if (obj == Py_None) {
    result = 1;
  }
  else {
    PyArrayObject *pya = toNumericArray(obj);
    if (pya) {
      int32_t dimen, lower[SIDL_MAX_ARRAY_DIMENSION],
        upper[SIDL_MAX_ARRAY_DIMENSION],
        stride[SIDL_MAX_ARRAY_DIMENSION];
      if (sidl_array__extract_python_info((PyObject *)pya, &dimen,
                                          lower, upper, stride)) {
        *array =
          sidl_long__array_createRow(dimen, lower, upper);
        clone_long_python(pya, *array, lower, upper, stride);
        result = 1;
      }
      Py_DECREF((PyObject *)pya);
    }
  }
  return result;
}
#endif
#endif
#endif

sidl_bool__clone_python_array_RETURN
sidl_bool__clone_python_array sidl_bool__clone_python_array_PROTO
ClonePython_impl(bool, sidl_bool, PyArray_INT, Row, sidl_general_order)

sidl_char__clone_python_array_RETURN
sidl_char__clone_python_array sidl_char__clone_python_array_PROTO
ClonePython_impl(char, char, PyArray_CHAR, Row, sidl_general_order)

sidl_dcomplex__clone_python_array_RETURN
sidl_dcomplex__clone_python_array sidl_dcomplex__clone_python_array_PROTO
ClonePython_impl(dcomplex, struct sidl_dcomplex, PyArray_CDOUBLE, Row, sidl_general_order)

sidl_double__clone_python_array_RETURN
sidl_double__clone_python_array sidl_double__clone_python_array_PROTO
ClonePython_impl(double, double, PyArray_DOUBLE, Row, sidl_general_order)

sidl_fcomplex__clone_python_array_RETURN
sidl_fcomplex__clone_python_array sidl_fcomplex__clone_python_array_PROTO
ClonePython_impl(fcomplex, struct sidl_fcomplex, PyArray_CFLOAT, Row, sidl_general_order)

sidl_float__clone_python_array_RETURN
sidl_float__clone_python_array sidl_float__clone_python_array_PROTO
ClonePython_impl(float, float, PyArray_FLOAT, Row, sidl_general_order)

sidl_int__clone_python_array_RETURN
sidl_int__clone_python_array sidl_int__clone_python_array_PROTO
ClonePython_impl(int, int, PyArray_INT, Row, sidl_general_order)

#if SIZEOF_SHORT == 8
sidl_long__clone_python_array_RETURN
sidl_long__clone_python_array sidl_long__clone_python_array_PROTO
ClonePython_impl(long, int64_t, PyArray_SHORT, Row, sidl_general_order)
#else
#if SIZEOF_INT == 8
sidl_long__clone_python_array_RETURN
sidl_long__clone_python_array sidl_long__clone_python_array_PROTO
ClonePython_impl(long, int64_t, PyArray_INT, Row, sidl_general_order)
#else
#if SIZEOF_LONG == 8
sidl_long__clone_python_array_RETURN
sidl_long__clone_python_array sidl_long__clone_python_array_PROTO
ClonePython_impl(long, int64_t, PyArray_LONG, Row, sidl_general_order)
#else
sidl_long__clone_python_array_RETURN
sidl_long__clone_python_array sidl_long__clone_python_array_PROTO
{
  int result = 0;
  *array = NULL;
  if (obj == Py_None) {
    result = 1;
  }
  else {
    PyArrayObject *pya = toNumericArray(obj);
    if (pya) {
      int32_t dimen, lower[SIDL_MAX_ARRAY_DIMENSION], 
        upper[SIDL_MAX_ARRAY_DIMENSION],
        stride[SIDL_MAX_ARRAY_DIMENSION];
      if (sidl_array__extract_python_info((PyObject *)pya, &dimen,
                                          lower, upper, stride)) {
        *array =
          sidl_long__array_createRow(dimen, lower, upper);
        clone_long_python(pya, *array, lower, upper, stride);
        result = 1;
      }
      Py_DECREF((PyObject *)pya);
    }
  }
  return result;
}
#endif
#endif
#endif


sidl_array__convert_python_RETURN
sidl_array__convert_python sidl_array__convert_python_PROTO
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
    sidl_opaque__array_set((struct sidl_opaque__array*)array,
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
    sidl_string__array_set((struct sidl_string__array*)array, ind,
                           PyString_AsString(str));
    Py_DECREF(str);
    return FALSE;
  }
  return TRUE;
}

sidl_opaque__clone_python_array_column_RETURN
sidl_opaque__clone_python_array_column sidl_opaque__clone_python_array_column_PROTO
{
  int result = 0;
  *array = NULL;
  if (obj == Py_None) {
    result = TRUE;
  }
  else {
    PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
    if (pya) {
      if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
        int dimen, lower[SIDL_MAX_ARRAY_DIMENSION], 
          upper[SIDL_MAX_ARRAY_DIMENSION],
          stride[SIDL_MAX_ARRAY_DIMENSION];
        if (sidl_array__extract_python_info(pya, &dimen, lower, upper,
                                            stride)) {
          *array = sidl_opaque__array_createCol(dimen, lower, upper);
          result = sidl_array__convert_python(pya, dimen, *array,
                                              CopyOpaquePointer);
          if (*array && !result) {
            sidl__array_deleteRef((struct sidl__array *)*array);
            *array = NULL;
          }
        }
      }
      Py_DECREF(pya);
    }
  }
  return result;
}

sidl_string__clone_python_array_column_RETURN
sidl_string__clone_python_array_column sidl_string__clone_python_array_column_PROTO
{
  int result = 0;
  *array = NULL;
  if (obj == Py_None) {
    result = TRUE;
  }
  else {
    PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
    if (pya) {
      if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
        int dimen, lower[SIDL_MAX_ARRAY_DIMENSION],
          upper[SIDL_MAX_ARRAY_DIMENSION],
          stride[SIDL_MAX_ARRAY_DIMENSION];
        if (sidl_array__extract_python_info(pya, &dimen, lower, upper,
                                            stride)) {
          *array = sidl_string__array_createCol(dimen, lower, upper);
          result = sidl_array__convert_python(pya, dimen, *array, 
                                              CopyStringPointer);
          if (*array && !result) {
            sidl__array_deleteRef((struct sidl__array *)*array);
            *array = NULL;
          }
        }
      }
      Py_DECREF(pya);
    }
  }
  return result;
}


sidl_opaque__clone_python_array_row_RETURN
sidl_opaque__clone_python_array_row sidl_opaque__clone_python_array_row_PROTO
{
  int result = 0;
  *array = NULL;
  if (obj == Py_None) {
    result = TRUE;
  }
  else {
    PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
    if (pya) {
      if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
        int dimen, lower[SIDL_MAX_ARRAY_DIMENSION], 
          upper[SIDL_MAX_ARRAY_DIMENSION],
          stride[SIDL_MAX_ARRAY_DIMENSION];
        if (sidl_array__extract_python_info(pya, &dimen, lower, upper,
                                            stride)) {
          *array = sidl_opaque__array_createRow(dimen, lower, upper);
          result = sidl_array__convert_python(pya, dimen, *array,
                                              CopyOpaquePointer);
          if (*array && !result) {
            sidl__array_deleteRef((struct sidl__array *)*array);
            *array = NULL;
          }
        }
      }
      Py_DECREF(pya);
    }
  }
  return result;
}

sidl_opaque__clone_python_array_RETURN
sidl_opaque__clone_python_array sidl_opaque__clone_python_array_PROTO
{
  return sidl_opaque__clone_python_array_row(obj,array);
}

sidl_string__clone_python_array_row_RETURN
sidl_string__clone_python_array_row sidl_string__clone_python_array_row_PROTO
{
  int result = 0;
  *array = NULL;
  if (obj == Py_None) {
    result = TRUE;
  }
  else {
    PyObject *pya = PyArray_FromObject(obj, PyArray_OBJECT, 0, 0);
    if (pya) {
      if (PyArray_OBJECT == ((PyArrayObject *)pya)->descr->type_num) {
        int dimen, lower[SIDL_MAX_ARRAY_DIMENSION], 
          upper[SIDL_MAX_ARRAY_DIMENSION], 
          stride[SIDL_MAX_ARRAY_DIMENSION];
        if (sidl_array__extract_python_info(pya, &dimen, lower, upper,
                                            stride)) {
          *array = sidl_string__array_createRow(dimen, lower, upper);
          result = sidl_array__convert_python(pya, dimen, *array, 
                                              CopyStringPointer);
          if (*array && !result) {
            sidl__array_deleteRef((struct sidl__array *)*array);
            *array = NULL;
          }
        }
      }
      Py_DECREF(pya);
    }
  }
  return result;
}

sidl_string__clone_python_array_RETURN
sidl_string__clone_python_array sidl_string__clone_python_array_PROTO
{
  return sidl_string__clone_python_array_row(obj, array);
}


#define CloneSIDL_impl(sidlname, sidltype, pyarraytype) \
static PyObject * \
clone_sidl_ ## sidlname ## _array(struct sidl_ ## sidlname ##__array *array) \
{ \
  PyArrayObject *pya = NULL; \
  if (array) { \
    const int dimen = sidlArrayDim(array); \
    int i; \
    int32_t *numelem = malloc(sizeof(int32_t)*dimen); \
    int *pynumelem = malloc(sizeof(int)*dimen); \
    int *bytestride = malloc(sizeof(int)*dimen); \
    for(i = 0; i < dimen; ++i){ \
      numelem[i] = sidlLength(array,i); \
      pynumelem[i] = (int)numelem[i]; \
      bytestride[i] = sizeof(sidltype)* \
        sidlStride(array, i); \
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
  else { \
    Py_INCREF(Py_None); \
    return Py_None; \
  } \
  return (PyObject *)pya; \
}

CloneSIDL_impl(bool, sidl_bool, PyArray_INT)
CloneSIDL_impl(char, char, PyArray_CHAR)
CloneSIDL_impl(double, double, PyArray_DOUBLE)
CloneSIDL_impl(dcomplex, struct sidl_dcomplex, PyArray_CDOUBLE)
CloneSIDL_impl(fcomplex, struct sidl_fcomplex, PyArray_CFLOAT)
CloneSIDL_impl(float, float, PyArray_FLOAT)
#if SIZEOF_SHORT == 4
CloneSIDL_impl(int, int32_t, PyArray_SHORT)
#else
#if SIZEOF_INT == 4
CloneSIDL_impl(int, int32_t, PyArray_INT)
#else
#if SIZEOF_LONG == 4
CloneSIDL_impl(int, int32_t, PyArray_LONG)
#else
#error No 32-bit integer type available.
#endif
#endif
#endif

#if SIZEOF_SHORT == 8
CloneSIDL_impl(long, int64_t, PyArray_SHORT)
#else
#if SIZEOF_INT == 8
CloneSIDL_impl(long, int64_t, PyArray_INT)
#else
#if SIZEOF_LONG == 8
CloneSIDL_impl(long, int64_t, PyArray_LONG)
#else
static int
getAndConvertLong(void *array,
                  const int32_t *ind,
                  PyObject **dest)
{
  int64_t val =
    sidl_long__array_get((struct sidl_long__array *)array, ind);
  *dest = PyLong_FromLongLong(val);
  return FALSE;
}

static PyObject *
clone_sidl_long_array(struct sidl_long__array *array)
{
  PyObject *pya = NULL;
  if (array) {
    const int dimen = sidl_long__array_dimen(array);
    int i;
    int32_t *lower = malloc(sizeof(int32_t) * dimen);
    int32_t *numelem = malloc(sizeof(int32_t) * dimen);
#if (SIZEOF_INT != 4)
    int *pynumelem = malloc(sizeof(int) * dimen);
#else
    int *pynumelem = (int *)numelem;
#endif
    for(i = 0 ; i < dimen; ++i ){
      lower[i] = sidl_long__array_lower(array, i);
      numelem[i] = sidlLength(array, i);
#if (SIZEOF_INT != 4)
      pynumelem[i] = numelem[i];
#endif
    }
    pya = PyArray_FromDims(dimen, pynumelem, PyArray_OBJECT);
    if (pya) {
      if (!sidl_array__convert_sidl(pya, dimen, lower, 
                                    array->d_metadata.d_upper,
                                    numelem, array, getAndConvertLong)) {
        Py_DECREF(pya);
        pya = NULL;
      }
    }
#if (SIZEOF_INT != 4)
    free(pynumelem);
#endif
    free(numelem);
    free(lower);
  }
  else {
    Py_INCREF(Py_None);
    pya = Py_None;
  }
  return pya;
}
#endif
#endif
#endif

static int
getAndConvertString(void *array, const int32_t *ind, PyObject **dest)
{
  char *str = 
    sidl_string__array_get((struct sidl_string__array *)array, ind);
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

static PyObject *
clone_sidl_string_array(struct sidl_string__array *array)
{
  PyObject *pya = NULL;
  if (array) {
    const int dimen = sidl_string__array_dimen(array);
    int i;
    int32_t *lower = malloc(sizeof(int32_t) * dimen);
    int32_t *upper = malloc(sizeof(int32_t) * dimen);
    int32_t *numelem = malloc(sizeof(int32_t) * dimen);
#if (SIZEOF_INT != 4)
    int *pynumelem = malloc(sizeof(int) * dimen);
#else
    int *pynumelem = (int *)numelem;
#endif
    for(i = 0; i < dimen; ++i) {
      lower[i] = sidl_string__array_lower(array, i);
      upper[i] = sidl_string__array_upper(array, i);
      numelem[i] = sidl_string__array_length(array, i);
#if (SIZEOF_INT != 4)
      pynumelem[i] = (int)numelem[i];
#endif
    }
    pya = PyArray_FromDims(dimen, pynumelem, PyArray_OBJECT);
    if (pya) {
      if (!sidl_array__convert_sidl(pya, dimen, lower, upper, 
                                    numelem, array, getAndConvertString)) {
        Py_DECREF(pya);
        pya = NULL;
      }
    }
#if (SIZEOF_INT != 4)
    free(pynumelem);
#endif
    free(numelem);
    free(upper);
    free(lower);
  }
  else {
    Py_INCREF(Py_None);
    pya = Py_None;
  }
  return pya;
}

static int
getAndConvertOpaque(void *array, const int32_t *ind, PyObject **dest)
{
  void *vptr =
    sidl_opaque__array_get((struct sidl_opaque__array *)array, ind);
  *dest = PyCObject_FromVoidPtr(vptr, NULL);
  return FALSE;
}

static PyObject *
clone_sidl_opaque_array(struct sidl_opaque__array *array)
{
  PyObject *pya = NULL;
  if (array) {
    const int dimen = sidl_opaque__array_dimen(array);
    int i;
    int32_t *lower = malloc(sizeof(int32_t) * dimen);
    int32_t *upper = malloc(sizeof(int32_t) * dimen);
    int32_t *numelem = malloc(sizeof(int32_t) * dimen);
#if (SIZEOF_INT != 4)
    int *pynumelem = malloc(sizeof(int) * dimen);
#else
    int *pynumelem = (int *)numelem;
#endif
    for(i = 0; i < dimen; ++i) {
      lower[i] = sidl_opaque__array_lower(array, i);
      upper[i] = sidl_opaque__array_upper(array, i);
      numelem[i] = 1 + upper[i] - lower[i];
#if (SIZEOF_INT != 4)
      pynumelem[i] = (int)numelem[i];
#endif
    }
    pya = PyArray_FromDims(dimen, pynumelem, PyArray_OBJECT);
    if (pya) {
      if (!sidl_array__convert_sidl(pya, dimen, lower, upper,
                                    numelem, array, getAndConvertOpaque)) {
        Py_DECREF(pya);
        pya = NULL;
      }
    }
#if (SIZEOF_INT != 4)
    free(pynumelem);
#endif
    free(numelem);
    free(upper);
    free(lower);
  }
  else {
    Py_INCREF(Py_None);
    pya = Py_None;
  }
  return pya;
}

sidl_python_clone_array_RETURN
sidl_python_clone_array sidl_python_clone_array_PROTO
{
  if (array) {
    switch (sidl__array_type(array)) {
    case sidl_bool_array:
      return clone_sidl_bool_array((struct sidl_bool__array *)array);
    case sidl_char_array:
      return clone_sidl_char_array((struct sidl_char__array *)array);
    case sidl_dcomplex_array:
      return clone_sidl_dcomplex_array((struct sidl_dcomplex__array *)array);
    case sidl_double_array:
      return clone_sidl_double_array((struct sidl_double__array *)array);
    case sidl_fcomplex_array:
      return clone_sidl_fcomplex_array((struct sidl_fcomplex__array *)array);
    case sidl_float_array:
      return clone_sidl_float_array((struct sidl_float__array *)array);
    case sidl_int_array:
      return clone_sidl_int_array((struct sidl_int__array *)array);
    case sidl_long_array:
      return clone_sidl_long_array((struct sidl_long__array *)array);
    case sidl_opaque_array:
      return clone_sidl_opaque_array((struct sidl_opaque__array *)array);
    case sidl_string_array:
      return clone_sidl_string_array((struct sidl_string__array *)array);
    default:
      return NULL;                /* indicate an error */
    }
  }
  else {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

sidl_python_copy_RETURN
sidl_python_copy sidl_python_copy_PROTO
{
  const int32_t src_type = sidl__array_type(src);
  const int32_t dest_type = sidl__array_type(dest);
  if (src_type == dest_type) {
    switch(src_type) {
    case sidl_bool_array:
      sidl_bool__array_copy((const struct sidl_bool__array *)src, 
                            (struct sidl_bool__array *)dest); break;
    case sidl_char_array:
      sidl_char__array_copy((const struct sidl_char__array *)src,
                            (struct sidl_char__array *)dest); break;
    case sidl_dcomplex_array:
      sidl_dcomplex__array_copy((const struct sidl_dcomplex__array *)src,
                                (struct sidl_dcomplex__array *)dest); break;
    case sidl_double_array:
      sidl_double__array_copy((const struct sidl_double__array *)src,
                              (struct sidl_double__array *)dest); break;
    case sidl_fcomplex_array:
      sidl_fcomplex__array_copy((const struct sidl_fcomplex__array *)src,
                                (struct sidl_fcomplex__array *)dest); break;
    case sidl_float_array:
      sidl_float__array_copy((const struct sidl_float__array *)src,
                             (struct sidl_float__array *)dest); break;
    case sidl_int_array:
      sidl_int__array_copy((const struct sidl_int__array *)src,
                           (struct sidl_int__array *)dest); break;
    case sidl_long_array:
      sidl_long__array_copy((const struct sidl_long__array *)src,
                            (struct sidl_long__array *)dest); break;
    case sidl_opaque_array:
      sidl_opaque__array_copy((const struct sidl_opaque__array *)src,
                              (struct sidl_opaque__array *)dest); break;
    case sidl_string_array:
      sidl_string__array_copy((const struct sidl_string__array *)src,
                              (struct sidl_string__array *)dest); break;
    }
  }
}

sidl_array__convert_sidl_RETURN
sidl_array__convert_sidl sidl_array__convert_sidl_PROTO
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

static PyObject*
createArrayHolder(struct sidl__array * const array)
{
  PyObject *result = sidlPyArrayType.tp_new(&sidlPyArrayType, NULL, NULL);
  if (result) {
    SIDLArrayObject *sao = (SIDLArrayObject *)result;  
    sao->d_array = array;
  }
  return result;
}

static PyObject *
borrow_sidl(struct sidl__array * const array,
            char * const dataPtr,
            const size_t dataSize,
            const int numpyType)
{
  const int dimen = sidlArrayDim(array);
  int extent[SIDL_MAX_ARRAY_DIMENSION], i;
  PyObject *result;
  for(i = 0; i < dimen; ++i) {
    extent[i] = sidlLength(array, i);
  }
  result = PyArray_FromDimsAndData(dimen, extent, numpyType, dataPtr);
  if (result) {
    PyObject *sidlRef = createArrayHolder(array);
    if (sidlRef) {
      PyArrayObject *numpy = (PyArrayObject *)result;
      /* fix the strides to match the SIDL array */
      for(i = 0; i < dimen; ++i) {
        numpy->strides[i] = sidlStride(array,i) * dataSize;
      }
      /* set the CONTIGUOUS flag */
      if (sidl__array_isRowOrder(array)){
        numpy->flags |= CONTIGUOUS;
      }
      else {
        numpy->flags &= ~CONTIGUOUS;
      }
      numpy->base = sidlRef;
    }
    else {
      sidl__array_deleteRef(array);
      Py_DECREF(result);
      result = NULL;
    }
  }
  else {
    sidl__array_deleteRef(array);
  }
  return result;
}

sidl_python_borrow_array_RETURN
sidl_python_borrow_array sidl_python_borrow_array_PROTO
{ 
  if (array) {
    switch(sidl__array_type(array)) {
    case sidl_char_array:
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)(((struct sidl_char__array*)array)->d_firstElement),
                         sizeof(char), PyArray_CHAR);
    case sidl_dcomplex_array:
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)(((struct sidl_dcomplex__array*)array)->d_firstElement),
                          sizeof(struct sidl_dcomplex), PyArray_CDOUBLE);
    case sidl_double_array:
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)((struct sidl_double__array*)array)->d_firstElement,
                         sizeof(double), PyArray_DOUBLE);
    case sidl_fcomplex_array:
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)(((struct sidl_fcomplex__array*)array)->d_firstElement),
                          sizeof(struct sidl_fcomplex), PyArray_CFLOAT);
    case sidl_float_array:
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)((struct sidl_float__array*)array)->d_firstElement,
                         sizeof(float), PyArray_FLOAT);
    case sidl_int_array:
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)((struct sidl_int__array*)array)->d_firstElement,
                         sizeof(int32_t),
#if SIZEOF_SHORT == 4
                         PyArray_SHORT
#else
#if SIZEOF_INT == 4
                         PyArray_INT
#else
#if SIZEOF_LONG == 4
                         PyArray_LONG
#else
#error No 32-bit integer available.
#endif
#endif
#endif
                         );
    case sidl_long_array:
#if SIZEOF_SHORT == 8
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)((struct sidl_long__array*)array)->d_firstElement,
                         sizeof(int64_t),
                         PyArray_SHORT);
#else
#if SIZEOF_INT == 8
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)((struct sidl_long__array*)array)->d_firstElement,
                         sizeof(int64_t),
                         PyArray_INT);
#else
#if SIZEOF_LONG == 8
      array = sidl__array_smartCopy(array);
      return borrow_sidl(array,
                         (char *)((struct sidl_long__array*)array)->d_firstElement,
                         sizeof(int64_t),
                         PyArray_LONG);
#else
      return sidl_python_clone_array(array); 
#endif
#endif
#endif
    }
  }
  return sidl_python_clone_array(array); 
}

sidl_generic_borrow_python_array_RETURN
sidl_generic_borrow_python_array sidl_generic_borrow_python_array_PROTO
{
  if (PyArray_Check(obj)) {
    PyArrayObject *pya = (PyArrayObject *)obj;
    switch (pya->descr->type_num) {
    case PyArray_CHAR:
    case PyArray_UBYTE:
    case PyArray_SBYTE:
      return sidl_char__borrow_python_array
        (obj, (struct sidl_char__array **)array);
    case PyArray_SHORT:
#ifdef PyArray_UNSIGNED_TYPES
    case PyArray_USHORT:
#endif
#if SIZEOF_SHORT == 4
      return sidl_int__borrow_python_array
        (obj, (struct sidl_int__array **)array);
#else
#if SIZEOF_SHORT == 8
      reutrn sidl_long__borrow_python_array
        (obj, (struct sidl_long__array **)array);
#else
      break; /* prevent default or fall through */
#endif
#endif
    case PyArray_INT:
#ifdef PyArray_UNSIGNED_TYPES
    case PyArray_UINT:
#endif
#if SIZEOF_INT == 4
      return sidl_int__borrow_python_array
        (obj, (struct sidl_int__array **)array);
#else
#if SIZEOF_INT == 8
      return sidl_long__borrow_python_array
        (obj, (struct sidl_long__array **)array);
#else
      break; /* prevent default or fall through */
#endif
#endif
    case PyArray_LONG:
#if SIZEOF_LONG == 4
      return sidl_int__borrow_python_array
        (obj, (struct sidl_int__array **)array);
#else
#if SIZEOF_LONG == 8
      return sidl_long__borrow_python_array
        (obj, (struct sidl_long__array **)array);
#else
      break; /* prevent default and fall through */
#endif
#endif
    case PyArray_FLOAT:
      return sidl_float__borrow_python_array
        (obj, (struct sidl_float__array **)array);
    case PyArray_DOUBLE:
      return sidl_double__borrow_python_array
        (obj, (struct sidl_double__array **)array);
    case PyArray_CFLOAT:
      return sidl_fcomplex__borrow_python_array
        (obj, (struct sidl_fcomplex__array **)array);
    case PyArray_CDOUBLE:
      return sidl_dcomplex__borrow_python_array
        (obj, (struct sidl_dcomplex__array **)array);
    case PyArray_OBJECT:
      break;
    default:
      *array = 0;
      return 0;
    }
  }
  return sidl_generic_clone_python_array(obj, array);
}

static int
convertObjectToBaseInterface(void *sidlarray, const int *ind, PyObject *pyobj)
{
  struct sidl_BaseInterface__object *sidlobj =
    sidl_Cast(pyobj, "sidl.BaseInterface");
  if (sidlobj || (Py_None == pyobj)) {
    sidl_interface__array_set((struct sidl_interface__array *)sidlarray,
                              ind, sidlobj);
    return FALSE;
  }
  return TRUE;
}

static
int
clone_object_array(PyArrayObject *pya, struct sidl__array **array)
{
  int result = 0;
  if (PyArray_Size((PyObject *)pya)) {
    PyObject *elem = *((PyObject **)(pya->data));
    if ((elem == Py_None) || 
        PyType_IsSubtype(elem->ob_type, sidl_PyType())) {
      /* an array of sidl objects */
      int32_t dimen;
      int32_t lower[SIDL_MAX_ARRAY_DIMENSION], 
        upper[SIDL_MAX_ARRAY_DIMENSION], 
        stride[SIDL_MAX_ARRAY_DIMENSION];
      if (sidl_array__extract_python_info((PyObject *)pya, &dimen,
                                          lower, upper, stride)) {
        *array = (struct sidl__array *)
          sidl_interface__array_createRow(dimen, lower, upper);
        result = sidl_array__convert_python
          ((PyObject *)pya, dimen, *array, convertObjectToBaseInterface);
        if (*array && !result ) {
          sidl__array_deleteRef(*array);
          *array = NULL;
        }
      }
    }
    else {
      /* otherwise treat it as an array of longs */
      result = sidl_long__clone_python_array
        ((PyObject *)pya, (struct sidl_long__array **)array);
    }
  }
  else { /* empty list of what? */
    result = sidl_int__clone_python_array
      ((PyObject *)pya, (struct sidl_int__array **)array);
  }
  return result;
}

sidl_generic_clone_python_array_RETURN
sidl_generic_clone_python_array sidl_generic_clone_python_array_PROTO
{
  int result = 0;
  *array = 0;
  if (obj == Py_None) {
    result = 1;
  }
  else {
    PyArrayObject *pya = NULL;
    if (PyArray_Check(obj)) {
      pya = (PyArrayObject *)obj;
      Py_INCREF(pya);
    }
    else {
      pya = (PyArrayObject *)
        PyArray_FromObject(obj, PyArray_NOTYPE, 0, 0);
    }
    if (pya) {
      switch (pya->descr->type_num) {
      case PyArray_CHAR:
      case PyArray_UBYTE:
      case PyArray_SBYTE:
        result = sidl_char__clone_python_array
          (obj, (struct sidl_char__array **)array);
        break;
      case PyArray_SHORT:
#ifdef PyArray_UNSIGNED_TYPES
      case PyArray_USHORT:
#endif
#if SIZEOF_SHORT == 4
        result = sidl_int__clone_python_array
          (obj, (struct sidl_int__array **)array);
#else
#if SIZEOF_SHORT == 8
        result = sidl_long__clone_python_array
          (obj, (struct sidl_long__array **)array);
#endif
#endif
        break;
      case PyArray_INT:
#ifdef PyArray_UNSIGNED_TYPES
      case PyArray_UINT:
#endif
#if SIZEOF_INT == 4
        result = sidl_int__clone_python_array
          (obj, (struct sidl_int__array **)array);
#else
#if SIZEOF_INT == 8
        result = sidl_long__clone_python_array
          (obj, (struct sidl_long__array **)array);
#endif
#endif
        break; /* prevent default or fall through */
      case PyArray_LONG:
#if SIZEOF_LONG == 4
        result = sidl_int__clone_python_array
          (obj, (struct sidl_int__array **)array);
#else
#if SIZEOF_LONG == 8
        result = sidl_long__clone_python_array
          (obj, (struct sidl_long__array **)array);
#endif
#endif
        break; /* prevent default and fall through */
      case PyArray_FLOAT:
        result = sidl_float__clone_python_array
          (obj, (struct sidl_float__array **)array);
      case PyArray_DOUBLE:
        result = sidl_double__clone_python_array
          (obj, (struct sidl_double__array **)array);
      case PyArray_CFLOAT:
        result = sidl_fcomplex__clone_python_array
          (obj, (struct sidl_fcomplex__array **)array);
      case PyArray_CDOUBLE:
        result = sidl_dcomplex__clone_python_array
          (obj, (struct sidl_dcomplex__array **)array);
      case PyArray_OBJECT:
        result =  clone_object_array(pya, array);
      default:
        *array = 0;
        result = 0;
      }
      Py_DECREF(pya);
    }
  }
  return result;
}


static struct PyMethodDef spa_methods[] = {
  /* this module exports no methods */
  { NULL, NULL }
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initsidlPyArrays(void) 
{
  PyObject *module, *dict, *c_api;
  static void *spa_api[sidlPyArrays_API_pointers];
  module = Py_InitModule("sidlPyArrays", spa_methods);
  import_array();
  dict = PyModule_GetDict(module);
  if (PyType_Ready(&sidlPyArrayType) < 0) return;
  Py_INCREF(&sidlPyArrayType);
  PyModule_AddObject(module, "SIDLArrayWrapper", (PyObject *)&sidlPyArrayType);
  spa_api[sidl_python_deleteRef_array_NUM] =
    (void *)sidl__array_deleteRef;
  spa_api[sidl_python_borrow_array_NUM] =
    (void *)sidl_python_borrow_array;
  spa_api[sidl_python_clone_array_NUM] =
    (void *)sidl_python_clone_array;
  spa_api[sidl_generic_clone_python_array_NUM] =
    (void *)sidl_generic_clone_python_array;
  spa_api[sidl_generic_borrow_python_array_NUM] =
    (void *)sidl_generic_borrow_python_array;
  spa_api[sidl_python_copy_NUM] =
    (void *)sidl_python_copy;
  spa_api[sidl_bool__borrow_python_array_NUM] =
    (void *)sidl_bool__borrow_python_array;
  spa_api[sidl_bool__clone_python_array_column_NUM] =
    (void *)sidl_bool__clone_python_array_column;
  spa_api[sidl_bool__clone_python_array_row_NUM] =
    (void *)sidl_bool__clone_python_array_row;
  spa_api[sidl_bool__clone_python_array_NUM] =
    (void *)sidl_bool__clone_python_array;
  spa_api[sidl_char__borrow_python_array_NUM] =
    (void *)sidl_char__borrow_python_array;
  spa_api[sidl_char__clone_python_array_column_NUM] =
    (void *)sidl_char__clone_python_array_column;
  spa_api[sidl_char__clone_python_array_row_NUM] =
    (void *)sidl_char__clone_python_array_row;
  spa_api[sidl_char__clone_python_array_NUM] =
    (void *)sidl_char__clone_python_array;
  spa_api[sidl_dcomplex__borrow_python_array_NUM] =
    (void *)sidl_dcomplex__borrow_python_array;
  spa_api[sidl_dcomplex__clone_python_array_column_NUM] =
    (void *)sidl_dcomplex__clone_python_array_column;
  spa_api[sidl_dcomplex__clone_python_array_row_NUM] =
    (void *)sidl_dcomplex__clone_python_array_row;
  spa_api[sidl_dcomplex__clone_python_array_NUM] =
    (void *)sidl_dcomplex__clone_python_array;
  spa_api[sidl_double__borrow_python_array_NUM] =
    (void *)sidl_double__borrow_python_array;
  spa_api[sidl_double__clone_python_array_column_NUM] =
    (void *)sidl_double__clone_python_array_column;
  spa_api[sidl_double__clone_python_array_row_NUM] =
    (void *)sidl_double__clone_python_array_row;
  spa_api[sidl_double__clone_python_array_NUM] =
    (void *)sidl_double__clone_python_array;
  spa_api[sidl_fcomplex__borrow_python_array_NUM] =
    (void *)sidl_fcomplex__borrow_python_array;
  spa_api[sidl_fcomplex__clone_python_array_column_NUM] =
    (void *)sidl_fcomplex__clone_python_array_column;
  spa_api[sidl_fcomplex__clone_python_array_row_NUM] =
    (void *)sidl_fcomplex__clone_python_array_row;
  spa_api[sidl_fcomplex__clone_python_array_NUM] =
    (void *)sidl_fcomplex__clone_python_array;
  spa_api[sidl_float__borrow_python_array_NUM] =
    (void *)sidl_float__borrow_python_array;
  spa_api[sidl_float__clone_python_array_column_NUM] =
    (void *)sidl_float__clone_python_array_column;
  spa_api[sidl_float__clone_python_array_row_NUM] =
    (void *)sidl_float__clone_python_array_row;
  spa_api[sidl_float__clone_python_array_NUM] =
    (void *)sidl_float__clone_python_array;
  spa_api[sidl_int__borrow_python_array_NUM] =
    (void *)sidl_int__borrow_python_array;
  spa_api[sidl_int__clone_python_array_column_NUM] =
    (void *)sidl_int__clone_python_array_column;
  spa_api[sidl_int__clone_python_array_row_NUM] =
    (void *)sidl_int__clone_python_array_row;
  spa_api[sidl_int__clone_python_array_NUM] =
    (void *)sidl_int__clone_python_array;
  spa_api[sidl_long__borrow_python_array_NUM] =
    (void *)sidl_long__borrow_python_array;
  spa_api[sidl_long__clone_python_array_column_NUM] =
    (void *)sidl_long__clone_python_array_column;
  spa_api[sidl_long__clone_python_array_row_NUM] =
    (void *)sidl_long__clone_python_array_row;
  spa_api[sidl_long__clone_python_array_NUM] =
    (void *)sidl_long__clone_python_array;
  spa_api[sidl_opaque__borrow_python_array_NUM] =
    (void *)sidl_opaque__borrow_python_array;
  spa_api[sidl_opaque__clone_python_array_column_NUM] =
    (void *)sidl_opaque__clone_python_array_column;
  spa_api[sidl_opaque__clone_python_array_row_NUM] =
    (void *)sidl_opaque__clone_python_array_row;
  spa_api[sidl_opaque__clone_python_array_NUM] =
    (void *)sidl_opaque__clone_python_array;
  spa_api[sidl_string__borrow_python_array_NUM] =
    (void *)sidl_string__borrow_python_array;
  spa_api[sidl_string__clone_python_array_column_NUM] =
    (void *)sidl_string__clone_python_array_column;
  spa_api[sidl_string__clone_python_array_row_NUM] =
    (void *)sidl_string__clone_python_array_row;
  spa_api[sidl_string__clone_python_array_NUM] =
    (void *)sidl_string__clone_python_array;
  spa_api[sidl_array__convert_python_NUM] =
    (void *)sidl_array__convert_python;
  spa_api[sidl_array__convert_sidl_NUM] =
    (void *)sidl_array__convert_sidl;
  spa_api[sidl_array__extract_python_info_NUM] =
    (void *)sidl_array__extract_python_info;
  c_api = PyCObject_FromVoidPtr((void *)spa_api, NULL);
  if (c_api) {
    PyDict_SetItemString(dict, "_C_API", c_api);
    Py_DECREF(c_api);
  }
  if (PyErr_Occurred()) {
    Py_FatalError("Can't initialize module sidlPyArrays.");
  }
}
