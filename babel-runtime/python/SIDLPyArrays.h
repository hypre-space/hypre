/*
 * File:        SIDLPyArrays.h
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name$
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Runtime support for routines to convert arrays to/from Python
 *
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

#ifndef included_SIDLPyArrays_h
#define included_SIDLPyArrays_h

/*
 * This header defines the external API for a Python C extension module.
 * It provides conversion routines between SIDL arrays and Numerical
 * Python arrays.
 *
 * For each type, 
 *      bool
 *      char
 *      dcomplex
 *      double
 *      fcomplex
 *      float
 *      int
 *      long
 *      opaque
 *      string
 * there are five methods related to borrowing, cloning or deleting a
 * reference on arrays. For each type x, the method names and meanings are
 * as follows: 
 *
 *
 * Attempt to convert a Python object into a SIDL array of x's. This
 * method will attempt to borrow data from the Python object if it's
 * possible; if borrowing is not possible, it will create an independent
 * copy. This function is designed for use as a convert function in a
 * PyArg_ParseTuple call with a format string entry of "O&".
 * Return value: 1 is returned if the conversion was successful; otherwise,
 *               0 is returned.
 * int
 * SIDL_x__borrow_python_array(PyObject *obj,
 *                             struct SIDL_x__array **array);
 *
 *
 * Attempt to convert a Python object into an independent, column-ordered
 * SIDL array of x's.  This will not borrow data from the Python object 
 * even if it is possible. This function is designed for use as a convert
 * function in a PyArg_ParseTuple call with a format string entry of "O&".
 * Return value: 1 is returned if the conversion was successful; otherwise,
 *               0 is returned.
 * int
 * SIDL_x__clone_python_array_column(PyObject *obj,
 *                                   struct SIDL_x__array **array);
 *
 *
 * Attempt to convert a Python object into an independent, row-ordered
 * SIDL array of x's.  This will not borrow data from the Python object
 * even if it is possible. This function is designed for use as a convert
 * function in a PyArg_ParseTuple call with a format string entry of "O&".
 * Return value: 1 is returned if the conversion was successful; otherwise,
 *               0 is returned.
 * int
 * SIDL_x__clone_python_array_row(PyObject *obj,
 *                                struct SIDL_x__array **array);
 *
 *
 * Attempt to convert a SIDL array of x's into a Python object.
 * This method will create a Python object that borrows data from
 * the SIDL array if it's possible; otherwise, it will create
 * an independent copy of the array. This function is designed
 * to be used in a Py_BuildValue call and a format string entry of "O&".
 * Return value: NULL means the conversion failed; otherwise,
 *               the conversion was successful.
 * PyObject *
 * SIDL_x__python_borrow_array(struct SIDL_x__array *array);
 *
 *
 * Attempt to convert a SIDL array of x's into an independent Python
 * object. This will create an independent Python object regardless
 * of whether borrowing is possible or not. This function is designed
 * to be used in a Py_BuildValue call and a format string entry of "O&".
 * Return value: NULL means the conversion failed; otherwise,
 *               the conversion was successful.
 * PyObject *
 * SIDL_x__python_clone_array(struct SIDL_x__array *array);
 *
 * This is a Python wrapper function for SIDL_x__array_deleteReference.
 * void
 * SIDL_x__python_deleteReference_array(struct SIDL_x__array *array);
 *
 *
 *
 * Here are the standalone functions.  These are useful for arrays of
 * object or interface pointers.
 *
 *
 *
 * This function will copy a Python array into a SIDL array.  The
 * SIDL array is assumed to have the same shape and index bounds
 * as the Python array.
 *
 * setfunc is called as follows:
 *     (*setfunc)(sidl_dest, ind, pyobj)
 *     sidl_dest is the void pointer passed to SIDL_array__convert_python.
 *     ind       is a pointer to dimen integers
 *     pyobj     is a pointer to a Python object from pya_src.
 * If setfunc returns a true value, SIDL_array__convert_python will 
 * immediately cease copying and return a false value.
 *
 * A true value is returned if the function was successful; otherwise,
 * a false value is returned to indicated that the operation failed.
 *
 * int
 * SIDL_array__convert_python(PyArrayObject          *pya_src,
 *                            const int32_t           dimen,
 *                            void                   *sidl_dest,
 *                            SIDL_array_set_py_elem  setfunc) 
 *
 *
 *
 * This function will copy a SIDL array into a Python array.  The
 * Python array is assumes to have the same shape as the SIDL array.
 * The Python array need not have index bounds.
 *
 * getfunc is called as follows:
 *     (*getfunc)(sidl_dest, ind, pyobj)
 *     sidl_src  is the void pointer passed to SIDL_array__convert_sidl.
 *     ind       is a pointer to dimen integers
 *     pyobj     is a pointer to a pointer to a Python object from pya_dest.
 *               This is where the return is stored
 * If setfunc returns a true value, SIDL_array__convert_sidl will 
 * immediately cease copying and return a false value.
 *
 * A true value is returned if the function was successful; otherwise,
 * a false value is returned to indicated that the operation failed.
 *
 * int
 * SIDL_array__convert_sidl(PyArrayObject          *pya_dest,
 *                          const int32_t           dimen,
 *                          int32_t                 lower[],
 *                          const int32_t           upper[],
 *                          const int32_t           numelem[],
 *                          void                   *sidl_src,
 *                          SIDL_array_get_py_elem  getfunc) 
 *
 *
 *
 *
 * Extract the dimension information from a Numerical Python array
 * into a form that SIDL can use.  If the return value is positive,
 * *lower, *upper, and *stride points to malloc'ed memory and must
 * be freed by the caller.
 *      dimension   on entry a pointer to a single integer
 *      lower       a pointer to a pointer to integers
 *      upper       a pointer to a pointer to integers
 *      stride      a pointer to a pointer to integers
 * int
 * SIDL_array__extract_python_info(PyArrayObject  *pya,
 *                                 int32_t        *dimension,
 *                                 int32_t       **lower,
 *                                 int32_t       **upper,
 *                                 int32_t       **stride)
 */

#include "babel_config.h"
#include "SIDLType.h"
#include <Python.h>

/*
 * Forward declarations of array types.
 */
struct SIDL_bool__array;
struct SIDL_char__array;
struct SIDL_dcomplex__array;
struct SIDL_double__array;
struct SIDL_fcomplex__array;
struct SIDL_float__array;
struct SIDL_int__array;
struct SIDL_long__array;
struct SIDL_opaque__array;
struct SIDL_string__array;
struct SIDL_BaseInterface__array;
typedef int (*SIDL_array_set_py_elem)(void *, const int32_t *, PyObject *);
typedef int (*SIDL_array_get_py_elem)(void *, const int32_t *, PyObject **);

/*
 * Here we use the standard Python method for exposing an external
 * C API.  It's not particular pretty, but it works.
 */

#define SIDL_bool__borrow_python_array_NUM 0
#define SIDL_bool__borrow_python_array_RETURN int
#define SIDL_bool__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_bool__array **array)

#define SIDL_bool__clone_python_array_column_NUM 1
#define SIDL_bool__clone_python_array_column_RETURN int
#define SIDL_bool__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_bool__array **array)

#define SIDL_bool__clone_python_array_row_NUM 2
#define SIDL_bool__clone_python_array_row_RETURN int
#define SIDL_bool__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_bool__array **array)

#define SIDL_bool__python_borrow_array_NUM 3
#define SIDL_bool__python_borrow_array_RETURN PyObject *
#define SIDL_bool__python_borrow_array_PROTO \
  (struct SIDL_bool__array *array)

#define SIDL_bool__python_clone_array_NUM 4
#define SIDL_bool__python_clone_array_RETURN PyObject *
#define SIDL_bool__python_clone_array_PROTO \
  (struct SIDL_bool__array *array)

#define SIDL_bool__python_deleteReference_array_NUM 5
#define SIDL_bool__python_deleteReference_array_RETURN void
#define SIDL_bool__python_deleteReference_array_PROTO \
  (struct SIDL_bool__array *array)


#define SIDL_char__borrow_python_array_NUM 6
#define SIDL_char__borrow_python_array_RETURN int
#define SIDL_char__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_char__array **array)

#define SIDL_char__clone_python_array_column_NUM 7
#define SIDL_char__clone_python_array_column_RETURN int
#define SIDL_char__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_char__array **array)

#define SIDL_char__clone_python_array_row_NUM 8
#define SIDL_char__clone_python_array_row_RETURN int
#define SIDL_char__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_char__array **array)

#define SIDL_char__python_borrow_array_NUM 9
#define SIDL_char__python_borrow_array_RETURN PyObject *
#define SIDL_char__python_borrow_array_PROTO \
  (struct SIDL_char__array *array)

#define SIDL_char__python_clone_array_NUM 10
#define SIDL_char__python_clone_array_RETURN PyObject *
#define SIDL_char__python_clone_array_PROTO \
  (struct SIDL_char__array *array)

#define SIDL_char__python_deleteReference_array_NUM 11
#define SIDL_char__python_deleteReference_array_RETURN void
#define SIDL_char__python_deleteReference_array_PROTO \
  (struct SIDL_char__array *array)


#define SIDL_dcomplex__borrow_python_array_NUM 12
#define SIDL_dcomplex__borrow_python_array_RETURN int
#define SIDL_dcomplex__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_dcomplex__array **array)

#define SIDL_dcomplex__clone_python_array_column_NUM 13
#define SIDL_dcomplex__clone_python_array_column_RETURN int
#define SIDL_dcomplex__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_dcomplex__array **array)

#define SIDL_dcomplex__clone_python_array_row_NUM 14
#define SIDL_dcomplex__clone_python_array_row_RETURN int
#define SIDL_dcomplex__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_dcomplex__array **array)

#define SIDL_dcomplex__python_borrow_array_NUM 15
#define SIDL_dcomplex__python_borrow_array_RETURN PyObject *
#define SIDL_dcomplex__python_borrow_array_PROTO \
  (struct SIDL_dcomplex__array *array)

#define SIDL_dcomplex__python_clone_array_NUM 16
#define SIDL_dcomplex__python_clone_array_RETURN PyObject *
#define SIDL_dcomplex__python_clone_array_PROTO \
  (struct SIDL_dcomplex__array *array)

#define SIDL_dcomplex__python_deleteReference_array_NUM 17
#define SIDL_dcomplex__python_deleteReference_array_RETURN void
#define SIDL_dcomplex__python_deleteReference_array_PROTO \
  (struct SIDL_dcomplex__array *array)


#define SIDL_double__borrow_python_array_NUM 18
#define SIDL_double__borrow_python_array_RETURN int
#define SIDL_double__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_double__array **array)

#define SIDL_double__clone_python_array_column_NUM 19
#define SIDL_double__clone_python_array_column_RETURN int
#define SIDL_double__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_double__array **array)

#define SIDL_double__clone_python_array_row_NUM 20
#define SIDL_double__clone_python_array_row_RETURN int
#define SIDL_double__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_double__array **array)

#define SIDL_double__python_borrow_array_NUM 21
#define SIDL_double__python_borrow_array_RETURN PyObject *
#define SIDL_double__python_borrow_array_PROTO \
  (struct SIDL_double__array *array)

#define SIDL_double__python_clone_array_NUM 22
#define SIDL_double__python_clone_array_RETURN PyObject *
#define SIDL_double__python_clone_array_PROTO \
  (struct SIDL_double__array *array)

#define SIDL_double__python_deleteReference_array_NUM 23
#define SIDL_double__python_deleteReference_array_RETURN void
#define SIDL_double__python_deleteReference_array_PROTO \
  (struct SIDL_double__array *array)


#define SIDL_fcomplex__borrow_python_array_NUM 24
#define SIDL_fcomplex__borrow_python_array_RETURN int
#define SIDL_fcomplex__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_fcomplex__array **array)

#define SIDL_fcomplex__clone_python_array_column_NUM 25
#define SIDL_fcomplex__clone_python_array_column_RETURN int
#define SIDL_fcomplex__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_fcomplex__array **array)

#define SIDL_fcomplex__clone_python_array_row_NUM 26
#define SIDL_fcomplex__clone_python_array_row_RETURN int
#define SIDL_fcomplex__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_fcomplex__array **array)

#define SIDL_fcomplex__python_borrow_array_NUM 27
#define SIDL_fcomplex__python_borrow_array_RETURN PyObject *
#define SIDL_fcomplex__python_borrow_array_PROTO \
  (struct SIDL_fcomplex__array *array)

#define SIDL_fcomplex__python_clone_array_NUM 28
#define SIDL_fcomplex__python_clone_array_RETURN PyObject *
#define SIDL_fcomplex__python_clone_array_PROTO \
  (struct SIDL_fcomplex__array *array)

#define SIDL_fcomplex__python_deleteReference_array_NUM 29
#define SIDL_fcomplex__python_deleteReference_array_RETURN void
#define SIDL_fcomplex__python_deleteReference_array_PROTO \
  (struct SIDL_fcomplex__array *array)


#define SIDL_float__borrow_python_array_NUM 30
#define SIDL_float__borrow_python_array_RETURN int
#define SIDL_float__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_float__array **array)

#define SIDL_float__clone_python_array_column_NUM 31
#define SIDL_float__clone_python_array_column_RETURN int
#define SIDL_float__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_float__array **array)

#define SIDL_float__clone_python_array_row_NUM 32
#define SIDL_float__clone_python_array_row_RETURN int
#define SIDL_float__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_float__array **array)

#define SIDL_float__python_borrow_array_NUM 33
#define SIDL_float__python_borrow_array_RETURN PyObject *
#define SIDL_float__python_borrow_array_PROTO \
  (struct SIDL_float__array *array)

#define SIDL_float__python_clone_array_NUM 34
#define SIDL_float__python_clone_array_RETURN PyObject *
#define SIDL_float__python_clone_array_PROTO \
  (struct SIDL_float__array *array)

#define SIDL_float__python_deleteReference_array_NUM 35
#define SIDL_float__python_deleteReference_array_RETURN void
#define SIDL_float__python_deleteReference_array_PROTO \
  (struct SIDL_float__array *array)


#define SIDL_int__borrow_python_array_NUM 36
#define SIDL_int__borrow_python_array_RETURN int
#define SIDL_int__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_int__array **array)

#define SIDL_int__clone_python_array_column_NUM 37
#define SIDL_int__clone_python_array_column_RETURN int
#define SIDL_int__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_int__array **array)

#define SIDL_int__clone_python_array_row_NUM 38
#define SIDL_int__clone_python_array_row_RETURN int
#define SIDL_int__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_int__array **array)

#define SIDL_int__python_borrow_array_NUM 39
#define SIDL_int__python_borrow_array_RETURN PyObject *
#define SIDL_int__python_borrow_array_PROTO \
  (struct SIDL_int__array *array)

#define SIDL_int__python_clone_array_NUM 40
#define SIDL_int__python_clone_array_RETURN PyObject *
#define SIDL_int__python_clone_array_PROTO \
  (struct SIDL_int__array *array)

#define SIDL_int__python_deleteReference_array_NUM 41
#define SIDL_int__python_deleteReference_array_RETURN void
#define SIDL_int__python_deleteReference_array_PROTO \
  (struct SIDL_int__array *array)


#define SIDL_long__borrow_python_array_NUM 42
#define SIDL_long__borrow_python_array_RETURN int
#define SIDL_long__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_long__array **array)

#define SIDL_long__clone_python_array_column_NUM 43
#define SIDL_long__clone_python_array_column_RETURN int
#define SIDL_long__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_long__array **array)

#define SIDL_long__clone_python_array_row_NUM 44
#define SIDL_long__clone_python_array_row_RETURN int
#define SIDL_long__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_long__array **array)

#define SIDL_long__python_borrow_array_NUM 45
#define SIDL_long__python_borrow_array_RETURN PyObject *
#define SIDL_long__python_borrow_array_PROTO \
  (struct SIDL_long__array *array)

#define SIDL_long__python_clone_array_NUM 46
#define SIDL_long__python_clone_array_RETURN PyObject *
#define SIDL_long__python_clone_array_PROTO \
  (struct SIDL_long__array *array)

#define SIDL_long__python_deleteReference_array_NUM 47
#define SIDL_long__python_deleteReference_array_RETURN void
#define SIDL_long__python_deleteReference_array_PROTO \
  (struct SIDL_long__array *array)


#define SIDL_opaque__borrow_python_array_NUM 48
#define SIDL_opaque__borrow_python_array_RETURN int
#define SIDL_opaque__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_opaque__array **array)

#define SIDL_opaque__clone_python_array_column_NUM 49
#define SIDL_opaque__clone_python_array_column_RETURN int
#define SIDL_opaque__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_opaque__array **array)

#define SIDL_opaque__clone_python_array_row_NUM 50
#define SIDL_opaque__clone_python_array_row_RETURN int
#define SIDL_opaque__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_opaque__array **array)

#define SIDL_opaque__python_borrow_array_NUM 51
#define SIDL_opaque__python_borrow_array_RETURN PyObject *
#define SIDL_opaque__python_borrow_array_PROTO \
  (struct SIDL_opaque__array *array)

#define SIDL_opaque__python_clone_array_NUM 52
#define SIDL_opaque__python_clone_array_RETURN PyObject *
#define SIDL_opaque__python_clone_array_PROTO \
  (struct SIDL_opaque__array *array)

#define SIDL_opaque__python_deleteReference_array_NUM 53
#define SIDL_opaque__python_deleteReference_array_RETURN void
#define SIDL_opaque__python_deleteReference_array_PROTO \
  (struct SIDL_opaque__array *array)


#define SIDL_string__borrow_python_array_NUM 54
#define SIDL_string__borrow_python_array_RETURN int
#define SIDL_string__borrow_python_array_PROTO \
  (PyObject *obj, struct SIDL_string__array **array)

#define SIDL_string__clone_python_array_column_NUM 55
#define SIDL_string__clone_python_array_column_RETURN int
#define SIDL_string__clone_python_array_column_PROTO \
  (PyObject *obj, struct SIDL_string__array **array)

#define SIDL_string__clone_python_array_row_NUM 56
#define SIDL_string__clone_python_array_row_RETURN int
#define SIDL_string__clone_python_array_row_PROTO \
  (PyObject *obj, struct SIDL_string__array **array)

#define SIDL_string__python_borrow_array_NUM 57
#define SIDL_string__python_borrow_array_RETURN PyObject *
#define SIDL_string__python_borrow_array_PROTO \
  (struct SIDL_string__array *array)

#define SIDL_string__python_clone_array_NUM 58
#define SIDL_string__python_clone_array_RETURN PyObject *
#define SIDL_string__python_clone_array_PROTO \
  (struct SIDL_string__array *array)

#define SIDL_string__python_deleteReference_array_NUM 59
#define SIDL_string__python_deleteReference_array_RETURN void
#define SIDL_string__python_deleteReference_array_PROTO \
  (struct SIDL_string__array *array)

#define SIDL_array__convert_python_NUM 60
#define SIDL_array__convert_python_RETURN int
#define SIDL_array__convert_python_PROTO \
  (PyObject *pya_src, const int32_t dimen, void *sidl_dest, \
   SIDL_array_set_py_elem setfunc)

#define SIDL_array__convert_sidl_NUM 61
#define SIDL_array__convert_sidl_RETURN int
#define SIDL_array__convert_sidl_PROTO \
  (PyObject *pya_dest, \
   const int32_t dimen, \
   int32_t lower[], \
   const int32_t upper[], \
   const int32_t numelem[], \
   void *sidl_src, \
   SIDL_array_get_py_elem getfunc)

#define SIDL_array__extract_python_info_NUM 62
#define SIDL_array__extract_python_info_RETURN int
#define SIDL_array__extract_python_info_PROTO \
  (PyObject *pya, int32_t *dimension, int32_t **lower, int32_t **upper,\
   int32_t **stride)

#define SIDLPyArrays_API_pointers 63


#ifdef SIDLPyArrays_MODULE

static SIDL_bool__borrow_python_array_RETURN
SIDL_bool__borrow_python_array SIDL_bool__borrow_python_array_PROTO;

static SIDL_bool__clone_python_array_column_RETURN
SIDL_bool__clone_python_array_column SIDL_bool__clone_python_array_column_PROTO;

static SIDL_bool__clone_python_array_row_RETURN
SIDL_bool__clone_python_array_row SIDL_bool__clone_python_array_row_PROTO;

static SIDL_bool__python_borrow_array_RETURN
SIDL_bool__python_borrow_array SIDL_bool__python_borrow_array_PROTO;

static SIDL_bool__python_clone_array_RETURN
SIDL_bool__python_clone_array SIDL_bool__python_clone_array_PROTO;

static SIDL_char__borrow_python_array_RETURN
SIDL_char__borrow_python_array SIDL_char__borrow_python_array_PROTO;

static SIDL_char__clone_python_array_column_RETURN
SIDL_char__clone_python_array_column SIDL_char__clone_python_array_column_PROTO;

static SIDL_char__clone_python_array_row_RETURN
SIDL_char__clone_python_array_row SIDL_char__clone_python_array_row_PROTO;

static SIDL_char__python_borrow_array_RETURN
SIDL_char__python_borrow_array SIDL_char__python_borrow_array_PROTO;

static SIDL_char__python_clone_array_RETURN
SIDL_char__python_clone_array SIDL_char__python_clone_array_PROTO;

static SIDL_dcomplex__borrow_python_array_RETURN
SIDL_dcomplex__borrow_python_array SIDL_dcomplex__borrow_python_array_PROTO;

static SIDL_dcomplex__clone_python_array_column_RETURN
SIDL_dcomplex__clone_python_array_column SIDL_dcomplex__clone_python_array_column_PROTO;

static SIDL_dcomplex__clone_python_array_row_RETURN
SIDL_dcomplex__clone_python_array_row SIDL_dcomplex__clone_python_array_row_PROTO;

static SIDL_dcomplex__python_borrow_array_RETURN
SIDL_dcomplex__python_borrow_array SIDL_dcomplex__python_borrow_array_PROTO;

static SIDL_dcomplex__python_clone_array_RETURN
SIDL_dcomplex__python_clone_array SIDL_dcomplex__python_clone_array_PROTO;

static SIDL_double__borrow_python_array_RETURN
SIDL_double__borrow_python_array SIDL_double__borrow_python_array_PROTO;

static SIDL_double__clone_python_array_column_RETURN
SIDL_double__clone_python_array_column SIDL_double__clone_python_array_column_PROTO;

static SIDL_double__clone_python_array_row_RETURN
SIDL_double__clone_python_array_row SIDL_double__clone_python_array_row_PROTO;

static SIDL_double__python_borrow_array_RETURN
SIDL_double__python_borrow_array SIDL_double__python_borrow_array_PROTO;

static SIDL_double__python_clone_array_RETURN
SIDL_double__python_clone_array SIDL_double__python_clone_array_PROTO;

static SIDL_fcomplex__borrow_python_array_RETURN
SIDL_fcomplex__borrow_python_array SIDL_fcomplex__borrow_python_array_PROTO;

static SIDL_fcomplex__clone_python_array_column_RETURN
SIDL_fcomplex__clone_python_array_column SIDL_fcomplex__clone_python_array_column_PROTO;

static SIDL_fcomplex__clone_python_array_row_RETURN
SIDL_fcomplex__clone_python_array_row SIDL_fcomplex__clone_python_array_row_PROTO;

static SIDL_fcomplex__python_borrow_array_RETURN
SIDL_fcomplex__python_borrow_array SIDL_fcomplex__python_borrow_array_PROTO;

static SIDL_fcomplex__python_clone_array_RETURN
SIDL_fcomplex__python_clone_array SIDL_fcomplex__python_clone_array_PROTO;

static SIDL_float__borrow_python_array_RETURN
SIDL_float__borrow_python_array SIDL_float__borrow_python_array_PROTO;

static SIDL_float__clone_python_array_column_RETURN
SIDL_float__clone_python_array_column SIDL_float__clone_python_array_column_PROTO;

static SIDL_float__clone_python_array_row_RETURN
SIDL_float__clone_python_array_row SIDL_float__clone_python_array_row_PROTO;

static SIDL_float__python_borrow_array_RETURN
SIDL_float__python_borrow_array SIDL_float__python_borrow_array_PROTO;

static SIDL_float__python_clone_array_RETURN
SIDL_float__python_clone_array SIDL_float__python_clone_array_PROTO;

static SIDL_int__borrow_python_array_RETURN
SIDL_int__borrow_python_array SIDL_int__borrow_python_array_PROTO;

static SIDL_int__clone_python_array_column_RETURN
SIDL_int__clone_python_array_column SIDL_int__clone_python_array_column_PROTO;

static SIDL_int__clone_python_array_row_RETURN
SIDL_int__clone_python_array_row SIDL_int__clone_python_array_row_PROTO;

static SIDL_int__python_borrow_array_RETURN
SIDL_int__python_borrow_array SIDL_int__python_borrow_array_PROTO;

static SIDL_int__python_clone_array_RETURN
SIDL_int__python_clone_array SIDL_int__python_clone_array_PROTO;

static SIDL_long__borrow_python_array_RETURN
SIDL_long__borrow_python_array SIDL_long__borrow_python_array_PROTO;

static SIDL_long__clone_python_array_column_RETURN
SIDL_long__clone_python_array_column SIDL_long__clone_python_array_column_PROTO;

static SIDL_long__clone_python_array_row_RETURN
SIDL_long__clone_python_array_row SIDL_long__clone_python_array_row_PROTO;

static SIDL_long__python_borrow_array_RETURN
SIDL_long__python_borrow_array SIDL_long__python_borrow_array_PROTO;

static SIDL_long__python_clone_array_RETURN
SIDL_long__python_clone_array SIDL_long__python_clone_array_PROTO;

static SIDL_opaque__borrow_python_array_RETURN
SIDL_opaque__borrow_python_array SIDL_opaque__borrow_python_array_PROTO;

static SIDL_opaque__clone_python_array_column_RETURN
SIDL_opaque__clone_python_array_column SIDL_opaque__clone_python_array_column_PROTO;

static SIDL_opaque__clone_python_array_row_RETURN
SIDL_opaque__clone_python_array_row SIDL_opaque__clone_python_array_row_PROTO;

static SIDL_opaque__python_borrow_array_RETURN
SIDL_opaque__python_borrow_array SIDL_opaque__python_borrow_array_PROTO;

static SIDL_opaque__python_clone_array_RETURN
SIDL_opaque__python_clone_array SIDL_opaque__python_clone_array_PROTO;

static SIDL_string__borrow_python_array_RETURN
SIDL_string__borrow_python_array SIDL_string__borrow_python_array_PROTO;

static SIDL_string__clone_python_array_column_RETURN
SIDL_string__clone_python_array_column SIDL_string__clone_python_array_column_PROTO;

static SIDL_string__clone_python_array_row_RETURN
SIDL_string__clone_python_array_row SIDL_string__clone_python_array_row_PROTO;

static SIDL_string__python_borrow_array_RETURN
SIDL_string__python_borrow_array SIDL_string__python_borrow_array_PROTO;

static SIDL_string__python_clone_array_RETURN
SIDL_string__python_clone_array SIDL_string__python_clone_array_PROTO;

static SIDL_array__convert_python_RETURN
SIDL_array__convert_python SIDL_array__convert_python_PROTO;

static SIDL_array__convert_sidl_RETURN
SIDL_array__convert_sidl SIDL_array__convert_sidl_PROTO;

static SIDL_array__extract_python_info_RETURN
SIDL_array__extract_python_info SIDL_array__extract_python_info_PROTO;

#else /* SIDLPyArrays_MODULE */

static void **SIDLPyArrays_API;

#define SIDL_bool__borrow_python_array  \
(*( SIDL_bool__borrow_python_array_RETURN (*) \
   SIDL_bool__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_bool__borrow_python_array_NUM])

#define SIDL_bool__clone_python_array_column  \
(*( SIDL_bool__clone_python_array_column_RETURN (*) \
   SIDL_bool__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_bool__clone_python_array_column_NUM])

#define SIDL_bool__clone_python_array_row  \
(*( SIDL_bool__clone_python_array_row_RETURN (*) \
   SIDL_bool__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_bool__clone_python_array_row_NUM])

#define SIDL_bool__python_borrow_array  \
(*( SIDL_bool__python_borrow_array_RETURN (*) \
   SIDL_bool__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_bool__python_borrow_array_NUM])

#define SIDL_bool__python_clone_array  \
(*( SIDL_bool__python_clone_array_RETURN (*) \
   SIDL_bool__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_bool__python_clone_array_NUM])

#define SIDL_bool__python_deleteReference_array  \
(*( SIDL_bool__python_deleteReference_array_RETURN (*) \
   SIDL_bool__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_bool__python_deleteReference_array_NUM])

#define SIDL_char__borrow_python_array  \
(*( SIDL_char__borrow_python_array_RETURN (*) \
   SIDL_char__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_char__borrow_python_array_NUM])

#define SIDL_char__clone_python_array_column  \
(*( SIDL_char__clone_python_array_column_RETURN (*) \
   SIDL_char__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_char__clone_python_array_column_NUM])

#define SIDL_char__clone_python_array_row  \
(*( SIDL_char__clone_python_array_row_RETURN (*) \
   SIDL_char__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_char__clone_python_array_row_NUM])

#define SIDL_char__python_borrow_array  \
(*( SIDL_char__python_borrow_array_RETURN (*) \
   SIDL_char__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_char__python_borrow_array_NUM])

#define SIDL_char__python_clone_array  \
(*( SIDL_char__python_clone_array_RETURN (*) \
   SIDL_char__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_char__python_clone_array_NUM])

#define SIDL_char__python_deleteReference_array  \
(*( SIDL_char__python_deleteReference_array_RETURN (*) \
   SIDL_char__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_char__python_deleteReference_array_NUM])

#define SIDL_dcomplex__borrow_python_array  \
(*( SIDL_dcomplex__borrow_python_array_RETURN (*) \
   SIDL_dcomplex__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_dcomplex__borrow_python_array_NUM])

#define SIDL_dcomplex__clone_python_array_column  \
(*( SIDL_dcomplex__clone_python_array_column_RETURN (*) \
   SIDL_dcomplex__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_dcomplex__clone_python_array_column_NUM])

#define SIDL_dcomplex__clone_python_array_row  \
(*( SIDL_dcomplex__clone_python_array_row_RETURN (*) \
   SIDL_dcomplex__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_dcomplex__clone_python_array_row_NUM])

#define SIDL_dcomplex__python_borrow_array  \
(*( SIDL_dcomplex__python_borrow_array_RETURN (*) \
   SIDL_dcomplex__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_dcomplex__python_borrow_array_NUM])

#define SIDL_dcomplex__python_clone_array  \
(*( SIDL_dcomplex__python_clone_array_RETURN (*) \
   SIDL_dcomplex__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_dcomplex__python_clone_array_NUM])

#define SIDL_dcomplex__python_deleteReference_array  \
(*( SIDL_dcomplex__python_deleteReference_array_RETURN (*) \
   SIDL_dcomplex__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_dcomplex__python_deleteReference_array_NUM])

#define SIDL_double__borrow_python_array  \
(*( SIDL_double__borrow_python_array_RETURN (*) \
   SIDL_double__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_double__borrow_python_array_NUM])

#define SIDL_double__clone_python_array_column  \
(*( SIDL_double__clone_python_array_column_RETURN (*) \
   SIDL_double__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_double__clone_python_array_column_NUM])

#define SIDL_double__clone_python_array_row  \
(*( SIDL_double__clone_python_array_row_RETURN (*) \
   SIDL_double__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_double__clone_python_array_row_NUM])

#define SIDL_double__python_borrow_array  \
(*( SIDL_double__python_borrow_array_RETURN (*) \
   SIDL_double__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_double__python_borrow_array_NUM])

#define SIDL_double__python_clone_array  \
(*( SIDL_double__python_clone_array_RETURN (*) \
   SIDL_double__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_double__python_clone_array_NUM])

#define SIDL_double__python_deleteReference_array  \
(*( SIDL_double__python_deleteReference_array_RETURN (*) \
   SIDL_double__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_double__python_deleteReference_array_NUM])

#define SIDL_fcomplex__borrow_python_array  \
(*( SIDL_fcomplex__borrow_python_array_RETURN (*) \
   SIDL_fcomplex__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_fcomplex__borrow_python_array_NUM])

#define SIDL_fcomplex__clone_python_array_column  \
(*( SIDL_fcomplex__clone_python_array_column_RETURN (*) \
   SIDL_fcomplex__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_fcomplex__clone_python_array_column_NUM])

#define SIDL_fcomplex__clone_python_array_row  \
(*( SIDL_fcomplex__clone_python_array_row_RETURN (*) \
   SIDL_fcomplex__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_fcomplex__clone_python_array_row_NUM])

#define SIDL_fcomplex__python_borrow_array  \
(*( SIDL_fcomplex__python_borrow_array_RETURN (*) \
   SIDL_fcomplex__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_fcomplex__python_borrow_array_NUM])

#define SIDL_fcomplex__python_clone_array  \
(*( SIDL_fcomplex__python_clone_array_RETURN (*) \
   SIDL_fcomplex__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_fcomplex__python_clone_array_NUM])

#define SIDL_fcomplex__python_deleteReference_array  \
(*( SIDL_fcomplex__python_deleteReference_array_RETURN (*) \
   SIDL_fcomplex__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_fcomplex__python_deleteReference_array_NUM])

#define SIDL_float__borrow_python_array  \
(*( SIDL_float__borrow_python_array_RETURN (*) \
   SIDL_float__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_float__borrow_python_array_NUM])

#define SIDL_float__clone_python_array_column  \
(*( SIDL_float__clone_python_array_column_RETURN (*) \
   SIDL_float__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_float__clone_python_array_column_NUM])

#define SIDL_float__clone_python_array_row  \
(*( SIDL_float__clone_python_array_row_RETURN (*) \
   SIDL_float__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_float__clone_python_array_row_NUM])

#define SIDL_float__python_borrow_array  \
(*( SIDL_float__python_borrow_array_RETURN (*) \
   SIDL_float__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_float__python_borrow_array_NUM])

#define SIDL_float__python_clone_array  \
(*( SIDL_float__python_clone_array_RETURN (*) \
   SIDL_float__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_float__python_clone_array_NUM])

#define SIDL_float__python_deleteReference_array  \
(*( SIDL_float__python_deleteReference_array_RETURN (*) \
   SIDL_float__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_float__python_deleteReference_array_NUM])

#define SIDL_int__borrow_python_array  \
(*( SIDL_int__borrow_python_array_RETURN (*) \
   SIDL_int__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_int__borrow_python_array_NUM])

#define SIDL_int__clone_python_array_column  \
(*( SIDL_int__clone_python_array_column_RETURN (*) \
   SIDL_int__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_int__clone_python_array_column_NUM])

#define SIDL_int__clone_python_array_row  \
(*( SIDL_int__clone_python_array_row_RETURN (*) \
   SIDL_int__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_int__clone_python_array_row_NUM])

#define SIDL_int__python_borrow_array  \
(*( SIDL_int__python_borrow_array_RETURN (*) \
   SIDL_int__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_int__python_borrow_array_NUM])

#define SIDL_int__python_clone_array  \
(*( SIDL_int__python_clone_array_RETURN (*) \
   SIDL_int__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_int__python_clone_array_NUM])

#define SIDL_int__python_deleteReference_array  \
(*( SIDL_int__python_deleteReference_array_RETURN (*) \
   SIDL_int__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_int__python_deleteReference_array_NUM])

#define SIDL_long__borrow_python_array  \
(*( SIDL_long__borrow_python_array_RETURN (*) \
   SIDL_long__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_long__borrow_python_array_NUM])

#define SIDL_long__clone_python_array_column  \
(*( SIDL_long__clone_python_array_column_RETURN (*) \
   SIDL_long__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_long__clone_python_array_column_NUM])

#define SIDL_long__clone_python_array_row  \
(*( SIDL_long__clone_python_array_row_RETURN (*) \
   SIDL_long__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_long__clone_python_array_row_NUM])

#define SIDL_long__python_borrow_array  \
(*( SIDL_long__python_borrow_array_RETURN (*) \
   SIDL_long__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_long__python_borrow_array_NUM])

#define SIDL_long__python_clone_array  \
(*( SIDL_long__python_clone_array_RETURN (*) \
   SIDL_long__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_long__python_clone_array_NUM])

#define SIDL_long__python_deleteReference_array  \
(*( SIDL_long__python_deleteReference_array_RETURN (*) \
   SIDL_long__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_long__python_deleteReference_array_NUM])

#define SIDL_opaque__borrow_python_array  \
(*( SIDL_opaque__borrow_python_array_RETURN (*) \
   SIDL_opaque__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_opaque__borrow_python_array_NUM])

#define SIDL_opaque__clone_python_array_column  \
(*( SIDL_opaque__clone_python_array_column_RETURN (*) \
   SIDL_opaque__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_opaque__clone_python_array_column_NUM])

#define SIDL_opaque__clone_python_array_row  \
(*( SIDL_opaque__clone_python_array_row_RETURN (*) \
   SIDL_opaque__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_opaque__clone_python_array_row_NUM])

#define SIDL_opaque__python_borrow_array  \
(*( SIDL_opaque__python_borrow_array_RETURN (*) \
   SIDL_opaque__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_opaque__python_borrow_array_NUM])

#define SIDL_opaque__python_clone_array  \
(*( SIDL_opaque__python_clone_array_RETURN (*) \
   SIDL_opaque__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_opaque__python_clone_array_NUM])

#define SIDL_opaque__python_deleteReference_array  \
(*( SIDL_opaque__python_deleteReference_array_RETURN (*) \
   SIDL_opaque__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_opaque__python_deleteReference_array_NUM])

#define SIDL_string__borrow_python_array  \
(*( SIDL_string__borrow_python_array_RETURN (*) \
   SIDL_string__borrow_python_array_PROTO) \
 SIDLPyArrays_API[SIDL_string__borrow_python_array_NUM])

#define SIDL_string__clone_python_array_column  \
(*( SIDL_string__clone_python_array_column_RETURN (*) \
   SIDL_string__clone_python_array_column_PROTO) \
 SIDLPyArrays_API[SIDL_string__clone_python_array_column_NUM])

#define SIDL_string__clone_python_array_row  \
(*( SIDL_string__clone_python_array_row_RETURN (*) \
   SIDL_string__clone_python_array_row_PROTO) \
 SIDLPyArrays_API[SIDL_string__clone_python_array_row_NUM])

#define SIDL_string__python_borrow_array  \
(*( SIDL_string__python_borrow_array_RETURN (*) \
   SIDL_string__python_borrow_array_PROTO) \
 SIDLPyArrays_API[SIDL_string__python_borrow_array_NUM])

#define SIDL_string__python_clone_array  \
(*( SIDL_string__python_clone_array_RETURN (*) \
   SIDL_string__python_clone_array_PROTO) \
 SIDLPyArrays_API[SIDL_string__python_clone_array_NUM])

#define SIDL_string__python_deleteReference_array  \
(*( SIDL_string__python_deleteReference_array_RETURN (*) \
   SIDL_string__python_deleteReference_array_PROTO) \
 SIDLPyArrays_API[SIDL_string__python_deleteReference_array_NUM])

#define SIDL_array__convert_python  \
(*( SIDL_array__convert_python_RETURN (*) \
   SIDL_array__convert_python_PROTO) \
 SIDLPyArrays_API[SIDL_array__convert_python_NUM])

#define SIDL_array__convert_sidl  \
(*( SIDL_array__convert_sidl_RETURN (*) \
   SIDL_array__convert_sidl_PROTO) \
 SIDLPyArrays_API[SIDL_array__convert_sidl_NUM])

#define SIDL_array__extract_python_info  \
(*( SIDL_array__extract_python_info_RETURN (*) \
   SIDL_array__extract_python_info_PROTO) \
 SIDLPyArrays_API[SIDL_array__extract_python_info_NUM])


#define import_SIDLPyArrays() \
{ \
  PyObject *module = PyImport_ImportModule("SIDLPyArrays"); \
  if (module != NULL) { \
    PyObject *module_dict = PyModule_GetDict(module); \
    PyObject *c_api_object = PyDict_GetItemString(module_dict, "_C_API"); \
    if (PyCObject_Check(c_api_object)) { \
       SIDLPyArrays_API = (void **)PyCObject_AsVoidPtr(c_api_object); \
    } \
    Py_DECREF(module); \
  } \
}

#endif /* SIDLPyArrays_MODULE */
#endif /* included_SIDLPyArrays_h */
