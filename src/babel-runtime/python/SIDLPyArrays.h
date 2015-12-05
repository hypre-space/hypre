/*
 * File:        sidlPyArrays.h
 * Copyright:   (c) 2001 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.5 $
 * Date:        $Date: 2006/08/29 22:29:27 $
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

#ifndef included_sidlPyArrays_h
#define included_sidlPyArrays_h

/*
 * This header defines the external API for a Python C extension module.
 * It provides conversion routines between sidl arrays and Numerical
 * Python arrays.
 * 
 * These functions apply to all arrays
 *
 * This is a Python wrapper function for sidl__array_deleteRef.
 * void
 * sidl_python_deleteRef_array(struct sidl__array *array);
 *
 * Attempt to convert a sidl array into a Python object.
 * This method will create a Python object that borrows data from
 * the sidl array if it's possible; otherwise, it will create
 * an independent copy of the array. This function is designed
 * to be used in a Py_BuildValue call and a format string entry of "O&".
 * Return value: NULL means the conversion failed; otherwise,
 *               the conversion was successful.
 * PyObject *
 * sidl_python_borrow_array(struct sidl__array *array);
 *
 *
 * Attempt to convert a sidl array into an independent Python
 * object. This will create an independent Python object regardless
 * of whether borrowing is possible or not. This function is designed
 * to be used in a Py_BuildValue call and a format string entry of "O&".
 * Return value: NULL means the conversion failed; otherwise,
 *               the conversion was successful.
 * PyObject *
 * sidl_python_clone_array(struct sidl__array *array);
 *
 * Copy from src to dest if they're the same type.
 * void
 * sidl_python_copy(struct sidl__array *src, struct sidl__array *dest);
 *
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
 * there are three methods related to converting python arrays into
 * sidl arrays.  For each type x, the method names and meanings are
 * as follows: 
 *
 *
 * Attempt to convert a Python object into a sidl array of x's. This
 * method will attempt to borrow data from the Python object if it's
 * possible; if borrowing is not possible, it will create an independent
 * copy. This function is designed for use as a convert function in a
 * PyArg_ParseTuple call with a format string entry of "O&".
 * Return value: 1 is returned if the conversion was successful; otherwise,
 *               0 is returned.
 * int
 * sidl_x__borrow_python_array(PyObject *obj,
 *                             struct sidl_x__array **array);
 *
 *
 * Attempt to convert a Python object into an independent, column-ordered
 * sidl array of x's.  This will always copy data from the Python object
 * unless the Python object is actually borrowing data from a SIDL array.
 * This function is designed for use as a convert
 * function in a PyArg_ParseTuple call with a format string entry of "O&".
 * Return value: 1 is returned if the conversion was successful; otherwise,
 *               0 is returned.
 * int
 * sidl_x__clone_python_array_column(PyObject *obj,
 *                                   struct sidl_x__array **array);
 *
 *
 * Attempt to convert a Python object into an independent, row-ordered
 * sidl array of x's.  This will always copy data from the Python object
 * unless the Python object is actually borrowing data from a SIDL array.
 * This function is designed for use as a convert
 * function in a PyArg_ParseTuple call with a format string entry of "O&".
 * Return value: 1 is returned if the conversion was successful; otherwise,
 *               0 is returned.
 * int
 * sidl_x__clone_python_array_row(PyObject *obj,
 *                                struct sidl_x__array **array);
 *
 * Attempt to convert a Python object into an independent
 * sidl array of x's.  This will always copy data from the Python object
 * unless the Python object is actually borrowing data from a SIDL array.
 * This function is designed for use as a convert
 * function in a PyArg_ParseTuple call with a format string entry of "O&".
 * Return value: 1 is returned if the conversion was successful; otherwise,
 *               0 is returned.
 * int
 * sidl_x__clone_python_array(PyObject *obj,
 *                            struct sidl_x__array **array);
 *
 *
 *
 *
 *
 *
 * Here are the standalone functions.  These are useful for arrays of
 * object or interface pointers.
 *
 *
 *
 * This function will copy a Python array into a sidl array.  The
 * sidl array is assumed to have the same shape and index bounds
 * as the Python array.
 *
 * setfunc is called as follows:
 *     (*setfunc)(sidl_dest, ind, pyobj)
 *     sidl_dest is the void pointer passed to sidl_array__convert_python.
 *     ind       is a pointer to dimen integers
 *     pyobj     is a pointer to a Python object from pya_src.
 * If setfunc returns a true value, sidl_array__convert_python will 
 * immediately cease copying and return a false value.
 *
 * A true value is returned if the function was successful; otherwise,
 * a false value is returned to indicated that the operation failed.
 *
 * int
 * sidl_array__convert_python(PyArrayObject          *pya_src,
 *                            const int32_t           dimen,
 *                            void                   *sidl_dest,
 *                            sidl_array_set_py_elem  setfunc) 
 *
 *
 *
 * This function will copy a sidl array into a Python array.  The
 * Python array is assumes to have the same shape as the sidl array.
 * The Python array need not have index bounds.
 *
 * getfunc is called as follows:
 *     (*getfunc)(sidl_dest, ind, pyobj)
 *     sidl_src  is the void pointer passed to sidl_array__convert_sidl.
 *     ind       is a pointer to dimen integers
 *     pyobj     is a pointer to a pointer to a Python object from pya_dest.
 *               This is where the return is stored
 * If setfunc returns a true value, sidl_array__convert_sidl will 
 * immediately cease copying and return a false value.
 *
 * A true value is returned if the function was successful; otherwise,
 * a false value is returned to indicated that the operation failed.
 *
 * int
 * sidl_array__convert_sidl(PyArrayObject          *pya_dest,
 *                          const int32_t           dimen,
 *                          int32_t                 lower[],
 *                          const int32_t           upper[],
 *                          const int32_t           numelem[],
 *                          void                   *sidl_src,
 *                          sidl_array_get_py_elem  getfunc) 
 *
 *
 *
 *
 * Extract the dimension information from a Numerical Python array
 * into a form that sidl can use.
 *      dimension   on entry a pointer to a single integer
 *      lower       an array of SIDL_MAX_ARRAY_DIMENSION integers
 *      upper       an array of SIDL_MAX_ARRAY_DIMENSION integers
 *      stride      an array of SIDL_MAX_ARRAY_DIMENSION integers
 * int
 * sidl_array__extract_python_info(PyArrayObject  *pya,
 *                                 int32_t        *dimension,
 *                                 int32_t       **lower,
 *                                 int32_t       **upper,
 *                                 int32_t       **stride)
 */

#include <Python.h>
#include "babel_config.h"
#ifndef included_sidlArray_h
#include "sidlArray.h"
#endif
#ifndef included_sidlType_h
#include "sidlType.h"
#endif

/*
 * Forward declarations of array types.
 */
struct sidl_bool__array;
struct sidl_char__array;
struct sidl_dcomplex__array;
struct sidl_double__array;
struct sidl_fcomplex__array;
struct sidl_float__array;
struct sidl_int__array;
struct sidl_long__array;
struct sidl_opaque__array;
struct sidl_string__array;
struct sidl_BaseInterface__array;

typedef int (*sidl_array_set_py_elem)(void *, const int32_t *, PyObject *);
typedef int (*sidl_array_get_py_elem)(void *, const int32_t *, PyObject **);

/*
 * Here we use the standard Python method for exposing an external
 * C API.  It's not particular pretty, but it works.
 */

#define sidl_python_deleteRef_array_NUM 0
#define sidl_python_deleteRef_array_RETURN void
#define sidl_python_deleteRef_array_PROTO \
  (struct sidl__array *array)

#define sidl_python_borrow_array_NUM 1
#define sidl_python_borrow_array_RETURN PyObject *
#define sidl_python_borrow_array_PROTO \
  (struct sidl__array *array)

#define sidl_python_clone_array_NUM 2
#define sidl_python_clone_array_RETURN PyObject *
#define sidl_python_clone_array_PROTO \
  (struct sidl__array *array)

#define sidl_generic_clone_python_array_NUM 3
#define sidl_generic_clone_python_array_RETURN int
#define sidl_generic_clone_python_array_PROTO \
  (PyObject *obj, struct sidl__array **array)

#define sidl_generic_borrow_python_array_NUM 4
#define sidl_generic_borrow_python_array_RETURN int
#define sidl_generic_borrow_python_array_PROTO \
  (PyObject *obj, struct sidl__array **array)

#define sidl_python_copy_NUM 5
#define sidl_python_copy_RETURN void
#define sidl_python_copy_PROTO \
  (const struct sidl__array *src, struct sidl__array *dest)


#define sidl_bool__borrow_python_array_NUM 6
#define sidl_bool__borrow_python_array_RETURN int
#define sidl_bool__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_bool__array **array)

#define sidl_bool__clone_python_array_column_NUM 7
#define sidl_bool__clone_python_array_column_RETURN int
#define sidl_bool__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_bool__array **array)

#define sidl_bool__clone_python_array_row_NUM 8
#define sidl_bool__clone_python_array_row_RETURN int
#define sidl_bool__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_bool__array **array)

#define sidl_bool__clone_python_array_NUM 9
#define sidl_bool__clone_python_array_RETURN int
#define sidl_bool__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_bool__array **array)



#define sidl_char__borrow_python_array_NUM 10
#define sidl_char__borrow_python_array_RETURN int
#define sidl_char__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_char__array **array)

#define sidl_char__clone_python_array_column_NUM 11
#define sidl_char__clone_python_array_column_RETURN int
#define sidl_char__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_char__array **array)

#define sidl_char__clone_python_array_row_NUM 12
#define sidl_char__clone_python_array_row_RETURN int
#define sidl_char__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_char__array **array)

#define sidl_char__clone_python_array_NUM 13
#define sidl_char__clone_python_array_RETURN int
#define sidl_char__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_char__array **array)

#define sidl_dcomplex__borrow_python_array_NUM 14
#define sidl_dcomplex__borrow_python_array_RETURN int
#define sidl_dcomplex__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_dcomplex__array **array)

#define sidl_dcomplex__clone_python_array_column_NUM 15
#define sidl_dcomplex__clone_python_array_column_RETURN int
#define sidl_dcomplex__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_dcomplex__array **array)

#define sidl_dcomplex__clone_python_array_row_NUM 16
#define sidl_dcomplex__clone_python_array_row_RETURN int
#define sidl_dcomplex__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_dcomplex__array **array)

#define sidl_dcomplex__clone_python_array_NUM 17
#define sidl_dcomplex__clone_python_array_RETURN int
#define sidl_dcomplex__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_dcomplex__array **array)


#define sidl_double__borrow_python_array_NUM 18
#define sidl_double__borrow_python_array_RETURN int
#define sidl_double__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_double__array **array)

#define sidl_double__clone_python_array_column_NUM 19
#define sidl_double__clone_python_array_column_RETURN int
#define sidl_double__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_double__array **array)

#define sidl_double__clone_python_array_row_NUM 20
#define sidl_double__clone_python_array_row_RETURN int
#define sidl_double__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_double__array **array)

#define sidl_double__clone_python_array_NUM 21
#define sidl_double__clone_python_array_RETURN int
#define sidl_double__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_double__array **array)



#define sidl_fcomplex__borrow_python_array_NUM 22
#define sidl_fcomplex__borrow_python_array_RETURN int
#define sidl_fcomplex__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_fcomplex__array **array)

#define sidl_fcomplex__clone_python_array_column_NUM 23
#define sidl_fcomplex__clone_python_array_column_RETURN int
#define sidl_fcomplex__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_fcomplex__array **array)

#define sidl_fcomplex__clone_python_array_row_NUM 24
#define sidl_fcomplex__clone_python_array_row_RETURN int
#define sidl_fcomplex__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_fcomplex__array **array)

#define sidl_fcomplex__clone_python_array_NUM 25
#define sidl_fcomplex__clone_python_array_RETURN int
#define sidl_fcomplex__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_fcomplex__array **array)



#define sidl_float__borrow_python_array_NUM 26
#define sidl_float__borrow_python_array_RETURN int
#define sidl_float__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_float__array **array)

#define sidl_float__clone_python_array_column_NUM 27
#define sidl_float__clone_python_array_column_RETURN int
#define sidl_float__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_float__array **array)

#define sidl_float__clone_python_array_row_NUM 28
#define sidl_float__clone_python_array_row_RETURN int
#define sidl_float__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_float__array **array)

#define sidl_float__clone_python_array_NUM 29
#define sidl_float__clone_python_array_RETURN int
#define sidl_float__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_float__array **array)


#define sidl_int__borrow_python_array_NUM 30
#define sidl_int__borrow_python_array_RETURN int
#define sidl_int__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_int__array **array)

#define sidl_int__clone_python_array_column_NUM 31
#define sidl_int__clone_python_array_column_RETURN int
#define sidl_int__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_int__array **array)

#define sidl_int__clone_python_array_row_NUM 32
#define sidl_int__clone_python_array_row_RETURN int
#define sidl_int__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_int__array **array)

#define sidl_int__clone_python_array_NUM 33
#define sidl_int__clone_python_array_RETURN int
#define sidl_int__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_int__array **array)



#define sidl_long__borrow_python_array_NUM 34
#define sidl_long__borrow_python_array_RETURN int
#define sidl_long__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_long__array **array)

#define sidl_long__clone_python_array_column_NUM 35
#define sidl_long__clone_python_array_column_RETURN int
#define sidl_long__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_long__array **array)

#define sidl_long__clone_python_array_row_NUM 36
#define sidl_long__clone_python_array_row_RETURN int
#define sidl_long__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_long__array **array)

#define sidl_long__clone_python_array_NUM 37
#define sidl_long__clone_python_array_RETURN int
#define sidl_long__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_long__array **array)



#define sidl_opaque__borrow_python_array_NUM 38
#define sidl_opaque__borrow_python_array_RETURN int
#define sidl_opaque__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_opaque__array **array)

#define sidl_opaque__clone_python_array_column_NUM 39
#define sidl_opaque__clone_python_array_column_RETURN int
#define sidl_opaque__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_opaque__array **array)

#define sidl_opaque__clone_python_array_row_NUM 40
#define sidl_opaque__clone_python_array_row_RETURN int
#define sidl_opaque__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_opaque__array **array)

#define sidl_opaque__clone_python_array_NUM 41
#define sidl_opaque__clone_python_array_RETURN int
#define sidl_opaque__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_opaque__array **array)



#define sidl_string__borrow_python_array_NUM 42
#define sidl_string__borrow_python_array_RETURN int
#define sidl_string__borrow_python_array_PROTO \
  (PyObject *obj, struct sidl_string__array **array)

#define sidl_string__clone_python_array_column_NUM 43
#define sidl_string__clone_python_array_column_RETURN int
#define sidl_string__clone_python_array_column_PROTO \
  (PyObject *obj, struct sidl_string__array **array)

#define sidl_string__clone_python_array_row_NUM 44
#define sidl_string__clone_python_array_row_RETURN int
#define sidl_string__clone_python_array_row_PROTO \
  (PyObject *obj, struct sidl_string__array **array)

#define sidl_string__clone_python_array_NUM 45
#define sidl_string__clone_python_array_RETURN int
#define sidl_string__clone_python_array_PROTO \
  (PyObject *obj, struct sidl_string__array **array)


#define sidl_array__convert_python_NUM 46
#define sidl_array__convert_python_RETURN int
#define sidl_array__convert_python_PROTO \
  (PyObject *pya_src, const int32_t dimen, void *sidl_dest, \
   sidl_array_set_py_elem setfunc)

#define sidl_array__convert_sidl_NUM 47
#define sidl_array__convert_sidl_RETURN int
#define sidl_array__convert_sidl_PROTO \
  (PyObject *pya_dest, \
   const int32_t dimen, \
   int32_t lower[], \
   const int32_t upper[], \
   const int32_t numelem[], \
   void *sidl_src, \
   sidl_array_get_py_elem getfunc)

#define sidl_array__extract_python_info_NUM 48
#define sidl_array__extract_python_info_RETURN int
#define sidl_array__extract_python_info_PROTO \
  (PyObject *pya, int32_t *dimension, int32_t lower[], int32_t upper[],\
   int32_t stride[])

#define sidlPyArrays_API_pointers 49


#ifdef sidlPyArrays_MODULE

static sidl_python_borrow_array_RETURN
sidl_python_borrow_array sidl_python_borrow_array_PROTO;

static sidl_python_clone_array_RETURN
sidl_python_clone_array sidl_python_clone_array_PROTO;

static sidl_generic_clone_python_array_RETURN
sidl_generic_clone_python_array sidl_generic_clone_python_array_PROTO;

static sidl_generic_borrow_python_array_RETURN
sidl_generic_borrow_python_array sidl_generic_borrow_python_array_PROTO;

static sidl_python_copy_RETURN
sidl_python_copy sidl_python_copy_PROTO;


static sidl_bool__borrow_python_array_RETURN
sidl_bool__borrow_python_array sidl_bool__borrow_python_array_PROTO;

static sidl_bool__clone_python_array_column_RETURN
sidl_bool__clone_python_array_column sidl_bool__clone_python_array_column_PROTO;

static sidl_bool__clone_python_array_row_RETURN
sidl_bool__clone_python_array_row sidl_bool__clone_python_array_row_PROTO;

static sidl_bool__clone_python_array_RETURN
sidl_bool__clone_python_array sidl_bool__clone_python_array_PROTO;

static sidl_char__borrow_python_array_RETURN
sidl_char__borrow_python_array sidl_char__borrow_python_array_PROTO;

static sidl_char__clone_python_array_column_RETURN
sidl_char__clone_python_array_column sidl_char__clone_python_array_column_PROTO;
static sidl_char__clone_python_array_row_RETURN
sidl_char__clone_python_array_row sidl_char__clone_python_array_row_PROTO;
static sidl_char__clone_python_array_RETURN
sidl_char__clone_python_array sidl_char__clone_python_array_PROTO;

static sidl_dcomplex__borrow_python_array_RETURN
sidl_dcomplex__borrow_python_array sidl_dcomplex__borrow_python_array_PROTO;

static sidl_dcomplex__clone_python_array_column_RETURN
sidl_dcomplex__clone_python_array_column sidl_dcomplex__clone_python_array_column_PROTO;

static sidl_dcomplex__clone_python_array_row_RETURN
sidl_dcomplex__clone_python_array_row sidl_dcomplex__clone_python_array_row_PROTO;

static sidl_dcomplex__clone_python_array_RETURN
sidl_dcomplex__clone_python_array sidl_dcomplex__clone_python_array_PROTO;

static sidl_double__borrow_python_array_RETURN
sidl_double__borrow_python_array sidl_double__borrow_python_array_PROTO;

static sidl_double__clone_python_array_column_RETURN
sidl_double__clone_python_array_column sidl_double__clone_python_array_column_PROTO;

static sidl_double__clone_python_array_row_RETURN
sidl_double__clone_python_array_row sidl_double__clone_python_array_row_PROTO;

static sidl_double__clone_python_array_RETURN
sidl_double__clone_python_array sidl_double__clone_python_array_PROTO;

static sidl_fcomplex__borrow_python_array_RETURN
sidl_fcomplex__borrow_python_array sidl_fcomplex__borrow_python_array_PROTO;

static sidl_fcomplex__clone_python_array_column_RETURN
sidl_fcomplex__clone_python_array_column sidl_fcomplex__clone_python_array_column_PROTO;

static sidl_fcomplex__clone_python_array_row_RETURN
sidl_fcomplex__clone_python_array_row sidl_fcomplex__clone_python_array_row_PROTO;

static sidl_fcomplex__clone_python_array_RETURN
sidl_fcomplex__clone_python_array sidl_fcomplex__clone_python_array_PROTO;

static sidl_float__borrow_python_array_RETURN
sidl_float__borrow_python_array sidl_float__borrow_python_array_PROTO;

static sidl_float__clone_python_array_column_RETURN
sidl_float__clone_python_array_column sidl_float__clone_python_array_column_PROTO;

static sidl_float__clone_python_array_row_RETURN
sidl_float__clone_python_array_row sidl_float__clone_python_array_row_PROTO;

static sidl_float__clone_python_array_RETURN
sidl_float__clone_python_array sidl_float__clone_python_array_PROTO;

static sidl_int__borrow_python_array_RETURN
sidl_int__borrow_python_array sidl_int__borrow_python_array_PROTO;

static sidl_int__clone_python_array_column_RETURN
sidl_int__clone_python_array_column sidl_int__clone_python_array_column_PROTO;

static sidl_int__clone_python_array_row_RETURN
sidl_int__clone_python_array_row sidl_int__clone_python_array_row_PROTO;

static sidl_int__clone_python_array_RETURN
sidl_int__clone_python_array sidl_int__clone_python_array_PROTO;

static sidl_long__borrow_python_array_RETURN
sidl_long__borrow_python_array sidl_long__borrow_python_array_PROTO;

static sidl_long__clone_python_array_column_RETURN
sidl_long__clone_python_array_column sidl_long__clone_python_array_column_PROTO;

static sidl_long__clone_python_array_row_RETURN
sidl_long__clone_python_array_row sidl_long__clone_python_array_row_PROTO;

static sidl_long__clone_python_array_RETURN
sidl_long__clone_python_array sidl_long__clone_python_array_PROTO;

static sidl_opaque__borrow_python_array_RETURN
sidl_opaque__borrow_python_array sidl_opaque__borrow_python_array_PROTO;

static sidl_opaque__clone_python_array_column_RETURN
sidl_opaque__clone_python_array_column sidl_opaque__clone_python_array_column_PROTO;

static sidl_opaque__clone_python_array_row_RETURN
sidl_opaque__clone_python_array_row sidl_opaque__clone_python_array_row_PROTO;

static sidl_opaque__clone_python_array_RETURN
sidl_opaque__clone_python_array sidl_opaque__clone_python_array_PROTO;

static sidl_string__borrow_python_array_RETURN
sidl_string__borrow_python_array sidl_string__borrow_python_array_PROTO;

static sidl_string__clone_python_array_column_RETURN
sidl_string__clone_python_array_column sidl_string__clone_python_array_column_PROTO;

static sidl_string__clone_python_array_row_RETURN
sidl_string__clone_python_array_row sidl_string__clone_python_array_row_PROTO;

static sidl_string__clone_python_array_RETURN
sidl_string__clone_python_array sidl_string__clone_python_array_PROTO;

static sidl_array__convert_python_RETURN
sidl_array__convert_python sidl_array__convert_python_PROTO;

static sidl_array__convert_sidl_RETURN
sidl_array__convert_sidl sidl_array__convert_sidl_PROTO;

static sidl_array__extract_python_info_RETURN
sidl_array__extract_python_info sidl_array__extract_python_info_PROTO;

#else /* sidlPyArrays_MODULE */

static void **sidlPyArrays_API;

#define sidl_python_deleteRef_array  \
(*( sidl_python_deleteRef_array_RETURN (*) \
   sidl_python_deleteRef_array_PROTO) \
 sidlPyArrays_API[sidl_python_deleteRef_array_NUM])

#define sidl_python_borrow_array  \
(*( sidl_python_borrow_array_RETURN (*) \
   sidl_python_borrow_array_PROTO) \
 sidlPyArrays_API[sidl_python_borrow_array_NUM])

#define sidl_python_clone_array  \
(*( sidl_python_clone_array_RETURN (*) \
   sidl_python_clone_array_PROTO) \
 sidlPyArrays_API[sidl_python_clone_array_NUM])

#define sidl_generic_clone_python_array \
(*( sidl_generic_clone_python_array_RETURN (*) \
   sidl_generic_clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_generic_clone_python_array_NUM])

#define sidl_generic_borrow_python_array \
(*( sidl_generic_borrow_python_array_RETURN (*) \
   sidl_generic_borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_generic_borrow_python_array_NUM])

#define sidl_python_copy \
(*( sidl_python_copy_RETURN (*) \
   sidl_python_copy_PROTO) \
 sidlPyArrays_API[sidl_python_copy_NUM])

#define sidl_bool__borrow_python_array  \
(*( sidl_bool__borrow_python_array_RETURN (*) \
   sidl_bool__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_bool__borrow_python_array_NUM])

#define sidl_bool__clone_python_array_column  \
(*( sidl_bool__clone_python_array_column_RETURN (*) \
   sidl_bool__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_bool__clone_python_array_column_NUM])

#define sidl_bool__clone_python_array_row  \
(*( sidl_bool__clone_python_array_row_RETURN (*) \
   sidl_bool__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_bool__clone_python_array_row_NUM])

#define sidl_bool__clone_python_array  \
(*( sidl_bool__clone_python_array_RETURN (*) \
   sidl_bool__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_bool__clone_python_array_NUM])

#define sidl_char__borrow_python_array  \
(*( sidl_char__borrow_python_array_RETURN (*) \
   sidl_char__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_char__borrow_python_array_NUM])

#define sidl_char__clone_python_array_column  \
(*( sidl_char__clone_python_array_column_RETURN (*) \
   sidl_char__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_char__clone_python_array_column_NUM])

#define sidl_char__clone_python_array_row  \
(*( sidl_char__clone_python_array_row_RETURN (*) \
   sidl_char__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_char__clone_python_array_row_NUM])

#define sidl_char__clone_python_array  \
(*( sidl_char__clone_python_array_RETURN (*) \
   sidl_char__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_char__clone_python_array_NUM])

#define sidl_dcomplex__borrow_python_array  \
(*( sidl_dcomplex__borrow_python_array_RETURN (*) \
   sidl_dcomplex__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_dcomplex__borrow_python_array_NUM])

#define sidl_dcomplex__clone_python_array_column  \
(*( sidl_dcomplex__clone_python_array_column_RETURN (*) \
   sidl_dcomplex__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_dcomplex__clone_python_array_column_NUM])

#define sidl_dcomplex__clone_python_array_row  \
(*( sidl_dcomplex__clone_python_array_row_RETURN (*) \
   sidl_dcomplex__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_dcomplex__clone_python_array_row_NUM])

#define sidl_dcomplex__clone_python_array  \
(*( sidl_dcomplex__clone_python_array_RETURN (*) \
   sidl_dcomplex__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_dcomplex__clone_python_array_NUM])

#define sidl_double__borrow_python_array  \
(*( sidl_double__borrow_python_array_RETURN (*) \
   sidl_double__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_double__borrow_python_array_NUM])

#define sidl_double__clone_python_array_column  \
(*( sidl_double__clone_python_array_column_RETURN (*) \
   sidl_double__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_double__clone_python_array_column_NUM])

#define sidl_double__clone_python_array_row  \
(*( sidl_double__clone_python_array_row_RETURN (*) \
   sidl_double__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_double__clone_python_array_row_NUM])

#define sidl_double__clone_python_array  \
(*( sidl_double__clone_python_array_RETURN (*) \
   sidl_double__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_double__clone_python_array_NUM])

#define sidl_fcomplex__borrow_python_array  \
(*( sidl_fcomplex__borrow_python_array_RETURN (*) \
   sidl_fcomplex__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_fcomplex__borrow_python_array_NUM])

#define sidl_fcomplex__clone_python_array_column  \
(*( sidl_fcomplex__clone_python_array_column_RETURN (*) \
   sidl_fcomplex__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_fcomplex__clone_python_array_column_NUM])

#define sidl_fcomplex__clone_python_array_row  \
(*( sidl_fcomplex__clone_python_array_row_RETURN (*) \
   sidl_fcomplex__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_fcomplex__clone_python_array_row_NUM])

#define sidl_fcomplex__clone_python_array  \
(*( sidl_fcomplex__clone_python_array_RETURN (*) \
   sidl_fcomplex__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_fcomplex__clone_python_array_NUM])

#define sidl_float__borrow_python_array  \
(*( sidl_float__borrow_python_array_RETURN (*) \
   sidl_float__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_float__borrow_python_array_NUM])

#define sidl_float__clone_python_array_column  \
(*( sidl_float__clone_python_array_column_RETURN (*) \
   sidl_float__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_float__clone_python_array_column_NUM])

#define sidl_float__clone_python_array_row  \
(*( sidl_float__clone_python_array_row_RETURN (*) \
   sidl_float__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_float__clone_python_array_row_NUM])

#define sidl_float__clone_python_array  \
(*( sidl_float__clone_python_array_RETURN (*) \
   sidl_float__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_float__clone_python_array_NUM])

#define sidl_int__borrow_python_array  \
(*( sidl_int__borrow_python_array_RETURN (*) \
   sidl_int__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_int__borrow_python_array_NUM])

#define sidl_int__clone_python_array_column  \
(*( sidl_int__clone_python_array_column_RETURN (*) \
   sidl_int__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_int__clone_python_array_column_NUM])

#define sidl_int__clone_python_array_row  \
(*( sidl_int__clone_python_array_row_RETURN (*) \
   sidl_int__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_int__clone_python_array_row_NUM])

#define sidl_int__clone_python_array  \
(*( sidl_int__clone_python_array_RETURN (*) \
   sidl_int__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_int__clone_python_array_NUM])

#define sidl_long__borrow_python_array  \
(*( sidl_long__borrow_python_array_RETURN (*) \
   sidl_long__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_long__borrow_python_array_NUM])

#define sidl_long__clone_python_array_column  \
(*( sidl_long__clone_python_array_column_RETURN (*) \
   sidl_long__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_long__clone_python_array_column_NUM])

#define sidl_long__clone_python_array_row  \
(*( sidl_long__clone_python_array_row_RETURN (*) \
   sidl_long__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_long__clone_python_array_row_NUM])

#define sidl_long__clone_python_array  \
(*( sidl_long__clone_python_array_RETURN (*) \
   sidl_long__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_long__clone_python_array_NUM])

#define sidl_opaque__borrow_python_array  \
(*( sidl_opaque__borrow_python_array_RETURN (*) \
   sidl_opaque__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_opaque__borrow_python_array_NUM])

#define sidl_opaque__clone_python_array_column  \
(*( sidl_opaque__clone_python_array_column_RETURN (*) \
   sidl_opaque__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_opaque__clone_python_array_column_NUM])

#define sidl_opaque__clone_python_array_row  \
(*( sidl_opaque__clone_python_array_row_RETURN (*) \
   sidl_opaque__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_opaque__clone_python_array_row_NUM])

#define sidl_opaque__clone_python_array  \
(*( sidl_opaque__clone_python_array_RETURN (*) \
   sidl_opaque__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_opaque__clone_python_array_NUM])

#define sidl_string__borrow_python_array  \
(*( sidl_string__borrow_python_array_RETURN (*) \
   sidl_string__borrow_python_array_PROTO) \
 sidlPyArrays_API[sidl_string__borrow_python_array_NUM])

#define sidl_string__clone_python_array_column  \
(*( sidl_string__clone_python_array_column_RETURN (*) \
   sidl_string__clone_python_array_column_PROTO) \
 sidlPyArrays_API[sidl_string__clone_python_array_column_NUM])

#define sidl_string__clone_python_array_row  \
(*( sidl_string__clone_python_array_row_RETURN (*) \
   sidl_string__clone_python_array_row_PROTO) \
 sidlPyArrays_API[sidl_string__clone_python_array_row_NUM])

#define sidl_string__clone_python_array  \
(*( sidl_string__clone_python_array_RETURN (*) \
   sidl_string__clone_python_array_PROTO) \
 sidlPyArrays_API[sidl_string__clone_python_array_NUM])

#define sidl_array__convert_python  \
(*( sidl_array__convert_python_RETURN (*) \
   sidl_array__convert_python_PROTO) \
 sidlPyArrays_API[sidl_array__convert_python_NUM])

#define sidl_array__convert_sidl  \
(*( sidl_array__convert_sidl_RETURN (*) \
   sidl_array__convert_sidl_PROTO) \
 sidlPyArrays_API[sidl_array__convert_sidl_NUM])

#define sidl_array__extract_python_info  \
(*( sidl_array__extract_python_info_RETURN (*) \
   sidl_array__extract_python_info_PROTO) \
 sidlPyArrays_API[sidl_array__extract_python_info_NUM])


#define import_SIDLPyArrays() \
{ \
  PyObject *module = PyImport_ImportModule("sidlPyArrays"); \
  if (module != NULL) { \
    PyObject *module_dict = PyModule_GetDict(module); \
    PyObject *c_api_object = PyDict_GetItemString(module_dict, "_C_API"); \
    if (PyCObject_Check(c_api_object)) { \
       sidlPyArrays_API = (void **)PyCObject_AsVoidPtr(c_api_object); \
    } \
    else { fprintf(stderr, "babel: import_sidlPyArrays failed to lookup _C_API (%p).\n", c_api_object); }\
    Py_DECREF(module); \
  } \
  else { fprintf(stderr, "babel: import_sidlPyArrays failed to import its module.\n"); }\
}

#endif /* sidlPyArrays_MODULE */
#endif /* included_sidlPyArrays_h */
