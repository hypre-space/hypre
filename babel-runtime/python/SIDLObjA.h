/*
 * File:        SIDLObjA.h
 * Package:     SIDL Python Object Adaptor
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name$
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: A Python C extension type to wrap up SIDL objects/interfaces
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

#ifndef included_SIDLObjA_h
#define included_SIDLObjA_h

/**
 * This header defines the external API for a Python
 * (c.f. http://www.python.org/ ) C extension types to expose
 * instances of SIDL classes and interfaces in Python. The C extension
 * type essentially wraps the SIDL class/interface in a Python object,
 * so it can be manipulated by a Python program.
 *
 * Here is a brief summary of the methods that external clients are
 * allowed to use: 
 *   SIDL_Object_Create    create a Python object to wrap an instance
 *                         of a SIDL class
 *   SIDL_Interface_Create create a Python object to wrap a SIDL
 *                         interface
 *   SIDL_Interface_Check  get abstract interface pointer or check if
 *                         SIDL interface 
 *   SIDL_Get_IOR          get the IOR (independent object
 *                         representation) pointer for an SIDL object
 *                         or interface. This can also be used to make
 *                         a quick check whether a Python object is
 *                         a wrapped SIDL object or interface.
 *   SIDL_Cast             see if a Python object/interface is
 *                         castable to a particular SIDL
 *                         object/interface.
 *   SIDL_Opaque_Create    create a PyCObject to wrap an SIDL
 *                         <code>opaque</code> type. The signature
 *                         of this method matches the signature
 *                         needed for a "O&" in a Py_BuildValue
 *                         call.
 *   SIDL_Opaque_Convert   convert a PyCObject to a void * with
 *                         a signature compatible for "O&" in
 *                         PyArg_ParseTuple call.
 *   SIDL_PyExceptionCast  a wrapper for SIDL_BaseException__cast.
 *                         It's defined to minimize the headers
 *                         that a Python module needs.
 *
 */

#include "babel_config.h"
#include <Python.h>

/* forward declarations of structures */
struct SIDL_BaseClass__object;
struct SIDL_BaseInterface__object;
struct SIDL_BaseException__object;

enum SIDL_PyRefTypes {
  SIDL_PyStealRef,
  SIDL_PyWeakRef,
  SIDL_PyNewRef
};

/* C API Functions */
#define SIDL_Interface_Create_NUM 0
#define SIDL_Interface_Create_RETURN PyObject *
#define SIDL_Interface_Create_PROTO \
  (struct SIDL_BaseInterface__object *abint, \
   PyMethodDef *methods, \
   int refType)

#define SIDL_Interface_Check_NUM 1
#define SIDL_Interface_Check_RETURN struct SIDL_BaseInterface__object *
#define SIDL_Interface_Check_PROTO (PyObject *obj)

#define SIDL_Object_Create_NUM 2
#define SIDL_Object_Create_RETURN PyObject *
#define SIDL_Object_Create_PROTO \
   (struct SIDL_BaseClass__object *ior, \
    PyMethodDef *methods, \
    int refType)

#define SIDL_Get_IOR_NUM 3
#define SIDL_Get_IOR_RETURN struct SIDL_BaseClass__object *
#define SIDL_Get_IOR_PROTO (PyObject *obj)

#define SIDL_Cast_NUM 4
#define SIDL_Cast_RETURN void *
#define SIDL_Cast_PROTO (PyObject *obj, char *name)

#define SIDL_Opaque_Create_NUM 5
#define SIDL_Opaque_Create_RETURN PyObject *
#define SIDL_Opaque_Create_PROTO (void *opaque_ptr)

#define SIDL_Opaque_Convert_NUM 6
#define SIDL_Opaque_Convert_RETURN int
#define SIDL_Opaque_Convert_PROTO (PyObject *obj, void **opaque_ptr)

#define SIDL_PyExceptionCast_NUM 7
#define SIDL_PyExceptionCast_RETURN void *
#define SIDL_PyExceptionCast_PROTO \
  (struct SIDL_BaseException__object *ex, const char *name)

#define SIDL_PyType_NUM 8
#define SIDL_PyType_RETURN PyObject *
#define SIDL_PyType_PROTO \
  (void)

#define SIDL_API_pointers 9

#ifdef SIDLOBJA_MODULE
/*
 * This branch should only be taken in the implementation of SIDLObjA.h in
 * SIDLObjA.c. No clients should define SIDLOBJA_MODULE!
 *
 */
static SIDL_Object_Create_RETURN
SIDL_Object_Create SIDL_Object_Create_PROTO;

static SIDL_Get_IOR_RETURN
SIDL_Get_IOR SIDL_Get_IOR_PROTO;

static SIDL_Interface_Create_RETURN
SIDL_Interface_Create SIDL_Interface_Create_PROTO;

static SIDL_Interface_Check_RETURN
SIDL_Interface_Check  SIDL_Interface_Check_PROTO;

static SIDL_Cast_RETURN
SIDL_Cast SIDL_Cast_PROTO;

static SIDL_Opaque_Create_RETURN
SIDL_Opaque_Create SIDL_Opaque_Create_PROTO;

static SIDL_Opaque_Convert_RETURN
SIDL_Opaque_Convert SIDL_Opaque_Convert_PROTO;

static SIDL_PyExceptionCast_RETURN
SIDL_PyExceptionCast SIDL_PyExceptionCast_PROTO;

static SIDL_PyType_RETURN
SIDL_PyType SIDL_PyType_PROTO;

#else
/*
 * This branch is the branch that clients should take
 *
 */

static void **SIDL_Object_Adaptor_API;

/**
 * PyObject *
 * SIDL_Object_Create(struct SIDL_BaseClass__object *ior,
 *                    PyMethodDef *methods,
 *                    int refType)
 *
 * This macro creates a wrapper object for a non-NULL SIDL ior.
 * If <code>methods</code> is NULL, it will return NULL and set
 * a Python AssertionError exception.
 * If <code>ior</code> is NULL, Py_None is returned.
 *
 * If <code>refType</code> is SIDL_PyStealRef, this takes ownership of one
 * reference to <code>ior</code>.  If <code>ior</code> is non-NULL and the
 * create fails for any reason, it will delete its reference to
 * <code>ior</code>. 
 *
 * If <code>refType</code> is SIDL_PyWeakRef, this function will borrow a
 * reference.  It will not increase or decrease the reference count of
 * <code>ior</code> under any circumstance (even when the Python object is
 * garbage collected). This behavior is needed primarily for server side
 * Python which provides a Python wrapper for its  own IOR pointer. 
 *
 * If <code>refType</code> is SIDL_PyNewRef, this function will increment
 * the reference count of <code>ior</code> if the wrapper can be created.
 * When the Python object is garbage collected, it will delete this
 * reference.
 */
#define SIDL_Object_Create \
(*(SIDL_Object_Create_RETURN (*)SIDL_Object_Create_PROTO) \
SIDL_Object_Adaptor_API[SIDL_Object_Create_NUM])

/**
 * struct SIDL_BaseClass__object *
 * SIDL_Get_IOR(PyObject *obj)
 * If obj is an instance of the SIDL object/interface C extension
 * class, return a non-NULL IOR pointer; otherwise, return NULL. This
 * is a safe and quick way to check if this is a wrapped SIDL object
 * or interface. 
 */
#define SIDL_Get_IOR \
(*(SIDL_Get_IOR_RETURN (*)SIDL_Get_IOR_PROTO) \
SIDL_Object_Adaptor_API[SIDL_Get_IOR_NUM])

/**
 * PyObject *
 * SIDL_Interface_Create(struct SIDL_BaseClass__object *abint,
 *                       PyMethodDef *methods,
 *                       int refType)
 *
 * This macro creates a wrapper object for a non-NULL SIDL abstract
 * interface. If methods is NULL, it will return NULL and set a 
 * Python AssertionError exception. If abint is NULL, Py_None is
 * returned.
 * 
 * If <code>refType</code> is SIDL_PyStealRef, this takes ownership of one
 * reference to <code>abint</code>.  If <code>abint</code> is non-NULL and
 * the create fails for any reason, it will delete its reference to
 * <code>abint</code>. 
 *
 * If <code>refType</code> is SIDL_PyWeakRef, this function will borrow a
 * reference.  It will not increase or decrease the reference count of
 * <code>abint</code> under any circumstance (even when the Python object is
 * garbage collected). This behavior is needed primarily for server side
 * Python which provides a Python wrapper for its  own IOR pointer. 
 *
 * If <code>refType</code> is SIDL_PyNewRef, this function will increment
 * the reference count of <code>abint</code> if the wrapper can be created.
 * When the Python object is garbage collected, it will delete this
 * reference.
 */
#define SIDL_Interface_Create \
(*(SIDL_Interface_Create_RETURN (*)SIDL_Interface_Create_PROTO) \
SIDL_Object_Adaptor_API[SIDL_Interface_Create_NUM])

/**
 * struct SIDL_BaseInterface__object *
 * SIDL_Interface_Check(PyObject *obj)
 * If obj is an instance of the SIDL object C extension class and it
 * wraps an interface, return a non-NULL IOR pointer; 
 * otherwise, return NULL.
 */
#define SIDL_Interface_Check \
(*(SIDL_Interface_Check_RETURN (*)SIDL_Interface_Check_PROTO) \
SIDL_Object_Adaptor_API[SIDL_Interface_Check_NUM])

/**
 * void *
 * SIDL_Cast(PyObject *obj, const char *typename)
 * If obj is an instance of the BABEL object C extension class and it is
 * castable to typename, return the IOR or interface pointer. The client
 * is expected to know to what the return value should be cast.
 * If obj is not an instance of a SIDL object C extension class or it
 * is not castable to typename, NULL is returned.
 */
#define SIDL_Cast \
(*(SIDL_Cast_RETURN (*)SIDL_Cast_PROTO) \
SIDL_Object_Adaptor_API[SIDL_Cast_NUM])

/**
 * PyObject *
 * SIDL_Opaque_Create(void *opaque_ptr)
 * Create a PyCObject to hold this SIDL opaque pointer. This function
 * is intended for use in a Py_BuildValue call associated with a "O&"
 * format element.
 */
#define SIDL_Opaque_Create \
(*(SIDL_Opaque_Create_RETURN (*)SIDL_Opaque_Create_PROTO) \
SIDL_Object_Adaptor_API[SIDL_Opaque_Create_NUM])

/**
 * int
 * SIDL_Opaque_Convert(PyObject *obj, void **opaque_ptr)
 * If obj is a PyCObject, put the void * from the PyCObject into
 * *opaque_ptr. If this succeeds, 1 is returned. If obj is not a
 * PyCObject is not a PyCObject, 0 is returned.
 */
#define SIDL_Opaque_Convert \
(*(SIDL_Opaque_Convert_RETURN (*)SIDL_Opaque_Convert_PROTO) \
SIDL_Object_Adaptor_API[SIDL_Opaque_Convert_NUM])

/**
 * void *
 * SIDL_PyExceptionCast(struct SIDL_BaseException__object *obj,
 *                      const char *name)
 * This is a wrapper for the SIDL_BaseException__cast method.
 * This will try to case obj to type name.  A NULL return value
 * means the cast was unsuccessful; a non-NULL return value
 * means the cast was successful.
 * 
 * It's defined to minimize the headers that a Python module needs.
 */
#define SIDL_PyExceptionCast \
(*(SIDL_PyExceptionCast_RETURN (*)SIDL_PyExceptionCast_PROTO) \
SIDL_Object_Adaptor_API[SIDL_PyExceptionCast_NUM])

#define SIDL_PyType \
(*(SIDL_PyType_RETURN (*)SIDL_PyType_PROTO) \
SIDL_Object_Adaptor_API[SIDL_PyType_NUM])


#define import_SIDLObjA() \
{ \
  PyObject *module = PyImport_ImportModule("SIDLObjA"); \
  if (module != NULL) { \
    PyObject *module_dict = PyModule_GetDict(module); \
    PyObject *c_api_object = PyDict_GetItemString(module_dict, "_C_API"); \
    if (PyCObject_Check(c_api_object)) { \
       SIDL_Object_Adaptor_API = (void **)PyCObject_AsVoidPtr(c_api_object); \
    } \
    Py_DECREF(module); \
  } \
}
#endif

#endif /*  included_SIDLObjA_h */
