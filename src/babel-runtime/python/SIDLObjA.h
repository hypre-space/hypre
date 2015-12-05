/*
 * File:        sidlObjA.h
 * Package:     sidl Python Object Adaptor
 * Copyright:   (c) 2001 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.6 $
 * Date:        $Date: 2007/09/27 19:35:21 $
 * Description: A Python C extension type to wrap up sidl objects/interfaces
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

#ifndef included_sidlObjA_h
#define included_sidlObjA_h

/**
 * This header defines the external API for a Python
 * (c.f. http://www.python.org/ ) C extension types to expose
 * instances of sidl classes and interfaces in Python. The C extension
 * type essentially wraps the sidl class/interface in a Python object,
 * so it can be manipulated by a Python program.
 *
 * Here is a brief summary of the methods that external clients are
 * allowed to use: 
 *   sidl_Object_Init      initialize a Python object to wrap an instance
 *                         of a sidl class
 *   sidl_Get_IOR          get the IOR (independent object
 *                         representation) pointer for an sidl object
 *                         or interface. This can also be used to make
 *                         a quick check whether a Python object is
 *                         a wrapped sidl object or interface.
 *   sidl_Cast             see if a Python object/interface is
 *                         castable to a particular sidl
 *                         object/interface.
 *   sidl_Opaque_Create    create a PyCObject to wrap an sidl
 *                         <code>opaque</code> type. The signature
 *                         of this method matches the signature
 *                         needed for a "O&" in a Py_BuildValue
 *                         call.
 *   sidl_Opaque_Convert   convert a PyCObject to a void * with
 *                         a signature compatible for "O&" in
 *                         PyArg_ParseTuple call.
 *   sidl_PyExceptionCast  a wrapper for sidl_BaseInterface__cast.
 *                         It's defined to minimize the headers
 *                         that a Python module needs.
 *
 */

#include <Python.h>
#include "babel_config.h"

/* forward declarations of structures */
struct sidl_BaseInterface__object;

enum sidl_PyRefTypes {
  sidl_PyStealRef,
  sidl_PyWeakRef,
  sidl_PyNewRef
};

/**
 * This defines the Python object wrapper for a sidl object or interface.
 * This one Python extension type written in C can support an arbitrary
 * class or interface. The peculiarities of a particular class or interface
 * are stored in the d_methods member which defines what the object
 * can do.
 */
struct sidlPythonObject {
  PyObject_HEAD                 /* standard Python object header */

  /* sidl specific extensions */
  struct sidl_BaseInterface__object   *d_ior;
  int				       d_refType;
};

/**
 * This <code>typedef</code> is required by PyObject_NEW; otherwise,
 * it is unused.
 */
typedef struct sidlPythonObject SPObject;

/* C API Functions */
#define sidl_Object_Init_NUM 0
#define sidl_Object_Init_RETURN int
#define sidl_Object_Init_PROTO \
   (SPObject *sidlObject, \
    struct sidl_BaseInterface__object *ior, \
    int refType)

#define sidl_Get_IOR_NUM 1
#define sidl_Get_IOR_RETURN struct sidl_BaseInterface__object *
#define sidl_Get_IOR_PROTO (PyObject *obj)

#define sidl_Cast_NUM 2
#define sidl_Cast_RETURN void *
#define sidl_Cast_PROTO (PyObject *obj, char *name)

#define sidl_Opaque_Create_NUM 3
#define sidl_Opaque_Create_RETURN PyObject *
#define sidl_Opaque_Create_PROTO (void *opaque_ptr)

#define sidl_Opaque_Convert_NUM 4
#define sidl_Opaque_Convert_RETURN int
#define sidl_Opaque_Convert_PROTO (PyObject *obj, void **opaque_ptr)

#define sidl_PyExceptionCast_NUM 5
#define sidl_PyExceptionCast_RETURN void *
#define sidl_PyExceptionCast_PROTO \
  (struct sidl_BaseInterface__object *ex, const char *name)

#define sidl_PyType_NUM 6
#define sidl_PyType_RETURN PyTypeObject *
#define sidl_PyType_PROTO \
  (void)

#define sidl_Handle_Unexpected_NUM 7
#define sidl_Handle_Unexpected_RETURN struct sidl_BaseInterface__object *
#define sidl_Handle_Unexpected_PROTO \
  (const char *func)

#define sidl_AddTrace_NUM 8
#define sidl_AddTrace_RETURN void
#define sidl_AddTrace_PROTO \
  (PyObject *exc, const char *func)

#define sidl_API_pointers 9

#ifdef sidlOBJA_MODULE
/*
 * This branch should only be taken in the implementation of sidlObjA.h in
 * sidlObjA.c. No clients should define sidlOBJA_MODULE!
 *
 */
static sidl_Object_Init_RETURN
sidl_Object_Init sidl_Object_Init_PROTO;

static sidl_Get_IOR_RETURN
sidl_Get_IOR sidl_Get_IOR_PROTO;

static sidl_Cast_RETURN
sidl_Cast sidl_Cast_PROTO;

static sidl_Opaque_Create_RETURN
sidl_Opaque_Create sidl_Opaque_Create_PROTO;

static sidl_Opaque_Convert_RETURN
sidl_Opaque_Convert sidl_Opaque_Convert_PROTO;

static sidl_PyExceptionCast_RETURN
sidl_PyExceptionCast sidl_PyExceptionCast_PROTO;

static sidl_PyType_RETURN
sidl_PyType sidl_PyType_PROTO;

static sidl_Handle_Unexpected_RETURN
sidl_Handle_Unexpected sidl_Handle_Unexpected_PROTO;

static sidl_AddTrace_RETURN
sidl_AddTrace sidl_AddTrace_PROTO;

#else
/*
 * This branch is the branch that clients should take
 *
 */

static void **sidl_Object_Adaptor_API;

/**
 * PyObject *
 * sidl_Object_Init(struct sidl_BaseInterface__object *ior,
 *                    int refType)
 *
 * This macro creates a wrapper object for a non-NULL sidl ior.
 * If <code>methods</code> is NULL, it will return NULL and set
 * a Python AssertionError exception.
 * If <code>ior</code> is NULL, Py_None is returned.
 *
 * If <code>refType</code> is sidl_PyStealRef, this takes ownership of one
 * reference to <code>ior</code>.  If <code>ior</code> is non-NULL and the
 * create fails for any reason, it will delete its reference to
 * <code>ior</code>. 
 *
 * If <code>refType</code> is sidl_PyWeakRef, this function will borrow a
 * reference.  It will not increase or decrease the reference count of
 * <code>ior</code> under any circumstance (even when the Python object is
 * garbage collected). This behavior is needed primarily for server side
 * Python which provides a Python wrapper for its  own IOR pointer. 
 *
 * If <code>refType</code> is sidl_PyNewRef, this function will increment
 * the reference count of <code>ior</code> if the wrapper can be created.
 * When the Python object is garbage collected, it will delete this
 * reference.
 */
#define sidl_Object_Init \
(*(sidl_Object_Init_RETURN (*)sidl_Object_Init_PROTO) \
sidl_Object_Adaptor_API[sidl_Object_Init_NUM])

/**
 * struct sidl_BaseInterface__object *
 * sidl_Get_IOR(PyObject *obj)
 * If obj is an instance of the sidl object/interface C extension
 * class, return a non-NULL IOR pointer; otherwise, return NULL. This
 * is a safe and quick way to check if this is a wrapped sidl object
 * or interface. 
 */
#define sidl_Get_IOR \
(*(sidl_Get_IOR_RETURN (*)sidl_Get_IOR_PROTO) \
sidl_Object_Adaptor_API[sidl_Get_IOR_NUM])

/**
 * void *
 * sidl_Cast(PyObject *obj, const char *typename)
 * If obj is an instance of the BABEL object C extension class and it is
 * castable to typename, return the IOR or interface pointer. The client
 * is expected to know to what the return value should be cast.
 * If obj is not an instance of a sidl object C extension class or it
 * is not castable to typename, NULL is returned.
 */
#define sidl_Cast \
(*(sidl_Cast_RETURN (*)sidl_Cast_PROTO) \
sidl_Object_Adaptor_API[sidl_Cast_NUM])

/**
 * PyObject *
 * sidl_Opaque_Create(void *opaque_ptr)
 * Create a PyCObject to hold this sidl opaque pointer. This function
 * is intended for use in a Py_BuildValue call associated with a "O&"
 * format element.
 */
#define sidl_Opaque_Create \
(*(sidl_Opaque_Create_RETURN (*)sidl_Opaque_Create_PROTO) \
sidl_Object_Adaptor_API[sidl_Opaque_Create_NUM])

/**
 * int
 * sidl_Opaque_Convert(PyObject *obj, void **opaque_ptr)
 * If obj is a PyCObject, put the void * from the PyCObject into
 * *opaque_ptr. If this succeeds, 1 is returned. If obj is not a
 * PyCObject is not a PyCObject, 0 is returned.
 */
#define sidl_Opaque_Convert \
(*(sidl_Opaque_Convert_RETURN (*)sidl_Opaque_Convert_PROTO) \
sidl_Object_Adaptor_API[sidl_Opaque_Convert_NUM])

/**
 * void *
 * sidl_PyExceptionCast(struct sidl_BaseInterface__object *obj,
 *                      const char *name)
 * This is a wrapper for the sidl_BaseInterface__cast method.
 * This will try to case obj to type name.  A NULL return value
 * means the cast was unsuccessful; a non-NULL return value
 * means the cast was successful.
 * 
 * It's defined to minimize the headers that a Python module needs.
 */
#define sidl_PyExceptionCast \
(*(sidl_PyExceptionCast_RETURN (*)sidl_PyExceptionCast_PROTO) \
sidl_Object_Adaptor_API[sidl_PyExceptionCast_NUM])

#define sidl_PyType \
(*(sidl_PyType_RETURN (*)sidl_PyType_PROTO) \
sidl_Object_Adaptor_API[sidl_PyType_NUM])

#define sidl_Handle_Unexpected \
(*(sidl_Handle_Unexpected_RETURN (*)sidl_Handle_Unexpected_PROTO) \
sidl_Object_Adaptor_API[sidl_Handle_Unexpected_NUM])

#define sidl_AddTrace \
(*(sidl_AddTrace_RETURN (*)sidl_AddTrace_PROTO) \
sidl_Object_Adaptor_API[sidl_AddTrace_NUM])


#define import_SIDLObjA() \
{ \
  PyObject *module = PyImport_ImportModule("sidlObjA"); \
  if (module != NULL) { \
    PyObject *module_dict = PyModule_GetDict(module); \
    PyObject *c_api_object = PyDict_GetItemString(module_dict, "_C_API"); \
    if (PyCObject_Check(c_api_object)) { \
       sidl_Object_Adaptor_API = (void **)PyCObject_AsVoidPtr(c_api_object); \
    } \
    else { fprintf(stderr, "babel: import_sidlObjA failed to lookup _C_API (%p).\n", c_api_object); }\
    Py_DECREF(module); \
  } \
  else { fprintf(stderr, "babel: import_sidlObjA failed to import its module.\n"); }\
}
#endif

#endif /*  included_sidlObjA_h */
