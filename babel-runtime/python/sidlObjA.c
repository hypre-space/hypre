/*
 * File:        sidlObjA.c
 * Package:     sidl Python Object Adaptor
 * Copyright:   (c) 2001 The Regents of the University of California
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Python extension type written in C for sidl object/interface
 *
 * This is a Python extension type written in C to wrap instances of
 * sidl objects or interfaces in a Python object.  If this looks
 * mysterious to you, look at Programming Python and the BABEL
 * documentation. 
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

#define sidlOBJA_MODULE 1
#include "sidlObjA.h"
#include "sidl_BaseClass_IOR.h"
#include "sidl_BaseException.h"

#include <string.h>

/**
 * Provide a forward declaration of the Python type object used
 * by <code>sidlPythonObject</code>'s.
 */
staticforward PyTypeObject sidlPythonObjectType;

#define is_SIDLobject(v) (PyType_IsSubtype((v)->ob_type,&sidlPythonObjectType))

/*
 * Exported C API methods
 */

static sidl_Object_Init_RETURN
sidl_Object_Init sidl_Object_Init_PROTO {
  if (ior) {
    sidlObject->d_ior = ior;
    if (refType == sidl_PyNewRef) {
      (*ior->d_epv->f_addRef)(ior);
    }
    sidlObject->d_refType = refType;
    return 0;
  }
  else {
    PyErr_SetString(PyExc_AssertionError,
		    "sidl object has NULL IOR or vtable");
    return -1;
  }
}

static sidl_Get_IOR_RETURN
sidl_Get_IOR sidl_Get_IOR_PROTO {
  if (is_SIDLobject(obj)) {
    return ((struct sidlPythonObject *)obj)->d_ior;
  }
  return NULL;
}

static sidl_Cast_RETURN
sidl_Cast sidl_Cast_PROTO {
  struct sidl_BaseInterface__object *ior = sidl_Get_IOR(obj);
  return (ior) ? ((*(ior->d_epv->f__cast))(ior->d_object, name)) : NULL;
}

static sidl_Opaque_Create_RETURN
sidl_Opaque_Create sidl_Opaque_Create_PROTO {
  return PyCObject_FromVoidPtr(opaque_ptr, NULL);
}

static sidl_Opaque_Convert_RETURN
sidl_Opaque_Convert sidl_Opaque_Convert_PROTO {
  if (PyCObject_Check(obj)) {
    *opaque_ptr = PyCObject_AsVoidPtr(obj);
    return 1;
  }
  return 0;
}

static sidl_PyType_RETURN
sidl_PyType sidl_PyType_PROTO {
  return (PyTypeObject *)&sidlPythonObjectType;
}

static sidl_PyExceptionCast_RETURN
sidl_PyExceptionCast sidl_PyExceptionCast_PROTO {
  return sidl_BaseInterface__cast2(ex, name);
}

/*
 * BASIC OBJECT METHODS
 */

static PyObject *
spoa_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  SPObject *self;
  self = (SPObject *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->d_ior = NULL;
    self->d_refType = sidl_PyWeakRef;
  }
  return (PyObject *)self;
}

static void
spoa_self_destruct(struct sidlPythonObject *self)
{
  struct sidl_BaseInterface__object *ior = sidl_Get_IOR((PyObject *)self);
  /* remove Python's reference to this sidl object */
  if (self->d_refType != sidl_PyWeakRef) {
    (*(ior->d_epv->f_deleteRef))(ior->d_object);
  }
  self->d_ior = NULL;
  self->d_refType = sidl_PyWeakRef;
  self->ob_type->tp_free((PyObject *)self);
}

static PyTypeObject sidlPythonObjectType = {
  /* type header */
  PyObject_HEAD_INIT(NULL)
  0,
  "sidlObjA.BaseWrapper",
  sizeof(struct sidlPythonObject),
  0,

  /* standard methods */
  (destructor) spoa_self_destruct, /* call when reference count == 0 */
  (printfunc) 0,
  (getattrfunc) 0,
  (setattrfunc) 0,
  (cmpfunc) 0,
  (reprfunc) 0,
 
  /* type categories */
  (PyNumberMethods *) 0,
  (PySequenceMethods *) 0,
  (PyMappingMethods *) 0,

  /* more methods */
  (hashfunc)   0,
  (ternaryfunc) 0,
  (reprfunc)   0, /* to string */
  (getattrofunc) 0,
  (setattrofunc) 0,
  
  (PyBufferProcs *) 0,
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  /* documentation string */
  "This is a Python wrapper for a SIDL object or interface.",
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
  spoa_new,                     /* tp_new */
};

static struct PyMethodDef spoa_methods[] = {
  /* this module exports no methods */
  { NULL, NULL }
};

#ifdef __cplusplus
extern "C" void initsidlObjA(void);
#else
extern void initsidlObjA(void);
#endif

void
initsidlObjA(void)
{
  PyObject *module, *dict, *c_api;
  static void *spoa_api[sidl_API_pointers];

  module = Py_InitModule("sidlObjA", spoa_methods);
  dict = PyModule_GetDict(module);

  if (PyType_Ready(&sidlPythonObjectType) < 0) return;
  Py_INCREF(&sidlPythonObjectType);
  PyModule_AddObject(module, "BaseWrapper", (PyObject *)&sidlPythonObjectType);

  spoa_api[sidl_Object_Init_NUM] = (void *)sidl_Object_Init;
  spoa_api[sidl_Get_IOR_NUM] = (void *)sidl_Get_IOR;
  spoa_api[sidl_Cast_NUM] = (void *)sidl_Cast;
  spoa_api[sidl_Opaque_Create_NUM] = (void *)sidl_Opaque_Create;
  spoa_api[sidl_Opaque_Convert_NUM] = (void *)sidl_Opaque_Convert;
  spoa_api[sidl_PyExceptionCast_NUM] = (void *)sidl_PyExceptionCast;
  spoa_api[sidl_PyType_NUM] = (void *)sidl_PyType;
  c_api = PyCObject_FromVoidPtr((void *)spoa_api, NULL);
  if (c_api) {
    PyDict_SetItemString(dict, "_C_API", c_api);
    Py_DECREF(c_api);
  }
  if (PyErr_Occurred()) {
    Py_FatalError("Can't initialize module sidlObjA.");
  }
}
