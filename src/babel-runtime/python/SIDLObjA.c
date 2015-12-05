/*
 * File:        SIDLObjA.c
 * Package:     SIDL Python Object Adaptor
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name: V1-9-0b $
 * Revision:    @(#) $Revision: 1.4 $
 * Date:        $Date: 2003/04/07 21:44:24 $
 * Description: Python extension type written in C for SIDL object/interface
 *
 * This is a Python extension type written in C to wrap instances of
 * SIDL objects or interfaces in a Python object.  If this looks
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

#define SIDLOBJA_MODULE 1
#include "SIDLObjA.h"
#include "SIDL_BaseClass_IOR.h"
#include "SIDL_BaseException.h"

#include <string.h>

/**
 * This defines the Python object wrapper for a SIDL object or interface.
 * This one Python extension type written in C can support an arbitrary
 * class or interface. The peculiarities of a particular class or interface
 * are stored in the d_methods member which defines what the object
 * can do.
 */
struct SIDLPythonObject {
  PyObject_HEAD                 /* standard Python object header */

  /* SIDL specific extensions */

  PyMethodDef                         *d_methods;
  struct SIDL_BaseClass__object *    (*d_getIOR)(struct SIDLPythonObject *);
  union {
    struct SIDL_BaseClass__object     *d_ior;
    struct SIDL_BaseInterface__object *d_abint;
  }                                    d_sidlobj;
  int				       d_refType;
};

/**
 * This <code>typedef</code> is required by PyObject_NEW; otherwise,
 * it is unused.
 */
typedef struct SIDLPythonObject SPObject;

/**
 * A function to get the independent object representation (IOR) for
 * a wrapped instance of a class.
 */
static struct SIDL_BaseClass__object *
SIDLPython_Object_getIOR(struct SIDLPythonObject *sobj) {
  return sobj->d_sidlobj.d_ior;
}

/**
 * A function to get the independent object representation (IOR) for
 * a wrapped instance of a interface.
 */
static struct SIDL_BaseClass__object *
SIDLPython_Interface_getIOR(struct SIDLPythonObject *sobj) {
  return sobj->d_sidlobj.d_abint->d_object;
}

/**
 * Provide a forward declaration of the Python type object used
 * by <code>SIDLPythonObject</code>'s.
 */
staticforward PyTypeObject SIDLPythonObjectType;

#define is_SIDLobject(v) ((v)->ob_type == &SIDLPythonObjectType)

/*
 * Exported C API methods
 */

static SIDL_Object_Create_RETURN
SIDL_Object_Create SIDL_Object_Create_PROTO {
  PyObject *result = NULL;
  if (methods) {
    if (ior) {
      struct SIDLPythonObject *self = 
	PyObject_NEW(SPObject, &SIDLPythonObjectType);
      if (self){
	self->d_sidlobj.d_ior = ior;
        if (refType == SIDL_PyNewRef) {
          (*ior->d_epv->f_addRef)(ior);
        }
	self->d_methods = methods;
	self->d_getIOR = SIDLPython_Object_getIOR;
        self->d_refType = refType;
	result = (PyObject *)self;
      }
      else {
        if (refType == SIDL_PyStealRef) {
          (*(ior->d_epv->f_deleteRef))(ior);
        }
      }
    }
    else {
      result = Py_None;
      Py_INCREF(result);
    }
  }
  else {
    PyErr_SetString(PyExc_AssertionError,
		    "SIDL object has NULL IOR or vtable");
  }
  return result;
}

static SIDL_Get_IOR_RETURN
SIDL_Get_IOR SIDL_Get_IOR_PROTO {
  if (is_SIDLobject(obj)) {
    return (*((struct SIDLPythonObject *)obj)->d_getIOR)
      ((struct SIDLPythonObject *)obj);
  }
  return NULL;
}

static SIDL_Cast_RETURN
SIDL_Cast SIDL_Cast_PROTO {
  struct SIDL_BaseClass__object *ior = SIDL_Get_IOR(obj);
  return (ior) ? ((*(ior->d_epv->f__cast))(ior, name)) : NULL;
}

static SIDL_Opaque_Create_RETURN
SIDL_Opaque_Create SIDL_Opaque_Create_PROTO {
  return PyCObject_FromVoidPtr(opaque_ptr, NULL);
}

static SIDL_Opaque_Convert_RETURN
SIDL_Opaque_Convert SIDL_Opaque_Convert_PROTO {
  if (PyCObject_Check(obj)) {
    *opaque_ptr = PyCObject_AsVoidPtr(obj);
    return 1;
  }
  return 0;
}

static SIDL_PyType_RETURN
SIDL_PyType SIDL_PyType_PROTO {
  return (PyObject *)&SIDLPythonObjectType;
}

static SIDL_Interface_Create_RETURN
SIDL_Interface_Create SIDL_Interface_Create_PROTO {
  PyObject *result = NULL;
  if (methods) {
    if (abint) {
      struct SIDLPythonObject *self = 
	PyObject_NEW(SPObject, &SIDLPythonObjectType);
      if (self){
	self->d_sidlobj.d_abint = abint;
        if (refType == SIDL_PyNewRef) {
          (*abint->d_epv->f_addRef)(abint->d_object);
        }
	self->d_methods = methods;
	self->d_getIOR = SIDLPython_Interface_getIOR;
        self->d_refType = refType;
	result = (PyObject *)self;
      }
      else {
        if (refType == SIDL_PyStealRef) {
          (*abint->d_epv->f_deleteRef)(abint->d_object);
        }
      }
    }
    else {
      result = Py_None;
      Py_INCREF(result);
    }
  }
  else {
    PyErr_SetString(PyExc_AssertionError,
		    "SIDL interface has NULL vtable");
  }
  return result;
}

static SIDL_Interface_Check_RETURN
SIDL_Interface_Check SIDL_Interface_Check_PROTO {
  if (is_SIDLobject(obj) && 
      (((struct SIDLPythonObject *)obj)->d_getIOR ==
       SIDLPython_Interface_getIOR)) {
    return ((struct SIDLPythonObject *)obj)->d_sidlobj.d_abint;
  }
  return NULL;
}

static SIDL_PyExceptionCast_RETURN
SIDL_PyExceptionCast SIDL_PyExceptionCast_PROTO {
  return SIDL_BaseException__cast2(ex, name);
}

/*
 * BASIC OBJECT METHODS
 */

static PyObject *
spoa_getattr(struct SIDLPythonObject *self,
            char *name)
{
  if (!strcmp(name, "__members__")){
    return Py_BuildValue("[]");
  }
  return Py_FindMethod(self->d_methods, (PyObject *)self, name);
}

static void
spoa_self_destruct(struct SIDLPythonObject *self)
{
  struct SIDL_BaseClass__object *ior = SIDL_Get_IOR((PyObject *)self);
  /* remove Python's reference to this SIDL object */
  if (self->d_refType != SIDL_PyWeakRef) {
    (*(ior->d_epv->f_deleteRef))(ior);
  }
  self->d_sidlobj.d_ior = NULL;
  self->d_sidlobj.d_abint = NULL;
  self->d_getIOR = NULL;
  self->d_methods = NULL;
  self->d_refType = SIDL_PyWeakRef;
  PyMem_DEL(self);
}

static PyTypeObject SIDLPythonObjectType = {
  /* type header */
  PyObject_HEAD_INIT(NULL)
  0,
  "SIDLObjA",
  sizeof(struct SIDLPythonObject),
  0,

  /* standard methods */
  (destructor) spoa_self_destruct, /* call when reference count == 0 */
  (printfunc) 0,
  (getattrfunc) spoa_getattr,
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
  0,
  /* documentation string */
  "This is a Python wrapper for a SIDL object or interface."
};

static struct PyMethodDef spoa_methods[] = {
  /* this module exports no methods */
  { NULL, NULL }
};

#ifdef __cplusplus
extern "C" void initSIDLObjA(void);
#else
extern void initSIDLObjA(void);
#endif

void
initSIDLObjA(void)
{
  PyObject *module, *dict, *c_api;
  static void *spoa_api[SIDL_API_pointers];

  SIDLPythonObjectType.ob_type = &PyType_Type;

  module = Py_InitModule("SIDLObjA", spoa_methods);
  dict = PyModule_GetDict(module);
  spoa_api[SIDL_Object_Create_NUM] = (void *)SIDL_Object_Create;
  spoa_api[SIDL_Get_IOR_NUM] = (void *)SIDL_Get_IOR;
  spoa_api[SIDL_Cast_NUM] = (void *)SIDL_Cast;
  spoa_api[SIDL_Interface_Create_NUM] = (void *)SIDL_Interface_Create;
  spoa_api[SIDL_Interface_Check_NUM] = (void *)SIDL_Interface_Check;
  spoa_api[SIDL_Opaque_Create_NUM] = (void *)SIDL_Opaque_Create;
  spoa_api[SIDL_Opaque_Convert_NUM] = (void *)SIDL_Opaque_Convert;
  spoa_api[SIDL_PyExceptionCast_NUM] = (void *)SIDL_PyExceptionCast;
  spoa_api[SIDL_PyType_NUM] = (void *)SIDL_PyType;
  c_api = PyCObject_FromVoidPtr((void *)spoa_api, NULL);
  if (c_api) {
    PyDict_SetItemString(dict, "_C_API", c_api);
    Py_DECREF(c_api);
  }
  if (PyErr_Occurred()) {
    Py_FatalError("Can't initialize module SIDLObjA.");
  }
}
