/*
 * File:          sidl_rmi_NoServerException_Stub.c
 * Symbol:        sidl.rmi.NoServerException-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V1-14-0b $
 * Revision:      @(#) $Id: sidl_rmi_NoServerException_Stub.c,v 1.1 2006/08/29 23:31:43 painter Exp $
 * Description:   Client-side glue code for sidl.rmi.NoServerException
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
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
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidl_rmi_NoServerException.h"
#include "sidl_rmi_NoServerException_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#include "sidl_Exception.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stddef.h>
#include <string.h>
#include "sidl_BaseInterface_IOR.h"

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

/*
 * Hold pointer to IOR functions.
 */

static const struct sidl_rmi_NoServerException__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct sidl_rmi_NoServerException__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
  _externals = sidl_rmi_NoServerException__externals();
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Constructor function for the class.
 */

sidl_rmi_NoServerException
sidl_rmi_NoServerException__create(sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(NULL,_ex);
}

static sidl_rmi_NoServerException 
  sidl_rmi_NoServerException__remoteCreate(const char* url,
  sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

sidl_rmi_NoServerException
sidl_rmi_NoServerException__createRemote(const char* url,
  sidl_BaseInterface *_ex)
{
  return sidl_rmi_NoServerException__remoteCreate(url, _ex);
}

static struct sidl_rmi_NoServerException__object* 
  sidl_rmi_NoServerException__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct sidl_rmi_NoServerException__object* 
  sidl_rmi_NoServerException__IHConnect(struct sidl_rmi_InstanceHandle__object* 
  instance, sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

sidl_rmi_NoServerException
sidl_rmi_NoServerException__connect(const char* url, sidl_BaseInterface *_ex)
{
  return sidl_rmi_NoServerException__remoteConnect(url, TRUE, _ex);
}

/*
 * Method:  getHopCount[]
 */

SIDL_C_INLINE_DEFN
int32_t
sidl_rmi_NoServerException_getHopCount(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_getHopCount)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException_addRef(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_addRef)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException_deleteRef(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_deleteRef)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_C_INLINE_DEFN
sidl_bool
sidl_rmi_NoServerException_isSame(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_C_INLINE_DEFN
sidl_bool
sidl_rmi_NoServerException_isType(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_isType)(
    self,
    name,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Return the meta-data about the class implementing this interface.
 */

SIDL_C_INLINE_DEFN
sidl_ClassInfo
sidl_rmi_NoServerException_getClassInfo(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_getClassInfo)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Return the message associated with the exception.
 */

SIDL_C_INLINE_DEFN
char*
sidl_rmi_NoServerException_getNote(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_getNote)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Set the message associated with the exception.
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException_setNote(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ const char* message,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_setNote)(
    self,
    message,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Returns formatted string containing the concatenation of all 
 * tracelines.
 */

SIDL_C_INLINE_DEFN
char*
sidl_rmi_NoServerException_getTrace(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_getTrace)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Adds a stringified entry/line to the stack trace.
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException_addLine(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ const char* traceline,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_addLine)(
    self,
    traceline,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Formats and adds an entry to the stack trace based on the 
 * file name, line number, and method name.
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException_add(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ const char* filename,
  /* in */ int32_t lineno,
  /* in */ const char* methodname,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_add)(
    self,
    filename,
    lineno,
    methodname,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Method:  packObj[]
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException_packObj(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ sidl_io_Serializer ser,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_packObj)(
    self,
    ser,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Method:  unpackObj[]
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException_unpackObj(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ sidl_io_Deserializer des,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_unpackObj)(
    self,
    des,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Cast method for interface and class type conversions.
 */

sidl_rmi_NoServerException
sidl_rmi_NoServerException__cast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  sidl_rmi_NoServerException cast = NULL;

  if(!connect_loaded) {
    connect_loaded = 1;
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.NoServerException",
      (void*)sidl_rmi_NoServerException__IHConnect,_ex);SIDL_CHECK(*_ex);
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (sidl_rmi_NoServerException) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.rmi.NoServerException", _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
sidl_rmi_NoServerException__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface* _ex)
{
  void* cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type, _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}
/*
 * Select and execute a method by name
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException__exec(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__exec)(
    self,
    methodName,
    inArgs,
    outArgs,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * Get the URL of the Implementation of this object (for RMI)
 */

SIDL_C_INLINE_DEFN
char*
sidl_rmi_NoServerException__getURL(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f__getURL)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * On a remote object, addrefs the remote instance.
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException__raddRef(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__raddRef)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * Method to set whether or not method hooks should be invoked.
 */

SIDL_C_INLINE_DEFN
void
sidl_rmi_NoServerException__set_hooks(
  /* in */ sidl_rmi_NoServerException self,
  /* in */ sidl_bool on,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__set_hooks)(
    self,
    on,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * TRUE if this object is remote, false if local
 */

SIDL_C_INLINE_DEFN
sidl_bool
sidl_rmi_NoServerException__isRemote(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f__isRemote)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * TRUE if this object is remote, false if local
 */

sidl_bool
sidl_rmi_NoServerException__isLocal(
  /* in */ sidl_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex)
{
  return !sidl_rmi_NoServerException__isRemote(self,_ex);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidl_rmi_NoServerException__array*)sidl_interface__array_createCol(dimen,
    lower, upper);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidl_rmi_NoServerException__array*)sidl_interface__array_createRow(dimen,
    lower, upper);
}

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_create1d(int32_t len)
{
  return (struct 
    sidl_rmi_NoServerException__array*)sidl_interface__array_create1d(len);
}

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_create1dInit(
  int32_t len, 
  sidl_rmi_NoServerException* data)
{
  return (struct 
    sidl_rmi_NoServerException__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    sidl_rmi_NoServerException__array*)sidl_interface__array_create2dCol(m, n);
}

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    sidl_rmi_NoServerException__array*)sidl_interface__array_create2dRow(m, n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_borrow(
  sidl_rmi_NoServerException* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct 
    sidl_rmi_NoServerException__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_smartCopy(
  struct sidl_rmi_NoServerException__array *array)
{
  return (struct sidl_rmi_NoServerException__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
sidl_rmi_NoServerException__array_addRef(
  struct sidl_rmi_NoServerException__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
sidl_rmi_NoServerException__array_deleteRef(
  struct sidl_rmi_NoServerException__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidl_rmi_NoServerException
sidl_rmi_NoServerException__array_get1(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t i1)
{
  return (sidl_rmi_NoServerException)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidl_rmi_NoServerException
sidl_rmi_NoServerException__array_get2(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (sidl_rmi_NoServerException)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidl_rmi_NoServerException
sidl_rmi_NoServerException__array_get3(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (sidl_rmi_NoServerException)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidl_rmi_NoServerException
sidl_rmi_NoServerException__array_get4(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (sidl_rmi_NoServerException)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidl_rmi_NoServerException
sidl_rmi_NoServerException__array_get5(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (sidl_rmi_NoServerException)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidl_rmi_NoServerException
sidl_rmi_NoServerException__array_get6(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (sidl_rmi_NoServerException)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidl_rmi_NoServerException
sidl_rmi_NoServerException__array_get7(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (sidl_rmi_NoServerException)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidl_rmi_NoServerException
sidl_rmi_NoServerException__array_get(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t indices[])
{
  return (sidl_rmi_NoServerException)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidl_rmi_NoServerException__array_set1(
  struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  sidl_rmi_NoServerException const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidl_rmi_NoServerException__array_set2(
  struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  sidl_rmi_NoServerException const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidl_rmi_NoServerException__array_set3(
  struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidl_rmi_NoServerException const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidl_rmi_NoServerException__array_set4(
  struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidl_rmi_NoServerException const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidl_rmi_NoServerException__array_set5(
  struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidl_rmi_NoServerException const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidl_rmi_NoServerException__array_set6(
  struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidl_rmi_NoServerException const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidl_rmi_NoServerException__array_set7(
  struct sidl_rmi_NoServerException__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidl_rmi_NoServerException const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidl_rmi_NoServerException__array_set(
  struct sidl_rmi_NoServerException__array* array,
  const int32_t indices[],
  sidl_rmi_NoServerException const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidl_rmi_NoServerException__array_dimen(
  const struct sidl_rmi_NoServerException__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_rmi_NoServerException__array_lower(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_rmi_NoServerException__array_upper(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

/**
 * Return the length of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_rmi_NoServerException__array_length(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_rmi_NoServerException__array_stride(
  const struct sidl_rmi_NoServerException__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidl_rmi_NoServerException__array_isColumnOrder(
  const struct sidl_rmi_NoServerException__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidl_rmi_NoServerException__array_isRowOrder(
  const struct sidl_rmi_NoServerException__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

/**
 * Copy the contents of one array (src) to a second array
 * (dest). For the copy to take place, both arrays must
 * exist and be of the same dimension. This method will
 * not modify dest's size, index bounds, or stride; only
 * the array element values of dest may be changed by
 * this function. No part of src is ever changed by copy.
 * 
 * On exit, dest[i][j][k]... = src[i][j][k]... for all
 * indices i,j,k...  that are in both arrays. If dest and
 * src have no indices in common, nothing is copied. For
 * example, if src is a 1-d array with elements 0-5 and
 * dest is a 1-d array with elements 2-3, this function
 * will make the following assignments:
 *   dest[2] = src[2],
 *   dest[3] = src[3].
 * The function copied the elements that both arrays have
 * in common.  If dest had elements 4-10, this function
 * will make the following assignments:
 *   dest[4] = src[4],
 *   dest[5] = src[5].
 */
void
sidl_rmi_NoServerException__array_copy(
  const struct sidl_rmi_NoServerException__array* src,
  struct sidl_rmi_NoServerException__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

/**
 * Create a sub-array of another array. This resulting
 * array shares data with the original array. The new
 * array can be of the same dimension or potentially
 * less assuming the original array has dimension
 * greater than 1.  If you are removing dimension,
 * indicate the dimensions to remove by setting
 * numElem[i] to zero for any dimension i wthat should
 * go away in the new array.  The meaning of each
 * argument is covered below.
 * 
 * src       the array to be created will be a subset
 *           of this array. If this argument is NULL,
 *           NULL will be returned. The array returned
 *           borrows data from src, so modifying src or
 *           the returned array will modify both
 *           arrays.
 * 
 * dimen     this argument must be greater than zero
 *           and less than or equal to the dimension of
 *           src. An illegal value will cause a NULL
 *           return value.
 * 
 * numElem   this specifies how many elements from src
 *           should be taken in each dimension. A zero
 *           entry indicates that the dimension should
 *           not appear in the new array.  This
 *           argument should be an array with an entry
 *           for each dimension of src.  Passing NULL
 *           here will cause NULL to be returned.  If
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           greater than upper[i] for src or if
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           less than lower[i] for src, NULL will be
 *           returned.
 * 
 * srcStart  this array holds the coordinates of the
 *           first element of the new array. If this
 *           argument is NULL, the first element of src
 *           will be the first element of the new
 *           array. If non-NULL, this argument should
 *           be an array with an entry for each
 *           dimension of src.  If srcStart[i] is less
 *           than lower[i] for the array src, NULL will
 *           be returned.
 * 
 * srcStride this array lets you specify the stride
 *           between elements in each dimension of
 *           src. This stride is relative to the
 *           coordinate system of the src array. If
 *           this argument is NULL, the stride is taken
 *           to be one in each dimension.  If non-NULL,
 *           this argument should be an array with an
 *           entry for each dimension of src.
 * 
 * newLower  this argument is like lower in a create
 *           method. It sets the coordinates for the
 *           first element in the new array.  If this
 *           argument is NULL, the values indicated by
 *           srcStart will be used. If non-NULL, this
 *           should be an array with dimen elements.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_slice(
  struct sidl_rmi_NoServerException__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct sidl_rmi_NoServerException__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

/**
 * If necessary, convert a general matrix into a matrix
 * with the required properties. This checks the
 * dimension and ordering of the matrix.  If both these
 * match, it simply returns a new reference to the
 * existing matrix. If the dimension of the incoming
 * array doesn't match, it returns NULL. If the ordering
 * of the incoming array doesn't match the specification,
 * a new array is created with the desired ordering and
 * the content of the incoming array is copied to the new
 * array.
 * 
 * The ordering parameter should be one of the constants
 * defined in enum sidl_array_ordering
 * (e.g. sidl_general_order, sidl_column_major_order, or
 * sidl_row_major_order). If you specify
 * sidl_general_order, this routine will only check the
 * dimension because any matrix is sidl_general_order.
 * 
 * The caller assumes ownership of the returned reference
 * unless it's NULL.
 */
struct sidl_rmi_NoServerException__array*
sidl_rmi_NoServerException__array_ensure(
  struct sidl_rmi_NoServerException__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct sidl_rmi_NoServerException__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

#include <stdlib.h>
#include <string.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidl_rmi_NoServerException__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_rmi_NoServerException__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_rmi_NoServerException__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_rmi_NoServerException__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 10;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct sidl_rmi_NoServerException__epv 
  s_rem_epv__sidl_rmi_noserverexception;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseException__epv  s_rem_epv__sidl_baseexception;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;

static struct sidl_RuntimeException__epv  s_rem_epv__sidl_runtimeexception;

static struct sidl_SIDLException__epv  s_rem_epv__sidl_sidlexception;

static struct sidl_io_IOException__epv  s_rem_epv__sidl_io_ioexception;

static struct sidl_io_Serializable__epv  s_rem_epv__sidl_io_serializable;

static struct sidl_rmi_NetworkException__epv  
  s_rem_epv__sidl_rmi_networkexception;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidl_rmi_NoServerException__cast(
  struct sidl_rmi_NoServerException__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2,
    cmp3;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.SIDLException");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = self;
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = 
        &((
        *self).d_sidl_rmi_networkexception.d_sidl_io_ioexception.d_sidl_sidlexception.d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.BaseException");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = 
          &((
          *self).d_sidl_rmi_networkexception.d_sidl_io_ioexception.d_sidl_sidlexception.d_sidl_baseexception);
        return cast;
      }
      else if (cmp2 < 0) {
        cmp3 = strcmp(name, "sidl.BaseClass");
        if (!cmp3) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = self;
          return cast;
        }
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.RuntimeException");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = 
          &((
          *self).d_sidl_rmi_networkexception.d_sidl_io_ioexception.d_sidl_runtimeexception);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.rmi.NetworkException");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.io.Serializable");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = 
          &((
          *self).d_sidl_rmi_networkexception.d_sidl_io_ioexception.d_sidl_sidlexception.d_sidl_io_serializable);
        return cast;
      }
      else if (cmp2 < 0) {
        cmp3 = strcmp(name, "sidl.io.IOException");
        if (!cmp3) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = self;
          return cast;
        }
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.rmi.NoServerException");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
        return cast;
      }
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*,
      struct sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidl_rmi_NoServerException__delete(
  struct sidl_rmi_NoServerException__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidl_rmi_NoServerException__getURL(
  struct sidl_rmi_NoServerException__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_sidl_rmi_NoServerException__raddRef(
  struct sidl_rmi_NoServerException__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    sidl_BaseInterface throwaway_exception = NULL;
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(netex,
      &throwaway_exception);
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_sidl_rmi_NoServerException__isRemote(
    struct sidl_rmi_NoServerException__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_sidl_rmi_NoServerException__set_hooks(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* in */ sidl_bool on,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException._set_hooks.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_sidl_rmi_NoServerException__exec(
  struct sidl_rmi_NoServerException__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:getHopCount */
static int32_t
remote_sidl_rmi_NoServerException_getHopCount(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getHopCount", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.getHopCount.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidl_rmi_NoServerException_addRef(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi_NoServerException__remote* r_obj = (struct 
      sidl_rmi_NoServerException__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidl_rmi_NoServerException_deleteRef(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct sidl_rmi_NoServerException__remote* r_obj = (struct 
      sidl_rmi_NoServerException__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount--;
    if(r_obj->d_refcount == 0) {
      sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
      free(r_obj);
      free(self);
    }
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_sidl_rmi_NoServerException_isSame(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.isSame.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_sidl_rmi_NoServerException_isType(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.isType.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_sidl_rmi_NoServerException_getClassInfo(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.getClassInfo.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getNote */
static char*
remote_sidl_rmi_NoServerException_getNote(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getNote", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.getNote.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:setNote */
static void
remote_sidl_rmi_NoServerException_setNote(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* in */ const char* message,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "setNote", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "message", message,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.setNote.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:getTrace */
static char*
remote_sidl_rmi_NoServerException_getTrace(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getTrace", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.getTrace.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:addLine */
static void
remote_sidl_rmi_NoServerException_addLine(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* in */ const char* traceline,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "addLine", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "traceline", traceline,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.addLine.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:add */
static void
remote_sidl_rmi_NoServerException_add(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* in */ const char* filename,
  /* in */ int32_t lineno,
  /* in */ const char* methodname,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "add", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "lineno", lineno, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packString( _inv, "methodname", methodname,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.add.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:packObj */
static void
remote_sidl_rmi_NoServerException_packObj(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* in */ struct sidl_io_Serializer__object* ser,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "packObj", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(ser){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)ser,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "ser", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "ser", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.packObj.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:unpackObj */
static void
remote_sidl_rmi_NoServerException_unpackObj(
  /* in */ struct sidl_rmi_NoServerException__object* self ,
  /* in */ struct sidl_io_Deserializer__object* des,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      sidl_rmi_NoServerException__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "unpackObj", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(des){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)des,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "des", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "des", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi.NoServerException.unpackObj.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void sidl_rmi_NoServerException__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidl_rmi_NoServerException__epv* epv = 
    &s_rem_epv__sidl_rmi_noserverexception;
  struct sidl_BaseClass__epv*             e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseException__epv*         e1  = &s_rem_epv__sidl_baseexception;
  struct sidl_BaseInterface__epv*         e2  = &s_rem_epv__sidl_baseinterface;
  struct sidl_RuntimeException__epv*      e3  = 
    &s_rem_epv__sidl_runtimeexception;
  struct sidl_SIDLException__epv*         e4  = &s_rem_epv__sidl_sidlexception;
  struct sidl_io_IOException__epv*        e5  = &s_rem_epv__sidl_io_ioexception;
  struct sidl_io_Serializable__epv*       e6  = 
    &s_rem_epv__sidl_io_serializable;
  struct sidl_rmi_NetworkException__epv*  e7  = 
    &s_rem_epv__sidl_rmi_networkexception;

  epv->f__cast             = remote_sidl_rmi_NoServerException__cast;
  epv->f__delete           = remote_sidl_rmi_NoServerException__delete;
  epv->f__exec             = remote_sidl_rmi_NoServerException__exec;
  epv->f__getURL           = remote_sidl_rmi_NoServerException__getURL;
  epv->f__raddRef          = remote_sidl_rmi_NoServerException__raddRef;
  epv->f__isRemote         = remote_sidl_rmi_NoServerException__isRemote;
  epv->f__set_hooks        = remote_sidl_rmi_NoServerException__set_hooks;
  epv->f__ctor             = NULL;
  epv->f__ctor2            = NULL;
  epv->f__dtor             = NULL;
  epv->f_getHopCount       = remote_sidl_rmi_NoServerException_getHopCount;
  epv->f_addRef            = remote_sidl_rmi_NoServerException_addRef;
  epv->f_deleteRef         = remote_sidl_rmi_NoServerException_deleteRef;
  epv->f_isSame            = remote_sidl_rmi_NoServerException_isSame;
  epv->f_isType            = remote_sidl_rmi_NoServerException_isType;
  epv->f_getClassInfo      = remote_sidl_rmi_NoServerException_getClassInfo;
  epv->f_getNote           = remote_sidl_rmi_NoServerException_getNote;
  epv->f_setNote           = remote_sidl_rmi_NoServerException_setNote;
  epv->f_getTrace          = remote_sidl_rmi_NoServerException_getTrace;
  epv->f_addLine           = remote_sidl_rmi_NoServerException_addLine;
  epv->f_add               = remote_sidl_rmi_NoServerException_add;
  epv->f_packObj           = remote_sidl_rmi_NoServerException_packObj;
  epv->f_unpackObj         = remote_sidl_rmi_NoServerException_unpackObj;

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL      = (char* (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef     = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote    = (sidl_bool (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks   = (void (*)(struct sidl_BaseClass__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_getNote      = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e1->f_setNote      = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_setNote;
  e1->f_getTrace     = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e1->f_addLine      = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_addLine;
  e1->f_add          = (void (*)(void*,const char*,int32_t,const char*,
    struct sidl_BaseInterface__object **)) epv->f_add;
  e1->f_packObj      = (void (*)(void*,struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e1->f_unpackObj    = (void (*)(void*,struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;
  e1->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e2->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e3->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e3->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e3->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e3->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e3->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e3->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e3->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_getNote      = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e3->f_setNote      = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_setNote;
  e3->f_getTrace     = (char* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e3->f_addLine      = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_addLine;
  e3->f_add          = (void (*)(void*,const char*,int32_t,const char*,
    struct sidl_BaseInterface__object **)) epv->f_add;
  e3->f_packObj      = (void (*)(void*,struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e3->f_unpackObj    = (void (*)(void*,struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;
  e3->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e3->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e4->f__cast        = (void* (*)(struct sidl_SIDLException__object*,
    const char*,sidl_BaseInterface*)) epv->f__cast;
  e4->f__delete      = (void (*)(struct sidl_SIDLException__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e4->f__getURL      = (char* (*)(struct sidl_SIDLException__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e4->f__raddRef     = (void (*)(struct sidl_SIDLException__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e4->f__isRemote    = (sidl_bool (*)(struct sidl_SIDLException__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e4->f__set_hooks   = (void (*)(struct sidl_SIDLException__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e4->f__exec        = (void (*)(struct sidl_SIDLException__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e4->f_addRef       = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e4->f_isType       = (sidl_bool (*)(struct sidl_SIDLException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e4->f_getNote      = (char* (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e4->f_setNote      = (void (*)(struct sidl_SIDLException__object*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_setNote;
  e4->f_getTrace     = (char* (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e4->f_addLine      = (void (*)(struct sidl_SIDLException__object*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_addLine;
  e4->f_add          = (void (*)(struct sidl_SIDLException__object*,const char*,
    int32_t,const char*,struct sidl_BaseInterface__object **)) epv->f_add;
  e4->f_packObj      = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e4->f_unpackObj    = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;

  e5->f__cast        = (void* (*)(struct sidl_io_IOException__object*,
    const char*,sidl_BaseInterface*)) epv->f__cast;
  e5->f__delete      = (void (*)(struct sidl_io_IOException__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e5->f__getURL      = (char* (*)(struct sidl_io_IOException__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e5->f__raddRef     = (void (*)(struct sidl_io_IOException__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e5->f__isRemote    = (sidl_bool (*)(struct sidl_io_IOException__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e5->f__set_hooks   = (void (*)(struct sidl_io_IOException__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e5->f__exec        = (void (*)(struct sidl_io_IOException__object*,
    const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e5->f_addRef       = (void (*)(struct sidl_io_IOException__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e5->f_deleteRef    = (void (*)(struct sidl_io_IOException__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e5->f_isSame       = (sidl_bool (*)(struct sidl_io_IOException__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e5->f_isType       = (sidl_bool (*)(struct sidl_io_IOException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e5->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_io_IOException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e5->f_getNote      = (char* (*)(struct sidl_io_IOException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e5->f_setNote      = (void (*)(struct sidl_io_IOException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_setNote;
  e5->f_getTrace     = (char* (*)(struct sidl_io_IOException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e5->f_addLine      = (void (*)(struct sidl_io_IOException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_addLine;
  e5->f_add          = (void (*)(struct sidl_io_IOException__object*,
    const char*,int32_t,const char*,
    struct sidl_BaseInterface__object **)) epv->f_add;
  e5->f_packObj      = (void (*)(struct sidl_io_IOException__object*,
    struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e5->f_unpackObj    = (void (*)(struct sidl_io_IOException__object*,
    struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;

  e6->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e6->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e6->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e6->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e6->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e6->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e6->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e6->f_packObj      = (void (*)(void*,struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e6->f_unpackObj    = (void (*)(void*,struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;
  e6->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e6->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e6->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e6->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e6->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e7->f__cast        = (void* (*)(struct sidl_rmi_NetworkException__object*,
    const char*,sidl_BaseInterface*)) epv->f__cast;
  e7->f__delete      = (void (*)(struct sidl_rmi_NetworkException__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e7->f__getURL      = (char* (*)(struct sidl_rmi_NetworkException__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e7->f__raddRef     = (void (*)(struct sidl_rmi_NetworkException__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e7->f__isRemote    = (sidl_bool (*)(struct sidl_rmi_NetworkException__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e7->f__set_hooks   = (void (*)(struct sidl_rmi_NetworkException__object*,
    int32_t, sidl_BaseInterface*)) epv->f__set_hooks;
  e7->f__exec        = (void (*)(struct sidl_rmi_NetworkException__object*,
    const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e7->f_getHopCount  = (int32_t (*)(struct sidl_rmi_NetworkException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getHopCount;
  e7->f_addRef       = (void (*)(struct sidl_rmi_NetworkException__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e7->f_deleteRef    = (void (*)(struct sidl_rmi_NetworkException__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e7->f_isSame       = (sidl_bool (*)(struct sidl_rmi_NetworkException__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e7->f_isType       = (sidl_bool (*)(struct sidl_rmi_NetworkException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e7->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_rmi_NetworkException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
  e7->f_getNote      = (char* (*)(struct sidl_rmi_NetworkException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e7->f_setNote      = (void (*)(struct sidl_rmi_NetworkException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_setNote;
  e7->f_getTrace     = (char* (*)(struct sidl_rmi_NetworkException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e7->f_addLine      = (void (*)(struct sidl_rmi_NetworkException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_addLine;
  e7->f_add          = (void (*)(struct sidl_rmi_NetworkException__object*,
    const char*,int32_t,const char*,
    struct sidl_BaseInterface__object **)) epv->f_add;
  e7->f_packObj      = (void (*)(struct sidl_rmi_NetworkException__object*,
    struct sidl_io_Serializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_packObj;
  e7->f_unpackObj    = (void (*)(struct sidl_rmi_NetworkException__object*,
    struct sidl_io_Deserializer__object*,
    struct sidl_BaseInterface__object **)) epv->f_unpackObj;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidl_rmi_NoServerException__object*
sidl_rmi_NoServerException__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi_NoServerException__object* self;

  struct sidl_rmi_NoServerException__object* s0;
  struct sidl_rmi_NetworkException__object* s1;
  struct sidl_io_IOException__object* s2;
  struct sidl_SIDLException__object* s3;
  struct sidl_BaseClass__object* s4;

  struct sidl_rmi_NoServerException__remote* r_obj;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = NULL;
  *_ex = NULL;
  if(url == NULL) {return NULL;}
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    sidl_BaseInterface bi = 
      (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
      objectID, _ex); SIDL_CHECK(*_ex);
    return sidl_rmi_NoServerException__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_rmi_NoServerException__object*) malloc(
      sizeof(struct sidl_rmi_NoServerException__object));

  r_obj =
    (struct sidl_rmi_NoServerException__remote*) malloc(
      sizeof(struct sidl_rmi_NoServerException__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_rmi_networkexception;
  s2 =                                      &s1->d_sidl_io_ioexception;
  s3 =                                      &s2->d_sidl_sidlexception;
  s4 =                                      &s3->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi_NoServerException__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s4->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s4->d_sidl_baseinterface.d_object = (void*) self;

  s4->d_data = (void*) r_obj;
  s4->d_epv  = &s_rem_epv__sidl_baseclass;

  s3->d_sidl_baseexception.d_epv    = &s_rem_epv__sidl_baseexception;
  s3->d_sidl_baseexception.d_object = (void*) self;

  s3->d_sidl_io_serializable.d_epv    = &s_rem_epv__sidl_io_serializable;
  s3->d_sidl_io_serializable.d_object = (void*) self;

  s3->d_data = (void*) r_obj;
  s3->d_epv  = &s_rem_epv__sidl_sidlexception;

  s2->d_sidl_runtimeexception.d_epv    = &s_rem_epv__sidl_runtimeexception;
  s2->d_sidl_runtimeexception.d_object = (void*) self;

  s2->d_data = (void*) r_obj;
  s2->d_epv  = &s_rem_epv__sidl_io_ioexception;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_rmi_networkexception;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi_noserverexception;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct sidl_rmi_NoServerException__object*
sidl_rmi_NoServerException__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidl_rmi_NoServerException__object* self;

  struct sidl_rmi_NoServerException__object* s0;
  struct sidl_rmi_NetworkException__object* s1;
  struct sidl_io_IOException__object* s2;
  struct sidl_SIDLException__object* s3;
  struct sidl_BaseClass__object* s4;

  struct sidl_rmi_NoServerException__remote* r_obj;
  self =
    (struct sidl_rmi_NoServerException__object*) malloc(
      sizeof(struct sidl_rmi_NoServerException__object));

  r_obj =
    (struct sidl_rmi_NoServerException__remote*) malloc(
      sizeof(struct sidl_rmi_NoServerException__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_rmi_networkexception;
  s2 =                                      &s1->d_sidl_io_ioexception;
  s3 =                                      &s2->d_sidl_sidlexception;
  s4 =                                      &s3->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi_NoServerException__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s4->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s4->d_sidl_baseinterface.d_object = (void*) self;

  s4->d_data = (void*) r_obj;
  s4->d_epv  = &s_rem_epv__sidl_baseclass;

  s3->d_sidl_baseexception.d_epv    = &s_rem_epv__sidl_baseexception;
  s3->d_sidl_baseexception.d_object = (void*) self;

  s3->d_sidl_io_serializable.d_epv    = &s_rem_epv__sidl_io_serializable;
  s3->d_sidl_io_serializable.d_object = (void*) self;

  s3->d_data = (void*) r_obj;
  s3->d_epv  = &s_rem_epv__sidl_sidlexception;

  s2->d_sidl_runtimeexception.d_epv    = &s_rem_epv__sidl_runtimeexception;
  s2->d_sidl_runtimeexception.d_object = (void*) self;

  s2->d_data = (void*) r_obj;
  s2->d_epv  = &s_rem_epv__sidl_io_ioexception;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_rmi_networkexception;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi_noserverexception;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct sidl_rmi_NoServerException__object*
sidl_rmi_NoServerException__remoteCreate(const char *url,
  sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct sidl_rmi_NoServerException__object* self;

  struct sidl_rmi_NoServerException__object* s0;
  struct sidl_rmi_NetworkException__object* s1;
  struct sidl_io_IOException__object* s2;
  struct sidl_SIDLException__object* s3;
  struct sidl_BaseClass__object* s4;

  struct sidl_rmi_NoServerException__remote* r_obj;
  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "sidl.rmi.NoServerException",
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidl_rmi_NoServerException__object*) malloc(
      sizeof(struct sidl_rmi_NoServerException__object));

  r_obj =
    (struct sidl_rmi_NoServerException__remote*) malloc(
      sizeof(struct sidl_rmi_NoServerException__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_rmi_networkexception;
  s2 =                                      &s1->d_sidl_io_ioexception;
  s3 =                                      &s2->d_sidl_sidlexception;
  s4 =                                      &s3->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidl_rmi_NoServerException__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s4->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s4->d_sidl_baseinterface.d_object = (void*) self;

  s4->d_data = (void*) r_obj;
  s4->d_epv  = &s_rem_epv__sidl_baseclass;

  s3->d_sidl_baseexception.d_epv    = &s_rem_epv__sidl_baseexception;
  s3->d_sidl_baseexception.d_object = (void*) self;

  s3->d_sidl_io_serializable.d_epv    = &s_rem_epv__sidl_io_serializable;
  s3->d_sidl_io_serializable.d_object = (void*) self;

  s3->d_data = (void*) r_obj;
  s3->d_epv  = &s_rem_epv__sidl_sidlexception;

  s2->d_sidl_runtimeexception.d_epv    = &s_rem_epv__sidl_runtimeexception;
  s2->d_sidl_runtimeexception.d_object = (void*) self;

  s2->d_data = (void*) r_obj;
  s2->d_epv  = &s_rem_epv__sidl_io_ioexception;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_rmi_networkexception;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__sidl_rmi_noserverexception;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance,
    &_throwaway_exception); }
  return NULL;
}
/*
 * Cast method for interface and class type conversions.
 */

struct sidl_rmi_NoServerException__object*
sidl_rmi_NoServerException__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct sidl_rmi_NoServerException__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.NoServerException",
      (void*)sidl_rmi_NoServerException__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct sidl_rmi_NoServerException__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "sidl.rmi.NoServerException", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct sidl_rmi_NoServerException__object*
sidl_rmi_NoServerException__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return sidl_rmi_NoServerException__remoteConnect(url, ar, _ex);
}

