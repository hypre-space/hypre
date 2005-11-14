/*
 * File:          sidlx_rmi_ChildSocket_Stub.c
 * Symbol:        sidlx.rmi.ChildSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for sidlx.rmi.ChildSocket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidlx_rmi_ChildSocket.h"
#include "sidlx_rmi_ChildSocket_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stddef.h>
#include <string.h>
#include "sidl_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.h"
#endif

/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

/*
 * Hold pointer to IOR functions.
 */

static const struct sidlx_rmi_ChildSocket__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct sidlx_rmi_ChildSocket__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = sidlx_rmi_ChildSocket__externals();
#else
  _externals = (struct 
    sidlx_rmi_ChildSocket__external*)sidl_dynamicLoadIOR(
    "sidlx.rmi.ChildSocket","sidlx_rmi_ChildSocket__externals") ;
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Constructor function for the class.
 */

sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__create()
{
  return (*(_getExternals()->createObject))();
}

static sidlx_rmi_ChildSocket sidlx_rmi_ChildSocket__remote(const char* url,
  sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__createRemote(const char* url, sidl_BaseInterface *_ex)
{
  return sidlx_rmi_ChildSocket__remote(url, _ex);
}

static struct sidlx_rmi_ChildSocket__object* 
  sidlx_rmi_ChildSocket__remoteConnect(const char* url,
  sidl_BaseInterface *_ex);
static struct sidlx_rmi_ChildSocket__object* 
  sidlx_rmi_ChildSocket__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__connect(const char* url, sidl_BaseInterface *_ex)
{
  return sidlx_rmi_ChildSocket__remoteConnect(url, _ex);
}

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

void
sidlx_rmi_ChildSocket_addRef(
  /* in */ sidlx_rmi_ChildSocket self)
{
  (*self->d_epv->f_addRef)(
    self);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
sidlx_rmi_ChildSocket_deleteRef(
  /* in */ sidlx_rmi_ChildSocket self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
sidlx_rmi_ChildSocket_isSame(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ sidl_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

sidl_BaseInterface
sidlx_rmi_ChildSocket_queryInt(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_queryInt)(
    self,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

sidl_bool
sidlx_rmi_ChildSocket_isType(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_isType)(
    self,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
sidlx_rmi_ChildSocket_getClassInfo(
  /* in */ sidlx_rmi_ChildSocket self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Method:  getsockname[]
 */

int32_t
sidlx_rmi_ChildSocket_getsockname(
  /* in */ sidlx_rmi_ChildSocket self,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_getsockname)(
    self,
    address,
    port,
    _ex);
}

/*
 * Method:  getpeername[]
 */

int32_t
sidlx_rmi_ChildSocket_getpeername(
  /* in */ sidlx_rmi_ChildSocket self,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_getpeername)(
    self,
    address,
    port,
    _ex);
}

/*
 * Method:  close[]
 */

int32_t
sidlx_rmi_ChildSocket_close(
  /* in */ sidlx_rmi_ChildSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_close)(
    self,
    _ex);
}

/*
 * Method:  readn[]
 */

int32_t
sidlx_rmi_ChildSocket_readn(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_readn)(
    self,
    nbytes,
    data,
    _ex);
}

/*
 * Method:  readline[]
 */

int32_t
sidlx_rmi_ChildSocket_readline(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_readline)(
    self,
    nbytes,
    data,
    _ex);
}

/*
 * Method:  readstring[]
 */

int32_t
sidlx_rmi_ChildSocket_readstring(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_readstring)(
    self,
    nbytes,
    data,
    _ex);
}

/*
 * Method:  readstring_alloc[]
 */

int32_t
sidlx_rmi_ChildSocket_readstring_alloc(
  /* in */ sidlx_rmi_ChildSocket self,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_readstring_alloc)(
    self,
    data,
    _ex);
}

/*
 * Method:  readint[]
 */

int32_t
sidlx_rmi_ChildSocket_readint(
  /* in */ sidlx_rmi_ChildSocket self,
  /* inout */ int32_t* data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_readint)(
    self,
    data,
    _ex);
}

/*
 * Method:  writen[]
 */

int32_t
sidlx_rmi_ChildSocket_writen(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_writen)(
    self,
    nbytes,
    data,
    _ex);
}

/*
 * Method:  writestring[]
 */

int32_t
sidlx_rmi_ChildSocket_writestring(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_writestring)(
    self,
    nbytes,
    data,
    _ex);
}

/*
 * Method:  writeint[]
 */

int32_t
sidlx_rmi_ChildSocket_writeint(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_writeint)(
    self,
    data,
    _ex);
}

/*
 * Method:  setFileDescriptor[]
 */

void
sidlx_rmi_ChildSocket_setFileDescriptor(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t fd,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_setFileDescriptor)(
    self,
    fd,
    _ex);
}

/*
 * Method:  getFileDescriptor[]
 */

int32_t
sidlx_rmi_ChildSocket_getFileDescriptor(
  /* in */ sidlx_rmi_ChildSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_getFileDescriptor)(
    self,
    _ex);
}

/*
 * Method:  init[]
 */

void
sidlx_rmi_ChildSocket_init(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t fileDes,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_init)(
    self,
    fileDes,
    _ex);
}

/*
 * Cast method for interface and class type conversions.
 */

sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__cast(
  void* obj)
{
  sidlx_rmi_ChildSocket cast = NULL;

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidlx.rmi.ChildSocket",
      (void*)sidlx_rmi_ChildSocket__IHConnect);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (sidlx_rmi_ChildSocket) (*base->d_epv->f__cast)(
      base->d_object,
      "sidlx.rmi.ChildSocket");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
sidlx_rmi_ChildSocket__cast2(
  void* obj,
  const char* type)
{
  void* cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type);
  }

  return cast;
}
/*
 * Select and execute a method by name
 */

void
sidlx_rmi_ChildSocket__exec(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs)
{
  (*self->d_epv->f__exec)(
  self,
  methodName,
  inArgs,
  outArgs);
}

/*
 * Get the URL of the Implementation of this object (for RMI)
 */

char*
sidlx_rmi_ChildSocket__getURL(
  /* in */ sidlx_rmi_ChildSocket self)
{
  return (*self->d_epv->f__getURL)(
  self);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidlx_rmi_ChildSocket__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidlx_rmi_ChildSocket__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_create1d(int32_t len)
{
  return (struct 
    sidlx_rmi_ChildSocket__array*)sidl_interface__array_create1d(len);
}

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_create1dInit(
  int32_t len, 
  sidlx_rmi_ChildSocket* data)
{
  return (struct 
    sidlx_rmi_ChildSocket__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    sidlx_rmi_ChildSocket__array*)sidl_interface__array_create2dCol(m, n);
}

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    sidlx_rmi_ChildSocket__array*)sidl_interface__array_create2dRow(m, n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_borrow(
  sidlx_rmi_ChildSocket* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct sidlx_rmi_ChildSocket__array*)sidl_interface__array_borrow(
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
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_smartCopy(
  struct sidlx_rmi_ChildSocket__array *array)
{
  return (struct sidlx_rmi_ChildSocket__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
sidlx_rmi_ChildSocket__array_addRef(
  struct sidlx_rmi_ChildSocket__array* array)
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
sidlx_rmi_ChildSocket__array_deleteRef(
  struct sidlx_rmi_ChildSocket__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__array_get1(
  const struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1)
{
  return (sidlx_rmi_ChildSocket)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__array_get2(
  const struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (sidlx_rmi_ChildSocket)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__array_get3(
  const struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (sidlx_rmi_ChildSocket)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__array_get4(
  const struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (sidlx_rmi_ChildSocket)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__array_get5(
  const struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (sidlx_rmi_ChildSocket)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__array_get6(
  const struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (sidlx_rmi_ChildSocket)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__array_get7(
  const struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (sidlx_rmi_ChildSocket)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidlx_rmi_ChildSocket
sidlx_rmi_ChildSocket__array_get(
  const struct sidlx_rmi_ChildSocket__array* array,
  const int32_t indices[])
{
  return (sidlx_rmi_ChildSocket)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidlx_rmi_ChildSocket__array_set1(
  struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  sidlx_rmi_ChildSocket const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidlx_rmi_ChildSocket__array_set2(
  struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  sidlx_rmi_ChildSocket const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidlx_rmi_ChildSocket__array_set3(
  struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidlx_rmi_ChildSocket const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidlx_rmi_ChildSocket__array_set4(
  struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidlx_rmi_ChildSocket const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidlx_rmi_ChildSocket__array_set5(
  struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidlx_rmi_ChildSocket const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidlx_rmi_ChildSocket__array_set6(
  struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidlx_rmi_ChildSocket const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidlx_rmi_ChildSocket__array_set7(
  struct sidlx_rmi_ChildSocket__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidlx_rmi_ChildSocket const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidlx_rmi_ChildSocket__array_set(
  struct sidlx_rmi_ChildSocket__array* array,
  const int32_t indices[],
  sidlx_rmi_ChildSocket const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidlx_rmi_ChildSocket__array_dimen(
  const struct sidlx_rmi_ChildSocket__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_rmi_ChildSocket__array_lower(
  const struct sidlx_rmi_ChildSocket__array* array,
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
sidlx_rmi_ChildSocket__array_upper(
  const struct sidlx_rmi_ChildSocket__array* array,
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
sidlx_rmi_ChildSocket__array_length(
  const struct sidlx_rmi_ChildSocket__array* array,
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
sidlx_rmi_ChildSocket__array_stride(
  const struct sidlx_rmi_ChildSocket__array* array,
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
sidlx_rmi_ChildSocket__array_isColumnOrder(
  const struct sidlx_rmi_ChildSocket__array* array)
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
sidlx_rmi_ChildSocket__array_isRowOrder(
  const struct sidlx_rmi_ChildSocket__array* array)
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
sidlx_rmi_ChildSocket__array_copy(
  const struct sidlx_rmi_ChildSocket__array* src,
  struct sidlx_rmi_ChildSocket__array* dest)
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
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_slice(
  struct sidlx_rmi_ChildSocket__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct sidlx_rmi_ChildSocket__array*)
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
struct sidlx_rmi_ChildSocket__array*
sidlx_rmi_ChildSocket__array_ensure(
  struct sidlx_rmi_ChildSocket__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct sidlx_rmi_ChildSocket__array*)
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
#include "sidl_rmi_ProtocolFactory.h"
#include "sidl_rmi_Invocation.h"
#include "sidl_rmi_Response.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidlx_rmi_ChildSocket__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_rmi_ChildSocket__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_rmi_ChildSocket__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_rmi_ChildSocket__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct sidlx_rmi_ChildSocket__epv s_rem_epv__sidlx_rmi_childsocket;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;

static struct sidlx_rmi_IPv4Socket__epv  s_rem_epv__sidlx_rmi_ipv4socket;

static struct sidlx_rmi_Socket__epv  s_rem_epv__sidlx_rmi_socket;

/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidlx_rmi_ChildSocket__cast(
struct sidlx_rmi_ChildSocket__object* self,
const char* name)
{
  void* cast = NULL;

  struct sidlx_rmi_ChildSocket__object* s0;
  struct sidlx_rmi_IPv4Socket__object* s1;
  struct sidl_BaseClass__object* s2;
   s0 =                                self;
   s1 =                                &s0->d_sidlx_rmi_ipv4socket;
   s2 =                                &s1->d_sidl_baseclass;

  if (!strcmp(name, "sidlx.rmi.ChildSocket")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidlx.rmi.IPv4Socket")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidlx.rmi.Socket")) {
    cast = (void*) &s1->d_sidlx_rmi_socket;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s2;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s2->d_sidl_baseinterface;
  }
  else if(sidlx_rmi_ChildSocket_isType(self, name)) {
    void* (*func)(sidl_rmi_InstanceHandle) = 
      (void* (*)(sidl_rmi_InstanceHandle)) 
      sidl_rmi_ConnectRegistry_getConnect(name);
    cast =  (*func)((sidl_rmi_InstanceHandle)self->d_data);
  }

  return cast;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidlx_rmi_ChildSocket__delete(
  struct sidlx_rmi_ChildSocket__object* self)
{
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidlx_rmi_ChildSocket__getURL(
  struct sidlx_rmi_ChildSocket__object* self)
{
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getURL(conn, &_ex);
  }
  return NULL;
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_sidlx_rmi_ChildSocket__exec(
  struct sidlx_rmi_ChildSocket__object* self,
  const char* methodName,
  sidl_io_Deserializer inArgs,
  sidl_io_Serializer outArgs)
{
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidlx_rmi_ChildSocket_addRef(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidlx_rmi_ChildSocket_deleteRef(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "deleteRef", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_sidlx_rmi_ChildSocket_isSame(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ struct sidl_BaseInterface__object* iobj)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:queryInt */
static struct sidl_BaseInterface__object*
remote_sidlx_rmi_ChildSocket_queryInt(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ const char* name)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_sidlx_rmi_ChildSocket_isType(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ const char* name)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "isType", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  sidl_bool _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_sidlx_rmi_ChildSocket_getClassInfo(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:getsockname */
static int32_t
remote_sidlx_rmi_ChildSocket_getsockname(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getsockname", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "address", *address, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "port", *port, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "address", address, _ex2);
  sidl_rmi_Response_unpackInt( _rsvp, "port", port, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:getpeername */
static int32_t
remote_sidlx_rmi_ChildSocket_getpeername(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getpeername", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "address", *address, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "port", *port, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "address", address, _ex2);
  sidl_rmi_Response_unpackInt( _rsvp, "port", port, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:close */
static int32_t
remote_sidlx_rmi_ChildSocket_close(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "close", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:readn */
static int32_t
remote_sidlx_rmi_ChildSocket_readn(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "readn", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "nbytes", nbytes, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:readline */
static int32_t
remote_sidlx_rmi_ChildSocket_readline(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "readline", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "nbytes", nbytes, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:readstring */
static int32_t
remote_sidlx_rmi_ChildSocket_readstring(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "readstring", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "nbytes", nbytes, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:readstring_alloc */
static int32_t
remote_sidlx_rmi_ChildSocket_readstring_alloc(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "readstring_alloc", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:readint */
static int32_t
remote_sidlx_rmi_ChildSocket_readint(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* inout */ int32_t* data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "readint", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "data", *data, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "data", data, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:writen */
static int32_t
remote_sidlx_rmi_ChildSocket_writen(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "writen", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "nbytes", nbytes, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:writestring */
static int32_t
remote_sidlx_rmi_ChildSocket_writestring(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "writestring", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "nbytes", nbytes, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:writeint */
static int32_t
remote_sidlx_rmi_ChildSocket_writeint(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ int32_t data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "writeint", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "data", data, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:setFileDescriptor */
static void
remote_sidlx_rmi_ChildSocket_setFileDescriptor(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ int32_t fd,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "setFileDescriptor", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "fd", fd, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getFileDescriptor */
static int32_t
remote_sidlx_rmi_ChildSocket_getFileDescriptor(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getFileDescriptor", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:init */
static void
remote_sidlx_rmi_ChildSocket_init(
  /* in */ struct sidlx_rmi_ChildSocket__object* self /* TLD */,
  /* in */ int32_t fileDes,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "init", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "fileDes", fileDes, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void sidlx_rmi_ChildSocket__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidlx_rmi_ChildSocket__epv* epv = &s_rem_epv__sidlx_rmi_childsocket;
  struct sidl_BaseClass__epv*        e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*    e1  = &s_rem_epv__sidl_baseinterface;
  struct sidlx_rmi_IPv4Socket__epv*  e2  = &s_rem_epv__sidlx_rmi_ipv4socket;
  struct sidlx_rmi_Socket__epv*      e3  = &s_rem_epv__sidlx_rmi_socket;

  epv->f__cast                  = remote_sidlx_rmi_ChildSocket__cast;
  epv->f__delete                = remote_sidlx_rmi_ChildSocket__delete;
  epv->f__exec                  = remote_sidlx_rmi_ChildSocket__exec;
  epv->f__getURL                = remote_sidlx_rmi_ChildSocket__getURL;
  epv->f__ctor                  = NULL;
  epv->f__dtor                  = NULL;
  epv->f_addRef                 = remote_sidlx_rmi_ChildSocket_addRef;
  epv->f_deleteRef              = remote_sidlx_rmi_ChildSocket_deleteRef;
  epv->f_isSame                 = remote_sidlx_rmi_ChildSocket_isSame;
  epv->f_queryInt               = remote_sidlx_rmi_ChildSocket_queryInt;
  epv->f_isType                 = remote_sidlx_rmi_ChildSocket_isType;
  epv->f_getClassInfo           = remote_sidlx_rmi_ChildSocket_getClassInfo;
  epv->f_getsockname            = remote_sidlx_rmi_ChildSocket_getsockname;
  epv->f_getpeername            = remote_sidlx_rmi_ChildSocket_getpeername;
  epv->f_close                  = remote_sidlx_rmi_ChildSocket_close;
  epv->f_readn                  = remote_sidlx_rmi_ChildSocket_readn;
  epv->f_readline               = remote_sidlx_rmi_ChildSocket_readline;
  epv->f_readstring             = remote_sidlx_rmi_ChildSocket_readstring;
  epv->f_readstring_alloc       = remote_sidlx_rmi_ChildSocket_readstring_alloc;
  epv->f_readint                = remote_sidlx_rmi_ChildSocket_readint;
  epv->f_writen                 = remote_sidlx_rmi_ChildSocket_writen;
  epv->f_writestring            = remote_sidlx_rmi_ChildSocket_writestring;
  epv->f_writeint               = remote_sidlx_rmi_ChildSocket_writeint;
  epv->f_setFileDescriptor      = 
    remote_sidlx_rmi_ChildSocket_setFileDescriptor;
  epv->f_getFileDescriptor      = 
    remote_sidlx_rmi_ChildSocket_getFileDescriptor;
  epv->f_init                   = remote_sidlx_rmi_ChildSocket_init;

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e0->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  e2->f__cast             = (void* (*)(struct sidlx_rmi_IPv4Socket__object*,
    const char*)) epv->f__cast;
  e2->f__delete           = (void (*)(struct sidlx_rmi_IPv4Socket__object*)) 
    epv->f__delete;
  e2->f__exec             = (void (*)(struct sidlx_rmi_IPv4Socket__object*,
    const char*,struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef            = (void (*)(struct sidlx_rmi_IPv4Socket__object*)) 
    epv->f_addRef;
  e2->f_deleteRef         = (void (*)(struct sidlx_rmi_IPv4Socket__object*)) 
    epv->f_deleteRef;
  e2->f_isSame            = (sidl_bool (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt          = (struct sidl_BaseInterface__object* (*)(struct 
    sidlx_rmi_IPv4Socket__object*,const char*)) epv->f_queryInt;
  e2->f_isType            = (sidl_bool (*)(struct sidlx_rmi_IPv4Socket__object*,
    const char*)) epv->f_isType;
  e2->f_getClassInfo      = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_rmi_IPv4Socket__object*)) epv->f_getClassInfo;
  e2->f_getsockname       = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t*,int32_t*,struct sidl_BaseInterface__object **)) epv->f_getsockname;
  e2->f_getpeername       = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t*,int32_t*,struct sidl_BaseInterface__object **)) epv->f_getpeername;
  e2->f_close             = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_BaseInterface__object **)) epv->f_close;
  e2->f_readn             = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readn;
  e2->f_readline          = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readline;
  e2->f_readstring        = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readstring;
  e2->f_readstring_alloc  = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readstring_alloc;
  e2->f_readint           = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t*,struct sidl_BaseInterface__object **)) epv->f_readint;
  e2->f_writen            = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_writen;
  e2->f_writestring       = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_writestring;
  e2->f_writeint          = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_BaseInterface__object **)) epv->f_writeint;
  e2->f_setFileDescriptor = (void (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_BaseInterface__object **)) epv->f_setFileDescriptor;
  e2->f_getFileDescriptor = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_BaseInterface__object **)) epv->f_getFileDescriptor;

  e3->f__cast             = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete           = (void (*)(void*)) epv->f__delete;
  e3->f__exec             = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef            = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef         = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame            = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt          = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType            = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo      = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_close             = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_close;
  e3->f_readn             = (int32_t (*)(void*,int32_t,
    struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readn;
  e3->f_readline          = (int32_t (*)(void*,int32_t,
    struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readline;
  e3->f_readstring        = (int32_t (*)(void*,int32_t,
    struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readstring;
  e3->f_readstring_alloc  = (int32_t (*)(void*,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readstring_alloc;
  e3->f_readint           = (int32_t (*)(void*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_readint;
  e3->f_writen            = (int32_t (*)(void*,int32_t,struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_writen;
  e3->f_writestring       = (int32_t (*)(void*,int32_t,struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_writestring;
  e3->f_writeint          = (int32_t (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_writeint;
  e3->f_setFileDescriptor = (void (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_setFileDescriptor;
  e3->f_getFileDescriptor = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getFileDescriptor;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidlx_rmi_ChildSocket__object*
sidlx_rmi_ChildSocket__remoteConnect(const char *url, sidl_BaseInterface *_ex)
{
  struct sidlx_rmi_ChildSocket__object* self;

  struct sidlx_rmi_ChildSocket__object* s0;
  struct sidlx_rmi_IPv4Socket__object* s1;
  struct sidl_BaseClass__object* s2;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_connectInstance(url, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidlx_rmi_ChildSocket__object*) malloc(
      sizeof(struct sidlx_rmi_ChildSocket__object));

   s0 =                                self;
   s1 =                                &s0->d_sidlx_rmi_ipv4socket;
   s2 =                                &s1->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_rmi_ChildSocket__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s2->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s2->d_sidl_baseinterface.d_object = (void*) self;

  s2->d_data = (void*) instance;
  s2->d_epv  = &s_rem_epv__sidl_baseclass;

  s1->d_sidlx_rmi_socket.d_epv    = &s_rem_epv__sidlx_rmi_socket;
  s1->d_sidlx_rmi_socket.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidlx_rmi_ipv4socket;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidlx_rmi_childsocket;

  self->d_data = (void*) instance;
  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_rmi_ChildSocket__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;


  return self;
}
/* Create an instance that uses an already existing InstanceHandel to connect 
  to an existing remote object. */
static struct sidlx_rmi_ChildSocket__object*
sidlx_rmi_ChildSocket__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidlx_rmi_ChildSocket__object* self;

  struct sidlx_rmi_ChildSocket__object* s0;
  struct sidlx_rmi_IPv4Socket__object* s1;
  struct sidl_BaseClass__object* s2;

  self =
    (struct sidlx_rmi_ChildSocket__object*) malloc(
      sizeof(struct sidlx_rmi_ChildSocket__object));

   s0 =                                self;
   s1 =                                &s0->d_sidlx_rmi_ipv4socket;
   s2 =                                &s1->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_rmi_ChildSocket__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s2->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s2->d_sidl_baseinterface.d_object = (void*) self;

  s2->d_data = (void*) instance;
  s2->d_epv  = &s_rem_epv__sidl_baseclass;

  s1->d_sidlx_rmi_socket.d_epv    = &s_rem_epv__sidlx_rmi_socket;
  s1->d_sidlx_rmi_socket.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidlx_rmi_ipv4socket;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidlx_rmi_childsocket;

  self->d_data = (void*) instance;

  sidl_rmi_InstanceHandle_addRef(instance);
  return self;
}
/* REMOTE: generate remote instance given URL string. */
static struct sidlx_rmi_ChildSocket__object*
sidlx_rmi_ChildSocket__remote(const char *url, sidl_BaseInterface *_ex)
{
  struct sidlx_rmi_ChildSocket__object* self;

  struct sidlx_rmi_ChildSocket__object* s0;
  struct sidlx_rmi_IPv4Socket__object* s1;
  struct sidl_BaseClass__object* s2;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "sidlx.rmi.ChildSocket", _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidlx_rmi_ChildSocket__object*) malloc(
      sizeof(struct sidlx_rmi_ChildSocket__object));

   s0 =                                self;
   s1 =                                &s0->d_sidlx_rmi_ipv4socket;
   s2 =                                &s1->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_rmi_ChildSocket__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s2->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s2->d_sidl_baseinterface.d_object = (void*) self;

  s2->d_data = (void*) instance;
  s2->d_epv  = &s_rem_epv__sidl_baseclass;

  s1->d_sidlx_rmi_socket.d_epv    = &s_rem_epv__sidlx_rmi_socket;
  s1->d_sidlx_rmi_socket.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidlx_rmi_ipv4socket;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidlx_rmi_childsocket;

  self->d_data = (void*) instance;

  return self;
}
