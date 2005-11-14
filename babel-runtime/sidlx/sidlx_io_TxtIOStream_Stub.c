/*
 * File:          sidlx_io_TxtIOStream_Stub.c
 * Symbol:        sidlx.io.TxtIOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for sidlx.io.TxtIOStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidlx_io_TxtIOStream.h"
#include "sidlx_io_TxtIOStream_IOR.h"
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

static const struct sidlx_io_TxtIOStream__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct sidlx_io_TxtIOStream__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = sidlx_io_TxtIOStream__externals();
#else
  _externals = (struct 
    sidlx_io_TxtIOStream__external*)sidl_dynamicLoadIOR("sidlx.io.TxtIOStream",
    "sidlx_io_TxtIOStream__externals") ;
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Constructor function for the class.
 */

sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__create()
{
  return (*(_getExternals()->createObject))();
}

static sidlx_io_TxtIOStream sidlx_io_TxtIOStream__remote(const char* url,
  sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__createRemote(const char* url, sidl_BaseInterface *_ex)
{
  return sidlx_io_TxtIOStream__remote(url, _ex);
}

static struct sidlx_io_TxtIOStream__object* 
  sidlx_io_TxtIOStream__remoteConnect(const char* url, sidl_BaseInterface *_ex);
static struct sidlx_io_TxtIOStream__object* 
  sidlx_io_TxtIOStream__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__connect(const char* url, sidl_BaseInterface *_ex)
{
  return sidlx_io_TxtIOStream__remoteConnect(url, _ex);
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
sidlx_io_TxtIOStream_addRef(
  /* in */ sidlx_io_TxtIOStream self)
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
sidlx_io_TxtIOStream_deleteRef(
  /* in */ sidlx_io_TxtIOStream self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
sidlx_io_TxtIOStream_isSame(
  /* in */ sidlx_io_TxtIOStream self,
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
sidlx_io_TxtIOStream_queryInt(
  /* in */ sidlx_io_TxtIOStream self,
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
sidlx_io_TxtIOStream_isType(
  /* in */ sidlx_io_TxtIOStream self,
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
sidlx_io_TxtIOStream_getClassInfo(
  /* in */ sidlx_io_TxtIOStream self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Method:  setFD[]
 */

void
sidlx_io_TxtIOStream_setFD(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t fd)
{
  (*self->d_epv->f_setFD)(
    self,
    fd);
}

/*
 * returns true iff the stream is at its end, or closed 
 */

sidl_bool
sidlx_io_TxtIOStream_atEnd(
  /* in */ sidlx_io_TxtIOStream self)
{
  return (*self->d_epv->f_atEnd)(
    self);
}

/*
 * low level read an array of bytes 
 */

int32_t
sidlx_io_TxtIOStream_read(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t nbytes,
  /* out array<char,row-major> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_read)(
    self,
    nbytes,
    data,
    _ex);
}

/*
 * low level read 
 */

int32_t
sidlx_io_TxtIOStream_readline(
  /* in */ sidlx_io_TxtIOStream self,
  /* out array<char,row-major> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_readline)(
    self,
    data,
    _ex);
}

/*
 * Method:  get[Bool]
 */

void
sidlx_io_TxtIOStream_getBool(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ sidl_bool* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getBool)(
    self,
    item,
    _ex);
}

/*
 * Method:  get[Char]
 */

void
sidlx_io_TxtIOStream_getChar(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ char* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getChar)(
    self,
    item,
    _ex);
}

/*
 * Method:  get[Int]
 */

void
sidlx_io_TxtIOStream_getInt(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ int32_t* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getInt)(
    self,
    item,
    _ex);
}

/*
 * Method:  get[Long]
 */

void
sidlx_io_TxtIOStream_getLong(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ int64_t* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getLong)(
    self,
    item,
    _ex);
}

/*
 * Method:  get[Float]
 */

void
sidlx_io_TxtIOStream_getFloat(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ float* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getFloat)(
    self,
    item,
    _ex);
}

/*
 * Method:  get[Double]
 */

void
sidlx_io_TxtIOStream_getDouble(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ double* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getDouble)(
    self,
    item,
    _ex);
}

/*
 * Method:  get[Fcomplex]
 */

void
sidlx_io_TxtIOStream_getFcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ struct sidl_fcomplex* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getFcomplex)(
    self,
    item,
    _ex);
}

/*
 * Method:  get[Dcomplex]
 */

void
sidlx_io_TxtIOStream_getDcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ struct sidl_dcomplex* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getDcomplex)(
    self,
    item,
    _ex);
}

/*
 * Method:  get[String]
 */

void
sidlx_io_TxtIOStream_getString(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ char** item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getString)(
    self,
    item,
    _ex);
}

/*
 * flushes the buffer, if any 
 */

void
sidlx_io_TxtIOStream_flush(
  /* in */ sidlx_io_TxtIOStream self)
{
  (*self->d_epv->f_flush)(
    self);
}

/*
 * low level write for an array of bytes 
 */

int32_t
sidlx_io_TxtIOStream_write(
  /* in */ sidlx_io_TxtIOStream self,
  /* in array<char,row-major> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_write)(
    self,
    data,
    _ex);
}

/*
 * Method:  put[Bool]
 */

void
sidlx_io_TxtIOStream_putBool(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ sidl_bool item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putBool)(
    self,
    item,
    _ex);
}

/*
 * Method:  put[Char]
 */

void
sidlx_io_TxtIOStream_putChar(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ char item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putChar)(
    self,
    item,
    _ex);
}

/*
 * Method:  put[Int]
 */

void
sidlx_io_TxtIOStream_putInt(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putInt)(
    self,
    item,
    _ex);
}

/*
 * Method:  put[Long]
 */

void
sidlx_io_TxtIOStream_putLong(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int64_t item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putLong)(
    self,
    item,
    _ex);
}

/*
 * Method:  put[Float]
 */

void
sidlx_io_TxtIOStream_putFloat(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ float item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putFloat)(
    self,
    item,
    _ex);
}

/*
 * Method:  put[Double]
 */

void
sidlx_io_TxtIOStream_putDouble(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ double item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putDouble)(
    self,
    item,
    _ex);
}

/*
 * Method:  put[Fcomplex]
 */

void
sidlx_io_TxtIOStream_putFcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ struct sidl_fcomplex item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putFcomplex)(
    self,
    item,
    _ex);
}

/*
 * Method:  put[Dcomplex]
 */

void
sidlx_io_TxtIOStream_putDcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ struct sidl_dcomplex item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putDcomplex)(
    self,
    item,
    _ex);
}

/*
 * Method:  put[String]
 */

void
sidlx_io_TxtIOStream_putString(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ const char* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putString)(
    self,
    item,
    _ex);
}

/*
 * Cast method for interface and class type conversions.
 */

sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__cast(
  void* obj)
{
  sidlx_io_TxtIOStream cast = NULL;

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidlx.io.TxtIOStream",
      (void*)sidlx_io_TxtIOStream__IHConnect);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (sidlx_io_TxtIOStream) (*base->d_epv->f__cast)(
      base->d_object,
      "sidlx.io.TxtIOStream");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
sidlx_io_TxtIOStream__cast2(
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
sidlx_io_TxtIOStream__exec(
  /* in */ sidlx_io_TxtIOStream self,
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
sidlx_io_TxtIOStream__getURL(
  /* in */ sidlx_io_TxtIOStream self)
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
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidlx_io_TxtIOStream__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidlx_io_TxtIOStream__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_create1d(int32_t len)
{
  return (struct 
    sidlx_io_TxtIOStream__array*)sidl_interface__array_create1d(len);
}

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_create1dInit(
  int32_t len, 
  sidlx_io_TxtIOStream* data)
{
  return (struct 
    sidlx_io_TxtIOStream__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    sidlx_io_TxtIOStream__array*)sidl_interface__array_create2dCol(m, n);
}

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    sidlx_io_TxtIOStream__array*)sidl_interface__array_create2dRow(m, n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_borrow(
  sidlx_io_TxtIOStream* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct sidlx_io_TxtIOStream__array*)sidl_interface__array_borrow(
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
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_smartCopy(
  struct sidlx_io_TxtIOStream__array *array)
{
  return (struct sidlx_io_TxtIOStream__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
sidlx_io_TxtIOStream__array_addRef(
  struct sidlx_io_TxtIOStream__array* array)
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
sidlx_io_TxtIOStream__array_deleteRef(
  struct sidlx_io_TxtIOStream__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__array_get1(
  const struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1)
{
  return (sidlx_io_TxtIOStream)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__array_get2(
  const struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (sidlx_io_TxtIOStream)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__array_get3(
  const struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (sidlx_io_TxtIOStream)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__array_get4(
  const struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (sidlx_io_TxtIOStream)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__array_get5(
  const struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (sidlx_io_TxtIOStream)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__array_get6(
  const struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (sidlx_io_TxtIOStream)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__array_get7(
  const struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (sidlx_io_TxtIOStream)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidlx_io_TxtIOStream
sidlx_io_TxtIOStream__array_get(
  const struct sidlx_io_TxtIOStream__array* array,
  const int32_t indices[])
{
  return (sidlx_io_TxtIOStream)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidlx_io_TxtIOStream__array_set1(
  struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  sidlx_io_TxtIOStream const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidlx_io_TxtIOStream__array_set2(
  struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  sidlx_io_TxtIOStream const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidlx_io_TxtIOStream__array_set3(
  struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidlx_io_TxtIOStream const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidlx_io_TxtIOStream__array_set4(
  struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidlx_io_TxtIOStream const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidlx_io_TxtIOStream__array_set5(
  struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidlx_io_TxtIOStream const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidlx_io_TxtIOStream__array_set6(
  struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidlx_io_TxtIOStream const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidlx_io_TxtIOStream__array_set7(
  struct sidlx_io_TxtIOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidlx_io_TxtIOStream const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidlx_io_TxtIOStream__array_set(
  struct sidlx_io_TxtIOStream__array* array,
  const int32_t indices[],
  sidlx_io_TxtIOStream const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidlx_io_TxtIOStream__array_dimen(
  const struct sidlx_io_TxtIOStream__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_io_TxtIOStream__array_lower(
  const struct sidlx_io_TxtIOStream__array* array,
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
sidlx_io_TxtIOStream__array_upper(
  const struct sidlx_io_TxtIOStream__array* array,
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
sidlx_io_TxtIOStream__array_length(
  const struct sidlx_io_TxtIOStream__array* array,
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
sidlx_io_TxtIOStream__array_stride(
  const struct sidlx_io_TxtIOStream__array* array,
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
sidlx_io_TxtIOStream__array_isColumnOrder(
  const struct sidlx_io_TxtIOStream__array* array)
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
sidlx_io_TxtIOStream__array_isRowOrder(
  const struct sidlx_io_TxtIOStream__array* array)
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
sidlx_io_TxtIOStream__array_copy(
  const struct sidlx_io_TxtIOStream__array* src,
  struct sidlx_io_TxtIOStream__array* dest)
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
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_slice(
  struct sidlx_io_TxtIOStream__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct sidlx_io_TxtIOStream__array*)
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
struct sidlx_io_TxtIOStream__array*
sidlx_io_TxtIOStream__array_ensure(
  struct sidlx_io_TxtIOStream__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct sidlx_io_TxtIOStream__array*)
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
static struct sidl_recursive_mutex_t sidlx_io_TxtIOStream__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_io_TxtIOStream__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_io_TxtIOStream__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_io_TxtIOStream__mutex )==EDEADLOCK) */
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

static struct sidlx_io_TxtIOStream__epv s_rem_epv__sidlx_io_txtiostream;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;

static struct sidlx_io_IOStream__epv s_rem_epv__sidlx_io_iostream;

static struct sidlx_io_IStream__epv s_rem_epv__sidlx_io_istream;

static struct sidlx_io_OStream__epv s_rem_epv__sidlx_io_ostream;

/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidlx_io_TxtIOStream__cast(
struct sidlx_io_TxtIOStream__object* self,
const char* name)
{
  void* cast = NULL;

  struct sidlx_io_TxtIOStream__object* s0;
  struct sidl_BaseClass__object* s1;
   s0 =                               self;
   s1 =                               &s0->d_sidl_baseclass;

  if (!strcmp(name, "sidlx.io.TxtIOStream")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidlx.io.IOStream")) {
    cast = (void*) &s0->d_sidlx_io_iostream;
  } else if (!strcmp(name, "sidlx.io.IStream")) {
    cast = (void*) &s0->d_sidlx_io_istream;
  } else if (!strcmp(name, "sidlx.io.OStream")) {
    cast = (void*) &s0->d_sidlx_io_ostream;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }
  else if(sidlx_io_TxtIOStream_isType(self, name)) {
    void* (*func)(sidl_rmi_InstanceHandle) = 
      (void* (*)(sidl_rmi_InstanceHandle)) 
      sidl_rmi_ConnectRegistry_getConnect(name);
    cast =  (*func)((sidl_rmi_InstanceHandle)self->d_data);
  }

  return cast;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidlx_io_TxtIOStream__delete(
  struct sidlx_io_TxtIOStream__object* self)
{
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidlx_io_TxtIOStream__getURL(
  struct sidlx_io_TxtIOStream__object* self)
{
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getURL(conn, &_ex);
  }
  return NULL;
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_sidlx_io_TxtIOStream__exec(
  struct sidlx_io_TxtIOStream__object* self,
  const char* methodName,
  sidl_io_Deserializer inArgs,
  sidl_io_Serializer outArgs)
{
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidlx_io_TxtIOStream_addRef(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidlx_io_TxtIOStream_deleteRef(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */)
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
remote_sidlx_io_TxtIOStream_isSame(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ struct sidl_BaseInterface__object* iobj)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:queryInt */
static struct sidl_BaseInterface__object*
remote_sidlx_io_TxtIOStream_queryInt(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ const char* name)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_sidlx_io_TxtIOStream_isType(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
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
remote_sidlx_io_TxtIOStream_getClassInfo(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:setFD */
static void
remote_sidlx_io_TxtIOStream_setFD(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ int32_t fd)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "setFD", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "fd", fd, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:atEnd */
static sidl_bool
remote_sidlx_io_TxtIOStream_atEnd(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "atEnd", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  sidl_bool _retval;

  /* pack in and inout arguments */

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

/* REMOTE METHOD STUB:read */
static int32_t
remote_sidlx_io_TxtIOStream_read(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ int32_t nbytes,
  /* out array<char,row-major> */ struct sidl_char__array** data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "read", _ex2 );
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
remote_sidlx_io_TxtIOStream_readline(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out array<char,row-major> */ struct sidl_char__array** data,
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

/* REMOTE METHOD STUB:getBool */
static void
remote_sidlx_io_TxtIOStream_getBool(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ sidl_bool* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getBool", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackBool( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getChar */
static void
remote_sidlx_io_TxtIOStream_getChar(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ char* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getChar", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackChar( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getInt */
static void
remote_sidlx_io_TxtIOStream_getInt(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ int32_t* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getInt", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getLong */
static void
remote_sidlx_io_TxtIOStream_getLong(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ int64_t* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getLong", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackLong( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getFloat */
static void
remote_sidlx_io_TxtIOStream_getFloat(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ float* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getFloat", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackFloat( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getDouble */
static void
remote_sidlx_io_TxtIOStream_getDouble(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ double* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getDouble", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackDouble( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getFcomplex */
static void
remote_sidlx_io_TxtIOStream_getFcomplex(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ struct sidl_fcomplex* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getFcomplex", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackFcomplex( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getDcomplex */
static void
remote_sidlx_io_TxtIOStream_getDcomplex(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ struct sidl_dcomplex* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getDcomplex", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackDcomplex( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:getString */
static void
remote_sidlx_io_TxtIOStream_getString(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* out */ char** item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "getString", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* check if exception thrown. */
  /* FIXME */

  /* extract return value */

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackString( _rsvp, "item", item, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:flush */
static void
remote_sidlx_io_TxtIOStream_flush(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "flush", _ex2 );
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

/* REMOTE METHOD STUB:write */
static int32_t
remote_sidlx_io_TxtIOStream_write(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in array<char,row-major> */ struct sidl_char__array* data,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "write", _ex2 );
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

/* REMOTE METHOD STUB:putBool */
static void
remote_sidlx_io_TxtIOStream_putBool(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ sidl_bool item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putBool", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packBool( _inv, "item", item, _ex2);

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

/* REMOTE METHOD STUB:putChar */
static void
remote_sidlx_io_TxtIOStream_putChar(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ char item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putChar", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packChar( _inv, "item", item, _ex2);

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

/* REMOTE METHOD STUB:putInt */
static void
remote_sidlx_io_TxtIOStream_putInt(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ int32_t item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putInt", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "item", item, _ex2);

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

/* REMOTE METHOD STUB:putLong */
static void
remote_sidlx_io_TxtIOStream_putLong(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ int64_t item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putLong", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packLong( _inv, "item", item, _ex2);

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

/* REMOTE METHOD STUB:putFloat */
static void
remote_sidlx_io_TxtIOStream_putFloat(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ float item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putFloat", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packFloat( _inv, "item", item, _ex2);

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

/* REMOTE METHOD STUB:putDouble */
static void
remote_sidlx_io_TxtIOStream_putDouble(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ double item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putDouble", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packDouble( _inv, "item", item, _ex2);

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

/* REMOTE METHOD STUB:putFcomplex */
static void
remote_sidlx_io_TxtIOStream_putFcomplex(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ struct sidl_fcomplex item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putFcomplex", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packFcomplex( _inv, "item", item, _ex2);

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

/* REMOTE METHOD STUB:putDcomplex */
static void
remote_sidlx_io_TxtIOStream_putDcomplex(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ struct sidl_dcomplex item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putDcomplex", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packDcomplex( _inv, "item", item, _ex2);

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

/* REMOTE METHOD STUB:putString */
static void
remote_sidlx_io_TxtIOStream_putString(
  /* in */ struct sidlx_io_TxtIOStream__object* self /* TLD */,
  /* in */ const char* item,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  sidl_BaseInterface *_ex2 =_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "putString", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "item", item, _ex2);

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
static void sidlx_io_TxtIOStream__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidlx_io_TxtIOStream__epv* epv = &s_rem_epv__sidlx_io_txtiostream;
  struct sidl_BaseClass__epv*       e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*   e1  = &s_rem_epv__sidl_baseinterface;
  struct sidlx_io_IOStream__epv*    e2  = &s_rem_epv__sidlx_io_iostream;
  struct sidlx_io_IStream__epv*     e3  = &s_rem_epv__sidlx_io_istream;
  struct sidlx_io_OStream__epv*     e4  = &s_rem_epv__sidlx_io_ostream;

  epv->f__cast             = remote_sidlx_io_TxtIOStream__cast;
  epv->f__delete           = remote_sidlx_io_TxtIOStream__delete;
  epv->f__exec             = remote_sidlx_io_TxtIOStream__exec;
  epv->f__getURL           = remote_sidlx_io_TxtIOStream__getURL;
  epv->f__ctor             = NULL;
  epv->f__dtor             = NULL;
  epv->f_addRef            = remote_sidlx_io_TxtIOStream_addRef;
  epv->f_deleteRef         = remote_sidlx_io_TxtIOStream_deleteRef;
  epv->f_isSame            = remote_sidlx_io_TxtIOStream_isSame;
  epv->f_queryInt          = remote_sidlx_io_TxtIOStream_queryInt;
  epv->f_isType            = remote_sidlx_io_TxtIOStream_isType;
  epv->f_getClassInfo      = remote_sidlx_io_TxtIOStream_getClassInfo;
  epv->f_setFD             = remote_sidlx_io_TxtIOStream_setFD;
  epv->f_atEnd             = remote_sidlx_io_TxtIOStream_atEnd;
  epv->f_read              = remote_sidlx_io_TxtIOStream_read;
  epv->f_readline          = remote_sidlx_io_TxtIOStream_readline;
  epv->f_getBool           = remote_sidlx_io_TxtIOStream_getBool;
  epv->f_getChar           = remote_sidlx_io_TxtIOStream_getChar;
  epv->f_getInt            = remote_sidlx_io_TxtIOStream_getInt;
  epv->f_getLong           = remote_sidlx_io_TxtIOStream_getLong;
  epv->f_getFloat          = remote_sidlx_io_TxtIOStream_getFloat;
  epv->f_getDouble         = remote_sidlx_io_TxtIOStream_getDouble;
  epv->f_getFcomplex       = remote_sidlx_io_TxtIOStream_getFcomplex;
  epv->f_getDcomplex       = remote_sidlx_io_TxtIOStream_getDcomplex;
  epv->f_getString         = remote_sidlx_io_TxtIOStream_getString;
  epv->f_flush             = remote_sidlx_io_TxtIOStream_flush;
  epv->f_write             = remote_sidlx_io_TxtIOStream_write;
  epv->f_putBool           = remote_sidlx_io_TxtIOStream_putBool;
  epv->f_putChar           = remote_sidlx_io_TxtIOStream_putChar;
  epv->f_putInt            = remote_sidlx_io_TxtIOStream_putInt;
  epv->f_putLong           = remote_sidlx_io_TxtIOStream_putLong;
  epv->f_putFloat          = remote_sidlx_io_TxtIOStream_putFloat;
  epv->f_putDouble         = remote_sidlx_io_TxtIOStream_putDouble;
  epv->f_putFcomplex       = remote_sidlx_io_TxtIOStream_putFcomplex;
  epv->f_putDcomplex       = remote_sidlx_io_TxtIOStream_putDcomplex;
  epv->f_putString         = remote_sidlx_io_TxtIOStream_putString;

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

  e2->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete      = (void (*)(void*)) epv->f__delete;
  e2->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_atEnd        = (sidl_bool (*)(void*)) epv->f_atEnd;
  e2->f_read         = (int32_t (*)(void*,int32_t,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_read;
  e2->f_readline     = (int32_t (*)(void*,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readline;
  e2->f_getBool      = (void (*)(void*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_getBool;
  e2->f_getChar      = (void (*)(void*,char*,
    struct sidl_BaseInterface__object **)) epv->f_getChar;
  e2->f_getInt       = (void (*)(void*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_getInt;
  e2->f_getLong      = (void (*)(void*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_getLong;
  e2->f_getFloat     = (void (*)(void*,float*,
    struct sidl_BaseInterface__object **)) epv->f_getFloat;
  e2->f_getDouble    = (void (*)(void*,double*,
    struct sidl_BaseInterface__object **)) epv->f_getDouble;
  e2->f_getFcomplex  = (void (*)(void*,struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_getFcomplex;
  e2->f_getDcomplex  = (void (*)(void*,struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_getDcomplex;
  e2->f_getString    = (void (*)(void*,char**,
    struct sidl_BaseInterface__object **)) epv->f_getString;
  e2->f_flush        = (void (*)(void*)) epv->f_flush;
  e2->f_write        = (int32_t (*)(void*,struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_write;
  e2->f_putBool      = (void (*)(void*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_putBool;
  e2->f_putChar      = (void (*)(void*,char,
    struct sidl_BaseInterface__object **)) epv->f_putChar;
  e2->f_putInt       = (void (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_putInt;
  e2->f_putLong      = (void (*)(void*,int64_t,
    struct sidl_BaseInterface__object **)) epv->f_putLong;
  e2->f_putFloat     = (void (*)(void*,float,
    struct sidl_BaseInterface__object **)) epv->f_putFloat;
  e2->f_putDouble    = (void (*)(void*,double,
    struct sidl_BaseInterface__object **)) epv->f_putDouble;
  e2->f_putFcomplex  = (void (*)(void*,struct sidl_fcomplex,
    struct sidl_BaseInterface__object **)) epv->f_putFcomplex;
  e2->f_putDcomplex  = (void (*)(void*,struct sidl_dcomplex,
    struct sidl_BaseInterface__object **)) epv->f_putDcomplex;
  e2->f_putString    = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_putString;

  e3->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete      = (void (*)(void*)) epv->f__delete;
  e3->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_atEnd        = (sidl_bool (*)(void*)) epv->f_atEnd;
  e3->f_read         = (int32_t (*)(void*,int32_t,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_read;
  e3->f_readline     = (int32_t (*)(void*,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readline;
  e3->f_getBool      = (void (*)(void*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_getBool;
  e3->f_getChar      = (void (*)(void*,char*,
    struct sidl_BaseInterface__object **)) epv->f_getChar;
  e3->f_getInt       = (void (*)(void*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_getInt;
  e3->f_getLong      = (void (*)(void*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_getLong;
  e3->f_getFloat     = (void (*)(void*,float*,
    struct sidl_BaseInterface__object **)) epv->f_getFloat;
  e3->f_getDouble    = (void (*)(void*,double*,
    struct sidl_BaseInterface__object **)) epv->f_getDouble;
  e3->f_getFcomplex  = (void (*)(void*,struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_getFcomplex;
  e3->f_getDcomplex  = (void (*)(void*,struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_getDcomplex;
  e3->f_getString    = (void (*)(void*,char**,
    struct sidl_BaseInterface__object **)) epv->f_getString;

  e4->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete      = (void (*)(void*)) epv->f__delete;
  e4->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e4->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e4->f_flush        = (void (*)(void*)) epv->f_flush;
  e4->f_write        = (int32_t (*)(void*,struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_write;
  e4->f_putBool      = (void (*)(void*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_putBool;
  e4->f_putChar      = (void (*)(void*,char,
    struct sidl_BaseInterface__object **)) epv->f_putChar;
  e4->f_putInt       = (void (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_putInt;
  e4->f_putLong      = (void (*)(void*,int64_t,
    struct sidl_BaseInterface__object **)) epv->f_putLong;
  e4->f_putFloat     = (void (*)(void*,float,
    struct sidl_BaseInterface__object **)) epv->f_putFloat;
  e4->f_putDouble    = (void (*)(void*,double,
    struct sidl_BaseInterface__object **)) epv->f_putDouble;
  e4->f_putFcomplex  = (void (*)(void*,struct sidl_fcomplex,
    struct sidl_BaseInterface__object **)) epv->f_putFcomplex;
  e4->f_putDcomplex  = (void (*)(void*,struct sidl_dcomplex,
    struct sidl_BaseInterface__object **)) epv->f_putDcomplex;
  e4->f_putString    = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_putString;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidlx_io_TxtIOStream__object*
sidlx_io_TxtIOStream__remoteConnect(const char *url, sidl_BaseInterface *_ex)
{
  struct sidlx_io_TxtIOStream__object* self;

  struct sidlx_io_TxtIOStream__object* s0;
  struct sidl_BaseClass__object* s1;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_connectInstance(url, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidlx_io_TxtIOStream__object*) malloc(
      sizeof(struct sidlx_io_TxtIOStream__object));

   s0 =                               self;
   s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_io_TxtIOStream__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_sidlx_io_iostream.d_epv    = &s_rem_epv__sidlx_io_iostream;
  s0->d_sidlx_io_iostream.d_object = (void*) self;

  s0->d_sidlx_io_istream.d_epv    = &s_rem_epv__sidlx_io_istream;
  s0->d_sidlx_io_istream.d_object = (void*) self;

  s0->d_sidlx_io_ostream.d_epv    = &s_rem_epv__sidlx_io_ostream;
  s0->d_sidlx_io_ostream.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidlx_io_txtiostream;

  self->d_data = (void*) instance;
  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_io_TxtIOStream__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;


  return self;
}
/* Create an instance that uses an already existing InstanceHandel to connect 
  to an existing remote object. */
static struct sidlx_io_TxtIOStream__object*
sidlx_io_TxtIOStream__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidlx_io_TxtIOStream__object* self;

  struct sidlx_io_TxtIOStream__object* s0;
  struct sidl_BaseClass__object* s1;

  self =
    (struct sidlx_io_TxtIOStream__object*) malloc(
      sizeof(struct sidlx_io_TxtIOStream__object));

   s0 =                               self;
   s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_io_TxtIOStream__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_sidlx_io_iostream.d_epv    = &s_rem_epv__sidlx_io_iostream;
  s0->d_sidlx_io_iostream.d_object = (void*) self;

  s0->d_sidlx_io_istream.d_epv    = &s_rem_epv__sidlx_io_istream;
  s0->d_sidlx_io_istream.d_object = (void*) self;

  s0->d_sidlx_io_ostream.d_epv    = &s_rem_epv__sidlx_io_ostream;
  s0->d_sidlx_io_ostream.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidlx_io_txtiostream;

  self->d_data = (void*) instance;

  sidl_rmi_InstanceHandle_addRef(instance);
  return self;
}
/* REMOTE: generate remote instance given URL string. */
static struct sidlx_io_TxtIOStream__object*
sidlx_io_TxtIOStream__remote(const char *url, sidl_BaseInterface *_ex)
{
  struct sidlx_io_TxtIOStream__object* self;

  struct sidlx_io_TxtIOStream__object* s0;
  struct sidl_BaseClass__object* s1;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "sidlx.io.TxtIOStream", _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidlx_io_TxtIOStream__object*) malloc(
      sizeof(struct sidlx_io_TxtIOStream__object));

   s0 =                               self;
   s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_io_TxtIOStream__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_sidlx_io_iostream.d_epv    = &s_rem_epv__sidlx_io_iostream;
  s0->d_sidlx_io_iostream.d_object = (void*) self;

  s0->d_sidlx_io_istream.d_epv    = &s_rem_epv__sidlx_io_istream;
  s0->d_sidlx_io_istream.d_object = (void*) self;

  s0->d_sidlx_io_ostream.d_epv    = &s_rem_epv__sidlx_io_ostream;
  s0->d_sidlx_io_ostream.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidlx_io_txtiostream;

  self->d_data = (void*) instance;

  return self;
}
