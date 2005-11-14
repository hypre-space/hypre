/*
 * File:          sidlx_io_IOStream_Stub.c
 * Symbol:        sidlx.io.IOStream-v0.1
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Description:   Client-side glue code for sidlx.io.IOStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidlx_io_IOStream.h"
#include "sidlx_io_IOStream_IOR.h"
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

static struct sidlx_io_IOStream__object* sidlx_io_IOStream__remoteConnect(const 
  char* url, sidl_BaseInterface *_ex);
static struct sidlx_io_IOStream__object* 
  sidlx_io_IOStream__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

sidlx_io_IOStream
sidlx_io_IOStream__connect(const char* url, sidl_BaseInterface *_ex)
{
  return sidlx_io_IOStream__remoteConnect(url, _ex);
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
sidlx_io_IOStream_addRef(
  /* in */ sidlx_io_IOStream self)
{
  (*self->d_epv->f_addRef)(
    self->d_object);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
sidlx_io_IOStream_deleteRef(
  /* in */ sidlx_io_IOStream self)
{
  (*self->d_epv->f_deleteRef)(
    self->d_object);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
sidlx_io_IOStream_isSame(
  /* in */ sidlx_io_IOStream self,
  /* in */ sidl_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
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
sidlx_io_IOStream_queryInt(
  /* in */ sidlx_io_IOStream self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_queryInt)(
    self->d_object,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

sidl_bool
sidlx_io_IOStream_isType(
  /* in */ sidlx_io_IOStream self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_isType)(
    self->d_object,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
sidlx_io_IOStream_getClassInfo(
  /* in */ sidlx_io_IOStream self)
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object);
}

/*
 * returns true iff the stream is at its end, or closed 
 */

sidl_bool
sidlx_io_IOStream_atEnd(
  /* in */ sidlx_io_IOStream self)
{
  return (*self->d_epv->f_atEnd)(
    self->d_object);
}

/*
 * low level read an array of bytes 
 */

int32_t
sidlx_io_IOStream_read(
  /* in */ sidlx_io_IOStream self,
  /* in */ int32_t nbytes,
  /* out array<char,row-major> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_read)(
    self->d_object,
    nbytes,
    data,
    _ex);
}

/*
 * low level read 
 */

int32_t
sidlx_io_IOStream_readline(
  /* in */ sidlx_io_IOStream self,
  /* out array<char,row-major> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_readline)(
    self->d_object,
    data,
    _ex);
}

/*
 * Method:  get[Bool]
 */

void
sidlx_io_IOStream_getBool(
  /* in */ sidlx_io_IOStream self,
  /* out */ sidl_bool* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getBool)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  get[Char]
 */

void
sidlx_io_IOStream_getChar(
  /* in */ sidlx_io_IOStream self,
  /* out */ char* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getChar)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  get[Int]
 */

void
sidlx_io_IOStream_getInt(
  /* in */ sidlx_io_IOStream self,
  /* out */ int32_t* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getInt)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  get[Long]
 */

void
sidlx_io_IOStream_getLong(
  /* in */ sidlx_io_IOStream self,
  /* out */ int64_t* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getLong)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  get[Float]
 */

void
sidlx_io_IOStream_getFloat(
  /* in */ sidlx_io_IOStream self,
  /* out */ float* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getFloat)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  get[Double]
 */

void
sidlx_io_IOStream_getDouble(
  /* in */ sidlx_io_IOStream self,
  /* out */ double* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getDouble)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  get[Fcomplex]
 */

void
sidlx_io_IOStream_getFcomplex(
  /* in */ sidlx_io_IOStream self,
  /* out */ struct sidl_fcomplex* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getFcomplex)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  get[Dcomplex]
 */

void
sidlx_io_IOStream_getDcomplex(
  /* in */ sidlx_io_IOStream self,
  /* out */ struct sidl_dcomplex* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getDcomplex)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  get[String]
 */

void
sidlx_io_IOStream_getString(
  /* in */ sidlx_io_IOStream self,
  /* out */ char** item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_getString)(
    self->d_object,
    item,
    _ex);
}

/*
 * flushes the buffer, if any 
 */

void
sidlx_io_IOStream_flush(
  /* in */ sidlx_io_IOStream self)
{
  (*self->d_epv->f_flush)(
    self->d_object);
}

/*
 * low level write for an array of bytes 
 */

int32_t
sidlx_io_IOStream_write(
  /* in */ sidlx_io_IOStream self,
  /* in array<char,row-major> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*self->d_epv->f_write)(
    self->d_object,
    data,
    _ex);
}

/*
 * Method:  put[Bool]
 */

void
sidlx_io_IOStream_putBool(
  /* in */ sidlx_io_IOStream self,
  /* in */ sidl_bool item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putBool)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  put[Char]
 */

void
sidlx_io_IOStream_putChar(
  /* in */ sidlx_io_IOStream self,
  /* in */ char item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putChar)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  put[Int]
 */

void
sidlx_io_IOStream_putInt(
  /* in */ sidlx_io_IOStream self,
  /* in */ int32_t item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putInt)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  put[Long]
 */

void
sidlx_io_IOStream_putLong(
  /* in */ sidlx_io_IOStream self,
  /* in */ int64_t item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putLong)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  put[Float]
 */

void
sidlx_io_IOStream_putFloat(
  /* in */ sidlx_io_IOStream self,
  /* in */ float item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putFloat)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  put[Double]
 */

void
sidlx_io_IOStream_putDouble(
  /* in */ sidlx_io_IOStream self,
  /* in */ double item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putDouble)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  put[Fcomplex]
 */

void
sidlx_io_IOStream_putFcomplex(
  /* in */ sidlx_io_IOStream self,
  /* in */ struct sidl_fcomplex item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putFcomplex)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  put[Dcomplex]
 */

void
sidlx_io_IOStream_putDcomplex(
  /* in */ sidlx_io_IOStream self,
  /* in */ struct sidl_dcomplex item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putDcomplex)(
    self->d_object,
    item,
    _ex);
}

/*
 * Method:  put[String]
 */

void
sidlx_io_IOStream_putString(
  /* in */ sidlx_io_IOStream self,
  /* in */ const char* item,
  /* out */ sidl_BaseInterface *_ex)
{
  (*self->d_epv->f_putString)(
    self->d_object,
    item,
    _ex);
}

/*
 * Cast method for interface and class type conversions.
 */

sidlx_io_IOStream
sidlx_io_IOStream__cast(
  void* obj)
{
  sidlx_io_IOStream cast = NULL;

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("sidlx.io.IOStream",
      (void*)sidlx_io_IOStream__IHConnect);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (sidlx_io_IOStream) (*base->d_epv->f__cast)(
      base->d_object,
      "sidlx.io.IOStream");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
sidlx_io_IOStream__cast2(
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
sidlx_io_IOStream__exec(
  /* in */ sidlx_io_IOStream self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs)
{
  (*self->d_epv->f__exec)(
  self->d_object,
  methodName,
  inArgs,
  outArgs);
}

/*
 * Get the URL of the Implementation of this object (for RMI)
 */

char*
sidlx_io_IOStream__getURL(
  /* in */ sidlx_io_IOStream self)
{
  return (*self->d_epv->f__getURL)(
  self->d_object);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidlx_io_IOStream__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    sidlx_io_IOStream__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_create1d(int32_t len)
{
  return (struct sidlx_io_IOStream__array*)sidl_interface__array_create1d(len);
}

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_create1dInit(
  int32_t len, 
  sidlx_io_IOStream* data)
{
  return (struct 
    sidlx_io_IOStream__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_create2dCol(int32_t m, int32_t n)
{
  return (struct sidlx_io_IOStream__array*)sidl_interface__array_create2dCol(m,
    n);
}

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_create2dRow(int32_t m, int32_t n)
{
  return (struct sidlx_io_IOStream__array*)sidl_interface__array_create2dRow(m,
    n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_borrow(
  sidlx_io_IOStream* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct sidlx_io_IOStream__array*)sidl_interface__array_borrow(
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
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_smartCopy(
  struct sidlx_io_IOStream__array *array)
{
  return (struct sidlx_io_IOStream__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
sidlx_io_IOStream__array_addRef(
  struct sidlx_io_IOStream__array* array)
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
sidlx_io_IOStream__array_deleteRef(
  struct sidlx_io_IOStream__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidlx_io_IOStream
sidlx_io_IOStream__array_get1(
  const struct sidlx_io_IOStream__array* array,
  const int32_t i1)
{
  return (sidlx_io_IOStream)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidlx_io_IOStream
sidlx_io_IOStream__array_get2(
  const struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (sidlx_io_IOStream)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidlx_io_IOStream
sidlx_io_IOStream__array_get3(
  const struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (sidlx_io_IOStream)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidlx_io_IOStream
sidlx_io_IOStream__array_get4(
  const struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (sidlx_io_IOStream)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidlx_io_IOStream
sidlx_io_IOStream__array_get5(
  const struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (sidlx_io_IOStream)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidlx_io_IOStream
sidlx_io_IOStream__array_get6(
  const struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (sidlx_io_IOStream)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidlx_io_IOStream
sidlx_io_IOStream__array_get7(
  const struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (sidlx_io_IOStream)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidlx_io_IOStream
sidlx_io_IOStream__array_get(
  const struct sidlx_io_IOStream__array* array,
  const int32_t indices[])
{
  return (sidlx_io_IOStream)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidlx_io_IOStream__array_set1(
  struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  sidlx_io_IOStream const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidlx_io_IOStream__array_set2(
  struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  sidlx_io_IOStream const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidlx_io_IOStream__array_set3(
  struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidlx_io_IOStream const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidlx_io_IOStream__array_set4(
  struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidlx_io_IOStream const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidlx_io_IOStream__array_set5(
  struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidlx_io_IOStream const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidlx_io_IOStream__array_set6(
  struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidlx_io_IOStream const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidlx_io_IOStream__array_set7(
  struct sidlx_io_IOStream__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidlx_io_IOStream const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidlx_io_IOStream__array_set(
  struct sidlx_io_IOStream__array* array,
  const int32_t indices[],
  sidlx_io_IOStream const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidlx_io_IOStream__array_dimen(
  const struct sidlx_io_IOStream__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_io_IOStream__array_lower(
  const struct sidlx_io_IOStream__array* array,
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
sidlx_io_IOStream__array_upper(
  const struct sidlx_io_IOStream__array* array,
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
sidlx_io_IOStream__array_length(
  const struct sidlx_io_IOStream__array* array,
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
sidlx_io_IOStream__array_stride(
  const struct sidlx_io_IOStream__array* array,
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
sidlx_io_IOStream__array_isColumnOrder(
  const struct sidlx_io_IOStream__array* array)
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
sidlx_io_IOStream__array_isRowOrder(
  const struct sidlx_io_IOStream__array* array)
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
sidlx_io_IOStream__array_copy(
  const struct sidlx_io_IOStream__array* src,
  struct sidlx_io_IOStream__array* dest)
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
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_slice(
  struct sidlx_io_IOStream__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct sidlx_io_IOStream__array*)
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
struct sidlx_io_IOStream__array*
sidlx_io_IOStream__array_ensure(
  struct sidlx_io_IOStream__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct sidlx_io_IOStream__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

#include <stdlib.h>
#include <string.h>
#include "sidl_rmi_ProtocolFactory.h"
#include "sidl_rmi_InstanceHandle.h"
#include "sidl_rmi_Invocation.h"
#include "sidl_rmi_Response.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidlx_io__IOStream__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_io__IOStream__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_io__IOStream__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_io__IOStream__mutex )==EDEADLOCK) */
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

static struct sidlx_io__IOStream__epv s_rem_epv__sidlx_io__iostream;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

static struct sidlx_io_IOStream__epv s_rem_epv__sidlx_io_iostream;

static struct sidlx_io_IStream__epv s_rem_epv__sidlx_io_istream;

static struct sidlx_io_OStream__epv s_rem_epv__sidlx_io_ostream;

/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_sidlx_io__IOStream__cast(
struct sidlx_io__IOStream__object* self,
const char* name)
{
  void* cast = NULL;

  struct sidlx_io__IOStream__object* s0;
   s0 =                             self;

  if (!strcmp(name, "sidlx.io._IOStream")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s0->d_sidl_baseinterface;
  } else if (!strcmp(name, "sidlx.io.IOStream")) {
    cast = (void*) &s0->d_sidlx_io_iostream;
  } else if (!strcmp(name, "sidlx.io.IStream")) {
    cast = (void*) &s0->d_sidlx_io_istream;
  } else if (!strcmp(name, "sidlx.io.OStream")) {
    cast = (void*) &s0->d_sidlx_io_ostream;
  }
  else if ((*self->d_epv->f_isType)(self,name)) {
    void* (*func)(sidl_rmi_InstanceHandle) = 
      (void* (*)(sidl_rmi_InstanceHandle)) 
      sidl_rmi_ConnectRegistry_getConnect(name);
    cast =  (*func)((sidl_rmi_InstanceHandle)self->d_data);
  }

  return cast;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_sidlx_io__IOStream__delete(
  struct sidlx_io__IOStream__object* self)
{
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_sidlx_io__IOStream__getURL(
  struct sidlx_io__IOStream__object* self)
{
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getURL(conn, &_ex);
  }
  return NULL;
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_sidlx_io__IOStream__exec(
  struct sidlx_io__IOStream__object* self,
  const char* methodName,
  sidl_io_Deserializer inArgs,
  sidl_io_Serializer outArgs)
{
}

/* REMOTE METHOD STUB:addRef */
static void
remote_sidlx_io__IOStream_addRef(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_sidlx_io__IOStream_deleteRef(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */)
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
remote_sidlx_io__IOStream_isSame(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
  /* in */ struct sidl_BaseInterface__object* iobj)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:queryInt */
static struct sidl_BaseInterface__object*
remote_sidlx_io__IOStream_queryInt(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
  /* in */ const char* name)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_sidlx_io__IOStream_isType(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getClassInfo(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:atEnd */
static sidl_bool
remote_sidlx_io__IOStream_atEnd(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */)
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
remote_sidlx_io__IOStream_read(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_readline(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getBool(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getChar(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getInt(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getLong(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getFloat(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getDouble(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getFcomplex(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getDcomplex(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_getString(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_flush(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */)
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
remote_sidlx_io__IOStream_write(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putBool(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putChar(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putInt(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putLong(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putFloat(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putDouble(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putFcomplex(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putDcomplex(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
remote_sidlx_io__IOStream_putString(
  /* in */ struct sidlx_io__IOStream__object* self /* TLD */,
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
static void sidlx_io__IOStream__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct sidlx_io__IOStream__epv* epv = &s_rem_epv__sidlx_io__iostream;
  struct sidl_BaseInterface__epv* e0  = &s_rem_epv__sidl_baseinterface;
  struct sidlx_io_IOStream__epv*  e1  = &s_rem_epv__sidlx_io_iostream;
  struct sidlx_io_IStream__epv*   e2  = &s_rem_epv__sidlx_io_istream;
  struct sidlx_io_OStream__epv*   e3  = &s_rem_epv__sidlx_io_ostream;

  epv->f__cast             = remote_sidlx_io__IOStream__cast;
  epv->f__delete           = remote_sidlx_io__IOStream__delete;
  epv->f__exec             = remote_sidlx_io__IOStream__exec;
  epv->f__getURL           = remote_sidlx_io__IOStream__getURL;
  epv->f__ctor             = NULL;
  epv->f__dtor             = NULL;
  epv->f_addRef            = remote_sidlx_io__IOStream_addRef;
  epv->f_deleteRef         = remote_sidlx_io__IOStream_deleteRef;
  epv->f_isSame            = remote_sidlx_io__IOStream_isSame;
  epv->f_queryInt          = remote_sidlx_io__IOStream_queryInt;
  epv->f_isType            = remote_sidlx_io__IOStream_isType;
  epv->f_getClassInfo      = remote_sidlx_io__IOStream_getClassInfo;
  epv->f_atEnd             = remote_sidlx_io__IOStream_atEnd;
  epv->f_read              = remote_sidlx_io__IOStream_read;
  epv->f_readline          = remote_sidlx_io__IOStream_readline;
  epv->f_getBool           = remote_sidlx_io__IOStream_getBool;
  epv->f_getChar           = remote_sidlx_io__IOStream_getChar;
  epv->f_getInt            = remote_sidlx_io__IOStream_getInt;
  epv->f_getLong           = remote_sidlx_io__IOStream_getLong;
  epv->f_getFloat          = remote_sidlx_io__IOStream_getFloat;
  epv->f_getDouble         = remote_sidlx_io__IOStream_getDouble;
  epv->f_getFcomplex       = remote_sidlx_io__IOStream_getFcomplex;
  epv->f_getDcomplex       = remote_sidlx_io__IOStream_getDcomplex;
  epv->f_getString         = remote_sidlx_io__IOStream_getString;
  epv->f_flush             = remote_sidlx_io__IOStream_flush;
  epv->f_write             = remote_sidlx_io__IOStream_write;
  epv->f_putBool           = remote_sidlx_io__IOStream_putBool;
  epv->f_putChar           = remote_sidlx_io__IOStream_putChar;
  epv->f_putInt            = remote_sidlx_io__IOStream_putInt;
  epv->f_putLong           = remote_sidlx_io__IOStream_putLong;
  epv->f_putFloat          = remote_sidlx_io__IOStream_putFloat;
  epv->f_putDouble         = remote_sidlx_io__IOStream_putDouble;
  epv->f_putFcomplex       = remote_sidlx_io__IOStream_putFcomplex;
  epv->f_putDcomplex       = remote_sidlx_io__IOStream_putDcomplex;
  epv->f_putString         = remote_sidlx_io__IOStream_putString;

  e0->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(void*)) epv->f__delete;
  e0->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

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
  e1->f_atEnd        = (sidl_bool (*)(void*)) epv->f_atEnd;
  e1->f_read         = (int32_t (*)(void*,int32_t,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_read;
  e1->f_readline     = (int32_t (*)(void*,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readline;
  e1->f_getBool      = (void (*)(void*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_getBool;
  e1->f_getChar      = (void (*)(void*,char*,
    struct sidl_BaseInterface__object **)) epv->f_getChar;
  e1->f_getInt       = (void (*)(void*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_getInt;
  e1->f_getLong      = (void (*)(void*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_getLong;
  e1->f_getFloat     = (void (*)(void*,float*,
    struct sidl_BaseInterface__object **)) epv->f_getFloat;
  e1->f_getDouble    = (void (*)(void*,double*,
    struct sidl_BaseInterface__object **)) epv->f_getDouble;
  e1->f_getFcomplex  = (void (*)(void*,struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_getFcomplex;
  e1->f_getDcomplex  = (void (*)(void*,struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_getDcomplex;
  e1->f_getString    = (void (*)(void*,char**,
    struct sidl_BaseInterface__object **)) epv->f_getString;
  e1->f_flush        = (void (*)(void*)) epv->f_flush;
  e1->f_write        = (int32_t (*)(void*,struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_write;
  e1->f_putBool      = (void (*)(void*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_putBool;
  e1->f_putChar      = (void (*)(void*,char,
    struct sidl_BaseInterface__object **)) epv->f_putChar;
  e1->f_putInt       = (void (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_putInt;
  e1->f_putLong      = (void (*)(void*,int64_t,
    struct sidl_BaseInterface__object **)) epv->f_putLong;
  e1->f_putFloat     = (void (*)(void*,float,
    struct sidl_BaseInterface__object **)) epv->f_putFloat;
  e1->f_putDouble    = (void (*)(void*,double,
    struct sidl_BaseInterface__object **)) epv->f_putDouble;
  e1->f_putFcomplex  = (void (*)(void*,struct sidl_fcomplex,
    struct sidl_BaseInterface__object **)) epv->f_putFcomplex;
  e1->f_putDcomplex  = (void (*)(void*,struct sidl_dcomplex,
    struct sidl_BaseInterface__object **)) epv->f_putDcomplex;
  e1->f_putString    = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_putString;

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
  e3->f_flush        = (void (*)(void*)) epv->f_flush;
  e3->f_write        = (int32_t (*)(void*,struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_write;
  e3->f_putBool      = (void (*)(void*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_putBool;
  e3->f_putChar      = (void (*)(void*,char,
    struct sidl_BaseInterface__object **)) epv->f_putChar;
  e3->f_putInt       = (void (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_putInt;
  e3->f_putLong      = (void (*)(void*,int64_t,
    struct sidl_BaseInterface__object **)) epv->f_putLong;
  e3->f_putFloat     = (void (*)(void*,float,
    struct sidl_BaseInterface__object **)) epv->f_putFloat;
  e3->f_putDouble    = (void (*)(void*,double,
    struct sidl_BaseInterface__object **)) epv->f_putDouble;
  e3->f_putFcomplex  = (void (*)(void*,struct sidl_fcomplex,
    struct sidl_BaseInterface__object **)) epv->f_putFcomplex;
  e3->f_putDcomplex  = (void (*)(void*,struct sidl_dcomplex,
    struct sidl_BaseInterface__object **)) epv->f_putDcomplex;
  e3->f_putString    = (void (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_putString;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct sidlx_io_IOStream__object*
sidlx_io_IOStream__remoteConnect(const char *url, sidl_BaseInterface *_ex)
{
  struct sidlx_io__IOStream__object* self;

  struct sidlx_io__IOStream__object* s0;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_connectInstance(url, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct sidlx_io__IOStream__object*) malloc(
      sizeof(struct sidlx_io__IOStream__object));

   s0 =                             self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_io__IOStream__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidlx_io_iostream.d_epv    = &s_rem_epv__sidlx_io_iostream;
  s0->d_sidlx_io_iostream.d_object = (void*) self;

  s0->d_sidlx_io_istream.d_epv    = &s_rem_epv__sidlx_io_istream;
  s0->d_sidlx_io_istream.d_object = (void*) self;

  s0->d_sidlx_io_ostream.d_epv    = &s_rem_epv__sidlx_io_ostream;
  s0->d_sidlx_io_ostream.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidlx_io__iostream;

  self->d_data = (void*) instance;

  return sidlx_io_IOStream__cast(self);
}
/* Create an instance that uses an already existing InstanceHandel to connect 
  to an existing remote object. */
static struct sidlx_io_IOStream__object*
sidlx_io_IOStream__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct sidlx_io__IOStream__object* self;

  struct sidlx_io__IOStream__object* s0;

  self =
    (struct sidlx_io__IOStream__object*) malloc(
      sizeof(struct sidlx_io__IOStream__object));

   s0 =                             self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    sidlx_io__IOStream__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_sidlx_io_iostream.d_epv    = &s_rem_epv__sidlx_io_iostream;
  s0->d_sidlx_io_iostream.d_object = (void*) self;

  s0->d_sidlx_io_istream.d_epv    = &s_rem_epv__sidlx_io_istream;
  s0->d_sidlx_io_istream.d_object = (void*) self;

  s0->d_sidlx_io_ostream.d_epv    = &s_rem_epv__sidlx_io_ostream;
  s0->d_sidlx_io_ostream.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__sidlx_io__iostream;

  self->d_data = (void*) instance;

  sidl_rmi_InstanceHandle_addRef(instance);
  return sidlx_io_IOStream__cast(self);
}
