/*
 * File:          bHYPRE_Vector_Stub.c
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:42 PST
 * Description:   Client-side glue code for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 672
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_Vector.h"
#include "bHYPRE_Vector_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stddef.h>
#include "sidl_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.h"
#endif

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
bHYPRE_Vector_addRef(
  bHYPRE_Vector self)
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
bHYPRE_Vector_deleteRef(
  bHYPRE_Vector self)
{
  (*self->d_epv->f_deleteRef)(
    self->d_object);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_Vector_isSame(
  bHYPRE_Vector self,
  /*in*/ sidl_BaseInterface iobj)
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
bHYPRE_Vector_queryInt(
  bHYPRE_Vector self,
  /*in*/ const char* name)
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
bHYPRE_Vector_isType(
  bHYPRE_Vector self,
  /*in*/ const char* name)
{
  return (*self->d_epv->f_isType)(
    self->d_object,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
bHYPRE_Vector_getClassInfo(
  bHYPRE_Vector self)
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object);
}

/*
 * Set {\tt self} to 0.
 * 
 */

int32_t
bHYPRE_Vector_Clear(
  bHYPRE_Vector self)
{
  return (*self->d_epv->f_Clear)(
    self->d_object);
}

/*
 * Copy x into {\tt self}.
 * 
 */

int32_t
bHYPRE_Vector_Copy(
  bHYPRE_Vector self,
  /*in*/ bHYPRE_Vector x)
{
  return (*self->d_epv->f_Copy)(
    self->d_object,
    x);
}

/*
 * Create an {\tt x} compatible with {\tt self}.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 * 
 */

int32_t
bHYPRE_Vector_Clone(
  bHYPRE_Vector self,
  /*out*/ bHYPRE_Vector* x)
{
  return (*self->d_epv->f_Clone)(
    self->d_object,
    x);
}

/*
 * Scale {\tt self} by {\tt a}.
 * 
 */

int32_t
bHYPRE_Vector_Scale(
  bHYPRE_Vector self,
  /*in*/ double a)
{
  return (*self->d_epv->f_Scale)(
    self->d_object,
    a);
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

int32_t
bHYPRE_Vector_Dot(
  bHYPRE_Vector self,
  /*in*/ bHYPRE_Vector x,
  /*out*/ double* d)
{
  return (*self->d_epv->f_Dot)(
    self->d_object,
    x,
    d);
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

int32_t
bHYPRE_Vector_Axpy(
  bHYPRE_Vector self,
  /*in*/ double a,
  /*in*/ bHYPRE_Vector x)
{
  return (*self->d_epv->f_Axpy)(
    self->d_object,
    a,
    x);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_Vector
bHYPRE_Vector__cast(
  void* obj)
{
  bHYPRE_Vector cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_Vector) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.Vector");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_Vector__cast2(
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
struct bHYPRE_Vector__array*
bHYPRE_Vector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct bHYPRE_Vector__array*)sidl_interface__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct bHYPRE_Vector__array*)sidl_interface__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create1d(int32_t len)
{
  return (struct bHYPRE_Vector__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create1dInit(
  int32_t len, 
  bHYPRE_Vector* data)
{
  return (struct bHYPRE_Vector__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create2dCol(int32_t m, int32_t n)
{
  return (struct bHYPRE_Vector__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create2dRow(int32_t m, int32_t n)
{
  return (struct bHYPRE_Vector__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_borrow(
  bHYPRE_Vector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_Vector__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_smartCopy(
  struct bHYPRE_Vector__array *array)
{
  return (struct bHYPRE_Vector__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_Vector__array_addRef(
  struct bHYPRE_Vector__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_Vector__array_deleteRef(
  struct bHYPRE_Vector__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_Vector
bHYPRE_Vector__array_get1(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1)
{
  return (bHYPRE_Vector)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_Vector
bHYPRE_Vector__array_get2(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_Vector)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_Vector
bHYPRE_Vector__array_get3(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_Vector)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_Vector
bHYPRE_Vector__array_get4(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_Vector)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_Vector
bHYPRE_Vector__array_get5(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_Vector)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_Vector
bHYPRE_Vector__array_get6(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_Vector)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_Vector
bHYPRE_Vector__array_get7(
  const struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_Vector)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_Vector
bHYPRE_Vector__array_get(
  const struct bHYPRE_Vector__array* array,
  const int32_t indices[])
{
  return (bHYPRE_Vector)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_Vector__array_set1(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  bHYPRE_Vector const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Vector__array_set2(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_Vector const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Vector__array_set3(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_Vector const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Vector__array_set4(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_Vector const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Vector__array_set5(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_Vector const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Vector__array_set6(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_Vector const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Vector__array_set7(
  struct bHYPRE_Vector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_Vector const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_Vector__array_set(
  struct bHYPRE_Vector__array* array,
  const int32_t indices[],
  bHYPRE_Vector const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_Vector__array_dimen(
  const struct bHYPRE_Vector__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_Vector__array_lower(
  const struct bHYPRE_Vector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_Vector__array_upper(
  const struct bHYPRE_Vector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_Vector__array_length(
  const struct bHYPRE_Vector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_Vector__array_stride(
  const struct bHYPRE_Vector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_Vector__array_isColumnOrder(
  const struct bHYPRE_Vector__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_Vector__array_isRowOrder(
  const struct bHYPRE_Vector__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_Vector__array_copy(
  const struct bHYPRE_Vector__array* src,
  struct bHYPRE_Vector__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_slice(
  struct bHYPRE_Vector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_Vector__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_ensure(
  struct bHYPRE_Vector__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_Vector__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

