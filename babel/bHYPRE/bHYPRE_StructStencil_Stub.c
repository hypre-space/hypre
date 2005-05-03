/*
 * File:          bHYPRE_StructStencil_Stub.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Client-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_StructStencil.h"
#include "bHYPRE_StructStencil_IOR.h"
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
 * Hold pointer to IOR functions.
 */

static const struct bHYPRE_StructStencil__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_StructStencil__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _ior = bHYPRE_StructStencil__externals();
#else
  sidl_DLL dll = sidl_DLL__create();
  const struct bHYPRE_StructStencil__external*(*dll_f)(void);
  /* check global namespace for symbol first */
  if (dll && sidl_DLL_loadLibrary(dll, "main:", TRUE, FALSE)) {
    dll_f =
      (const struct bHYPRE_StructStencil__external*(*)(void)) 
        sidl_DLL_lookupSymbol(
        dll, "bHYPRE_StructStencil__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
  }
  if (dll) sidl_DLL_deleteRef(dll);
  if (!_ior) {
    dll = sidl_Loader_findLibrary("bHYPRE.StructStencil",
      "ior/impl", sidl_Scope_SCLSCOPE,
      sidl_Resolve_SCLRESOLVE);
    if (dll) {
      dll_f =
        (const struct bHYPRE_StructStencil__external*(*)(void)) 
          sidl_DLL_lookupSymbol(
          dll, "bHYPRE_StructStencil__externals");
      _ior = (dll_f ? (*dll_f)() : NULL);
      sidl_DLL_deleteRef(dll);
    }
  }
  if (!_ior) {
    fputs("Babel: unable to load the implementation for bHYPRE.StructStencil; please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
#endif
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Constructor function for the class.
 */

bHYPRE_StructStencil
bHYPRE_StructStencil__create()
{
  return (*(_getIOR()->createObject))();
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
bHYPRE_StructStencil_addRef(
  bHYPRE_StructStencil self)
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
bHYPRE_StructStencil_deleteRef(
  bHYPRE_StructStencil self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_StructStencil_isSame(
  bHYPRE_StructStencil self,
  /*in*/ sidl_BaseInterface iobj)
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
bHYPRE_StructStencil_queryInt(
  bHYPRE_StructStencil self,
  /*in*/ const char* name)
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
bHYPRE_StructStencil_isType(
  bHYPRE_StructStencil self,
  /*in*/ const char* name)
{
  return (*self->d_epv->f_isType)(
    self,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
bHYPRE_StructStencil_getClassInfo(
  bHYPRE_StructStencil self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Method:  SetDimension[]
 */

int32_t
bHYPRE_StructStencil_SetDimension(
  bHYPRE_StructStencil self,
  /*in*/ int32_t dim)
{
  return (*self->d_epv->f_SetDimension)(
    self,
    dim);
}

/*
 * Method:  SetSize[]
 */

int32_t
bHYPRE_StructStencil_SetSize(
  bHYPRE_StructStencil self,
  /*in*/ int32_t size)
{
  return (*self->d_epv->f_SetSize)(
    self,
    size);
}

/*
 * Method:  SetElement[]
 */

int32_t
bHYPRE_StructStencil_SetElement(
  bHYPRE_StructStencil self,
  /*in*/ int32_t index,
  /*in*/ struct sidl_int__array* offset)
{
  return (*self->d_epv->f_SetElement)(
    self,
    index,
    offset);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_StructStencil
bHYPRE_StructStencil__cast(
  void* obj)
{
  bHYPRE_StructStencil cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_StructStencil) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.StructStencil");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_StructStencil__cast2(
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
struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create1dInit(
  int32_t len, 
  bHYPRE_StructStencil* data)
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_borrow(
  bHYPRE_StructStencil* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_StructStencil__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_smartCopy(
  struct bHYPRE_StructStencil__array *array)
{
  return (struct bHYPRE_StructStencil__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_StructStencil__array_addRef(
  struct bHYPRE_StructStencil__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_StructStencil__array_deleteRef(
  struct bHYPRE_StructStencil__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get1(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get2(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get3(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get4(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get5(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get6(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get7(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t indices[])
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_StructStencil__array_set1(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set2(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set3(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set4(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set5(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set6(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set7(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set(
  struct bHYPRE_StructStencil__array* array,
  const int32_t indices[],
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_StructStencil__array_dimen(
  const struct bHYPRE_StructStencil__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_StructStencil__array_lower(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructStencil__array_upper(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructStencil__array_length(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructStencil__array_stride(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_StructStencil__array_isColumnOrder(
  const struct bHYPRE_StructStencil__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_StructStencil__array_isRowOrder(
  const struct bHYPRE_StructStencil__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_StructStencil__array_copy(
  const struct bHYPRE_StructStencil__array* src,
  struct bHYPRE_StructStencil__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_slice(
  struct bHYPRE_StructStencil__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_StructStencil__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_ensure(
  struct bHYPRE_StructStencil__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_StructStencil__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

