/*
 * File:          bHYPRE_StructGrid_Stub.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:38 PST
 * Description:   Client-side glue code for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 1101
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_StructGrid.h"
#include "bHYPRE_StructGrid_IOR.h"
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

static const struct bHYPRE_StructGrid__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_StructGrid__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _ior = bHYPRE_StructGrid__externals();
#else
  sidl_DLL dll = sidl_DLL__create();
  const struct bHYPRE_StructGrid__external*(*dll_f)(void);
  /* check global namespace for symbol first */
  if (dll && sidl_DLL_loadLibrary(dll, "main:", TRUE, FALSE)) {
    dll_f =
      (const struct bHYPRE_StructGrid__external*(*)(void)) 
        sidl_DLL_lookupSymbol(
        dll, "bHYPRE_StructGrid__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
  }
  if (dll) sidl_DLL_deleteRef(dll);
  if (!_ior) {
    dll = sidl_Loader_findLibrary("bHYPRE.StructGrid",
      "ior/impl", sidl_Scope_SCLSCOPE,
      sidl_Resolve_SCLRESOLVE);
    if (dll) {
      dll_f =
        (const struct bHYPRE_StructGrid__external*(*)(void)) 
          sidl_DLL_lookupSymbol(
          dll, "bHYPRE_StructGrid__externals");
      _ior = (dll_f ? (*dll_f)() : NULL);
      sidl_DLL_deleteRef(dll);
    }
  }
  if (!_ior) {
    fputs("Babel: unable to load the implementation for bHYPRE.StructGrid; please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
#endif
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Constructor function for the class.
 */

bHYPRE_StructGrid
bHYPRE_StructGrid__create()
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
bHYPRE_StructGrid_addRef(
  bHYPRE_StructGrid self)
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
bHYPRE_StructGrid_deleteRef(
  bHYPRE_StructGrid self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_StructGrid_isSame(
  bHYPRE_StructGrid self,
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
bHYPRE_StructGrid_queryInt(
  bHYPRE_StructGrid self,
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
bHYPRE_StructGrid_isType(
  bHYPRE_StructGrid self,
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
bHYPRE_StructGrid_getClassInfo(
  bHYPRE_StructGrid self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Set the MPI Communicator.
 * 
 */

int32_t
bHYPRE_StructGrid_SetCommunicator(
  bHYPRE_StructGrid self,
  /*in*/ void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
}

/*
 * Method:  SetDimension[]
 */

int32_t
bHYPRE_StructGrid_SetDimension(
  bHYPRE_StructGrid self,
  /*in*/ int32_t dim)
{
  return (*self->d_epv->f_SetDimension)(
    self,
    dim);
}

/*
 * Method:  SetExtents[]
 */

int32_t
bHYPRE_StructGrid_SetExtents(
  bHYPRE_StructGrid self,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper)
{
  return (*self->d_epv->f_SetExtents)(
    self,
    ilower,
    iupper);
}

/*
 * Method:  SetPeriodic[]
 */

int32_t
bHYPRE_StructGrid_SetPeriodic(
  bHYPRE_StructGrid self,
  /*in*/ struct sidl_int__array* periodic)
{
  return (*self->d_epv->f_SetPeriodic)(
    self,
    periodic);
}

/*
 * Method:  Assemble[]
 */

int32_t
bHYPRE_StructGrid_Assemble(
  bHYPRE_StructGrid self)
{
  return (*self->d_epv->f_Assemble)(
    self);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_StructGrid
bHYPRE_StructGrid__cast(
  void* obj)
{
  bHYPRE_StructGrid cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_StructGrid) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.StructGrid");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_StructGrid__cast2(
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
struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_StructGrid__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_StructGrid__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create1d(int32_t len)
{
  return (struct bHYPRE_StructGrid__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create1dInit(
  int32_t len, 
  bHYPRE_StructGrid* data)
{
  return (struct 
    bHYPRE_StructGrid__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create2dCol(int32_t m, int32_t n)
{
  return (struct bHYPRE_StructGrid__array*)sidl_interface__array_create2dCol(m,
    n);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create2dRow(int32_t m, int32_t n)
{
  return (struct bHYPRE_StructGrid__array*)sidl_interface__array_create2dRow(m,
    n);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_borrow(
  bHYPRE_StructGrid* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_StructGrid__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_smartCopy(
  struct bHYPRE_StructGrid__array *array)
{
  return (struct bHYPRE_StructGrid__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_StructGrid__array_addRef(
  struct bHYPRE_StructGrid__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_StructGrid__array_deleteRef(
  struct bHYPRE_StructGrid__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get1(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1)
{
  return (bHYPRE_StructGrid)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get2(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_StructGrid)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get3(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_StructGrid)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get4(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_StructGrid)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get5(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_StructGrid)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get6(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_StructGrid)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get7(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_StructGrid)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t indices[])
{
  return (bHYPRE_StructGrid)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_StructGrid__array_set1(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  bHYPRE_StructGrid const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructGrid__array_set2(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructGrid const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructGrid__array_set3(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructGrid const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructGrid__array_set4(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructGrid const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructGrid__array_set5(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructGrid const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructGrid__array_set6(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructGrid const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructGrid__array_set7(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructGrid const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructGrid__array_set(
  struct bHYPRE_StructGrid__array* array,
  const int32_t indices[],
  bHYPRE_StructGrid const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_StructGrid__array_dimen(
  const struct bHYPRE_StructGrid__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_StructGrid__array_lower(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructGrid__array_upper(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructGrid__array_length(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructGrid__array_stride(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_StructGrid__array_isColumnOrder(
  const struct bHYPRE_StructGrid__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_StructGrid__array_isRowOrder(
  const struct bHYPRE_StructGrid__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_StructGrid__array_copy(
  const struct bHYPRE_StructGrid__array* src,
  struct bHYPRE_StructGrid__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_slice(
  struct bHYPRE_StructGrid__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_StructGrid__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_ensure(
  struct bHYPRE_StructGrid__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_StructGrid__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

