/*
 * File:          bHYPRE_SStructGraph_Stub.c
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:27 PST
 * Description:   Client-side glue code for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1022
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructGraph.h"
#include "bHYPRE_SStructGraph_IOR.h"
#ifndef included_SIDL_interface_IOR_h
#include "SIDL_interface_IOR.h"
#endif
#include <stddef.h>
#include "SIDL_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "SIDL_Loader.h"
#endif

/*
 * Hold pointer to IOR functions.
 */

static const struct bHYPRE_SStructGraph__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_SStructGraph__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _ior = bHYPRE_SStructGraph__externals();
#else
  const struct bHYPRE_SStructGraph__external*(*dll_f)(void) =
    (const struct bHYPRE_SStructGraph__external*(*)(void)) 
      SIDL_Loader_lookupSymbol(
      "bHYPRE_SStructGraph__externals");
  _ior = (dll_f ? (*dll_f)() : NULL);
  if (!_ior) {
    fputs("Babel: unable to load the implementation for bHYPRE.SStructGraph; please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
#endif
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Constructor function for the class.
 */

bHYPRE_SStructGraph
bHYPRE_SStructGraph__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>SIDL</code> have an intrinsic reference count.
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
bHYPRE_SStructGraph_addRef(
  bHYPRE_SStructGraph self)
{
  (*self->d_epv->f_addRef)(
    self);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
bHYPRE_SStructGraph_deleteRef(
  bHYPRE_SStructGraph self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
bHYPRE_SStructGraph_isSame(
  bHYPRE_SStructGraph self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

SIDL_BaseInterface
bHYPRE_SStructGraph_queryInt(
  bHYPRE_SStructGraph self,
  const char* name)
{
  return (*self->d_epv->f_queryInt)(
    self,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
bHYPRE_SStructGraph_isType(
  bHYPRE_SStructGraph self,
  const char* name)
{
  return (*self->d_epv->f_isType)(
    self,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

SIDL_ClassInfo
bHYPRE_SStructGraph_getClassInfo(
  bHYPRE_SStructGraph self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Set the grid.
 * 
 */

int32_t
bHYPRE_SStructGraph_SetGrid(
  bHYPRE_SStructGraph self,
  bHYPRE_SStructGrid grid)
{
  return (*self->d_epv->f_SetGrid)(
    self,
    grid);
}

/*
 * Set the stencil for a variable on a structured part of the
 * grid.
 * 
 */

int32_t
bHYPRE_SStructGraph_SetStencil(
  bHYPRE_SStructGraph self,
  int32_t part,
  int32_t var,
  bHYPRE_SStructStencil stencil)
{
  return (*self->d_epv->f_SetStencil)(
    self,
    part,
    var,
    stencil);
}

/*
 * Add a non-stencil graph entry at a particular index.  This
 * graph entry is appended to the existing graph entries, and is
 * referenced as such.
 * 
 * NOTE: Users are required to set graph entries on all
 * processes that own the associated variables.  This means that
 * some data will be multiply defined.
 * 
 */

int32_t
bHYPRE_SStructGraph_AddEntries(
  bHYPRE_SStructGraph self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  int32_t to_part,
  struct SIDL_int__array* to_index,
  int32_t to_var)
{
  return (*self->d_epv->f_AddEntries)(
    self,
    part,
    index,
    var,
    to_part,
    to_index,
    to_var);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_SStructGraph
bHYPRE_SStructGraph__cast(
  void* obj)
{
  bHYPRE_SStructGraph cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (bHYPRE_SStructGraph) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructGraph");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_SStructGraph__cast2(
  void* obj,
  const char* type)
{
  void* cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type);
  }

  return cast;
}
struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_createCol(int32_t        dimen,
                                     const int32_t lower[],
                                     const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructGraph__array*)SIDL_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_createRow(int32_t        dimen,
                                     const int32_t lower[],
                                     const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructGraph__array*)SIDL_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_SStructGraph__array*)SIDL_interface__array_create1d(len);
}

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructGraph__array*)SIDL_interface__array_create2dCol(m, n);
}

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructGraph__array*)SIDL_interface__array_create2dRow(m, n);
}

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_borrow(bHYPRE_SStructGraph*firstElement,
                                  int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct bHYPRE_SStructGraph__array*)SIDL_interface__array_borrow(
    (struct SIDL_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_smartCopy(struct bHYPRE_SStructGraph__array *array)
{
  return (struct bHYPRE_SStructGraph__array*)
    SIDL_interface__array_smartCopy((struct SIDL_interface__array *)array);
}

void
bHYPRE_SStructGraph__array_addRef(struct bHYPRE_SStructGraph__array* array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array *)array);
}

void
bHYPRE_SStructGraph__array_deleteRef(struct bHYPRE_SStructGraph__array* array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array *)array);
}

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get1(const struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1)
{
  return (bHYPRE_SStructGraph)
    SIDL_interface__array_get1((const struct SIDL_interface__array *)array
    , i1);
}

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get2(const struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2)
{
  return (bHYPRE_SStructGraph)
    SIDL_interface__array_get2((const struct SIDL_interface__array *)array
    , i1, i2);
}

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get3(const struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3)
{
  return (bHYPRE_SStructGraph)
    SIDL_interface__array_get3((const struct SIDL_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get4(const struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                const int32_t i4)
{
  return (bHYPRE_SStructGraph)
    SIDL_interface__array_get4((const struct SIDL_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get(const struct bHYPRE_SStructGraph__array* array,
                               const int32_t indices[])
{
  return (bHYPRE_SStructGraph)
    SIDL_interface__array_get((const struct SIDL_interface__array *)array,
      indices);
}

void
bHYPRE_SStructGraph__array_set1(struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                bHYPRE_SStructGraph const value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)array
  , i1, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_SStructGraph__array_set2(struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                bHYPRE_SStructGraph const value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)array
  , i1, i2, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_SStructGraph__array_set3(struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                bHYPRE_SStructGraph const value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)array
  , i1, i2, i3, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_SStructGraph__array_set4(struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                const int32_t i4,
                                bHYPRE_SStructGraph const value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)array
  , i1, i2, i3, i4, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_SStructGraph__array_set(struct bHYPRE_SStructGraph__array* array,
                               const int32_t indices[],
                               bHYPRE_SStructGraph const value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)array, indices,
    (struct SIDL_BaseInterface__object *)value);
}

int32_t
bHYPRE_SStructGraph__array_dimen(const struct bHYPRE_SStructGraph__array* array)
{
  return SIDL_interface__array_dimen((struct SIDL_interface__array *)array);
}

int32_t
bHYPRE_SStructGraph__array_lower(const struct bHYPRE_SStructGraph__array* array,
                                 const int32_t ind)
{
  return SIDL_interface__array_lower((struct SIDL_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructGraph__array_upper(const struct bHYPRE_SStructGraph__array* array,
                                 const int32_t ind)
{
  return SIDL_interface__array_upper((struct SIDL_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructGraph__array_stride(const struct bHYPRE_SStructGraph__array* 
  array,
                                  const int32_t ind)
{
  return SIDL_interface__array_stride((struct SIDL_interface__array *)array,
    ind);
}

int
bHYPRE_SStructGraph__array_isColumnOrder(const struct 
  bHYPRE_SStructGraph__array* array)
{
  return SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)array);
}

int
bHYPRE_SStructGraph__array_isRowOrder(const struct bHYPRE_SStructGraph__array* 
  array)
{
  return SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)array);
}

void
bHYPRE_SStructGraph__array_copy(const struct bHYPRE_SStructGraph__array* src,
                                      struct bHYPRE_SStructGraph__array* dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array *)src,
                             (struct SIDL_interface__array *)dest);
}

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_ensure(struct bHYPRE_SStructGraph__array* src,
                                  int32_t dimen,
                                  int     ordering)
{
  return (struct bHYPRE_SStructGraph__array*)
    SIDL_interface__array_ensure((struct SIDL_interface__array *)src, dimen,
      ordering);
}

