/*
 * File:          bHYPRE_SStructParCSRVector_Stub.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Client-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#include "bHYPRE_SStructParCSRVector.h"
#include "bHYPRE_SStructParCSRVector_IOR.h"
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

static const struct bHYPRE_SStructParCSRVector__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_SStructParCSRVector__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _ior = bHYPRE_SStructParCSRVector__externals();
#else
  sidl_DLL dll = sidl_DLL__create();
  const struct bHYPRE_SStructParCSRVector__external*(*dll_f)(void);
  /* check global namespace for symbol first */
  if (dll && sidl_DLL_loadLibrary(dll, "main:", TRUE, FALSE)) {
    dll_f =
      (const struct bHYPRE_SStructParCSRVector__external*(*)(void)) 
        sidl_DLL_lookupSymbol(
        dll, "bHYPRE_SStructParCSRVector__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
  }
  if (dll) sidl_DLL_deleteRef(dll);
  if (!_ior) {
    dll = sidl_Loader_findLibrary("bHYPRE.SStructParCSRVector",
      "ior/impl", sidl_Scope_SCLSCOPE,
      sidl_Resolve_SCLRESOLVE);
    if (dll) {
      dll_f =
        (const struct bHYPRE_SStructParCSRVector__external*(*)(void)) 
          sidl_DLL_lookupSymbol(
          dll, "bHYPRE_SStructParCSRVector__externals");
      _ior = (dll_f ? (*dll_f)() : NULL);
      sidl_DLL_deleteRef(dll);
    }
  }
  if (!_ior) {
    fputs("Babel: unable to load the implementation for bHYPRE.SStructParCSRVector; please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
#endif
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Constructor function for the class.
 */

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__create()
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
bHYPRE_SStructParCSRVector_addRef(
  bHYPRE_SStructParCSRVector self)
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
bHYPRE_SStructParCSRVector_deleteRef(
  bHYPRE_SStructParCSRVector self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_SStructParCSRVector_isSame(
  bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_queryInt(
  bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_isType(
  bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_getClassInfo(
  bHYPRE_SStructParCSRVector self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Set {\tt self} to 0.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Clear(
  bHYPRE_SStructParCSRVector self)
{
  return (*self->d_epv->f_Clear)(
    self);
}

/*
 * Copy x into {\tt self}.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Copy(
  bHYPRE_SStructParCSRVector self,
  /*in*/ bHYPRE_Vector x)
{
  return (*self->d_epv->f_Copy)(
    self,
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
bHYPRE_SStructParCSRVector_Clone(
  bHYPRE_SStructParCSRVector self,
  /*out*/ bHYPRE_Vector* x)
{
  return (*self->d_epv->f_Clone)(
    self,
    x);
}

/*
 * Scale {\tt self} by {\tt a}.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Scale(
  bHYPRE_SStructParCSRVector self,
  /*in*/ double a)
{
  return (*self->d_epv->f_Scale)(
    self,
    a);
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Dot(
  bHYPRE_SStructParCSRVector self,
  /*in*/ bHYPRE_Vector x,
  /*out*/ double* d)
{
  return (*self->d_epv->f_Dot)(
    self,
    x,
    d);
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Axpy(
  bHYPRE_SStructParCSRVector self,
  /*in*/ double a,
  /*in*/ bHYPRE_Vector x)
{
  return (*self->d_epv->f_Axpy)(
    self,
    a,
    x);
}

/*
 * Set the MPI Communicator.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_SetCommunicator(
  bHYPRE_SStructParCSRVector self,
  /*in*/ void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Initialize(
  bHYPRE_SStructParCSRVector self)
{
  return (*self->d_epv->f_Initialize)(
    self);
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Assemble(
  bHYPRE_SStructParCSRVector self)
{
  return (*self->d_epv->f_Assemble)(
    self);
}

/*
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_GetObject(
  bHYPRE_SStructParCSRVector self,
  /*out*/ sidl_BaseInterface* A)
{
  return (*self->d_epv->f_GetObject)(
    self,
    A);
}

/*
 * Set the vector grid.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_SetGrid(
  bHYPRE_SStructParCSRVector self,
  /*in*/ bHYPRE_SStructGrid grid)
{
  return (*self->d_epv->f_SetGrid)(
    self,
    grid);
}

/*
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_SetValues(
  bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* values)
{
  return (*self->d_epv->f_SetValues)(
    self,
    part,
    index,
    var,
    values);
}

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_SetBoxValues(
  bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* values)
{
  return (*self->d_epv->f_SetBoxValues)(
    self,
    part,
    ilower,
    iupper,
    var,
    values);
}

/*
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_AddToValues(
  bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* values)
{
  return (*self->d_epv->f_AddToValues)(
    self,
    part,
    index,
    var,
    values);
}

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_AddToBoxValues(
  bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* values)
{
  return (*self->d_epv->f_AddToBoxValues)(
    self,
    part,
    ilower,
    iupper,
    var,
    values);
}

/*
 * Gather vector data before calling {\tt GetValues}.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Gather(
  bHYPRE_SStructParCSRVector self)
{
  return (*self->d_epv->f_Gather)(
    self);
}

/*
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_GetValues(
  bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*out*/ double* value)
{
  return (*self->d_epv->f_GetValues)(
    self,
    part,
    index,
    var,
    value);
}

/*
 * Get vector coefficients a box at a time.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_GetBoxValues(
  bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*inout*/ struct sidl_double__array** values)
{
  return (*self->d_epv->f_GetBoxValues)(
    self,
    part,
    ilower,
    iupper,
    var,
    values);
}

/*
 * Set the vector to be complex.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_SetComplex(
  bHYPRE_SStructParCSRVector self)
{
  return (*self->d_epv->f_SetComplex)(
    self);
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

int32_t
bHYPRE_SStructParCSRVector_Print(
  bHYPRE_SStructParCSRVector self,
  /*in*/ const char* filename,
  /*in*/ int32_t all)
{
  return (*self->d_epv->f_Print)(
    self,
    filename,
    all);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__cast(
  void* obj)
{
  bHYPRE_SStructParCSRVector cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_SStructParCSRVector) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructParCSRVector");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_SStructParCSRVector__cast2(
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
struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructParCSRVector* data)
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_borrow(
  bHYPRE_SStructParCSRVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_smartCopy(
  struct bHYPRE_SStructParCSRVector__array *array)
{
  return (struct bHYPRE_SStructParCSRVector__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructParCSRVector__array_addRef(
  struct bHYPRE_SStructParCSRVector__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructParCSRVector__array_deleteRef(
  struct bHYPRE_SStructParCSRVector__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get1(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get2(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get3(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get4(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get5(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get6(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get7(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t indices[])
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_SStructParCSRVector__array_set1(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set2(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set3(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set4(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set5(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set6(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set7(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t indices[],
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_SStructParCSRVector__array_dimen(
  const struct bHYPRE_SStructParCSRVector__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_SStructParCSRVector__array_lower(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRVector__array_upper(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRVector__array_length(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRVector__array_stride(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_SStructParCSRVector__array_isColumnOrder(
  const struct bHYPRE_SStructParCSRVector__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_SStructParCSRVector__array_isRowOrder(
  const struct bHYPRE_SStructParCSRVector__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_SStructParCSRVector__array_copy(
  const struct bHYPRE_SStructParCSRVector__array* src,
  struct bHYPRE_SStructParCSRVector__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_slice(
  struct bHYPRE_SStructParCSRVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_SStructParCSRVector__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_ensure(
  struct bHYPRE_SStructParCSRVector__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_SStructParCSRVector__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

