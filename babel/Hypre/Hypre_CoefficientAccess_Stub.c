/*
 * File:          Hypre_CoefficientAccess_Stub.c
 * Symbol:        Hypre.CoefficientAccess-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:30 PDT
 * Description:   Client-side glue code for Hypre.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_CoefficientAccess.h"
#include "Hypre_CoefficientAccess_IOR.h"
#include <stddef.h>
#include "SIDL_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "SIDL_Loader.h"
#endif

/*
 * Return pointer to internal IOR functions.
 */

static const struct Hypre_CoefficientAccess__external* _getIOR(void)
{
  static const struct Hypre_CoefficientAccess__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_CoefficientAccess__externals();
#else
    const struct Hypre_CoefficientAccess__external*(*dll_f)(void) =
      (const struct Hypre_CoefficientAccess__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_CoefficientAccess__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.CoefficientAccess; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
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
Hypre_CoefficientAccess_addReference(
  Hypre_CoefficientAccess self)
{
  (*self->d_epv->f_addReference)(
    self->d_object);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
Hypre_CoefficientAccess_isInstanceOf(
  Hypre_CoefficientAccess self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self->d_object,
    name);
}

/*
 * Method:  GetRow
 */

int32_t
Hypre_CoefficientAccess_GetRow(
  Hypre_CoefficientAccess self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  return (*self->d_epv->f_GetRow)(
    self->d_object,
    row,
    size,
    col_ind,
    values);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteReference</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

SIDL_BaseInterface
Hypre_CoefficientAccess_queryInterface(
  Hypre_CoefficientAccess self,
  const char* name)
{
  return (*self->d_epv->f_queryInterface)(
    self->d_object,
    name);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
Hypre_CoefficientAccess_isSame(
  Hypre_CoefficientAccess self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
    iobj);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
Hypre_CoefficientAccess_deleteReference(
  Hypre_CoefficientAccess self)
{
  (*self->d_epv->f_deleteReference)(
    self->d_object);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_CoefficientAccess
Hypre_CoefficientAccess__cast(
  void* obj)
{
  Hypre_CoefficientAccess cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_CoefficientAccess) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.CoefficientAccess");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_CoefficientAccess__cast2(
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
/*
 * Create a new copy of the array.
 */

struct Hypre_CoefficientAccess__array*
Hypre_CoefficientAccess__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_CoefficientAccess__array*
Hypre_CoefficientAccess__array_borrow(
  struct Hypre_CoefficientAccess__object** firstElement,
  int32_t                                  dimen,
  const int32_t                            lower[],
  const int32_t                            upper[],
  const int32_t                            stride[])
{
  return (*(_getIOR()->borrowArray))
    (firstElement, dimen, lower, upper, stride);
}

/*
 * Destructor for the array.
 */

void
Hypre_CoefficientAccess__array_destroy(
  struct Hypre_CoefficientAccess__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_CoefficientAccess__array_dimen(const struct 
  Hypre_CoefficientAccess__array *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_CoefficientAccess__array_lower(const struct 
  Hypre_CoefficientAccess__array *array, int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_CoefficientAccess__array_upper(const struct 
  Hypre_CoefficientAccess__array *array, int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_CoefficientAccess__object*
Hypre_CoefficientAccess__array_get(
  const struct Hypre_CoefficientAccess__array* array,
  const int32_t                                indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_CoefficientAccess__object*
Hypre_CoefficientAccess__array_get4(
  const struct Hypre_CoefficientAccess__array* array,
  int32_t                                      i1,
  int32_t                                      i2,
  int32_t                                      i3,
  int32_t                                      i4)
{
  return (*(_getIOR()->getElement4))
    (array, i1, i2, i3, i4);
}

/*
 * Set an element of the array.
 */

void
Hypre_CoefficientAccess__array_set(
  struct Hypre_CoefficientAccess__array*  array,
  const int32_t                           indices[],
  struct Hypre_CoefficientAccess__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_CoefficientAccess__array_set4(
  struct Hypre_CoefficientAccess__array*  array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4,
  struct Hypre_CoefficientAccess__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
