/*
 * File:          Hypre_StructStencil_Stub.c
 * Symbol:        Hypre.StructStencil-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:50 PDT
 * Description:   Client-side glue code for Hypre.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructStencil.h"
#include "Hypre_StructStencil_IOR.h"
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

static const struct Hypre_StructStencil__external* _getIOR(void)
{
  static const struct Hypre_StructStencil__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_StructStencil__externals();
#else
    const struct Hypre_StructStencil__external*(*dll_f)(void) =
      (const struct Hypre_StructStencil__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_StructStencil__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.StructStencil; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor function for the class.
 */

Hypre_StructStencil
Hypre_StructStencil__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * Method:  SetElement
 */

int32_t
Hypre_StructStencil_SetElement(
  Hypre_StructStencil self,
  int32_t index,
  struct SIDL_int__array* offset)
{
  return (*self->d_epv->f_SetElement)(
    self,
    index,
    offset);
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
Hypre_StructStencil_addReference(
  Hypre_StructStencil self)
{
  (*self->d_epv->f_addReference)(
    self);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
Hypre_StructStencil_isInstanceOf(
  Hypre_StructStencil self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self,
    name);
}

/*
 * Method:  SetDimension
 */

int32_t
Hypre_StructStencil_SetDimension(
  Hypre_StructStencil self,
  int32_t dim)
{
  return (*self->d_epv->f_SetDimension)(
    self,
    dim);
}

/*
 * Method:  SetSize
 */

int32_t
Hypre_StructStencil_SetSize(
  Hypre_StructStencil self,
  int32_t size)
{
  return (*self->d_epv->f_SetSize)(
    self,
    size);
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
Hypre_StructStencil_queryInterface(
  Hypre_StructStencil self,
  const char* name)
{
  return (*self->d_epv->f_queryInterface)(
    self,
    name);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
Hypre_StructStencil_deleteReference(
  Hypre_StructStencil self)
{
  (*self->d_epv->f_deleteReference)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
Hypre_StructStencil_isSame(
  Hypre_StructStencil self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_StructStencil
Hypre_StructStencil__cast(
  void* obj)
{
  Hypre_StructStencil cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_StructStencil) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.StructStencil");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_StructStencil__cast2(
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

struct Hypre_StructStencil__array*
Hypre_StructStencil__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_StructStencil__array*
Hypre_StructStencil__array_borrow(
  struct Hypre_StructStencil__object** firstElement,
  int32_t                              dimen,
  const int32_t                        lower[],
  const int32_t                        upper[],
  const int32_t                        stride[])
{
  return (*(_getIOR()->borrowArray))
    (firstElement, dimen, lower, upper, stride);
}

/*
 * Destructor for the array.
 */

void
Hypre_StructStencil__array_destroy(
  struct Hypre_StructStencil__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_StructStencil__array_dimen(const struct Hypre_StructStencil__array *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_StructStencil__array_lower(const struct Hypre_StructStencil__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_StructStencil__array_upper(const struct Hypre_StructStencil__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructStencil__object*
Hypre_StructStencil__array_get(
  const struct Hypre_StructStencil__array* array,
  const int32_t                            indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructStencil__object*
Hypre_StructStencil__array_get4(
  const struct Hypre_StructStencil__array* array,
  int32_t                                  i1,
  int32_t                                  i2,
  int32_t                                  i3,
  int32_t                                  i4)
{
  return (*(_getIOR()->getElement4))
    (array, i1, i2, i3, i4);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructStencil__array_set(
  struct Hypre_StructStencil__array*  array,
  const int32_t                       indices[],
  struct Hypre_StructStencil__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructStencil__array_set4(
  struct Hypre_StructStencil__array*  array,
  int32_t                             i1,
  int32_t                             i2,
  int32_t                             i3,
  int32_t                             i4,
  struct Hypre_StructStencil__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
