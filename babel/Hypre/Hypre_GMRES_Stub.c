/*
 * File:          Hypre_GMRES_Stub.c
 * Symbol:        Hypre.GMRES-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:40 PDT
 * Description:   Client-side glue code for Hypre.GMRES
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_GMRES.h"
#include "Hypre_GMRES_IOR.h"
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

static const struct Hypre_GMRES__external* _getIOR(void)
{
  static const struct Hypre_GMRES__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_GMRES__externals();
#else
    const struct Hypre_GMRES__external*(*dll_f)(void) =
      (const struct Hypre_GMRES__external*(*)(void)) SIDL_Loader_lookupSymbol(
        "Hypre_GMRES__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.GMRES; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor function for the class.
 */

Hypre_GMRES
Hypre_GMRES__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * Method:  Setup
 */

int32_t
Hypre_GMRES_Setup(
  Hypre_GMRES self)
{
  return (*self->d_epv->f_Setup)(
    self);
}

/*
 * Method:  SetIntArrayParameter
 */

int32_t
Hypre_GMRES_SetIntArrayParameter(
  Hypre_GMRES self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArrayParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetLogging
 */

int32_t
Hypre_GMRES_SetLogging(
  Hypre_GMRES self,
  int32_t level)
{
  return (*self->d_epv->f_SetLogging)(
    self,
    level);
}

/*
 * Method:  SetIntParameter
 */

int32_t
Hypre_GMRES_SetIntParameter(
  Hypre_GMRES self,
  const char* name,
  int32_t value)
{
  return (*self->d_epv->f_SetIntParameter)(
    self,
    name,
    value);
}

/*
 * Method:  GetResidual
 */

int32_t
Hypre_GMRES_GetResidual(
  Hypre_GMRES self,
  Hypre_Vector* r)
{
  return (*self->d_epv->f_GetResidual)(
    self,
    r);
}

/*
 * Method:  SetPrintLevel
 */

int32_t
Hypre_GMRES_SetPrintLevel(
  Hypre_GMRES self,
  int32_t level)
{
  return (*self->d_epv->f_SetPrintLevel)(
    self,
    level);
}

/*
 * Method:  SetCommunicator
 */

int32_t
Hypre_GMRES_SetCommunicator(
  Hypre_GMRES self,
  void* comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    comm);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
Hypre_GMRES_isInstanceOf(
  Hypre_GMRES self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self,
    name);
}

/*
 * Method:  SetStringParameter
 */

int32_t
Hypre_GMRES_SetStringParameter(
  Hypre_GMRES self,
  const char* name,
  const char* value)
{
  return (*self->d_epv->f_SetStringParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetDoubleParameter
 */

int32_t
Hypre_GMRES_SetDoubleParameter(
  Hypre_GMRES self,
  const char* name,
  double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);
}

/*
 * Method:  GetPreconditionedResidual
 */

int32_t
Hypre_GMRES_GetPreconditionedResidual(
  Hypre_GMRES self,
  Hypre_Vector* r)
{
  return (*self->d_epv->f_GetPreconditionedResidual)(
    self,
    r);
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
Hypre_GMRES_queryInterface(
  Hypre_GMRES self,
  const char* name)
{
  return (*self->d_epv->f_queryInterface)(
    self,
    name);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
Hypre_GMRES_isSame(
  Hypre_GMRES self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Method:  Apply
 */

int32_t
Hypre_GMRES_Apply(
  Hypre_GMRES self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  return (*self->d_epv->f_Apply)(
    self,
    x,
    y);
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
Hypre_GMRES_addReference(
  Hypre_GMRES self)
{
  (*self->d_epv->f_addReference)(
    self);
}

/*
 * Method:  SetPreconditioner
 */

int32_t
Hypre_GMRES_SetPreconditioner(
  Hypre_GMRES self,
  Hypre_Solver s)
{
  return (*self->d_epv->f_SetPreconditioner)(
    self,
    s);
}

/*
 * Method:  SetOperator
 */

int32_t
Hypre_GMRES_SetOperator(
  Hypre_GMRES self,
  Hypre_Operator A)
{
  return (*self->d_epv->f_SetOperator)(
    self,
    A);
}

/*
 * Method:  SetDoubleArrayParameter
 */

int32_t
Hypre_GMRES_SetDoubleArrayParameter(
  Hypre_GMRES self,
  const char* name,
  struct SIDL_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArrayParameter)(
    self,
    name,
    value);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
Hypre_GMRES_deleteReference(
  Hypre_GMRES self)
{
  (*self->d_epv->f_deleteReference)(
    self);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_GMRES
Hypre_GMRES__cast(
  void* obj)
{
  Hypre_GMRES cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_GMRES) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.GMRES");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_GMRES__cast2(
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

struct Hypre_GMRES__array*
Hypre_GMRES__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_GMRES__array*
Hypre_GMRES__array_borrow(
  struct Hypre_GMRES__object** firstElement,
  int32_t                      dimen,
  const int32_t                lower[],
  const int32_t                upper[],
  const int32_t                stride[])
{
  return (*(_getIOR()->borrowArray))
    (firstElement, dimen, lower, upper, stride);
}

/*
 * Destructor for the array.
 */

void
Hypre_GMRES__array_destroy(
  struct Hypre_GMRES__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_GMRES__array_dimen(const struct Hypre_GMRES__array *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_GMRES__array_lower(const struct Hypre_GMRES__array *array, int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_GMRES__array_upper(const struct Hypre_GMRES__array *array, int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_GMRES__object*
Hypre_GMRES__array_get(
  const struct Hypre_GMRES__array* array,
  const int32_t                    indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_GMRES__object*
Hypre_GMRES__array_get4(
  const struct Hypre_GMRES__array* array,
  int32_t                          i1,
  int32_t                          i2,
  int32_t                          i3,
  int32_t                          i4)
{
  return (*(_getIOR()->getElement4))
    (array, i1, i2, i3, i4);
}

/*
 * Set an element of the array.
 */

void
Hypre_GMRES__array_set(
  struct Hypre_GMRES__array*  array,
  const int32_t               indices[],
  struct Hypre_GMRES__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_GMRES__array_set4(
  struct Hypre_GMRES__array*  array,
  int32_t                     i1,
  int32_t                     i2,
  int32_t                     i3,
  int32_t                     i4,
  struct Hypre_GMRES__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
