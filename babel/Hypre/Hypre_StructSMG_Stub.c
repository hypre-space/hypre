/*
 * File:          Hypre_StructSMG_Stub.c
 * Symbol:        Hypre.StructSMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020904 10:05:22 PDT
 * Generated:     20020904 10:05:27 PDT
 * Description:   Client-side glue code for Hypre.StructSMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructSMG.h"
#include "Hypre_StructSMG_IOR.h"
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

static const struct Hypre_StructSMG__external* _getIOR(void)
{
  static const struct Hypre_StructSMG__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_StructSMG__externals();
#else
    const struct Hypre_StructSMG__external*(*dll_f)(void) =
      (const struct Hypre_StructSMG__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_StructSMG__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.StructSMG; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor function for the class.
 */

Hypre_StructSMG
Hypre_StructSMG__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * Method:  SetLogging
 */

int32_t
Hypre_StructSMG_SetLogging(
  Hypre_StructSMG self,
  int32_t level)
{
  return (*self->d_epv->f_SetLogging)(
    self,
    level);
}

/*
 * Method:  Setup
 */

int32_t
Hypre_StructSMG_Setup(
  Hypre_StructSMG self,
  Hypre_Vector x,
  Hypre_Vector y)
{
  return (*self->d_epv->f_Setup)(
    self,
    x,
    y);
}

/*
 * Method:  SetIntArrayParameter
 */

int32_t
Hypre_StructSMG_SetIntArrayParameter(
  Hypre_StructSMG self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArrayParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetIntParameter
 */

int32_t
Hypre_StructSMG_SetIntParameter(
  Hypre_StructSMG self,
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
Hypre_StructSMG_GetResidual(
  Hypre_StructSMG self,
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
Hypre_StructSMG_SetPrintLevel(
  Hypre_StructSMG self,
  int32_t level)
{
  return (*self->d_epv->f_SetPrintLevel)(
    self,
    level);
}

/*
 * Method:  GetIntValue
 */

int32_t
Hypre_StructSMG_GetIntValue(
  Hypre_StructSMG self,
  const char* name,
  int32_t* value)
{
  return (*self->d_epv->f_GetIntValue)(
    self,
    name,
    value);
}

/*
 * Method:  SetCommunicator
 */

int32_t
Hypre_StructSMG_SetCommunicator(
  Hypre_StructSMG self,
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
Hypre_StructSMG_isInstanceOf(
  Hypre_StructSMG self,
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
Hypre_StructSMG_SetStringParameter(
  Hypre_StructSMG self,
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
Hypre_StructSMG_SetDoubleParameter(
  Hypre_StructSMG self,
  const char* name,
  double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);
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
Hypre_StructSMG_queryInterface(
  Hypre_StructSMG self,
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
Hypre_StructSMG_isSame(
  Hypre_StructSMG self,
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
Hypre_StructSMG_Apply(
  Hypre_StructSMG self,
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
Hypre_StructSMG_addReference(
  Hypre_StructSMG self)
{
  (*self->d_epv->f_addReference)(
    self);
}

/*
 * Method:  GetDoubleValue
 */

int32_t
Hypre_StructSMG_GetDoubleValue(
  Hypre_StructSMG self,
  const char* name,
  double* value)
{
  return (*self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value);
}

/*
 * Method:  SetOperator
 */

int32_t
Hypre_StructSMG_SetOperator(
  Hypre_StructSMG self,
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
Hypre_StructSMG_SetDoubleArrayParameter(
  Hypre_StructSMG self,
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
Hypre_StructSMG_deleteReference(
  Hypre_StructSMG self)
{
  (*self->d_epv->f_deleteReference)(
    self);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_StructSMG
Hypre_StructSMG__cast(
  void* obj)
{
  Hypre_StructSMG cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_StructSMG) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.StructSMG");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_StructSMG__cast2(
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

struct Hypre_StructSMG__array*
Hypre_StructSMG__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_StructSMG__array*
Hypre_StructSMG__array_borrow(
  struct Hypre_StructSMG__object** firstElement,
  int32_t                          dimen,
  const int32_t                    lower[],
  const int32_t                    upper[],
  const int32_t                    stride[])
{
  return (*(_getIOR()->borrowArray))
    (firstElement, dimen, lower, upper, stride);
}

/*
 * Destructor for the array.
 */

void
Hypre_StructSMG__array_destroy(
  struct Hypre_StructSMG__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_StructSMG__array_dimen(const struct Hypre_StructSMG__array *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_StructSMG__array_lower(const struct Hypre_StructSMG__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_StructSMG__array_upper(const struct Hypre_StructSMG__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructSMG__object*
Hypre_StructSMG__array_get(
  const struct Hypre_StructSMG__array* array,
  const int32_t                        indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructSMG__object*
Hypre_StructSMG__array_get4(
  const struct Hypre_StructSMG__array* array,
  int32_t                              i1,
  int32_t                              i2,
  int32_t                              i3,
  int32_t                              i4)
{
  return (*(_getIOR()->getElement4))
    (array, i1, i2, i3, i4);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructSMG__array_set(
  struct Hypre_StructSMG__array*  array,
  const int32_t                   indices[],
  struct Hypre_StructSMG__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructSMG__array_set4(
  struct Hypre_StructSMG__array*  array,
  int32_t                         i1,
  int32_t                         i2,
  int32_t                         i3,
  int32_t                         i4,
  struct Hypre_StructSMG__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
