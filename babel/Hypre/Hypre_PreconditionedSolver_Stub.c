/*
 * File:          Hypre_PreconditionedSolver_Stub.c
 * Symbol:        Hypre.PreconditionedSolver-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:31 PDT
 * Description:   Client-side glue code for Hypre.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_PreconditionedSolver.h"
#include "Hypre_PreconditionedSolver_IOR.h"
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

static const struct Hypre_PreconditionedSolver__external* _getIOR(void)
{
  static const struct Hypre_PreconditionedSolver__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_PreconditionedSolver__externals();
#else
    const struct Hypre_PreconditionedSolver__external*(*dll_f)(void) =
      (const struct Hypre_PreconditionedSolver__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_PreconditionedSolver__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.PreconditionedSolver; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Method:  SetLogging
 */

int32_t
Hypre_PreconditionedSolver_SetLogging(
  Hypre_PreconditionedSolver self,
  int32_t level)
{
  return (*self->d_epv->f_SetLogging)(
    self->d_object,
    level);
}

/*
 * Method:  Setup
 */

int32_t
Hypre_PreconditionedSolver_Setup(
  Hypre_PreconditionedSolver self)
{
  return (*self->d_epv->f_Setup)(
    self->d_object);
}

/*
 * Method:  SetIntArrayParameter
 */

int32_t
Hypre_PreconditionedSolver_SetIntArrayParameter(
  Hypre_PreconditionedSolver self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArrayParameter)(
    self->d_object,
    name,
    value);
}

/*
 * Method:  SetIntParameter
 */

int32_t
Hypre_PreconditionedSolver_SetIntParameter(
  Hypre_PreconditionedSolver self,
  const char* name,
  int32_t value)
{
  return (*self->d_epv->f_SetIntParameter)(
    self->d_object,
    name,
    value);
}

/*
 * Method:  GetResidual
 */

int32_t
Hypre_PreconditionedSolver_GetResidual(
  Hypre_PreconditionedSolver self,
  Hypre_Vector* r)
{
  return (*self->d_epv->f_GetResidual)(
    self->d_object,
    r);
}

/*
 * Method:  SetPrintLevel
 */

int32_t
Hypre_PreconditionedSolver_SetPrintLevel(
  Hypre_PreconditionedSolver self,
  int32_t level)
{
  return (*self->d_epv->f_SetPrintLevel)(
    self->d_object,
    level);
}

/*
 * Method:  GetIntValue
 */

int32_t
Hypre_PreconditionedSolver_GetIntValue(
  Hypre_PreconditionedSolver self,
  const char* name,
  int32_t* value)
{
  return (*self->d_epv->f_GetIntValue)(
    self->d_object,
    name,
    value);
}

/*
 * Method:  SetCommunicator
 */

int32_t
Hypre_PreconditionedSolver_SetCommunicator(
  Hypre_PreconditionedSolver self,
  void* comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self->d_object,
    comm);
}

/*
 * Method:  SetStringParameter
 */

int32_t
Hypre_PreconditionedSolver_SetStringParameter(
  Hypre_PreconditionedSolver self,
  const char* name,
  const char* value)
{
  return (*self->d_epv->f_SetStringParameter)(
    self->d_object,
    name,
    value);
}

/*
 * Method:  SetDoubleParameter
 */

int32_t
Hypre_PreconditionedSolver_SetDoubleParameter(
  Hypre_PreconditionedSolver self,
  const char* name,
  double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self->d_object,
    name,
    value);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
Hypre_PreconditionedSolver_isInstanceOf(
  Hypre_PreconditionedSolver self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self->d_object,
    name);
}

/*
 * Method:  GetPreconditionedResidual
 */

int32_t
Hypre_PreconditionedSolver_GetPreconditionedResidual(
  Hypre_PreconditionedSolver self,
  Hypre_Vector* r)
{
  return (*self->d_epv->f_GetPreconditionedResidual)(
    self->d_object,
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
Hypre_PreconditionedSolver_queryInterface(
  Hypre_PreconditionedSolver self,
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
Hypre_PreconditionedSolver_isSame(
  Hypre_PreconditionedSolver self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
    iobj);
}

/*
 * Method:  Apply
 */

int32_t
Hypre_PreconditionedSolver_Apply(
  Hypre_PreconditionedSolver self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  return (*self->d_epv->f_Apply)(
    self->d_object,
    x,
    y);
}

/*
 * Method:  GetDoubleValue
 */

int32_t
Hypre_PreconditionedSolver_GetDoubleValue(
  Hypre_PreconditionedSolver self,
  const char* name,
  double* value)
{
  return (*self->d_epv->f_GetDoubleValue)(
    self->d_object,
    name,
    value);
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
Hypre_PreconditionedSolver_addReference(
  Hypre_PreconditionedSolver self)
{
  (*self->d_epv->f_addReference)(
    self->d_object);
}

/*
 * Method:  SetPreconditioner
 */

int32_t
Hypre_PreconditionedSolver_SetPreconditioner(
  Hypre_PreconditionedSolver self,
  Hypre_Solver s)
{
  return (*self->d_epv->f_SetPreconditioner)(
    self->d_object,
    s);
}

/*
 * Method:  SetOperator
 */

int32_t
Hypre_PreconditionedSolver_SetOperator(
  Hypre_PreconditionedSolver self,
  Hypre_Operator A)
{
  return (*self->d_epv->f_SetOperator)(
    self->d_object,
    A);
}

/*
 * Method:  SetDoubleArrayParameter
 */

int32_t
Hypre_PreconditionedSolver_SetDoubleArrayParameter(
  Hypre_PreconditionedSolver self,
  const char* name,
  struct SIDL_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArrayParameter)(
    self->d_object,
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
Hypre_PreconditionedSolver_deleteReference(
  Hypre_PreconditionedSolver self)
{
  (*self->d_epv->f_deleteReference)(
    self->d_object);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_PreconditionedSolver
Hypre_PreconditionedSolver__cast(
  void* obj)
{
  Hypre_PreconditionedSolver cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_PreconditionedSolver) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.PreconditionedSolver");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_PreconditionedSolver__cast2(
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

struct Hypre_PreconditionedSolver__array*
Hypre_PreconditionedSolver__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_PreconditionedSolver__array*
Hypre_PreconditionedSolver__array_borrow(
  struct Hypre_PreconditionedSolver__object** firstElement,
  int32_t                                     dimen,
  const int32_t                               lower[],
  const int32_t                               upper[],
  const int32_t                               stride[])
{
  return (*(_getIOR()->borrowArray))
    (firstElement, dimen, lower, upper, stride);
}

/*
 * Destructor for the array.
 */

void
Hypre_PreconditionedSolver__array_destroy(
  struct Hypre_PreconditionedSolver__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_PreconditionedSolver__array_dimen(const struct 
  Hypre_PreconditionedSolver__array *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_PreconditionedSolver__array_lower(const struct 
  Hypre_PreconditionedSolver__array *array, int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_PreconditionedSolver__array_upper(const struct 
  Hypre_PreconditionedSolver__array *array, int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_PreconditionedSolver__object*
Hypre_PreconditionedSolver__array_get(
  const struct Hypre_PreconditionedSolver__array* array,
  const int32_t                                   indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_PreconditionedSolver__object*
Hypre_PreconditionedSolver__array_get4(
  const struct Hypre_PreconditionedSolver__array* array,
  int32_t                                         i1,
  int32_t                                         i2,
  int32_t                                         i3,
  int32_t                                         i4)
{
  return (*(_getIOR()->getElement4))
    (array, i1, i2, i3, i4);
}

/*
 * Set an element of the array.
 */

void
Hypre_PreconditionedSolver__array_set(
  struct Hypre_PreconditionedSolver__array*  array,
  const int32_t                              indices[],
  struct Hypre_PreconditionedSolver__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_PreconditionedSolver__array_set4(
  struct Hypre_PreconditionedSolver__array*  array,
  int32_t                                    i1,
  int32_t                                    i2,
  int32_t                                    i3,
  int32_t                                    i4,
  struct Hypre_PreconditionedSolver__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
