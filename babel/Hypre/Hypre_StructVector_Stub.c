/*
 * File:          Hypre_StructVector_Stub.c
 * Symbol:        Hypre.StructVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:17 PST
 * Description:   Client-side glue code for Hypre.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructVector.h"
#include "Hypre_StructVector_IOR.h"
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

static const struct Hypre_StructVector__external* _getIOR(void)
{
  static const struct Hypre_StructVector__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_StructVector__externals();
#else
    const struct Hypre_StructVector__external*(*dll_f)(void) =
      (const struct Hypre_StructVector__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_StructVector__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.StructVector; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor function for the class.
 */

Hypre_StructVector
Hypre_StructVector__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * y <- a*x + y
 */

int32_t
Hypre_StructVector_Axpy(
  Hypre_StructVector self,
  double a,
  Hypre_Vector x)
{
  return (*self->d_epv->f_Axpy)(
    self,
    a,
    x);
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

int32_t
Hypre_StructVector_Initialize(
  Hypre_StructVector self)
{
  return (*self->d_epv->f_Initialize)(
    self);
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
Hypre_StructVector_addReference(
  Hypre_StructVector self)
{
  (*self->d_epv->f_addReference)(
    self);
}

/*
 * Method:  SetCommunicator
 */

int32_t
Hypre_StructVector_SetCommunicator(
  Hypre_StructVector self,
  void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
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
Hypre_StructVector_queryInterface(
  Hypre_StructVector self,
  const char* name)
{
  return (*self->d_epv->f_queryInterface)(
    self,
    name);
}

/*
 * Method:  SetStencil
 */

int32_t
Hypre_StructVector_SetStencil(
  Hypre_StructVector self,
  Hypre_StructStencil stencil)
{
  return (*self->d_epv->f_SetStencil)(
    self,
    stencil);
}

/*
 * Method:  SetValue
 */

int32_t
Hypre_StructVector_SetValue(
  Hypre_StructVector self,
  struct SIDL_int__array* grid_index,
  double value)
{
  return (*self->d_epv->f_SetValue)(
    self,
    grid_index,
    value);
}

/*
 * create an x compatible with y
 */

int32_t
Hypre_StructVector_Clone(
  Hypre_StructVector self,
  Hypre_Vector* x)
{
  return (*self->d_epv->f_Clone)(
    self,
    x);
}

/*
 * y <- 0 (where y=self)
 */

int32_t
Hypre_StructVector_Clear(
  Hypre_StructVector self)
{
  return (*self->d_epv->f_Clear)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
Hypre_StructVector_isSame(
  Hypre_StructVector self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */

int32_t
Hypre_StructVector_Assemble(
  Hypre_StructVector self)
{
  return (*self->d_epv->f_Assemble)(
    self);
}

/*
 * Method:  SetBoxValues
 */

int32_t
Hypre_StructVector_SetBoxValues(
  Hypre_StructVector self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_SetBoxValues)(
    self,
    ilower,
    iupper,
    values);
}

/*
 * y <- a*y 
 */

int32_t
Hypre_StructVector_Scale(
  Hypre_StructVector self,
  double a)
{
  return (*self->d_epv->f_Scale)(
    self,
    a);
}

/*
 * d <- (y,x)
 */

int32_t
Hypre_StructVector_Dot(
  Hypre_StructVector self,
  Hypre_Vector x,
  double* d)
{
  return (*self->d_epv->f_Dot)(
    self,
    x,
    d);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
Hypre_StructVector_deleteReference(
  Hypre_StructVector self)
{
  (*self->d_epv->f_deleteReference)(
    self);
}

/*
 * The problem definition interface is a "builder" that creates an object
 * that contains the problem definition information, e.g. a matrix. To
 * perform subsequent operations with that object, it must be returned from
 * the problem definition object. "GetObject" performs this function.
 * <note>At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface. QueryInterface or Cast must
 * be used on the returned object to convert it into a known type.</note>
 * 
 * 
 */

int32_t
Hypre_StructVector_GetObject(
  Hypre_StructVector self,
  SIDL_BaseInterface* A)
{
  return (*self->d_epv->f_GetObject)(
    self,
    A);
}

/*
 * y <- x 
 */

int32_t
Hypre_StructVector_Copy(
  Hypre_StructVector self,
  Hypre_Vector x)
{
  return (*self->d_epv->f_Copy)(
    self,
    x);
}

/*
 * Method:  SetGrid
 */

int32_t
Hypre_StructVector_SetGrid(
  Hypre_StructVector self,
  Hypre_StructGrid grid)
{
  return (*self->d_epv->f_SetGrid)(
    self,
    grid);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
Hypre_StructVector_isInstanceOf(
  Hypre_StructVector self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self,
    name);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_StructVector
Hypre_StructVector__cast(
  void* obj)
{
  Hypre_StructVector cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_StructVector) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.StructVector");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_StructVector__cast2(
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

struct Hypre_StructVector__array*
Hypre_StructVector__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_StructVector__array*
Hypre_StructVector__array_borrow(
  struct Hypre_StructVector__object** firstElement,
  int32_t                             dimen,
  const int32_t                       lower[],
  const int32_t                       upper[],
  const int32_t                       stride[])
{
  return (*(_getIOR()->borrowArray))
    (firstElement, dimen, lower, upper, stride);
}

/*
 * Destructor for the array.
 */

void
Hypre_StructVector__array_destroy(
  struct Hypre_StructVector__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_StructVector__array_dimen(const struct Hypre_StructVector__array *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_StructVector__array_lower(const struct Hypre_StructVector__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_StructVector__array_upper(const struct Hypre_StructVector__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructVector__object*
Hypre_StructVector__array_get(
  const struct Hypre_StructVector__array* array,
  const int32_t                           indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructVector__object*
Hypre_StructVector__array_get4(
  const struct Hypre_StructVector__array* array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4)
{
  return (*(_getIOR()->getElement4))
    (array, i1, i2, i3, i4);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructVector__array_set(
  struct Hypre_StructVector__array*  array,
  const int32_t                      indices[],
  struct Hypre_StructVector__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructVector__array_set4(
  struct Hypre_StructVector__array*  array,
  int32_t                            i1,
  int32_t                            i2,
  int32_t                            i3,
  int32_t                            i4,
  struct Hypre_StructVector__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
