/*
 * File:          Hypre_StructToIJVector_Stub.c
 * Symbol:        Hypre.StructToIJVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:49 PDT
 * Description:   Client-side glue code for Hypre.StructToIJVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructToIJVector.h"
#include "Hypre_StructToIJVector_IOR.h"
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

static const struct Hypre_StructToIJVector__external* _getIOR(void)
{
  static const struct Hypre_StructToIJVector__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_StructToIJVector__externals();
#else
    const struct Hypre_StructToIJVector__external*(*dll_f)(void) =
      (const struct Hypre_StructToIJVector__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_StructToIJVector__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.StructToIJVector; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor function for the class.
 */

Hypre_StructToIJVector
Hypre_StructToIJVector__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * Method:  SetValue
 */

int32_t
Hypre_StructToIJVector_SetValue(
  Hypre_StructToIJVector self,
  struct SIDL_int__array* grid_index,
  double value)
{
  return (*self->d_epv->f_SetValue)(
    self,
    grid_index,
    value);
}

/*
 * Method:  SetBoxValues
 */

int32_t
Hypre_StructToIJVector_SetBoxValues(
  Hypre_StructToIJVector self,
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
 * Method:  SetStencil
 */

int32_t
Hypre_StructToIJVector_SetStencil(
  Hypre_StructToIJVector self,
  Hypre_StructStencil stencil)
{
  return (*self->d_epv->f_SetStencil)(
    self,
    stencil);
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
Hypre_StructToIJVector_addReference(
  Hypre_StructToIJVector self)
{
  (*self->d_epv->f_addReference)(
    self);
}

/*
 * Method:  SetGrid
 */

int32_t
Hypre_StructToIJVector_SetGrid(
  Hypre_StructToIJVector self,
  Hypre_StructGrid grid)
{
  return (*self->d_epv->f_SetGrid)(
    self,
    grid);
}

/*
 * Method:  SetCommunicator
 */

int32_t
Hypre_StructToIJVector_SetCommunicator(
  Hypre_StructToIJVector self,
  void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
}

/*
 * Method:  SetIJVector
 */

int32_t
Hypre_StructToIJVector_SetIJVector(
  Hypre_StructToIJVector self,
  Hypre_IJBuildVector I)
{
  return (*self->d_epv->f_SetIJVector)(
    self,
    I);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
Hypre_StructToIJVector_isInstanceOf(
  Hypre_StructToIJVector self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self,
    name);
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
Hypre_StructToIJVector_GetObject(
  Hypre_StructToIJVector self,
  SIDL_BaseInterface* A)
{
  return (*self->d_epv->f_GetObject)(
    self,
    A);
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
Hypre_StructToIJVector_Assemble(
  Hypre_StructToIJVector self)
{
  return (*self->d_epv->f_Assemble)(
    self);
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
Hypre_StructToIJVector_queryInterface(
  Hypre_StructToIJVector self,
  const char* name)
{
  return (*self->d_epv->f_queryInterface)(
    self,
    name);
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

int32_t
Hypre_StructToIJVector_Initialize(
  Hypre_StructToIJVector self)
{
  return (*self->d_epv->f_Initialize)(
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
Hypre_StructToIJVector_deleteReference(
  Hypre_StructToIJVector self)
{
  (*self->d_epv->f_deleteReference)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
Hypre_StructToIJVector_isSame(
  Hypre_StructToIJVector self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_StructToIJVector
Hypre_StructToIJVector__cast(
  void* obj)
{
  Hypre_StructToIJVector cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_StructToIJVector) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.StructToIJVector");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_StructToIJVector__cast2(
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

struct Hypre_StructToIJVector__array*
Hypre_StructToIJVector__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_StructToIJVector__array*
Hypre_StructToIJVector__array_borrow(
  struct Hypre_StructToIJVector__object** firstElement,
  int32_t                                 dimen,
  const int32_t                           lower[],
  const int32_t                           upper[],
  const int32_t                           stride[])
{
  return (*(_getIOR()->borrowArray))
    (firstElement, dimen, lower, upper, stride);
}

/*
 * Destructor for the array.
 */

void
Hypre_StructToIJVector__array_destroy(
  struct Hypre_StructToIJVector__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_StructToIJVector__array_dimen(const struct Hypre_StructToIJVector__array 
  *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_StructToIJVector__array_lower(const struct Hypre_StructToIJVector__array 
  *array, int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_StructToIJVector__array_upper(const struct Hypre_StructToIJVector__array 
  *array, int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__array_get(
  const struct Hypre_StructToIJVector__array* array,
  const int32_t                               indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__array_get4(
  const struct Hypre_StructToIJVector__array* array,
  int32_t                                     i1,
  int32_t                                     i2,
  int32_t                                     i3,
  int32_t                                     i4)
{
  return (*(_getIOR()->getElement4))
    (array, i1, i2, i3, i4);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructToIJVector__array_set(
  struct Hypre_StructToIJVector__array*  array,
  const int32_t                          indices[],
  struct Hypre_StructToIJVector__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructToIJVector__array_set4(
  struct Hypre_StructToIJVector__array*  array,
  int32_t                                i1,
  int32_t                                i2,
  int32_t                                i3,
  int32_t                                i4,
  struct Hypre_StructToIJVector__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
