/*
 * File:          Hypre_StructMatrix_Stub.c
 * Symbol:        Hypre.StructMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:42 PDT
 * Description:   Client-side glue code for Hypre.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructMatrix.h"
#include "Hypre_StructMatrix_IOR.h"
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

static const struct Hypre_StructMatrix__external* _getIOR(void)
{
  static const struct Hypre_StructMatrix__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_StructMatrix__externals();
#else
    const struct Hypre_StructMatrix__external*(*dll_f)(void) =
      (const struct Hypre_StructMatrix__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_StructMatrix__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.StructMatrix; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor function for the class.
 */

Hypre_StructMatrix
Hypre_StructMatrix__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * Method:  SetIntArrayParameter
 */

int32_t
Hypre_StructMatrix_SetIntArrayParameter(
  Hypre_StructMatrix self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArrayParameter)(
    self,
    name,
    value);
}

/*
 * Method:  Setup
 */

int32_t
Hypre_StructMatrix_Setup(
  Hypre_StructMatrix self)
{
  return (*self->d_epv->f_Setup)(
    self);
}

/*
 * Method:  SetIntParameter
 */

int32_t
Hypre_StructMatrix_SetIntParameter(
  Hypre_StructMatrix self,
  const char* name,
  int32_t value)
{
  return (*self->d_epv->f_SetIntParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetStencil
 */

int32_t
Hypre_StructMatrix_SetStencil(
  Hypre_StructMatrix self,
  Hypre_StructStencil stencil)
{
  return (*self->d_epv->f_SetStencil)(
    self,
    stencil);
}

/*
 * Method:  SetCommunicator
 */

int32_t
Hypre_StructMatrix_SetCommunicator(
  Hypre_StructMatrix self,
  void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
}

/*
 * Method:  SetDoubleParameter
 */

int32_t
Hypre_StructMatrix_SetDoubleParameter(
  Hypre_StructMatrix self,
  const char* name,
  double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetStringParameter
 */

int32_t
Hypre_StructMatrix_SetStringParameter(
  Hypre_StructMatrix self,
  const char* name,
  const char* value)
{
  return (*self->d_epv->f_SetStringParameter)(
    self,
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
Hypre_StructMatrix_isInstanceOf(
  Hypre_StructMatrix self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self,
    name);
}

/*
 * Method:  SetSymmetric
 */

int32_t
Hypre_StructMatrix_SetSymmetric(
  Hypre_StructMatrix self,
  int32_t symmetric)
{
  return (*self->d_epv->f_SetSymmetric)(
    self,
    symmetric);
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
Hypre_StructMatrix_GetObject(
  Hypre_StructMatrix self,
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
Hypre_StructMatrix_Assemble(
  Hypre_StructMatrix self)
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
Hypre_StructMatrix_queryInterface(
  Hypre_StructMatrix self,
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
Hypre_StructMatrix_isSame(
  Hypre_StructMatrix self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

int32_t
Hypre_StructMatrix_Initialize(
  Hypre_StructMatrix self)
{
  return (*self->d_epv->f_Initialize)(
    self);
}

/*
 * Method:  Apply
 */

int32_t
Hypre_StructMatrix_Apply(
  Hypre_StructMatrix self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  return (*self->d_epv->f_Apply)(
    self,
    x,
    y);
}

/*
 * Method:  SetNumGhost
 */

int32_t
Hypre_StructMatrix_SetNumGhost(
  Hypre_StructMatrix self,
  struct SIDL_int__array* num_ghost)
{
  return (*self->d_epv->f_SetNumGhost)(
    self,
    num_ghost);
}

/*
 * Method:  SetBoxValues
 */

int32_t
Hypre_StructMatrix_SetBoxValues(
  Hypre_StructMatrix self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_SetBoxValues)(
    self,
    ilower,
    iupper,
    num_stencil_indices,
    stencil_indices,
    values);
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
Hypre_StructMatrix_addReference(
  Hypre_StructMatrix self)
{
  (*self->d_epv->f_addReference)(
    self);
}

/*
 * Method:  SetGrid
 */

int32_t
Hypre_StructMatrix_SetGrid(
  Hypre_StructMatrix self,
  Hypre_StructGrid grid)
{
  return (*self->d_epv->f_SetGrid)(
    self,
    grid);
}

/*
 * Method:  SetValues
 */

int32_t
Hypre_StructMatrix_SetValues(
  Hypre_StructMatrix self,
  struct SIDL_int__array* index,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_SetValues)(
    self,
    index,
    num_stencil_indices,
    stencil_indices,
    values);
}

/*
 * Method:  SetDoubleArrayParameter
 */

int32_t
Hypre_StructMatrix_SetDoubleArrayParameter(
  Hypre_StructMatrix self,
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
Hypre_StructMatrix_deleteReference(
  Hypre_StructMatrix self)
{
  (*self->d_epv->f_deleteReference)(
    self);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_StructMatrix
Hypre_StructMatrix__cast(
  void* obj)
{
  Hypre_StructMatrix cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_StructMatrix) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.StructMatrix");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_StructMatrix__cast2(
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

struct Hypre_StructMatrix__array*
Hypre_StructMatrix__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_StructMatrix__array*
Hypre_StructMatrix__array_borrow(
  struct Hypre_StructMatrix__object** firstElement,
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
Hypre_StructMatrix__array_destroy(
  struct Hypre_StructMatrix__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_StructMatrix__array_dimen(const struct Hypre_StructMatrix__array *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_StructMatrix__array_lower(const struct Hypre_StructMatrix__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_StructMatrix__array_upper(const struct Hypre_StructMatrix__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructMatrix__object*
Hypre_StructMatrix__array_get(
  const struct Hypre_StructMatrix__array* array,
  const int32_t                           indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_StructMatrix__object*
Hypre_StructMatrix__array_get4(
  const struct Hypre_StructMatrix__array* array,
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
Hypre_StructMatrix__array_set(
  struct Hypre_StructMatrix__array*  array,
  const int32_t                      indices[],
  struct Hypre_StructMatrix__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_StructMatrix__array_set4(
  struct Hypre_StructMatrix__array*  array,
  int32_t                            i1,
  int32_t                            i2,
  int32_t                            i3,
  int32_t                            i4,
  struct Hypre_StructMatrix__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
