/*
 * File:          bHYPRE_ParCSRDiagScale_Stub.c
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:35 PST
 * Generated:     20030401 14:47:42 PST
 * Description:   Client-side glue code for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1140
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_ParCSRDiagScale.h"
#include "bHYPRE_ParCSRDiagScale_IOR.h"
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

static const struct bHYPRE_ParCSRDiagScale__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_ParCSRDiagScale__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _ior = bHYPRE_ParCSRDiagScale__externals();
#else
  const struct bHYPRE_ParCSRDiagScale__external*(*dll_f)(void) =
    (const struct bHYPRE_ParCSRDiagScale__external*(*)(void)) 
      SIDL_Loader_lookupSymbol(
      "bHYPRE_ParCSRDiagScale__externals");
  _ior = (dll_f ? (*dll_f)() : NULL);
  if (!_ior) {
    fputs("Babel: unable to load the implementation for bHYPRE.ParCSRDiagScale; please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
#endif
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Constructor function for the class.
 */

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__create()
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
bHYPRE_ParCSRDiagScale_addRef(
  bHYPRE_ParCSRDiagScale self)
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
bHYPRE_ParCSRDiagScale_deleteRef(
  bHYPRE_ParCSRDiagScale self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
bHYPRE_ParCSRDiagScale_isSame(
  bHYPRE_ParCSRDiagScale self,
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
bHYPRE_ParCSRDiagScale_queryInt(
  bHYPRE_ParCSRDiagScale self,
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
bHYPRE_ParCSRDiagScale_isType(
  bHYPRE_ParCSRDiagScale self,
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
bHYPRE_ParCSRDiagScale_getClassInfo(
  bHYPRE_ParCSRDiagScale self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Set the MPI Communicator.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetCommunicator(
  bHYPRE_ParCSRDiagScale self,
  void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetIntParameter(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  int32_t value)
{
  return (*self->d_epv->f_SetIntParameter)(
    self,
    name,
    value);
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetDoubleParameter(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetStringParameter(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  const char* value)
{
  return (*self->d_epv->f_SetStringParameter)(
    self,
    name,
    value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArray1Parameter)(
    self,
    name,
    value);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArray2Parameter)(
    self,
    name,
    value);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  struct SIDL_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArray1Parameter)(
    self,
    name,
    value);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  struct SIDL_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArray2Parameter)(
    self,
    name,
    value);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_GetIntValue(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  int32_t* value)
{
  return (*self->d_epv->f_GetIntValue)(
    self,
    name,
    value);
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_GetDoubleValue(
  bHYPRE_ParCSRDiagScale self,
  const char* name,
  double* value)
{
  return (*self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_Setup(
  bHYPRE_ParCSRDiagScale self,
  bHYPRE_Vector b,
  bHYPRE_Vector x)
{
  return (*self->d_epv->f_Setup)(
    self,
    b,
    x);
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_Apply(
  bHYPRE_ParCSRDiagScale self,
  bHYPRE_Vector b,
  bHYPRE_Vector* x)
{
  return (*self->d_epv->f_Apply)(
    self,
    b,
    x);
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetOperator(
  bHYPRE_ParCSRDiagScale self,
  bHYPRE_Operator A)
{
  return (*self->d_epv->f_SetOperator)(
    self,
    A);
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetTolerance(
  bHYPRE_ParCSRDiagScale self,
  double tolerance)
{
  return (*self->d_epv->f_SetTolerance)(
    self,
    tolerance);
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetMaxIterations(
  bHYPRE_ParCSRDiagScale self,
  int32_t max_iterations)
{
  return (*self->d_epv->f_SetMaxIterations)(
    self,
    max_iterations);
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetLogging(
  bHYPRE_ParCSRDiagScale self,
  int32_t level)
{
  return (*self->d_epv->f_SetLogging)(
    self,
    level);
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_SetPrintLevel(
  bHYPRE_ParCSRDiagScale self,
  int32_t level)
{
  return (*self->d_epv->f_SetPrintLevel)(
    self,
    level);
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_GetNumIterations(
  bHYPRE_ParCSRDiagScale self,
  int32_t* num_iterations)
{
  return (*self->d_epv->f_GetNumIterations)(
    self,
    num_iterations);
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

int32_t
bHYPRE_ParCSRDiagScale_GetRelResidualNorm(
  bHYPRE_ParCSRDiagScale self,
  double* norm)
{
  return (*self->d_epv->f_GetRelResidualNorm)(
    self,
    norm);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__cast(
  void* obj)
{
  bHYPRE_ParCSRDiagScale cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (bHYPRE_ParCSRDiagScale) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.ParCSRDiagScale");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_ParCSRDiagScale__cast2(
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
struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_createCol(int32_t        dimen,
                                        const int32_t lower[],
                                        const int32_t upper[])
{
  return (struct 
    bHYPRE_ParCSRDiagScale__array*)SIDL_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_createRow(int32_t        dimen,
                                        const int32_t lower[],
                                        const int32_t upper[])
{
  return (struct 
    bHYPRE_ParCSRDiagScale__array*)SIDL_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_ParCSRDiagScale__array*)SIDL_interface__array_create1d(len);
}

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_ParCSRDiagScale__array*)SIDL_interface__array_create2dCol(m, n);
}

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_ParCSRDiagScale__array*)SIDL_interface__array_create2dRow(m, n);
}

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_borrow(bHYPRE_ParCSRDiagScale*firstElement,
                                     int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct bHYPRE_ParCSRDiagScale__array*)SIDL_interface__array_borrow(
    (struct SIDL_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_smartCopy(struct bHYPRE_ParCSRDiagScale__array 
  *array)
{
  return (struct bHYPRE_ParCSRDiagScale__array*)
    SIDL_interface__array_smartCopy((struct SIDL_interface__array *)array);
}

void
bHYPRE_ParCSRDiagScale__array_addRef(struct bHYPRE_ParCSRDiagScale__array* 
  array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array *)array);
}

void
bHYPRE_ParCSRDiagScale__array_deleteRef(struct bHYPRE_ParCSRDiagScale__array* 
  array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array *)array);
}

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get1(const struct bHYPRE_ParCSRDiagScale__array* 
  array,
                                   const int32_t i1)
{
  return (bHYPRE_ParCSRDiagScale)
    SIDL_interface__array_get1((const struct SIDL_interface__array *)array
    , i1);
}

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get2(const struct bHYPRE_ParCSRDiagScale__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2)
{
  return (bHYPRE_ParCSRDiagScale)
    SIDL_interface__array_get2((const struct SIDL_interface__array *)array
    , i1, i2);
}

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get3(const struct bHYPRE_ParCSRDiagScale__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3)
{
  return (bHYPRE_ParCSRDiagScale)
    SIDL_interface__array_get3((const struct SIDL_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get4(const struct bHYPRE_ParCSRDiagScale__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   const int32_t i4)
{
  return (bHYPRE_ParCSRDiagScale)
    SIDL_interface__array_get4((const struct SIDL_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_ParCSRDiagScale
bHYPRE_ParCSRDiagScale__array_get(const struct bHYPRE_ParCSRDiagScale__array* 
  array,
                                  const int32_t indices[])
{
  return (bHYPRE_ParCSRDiagScale)
    SIDL_interface__array_get((const struct SIDL_interface__array *)array,
      indices);
}

void
bHYPRE_ParCSRDiagScale__array_set1(struct bHYPRE_ParCSRDiagScale__array* array,
                                   const int32_t i1,
                                   bHYPRE_ParCSRDiagScale const value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)array
  , i1, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_ParCSRDiagScale__array_set2(struct bHYPRE_ParCSRDiagScale__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   bHYPRE_ParCSRDiagScale const value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)array
  , i1, i2, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_ParCSRDiagScale__array_set3(struct bHYPRE_ParCSRDiagScale__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   bHYPRE_ParCSRDiagScale const value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)array
  , i1, i2, i3, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_ParCSRDiagScale__array_set4(struct bHYPRE_ParCSRDiagScale__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   const int32_t i4,
                                   bHYPRE_ParCSRDiagScale const value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)array
  , i1, i2, i3, i4, (struct SIDL_BaseInterface__object *)value);
}

void
bHYPRE_ParCSRDiagScale__array_set(struct bHYPRE_ParCSRDiagScale__array* array,
                                  const int32_t indices[],
                                  bHYPRE_ParCSRDiagScale const value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)array, indices,
    (struct SIDL_BaseInterface__object *)value);
}

int32_t
bHYPRE_ParCSRDiagScale__array_dimen(const struct bHYPRE_ParCSRDiagScale__array* 
  array)
{
  return SIDL_interface__array_dimen((struct SIDL_interface__array *)array);
}

int32_t
bHYPRE_ParCSRDiagScale__array_lower(const struct bHYPRE_ParCSRDiagScale__array* 
  array,
                                    const int32_t ind)
{
  return SIDL_interface__array_lower((struct SIDL_interface__array *)array,
    ind);
}

int32_t
bHYPRE_ParCSRDiagScale__array_upper(const struct bHYPRE_ParCSRDiagScale__array* 
  array,
                                    const int32_t ind)
{
  return SIDL_interface__array_upper((struct SIDL_interface__array *)array,
    ind);
}

int32_t
bHYPRE_ParCSRDiagScale__array_stride(const struct 
  bHYPRE_ParCSRDiagScale__array* array,
                                     const int32_t ind)
{
  return SIDL_interface__array_stride((struct SIDL_interface__array *)array,
    ind);
}

int
bHYPRE_ParCSRDiagScale__array_isColumnOrder(const struct 
  bHYPRE_ParCSRDiagScale__array* array)
{
  return SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)array);
}

int
bHYPRE_ParCSRDiagScale__array_isRowOrder(const struct 
  bHYPRE_ParCSRDiagScale__array* array)
{
  return SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)array);
}

void
bHYPRE_ParCSRDiagScale__array_copy(const struct bHYPRE_ParCSRDiagScale__array* 
  src,
                                         struct bHYPRE_ParCSRDiagScale__array* 
  dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array *)src,
                             (struct SIDL_interface__array *)dest);
}

struct bHYPRE_ParCSRDiagScale__array*
bHYPRE_ParCSRDiagScale__array_ensure(struct bHYPRE_ParCSRDiagScale__array* src,
                                     int32_t dimen,
                                     int     ordering)
{
  return (struct bHYPRE_ParCSRDiagScale__array*)
    SIDL_interface__array_ensure((struct SIDL_interface__array *)src, dimen,
      ordering);
}

