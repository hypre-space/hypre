/*
 * File:          Hypre_PCG_Stub.c
 * Symbol:        Hypre.PCG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:28 PST
 * Generated:     20021101 15:14:33 PST
 * Description:   Client-side glue code for Hypre.PCG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 463
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_PCG.h"
#include "Hypre_PCG_IOR.h"
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

static const struct Hypre_PCG__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct Hypre_PCG__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _ior = Hypre_PCG__externals();
#else
  const struct Hypre_PCG__external*(*dll_f)(void) =
    (const struct Hypre_PCG__external*(*)(void)) SIDL_Loader_lookupSymbol(
      "Hypre_PCG__externals");
  _ior = (dll_f ? (*dll_f)() : NULL);
  if (!_ior) {
    fputs("Babel: unable to load the implementation for Hypre.PCG; please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
#endif
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Constructor function for the class.
 */

Hypre_PCG
Hypre_PCG__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * &lt;p&gt;
 * Add one to the intrinsic reference count in the underlying object.
 * Object in &lt;code&gt;SIDL&lt;/code&gt; have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * &lt;/p&gt;
 * &lt;p&gt;
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * &lt;/p&gt;
 */

void
Hypre_PCG_addReference(
  Hypre_PCG self)
{
  (*self->d_epv->f_addReference)(
    self);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in &lt;code&gt;SIDL&lt;/code&gt; have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
Hypre_PCG_deleteReference(
  Hypre_PCG self)
{
  (*self->d_epv->f_deleteReference)(
    self);
}

/*
 * Return true if and only if &lt;code&gt;obj&lt;/code&gt; refers to the same
 * object as this object.
 */

SIDL_bool
Hypre_PCG_isSame(
  Hypre_PCG self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the &lt;code&gt;SIDL&lt;/code&gt; type name in &lt;code&gt;name&lt;/code&gt;
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling &lt;code&gt;deleteReference&lt;/code&gt; on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

SIDL_BaseInterface
Hypre_PCG_queryInterface(
  Hypre_PCG self,
  const char* name)
{
  return (*self->d_epv->f_queryInterface)(
    self,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the &lt;code&gt;SIDL&lt;/code&gt; type name.  This
 * routine will return &lt;code&gt;true&lt;/code&gt; if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
Hypre_PCG_isInstanceOf(
  Hypre_PCG self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self,
    name);
}

/*
 * Method:  SetCommunicator[]
 */

int32_t
Hypre_PCG_SetCommunicator(
  Hypre_PCG self,
  void* comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    comm);
}

/*
 * Method:  GetDoubleValue[]
 */

int32_t
Hypre_PCG_GetDoubleValue(
  Hypre_PCG self,
  const char* name,
  double* value)
{
  return (*self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value);
}

/*
 * Method:  GetIntValue[]
 */

int32_t
Hypre_PCG_GetIntValue(
  Hypre_PCG self,
  const char* name,
  int32_t* value)
{
  return (*self->d_epv->f_GetIntValue)(
    self,
    name,
    value);
}

/*
 * Method:  SetDoubleParameter[]
 */

int32_t
Hypre_PCG_SetDoubleParameter(
  Hypre_PCG self,
  const char* name,
  double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetIntParameter[]
 */

int32_t
Hypre_PCG_SetIntParameter(
  Hypre_PCG self,
  const char* name,
  int32_t value)
{
  return (*self->d_epv->f_SetIntParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetStringParameter[]
 */

int32_t
Hypre_PCG_SetStringParameter(
  Hypre_PCG self,
  const char* name,
  const char* value)
{
  return (*self->d_epv->f_SetStringParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetIntArrayParameter[]
 */

int32_t
Hypre_PCG_SetIntArrayParameter(
  Hypre_PCG self,
  const char* name,
  struct SIDL_int__array* value)
{
  return (*self->d_epv->f_SetIntArrayParameter)(
    self,
    name,
    value);
}

/*
 * Method:  SetDoubleArrayParameter[]
 */

int32_t
Hypre_PCG_SetDoubleArrayParameter(
  Hypre_PCG self,
  const char* name,
  struct SIDL_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArrayParameter)(
    self,
    name,
    value);
}

/*
 * Method:  Setup[]
 */

int32_t
Hypre_PCG_Setup(
  Hypre_PCG self,
  Hypre_Vector x,
  Hypre_Vector y)
{
  return (*self->d_epv->f_Setup)(
    self,
    x,
    y);
}

/*
 * Method:  Apply[]
 */

int32_t
Hypre_PCG_Apply(
  Hypre_PCG self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  return (*self->d_epv->f_Apply)(
    self,
    x,
    y);
}

/*
 * Method:  SetOperator[]
 */

int32_t
Hypre_PCG_SetOperator(
  Hypre_PCG self,
  Hypre_Operator A)
{
  return (*self->d_epv->f_SetOperator)(
    self,
    A);
}

/*
 * Method:  GetResidual[]
 */

int32_t
Hypre_PCG_GetResidual(
  Hypre_PCG self,
  Hypre_Vector* r)
{
  return (*self->d_epv->f_GetResidual)(
    self,
    r);
}

/*
 * Method:  SetLogging[]
 */

int32_t
Hypre_PCG_SetLogging(
  Hypre_PCG self,
  int32_t level)
{
  return (*self->d_epv->f_SetLogging)(
    self,
    level);
}

/*
 * Method:  SetPrintLevel[]
 */

int32_t
Hypre_PCG_SetPrintLevel(
  Hypre_PCG self,
  int32_t level)
{
  return (*self->d_epv->f_SetPrintLevel)(
    self,
    level);
}

/*
 * Method:  SetPreconditioner[]
 */

int32_t
Hypre_PCG_SetPreconditioner(
  Hypre_PCG self,
  Hypre_Solver s)
{
  return (*self->d_epv->f_SetPreconditioner)(
    self,
    s);
}

/*
 * Method:  GetPreconditionedResidual[]
 */

int32_t
Hypre_PCG_GetPreconditionedResidual(
  Hypre_PCG self,
  Hypre_Vector* r)
{
  return (*self->d_epv->f_GetPreconditionedResidual)(
    self,
    r);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_PCG
Hypre_PCG__cast(
  void* obj)
{
  Hypre_PCG cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_PCG) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.PCG");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_PCG__cast2(
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
/**
 * Create a dense array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_PCG__array*
Hypre_PCG__array_createCol(int32_t        dimen,
                           const int32_t lower[],
                           const int32_t upper[])
{
  return (struct Hypre_PCG__array*)SIDL_interface__array_createCol(dimen, lower,
    upper);
}

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_PCG__array*
Hypre_PCG__array_createRow(int32_t        dimen,
                           const int32_t lower[],
                           const int32_t upper[])
{
  return (struct Hypre_PCG__array*)SIDL_interface__array_createRow(dimen, lower,
    upper);
}

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_PCG__array*
Hypre_PCG__array_create1d(int32_t len)
{
  return (struct Hypre_PCG__array*)SIDL_interface__array_create1d(len);
}

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_PCG__array*
Hypre_PCG__array_create2dCol(int32_t m, int32_t n)
{
  return (struct Hypre_PCG__array*)SIDL_interface__array_create2dCol(m, n);
}

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_PCG__array*
Hypre_PCG__array_create2dRow(int32_t m, int32_t n)
{
  return (struct Hypre_PCG__array*)SIDL_interface__array_create2dRow(m, n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteReference will be called on the
 * value being replaced if it is not NULL.
 */
struct Hypre_PCG__array*
Hypre_PCG__array_borrow(Hypre_PCG*firstElement,
                        int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct Hypre_PCG__array*)SIDL_interface__array_borrow(
    (struct SIDL_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct Hypre_PCG__array*
Hypre_PCG__array_smartCopy(struct Hypre_PCG__array *array)
{
  return (struct Hypre_PCG__array*)
    SIDL_interface__array_smartCopy((struct SIDL_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
Hypre_PCG__array_addReference(struct Hypre_PCG__array* array)
{
  SIDL_interface__array_addReference((struct SIDL_interface__array *)array);
}

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
Hypre_PCG__array_deleteReference(struct Hypre_PCG__array* array)
{
  SIDL_interface__array_deleteReference((struct SIDL_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
Hypre_PCG
Hypre_PCG__array_get1(const struct Hypre_PCG__array* array,
                      const int32_t i1)
{
  return (Hypre_PCG)
    SIDL_interface__array_get1((const struct SIDL_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
Hypre_PCG
Hypre_PCG__array_get2(const struct Hypre_PCG__array* array,
                      const int32_t i1,
                      const int32_t i2)
{
  return (Hypre_PCG)
    SIDL_interface__array_get2((const struct SIDL_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
Hypre_PCG
Hypre_PCG__array_get3(const struct Hypre_PCG__array* array,
                      const int32_t i1,
                      const int32_t i2,
                      const int32_t i3)
{
  return (Hypre_PCG)
    SIDL_interface__array_get3((const struct SIDL_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
Hypre_PCG
Hypre_PCG__array_get4(const struct Hypre_PCG__array* array,
                      const int32_t i1,
                      const int32_t i2,
                      const int32_t i3,
                      const int32_t i4)
{
  return (Hypre_PCG)
    SIDL_interface__array_get4((const struct SIDL_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
Hypre_PCG
Hypre_PCG__array_get(const struct Hypre_PCG__array* array,
                     const int32_t indices[])
{
  return (Hypre_PCG)
    SIDL_interface__array_get((const struct SIDL_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
Hypre_PCG__array_set1(struct Hypre_PCG__array* array,
                      const int32_t i1,
                      Hypre_PCG const value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)array
  , i1, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
Hypre_PCG__array_set2(struct Hypre_PCG__array* array,
                      const int32_t i1,
                      const int32_t i2,
                      Hypre_PCG const value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)array
  , i1, i2, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
Hypre_PCG__array_set3(struct Hypre_PCG__array* array,
                      const int32_t i1,
                      const int32_t i2,
                      const int32_t i3,
                      Hypre_PCG const value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)array
  , i1, i2, i3, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
Hypre_PCG__array_set4(struct Hypre_PCG__array* array,
                      const int32_t i1,
                      const int32_t i2,
                      const int32_t i3,
                      const int32_t i4,
                      Hypre_PCG const value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)array
  , i1, i2, i3, i4, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
Hypre_PCG__array_set(struct Hypre_PCG__array* array,
                     const int32_t indices[],
                     Hypre_PCG const value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)array, indices,
    (struct SIDL_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
Hypre_PCG__array_dimen(const struct Hypre_PCG__array* array)
{
  return SIDL_interface__array_dimen((struct SIDL_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_PCG__array_lower(const struct Hypre_PCG__array* array,
                       const int32_t ind)
{
  return SIDL_interface__array_lower((struct SIDL_interface__array *)array,
    ind);
}

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_PCG__array_upper(const struct Hypre_PCG__array* array,
                       const int32_t ind)
{
  return SIDL_interface__array_upper((struct SIDL_interface__array *)array,
    ind);
}

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_PCG__array_stride(const struct Hypre_PCG__array* array,
                        const int32_t ind)
{
  return SIDL_interface__array_stride((struct SIDL_interface__array *)array,
    ind);
}

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_PCG__array_isColumnOrder(const struct Hypre_PCG__array* array)
{
  return SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)array);
}

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_PCG__array_isRowOrder(const struct Hypre_PCG__array* array)
{
  return SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)array);
}

/**
 * Copy the contents of one array (src) to a second array
 * (dest). For the copy to take place, both arrays must
 * exist and be of the same dimension. This method will
 * not modify dest's size, index bounds, or stride; only
 * the array element values of dest may be changed by
 * this function. No part of src is ever changed by copy.
 * 
 * On exit, dest[i][j][k]... = src[i][j][k]... for all
 * indices i,j,k...  that are in both arrays. If dest and
 * src have no indices in common, nothing is copied. For
 * example, if src is a 1-d array with elements 0-5 and
 * dest is a 1-d array with elements 2-3, this function
 * will make the following assignments:
 *   dest[2] = src[2],
 *   dest[3] = src[3].
 * The function copied the elements that both arrays have
 * in common.  If dest had elements 4-10, this function
 * will make the following assignments:
 *   dest[4] = src[4],
 *   dest[5] = src[5].
 */
void
Hypre_PCG__array_copy(const struct Hypre_PCG__array* src,
                            struct Hypre_PCG__array* dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array *)src,
                             (struct SIDL_interface__array *)dest);
}

/**
 * If necessary, convert a general matrix into a matrix
 * with the required properties. This checks the
 * dimension and ordering of the matrix.  If both these
 * match, it simply returns a new reference to the
 * existing matrix. If the dimension of the incoming
 * array doesn't match, it returns NULL. If the ordering
 * of the incoming array doesn't match the specification,
 * a new array is created with the desired ordering and
 * the content of the incoming array is copied to the new
 * array.
 * 
 * The ordering parameter should be one of the constants
 * defined in enum SIDL_array_ordering
 * (e.g. SIDL_general_order, SIDL_column_major_order, or
 * SIDL_row_major_order). If you specify
 * SIDL_general_order, this routine will only check the
 * dimension because any matrix is SIDL_general_order.
 * 
 * The caller assumes ownership of the returned reference
 * unless it's NULL.
 */
struct Hypre_PCG__array*
Hypre_PCG__array_ensure(struct Hypre_PCG__array* src,
                        int32_t dimen,
int     ordering)
{
  return (struct Hypre_PCG__array*)
    SIDL_interface__array_ensure((struct SIDL_interface__array *)src, dimen,
      ordering);
}

