/*
 * File:          bHYPRE_IJBuildVector_Stub.c
 * Symbol:        bHYPRE.IJBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:36 PST
 * Generated:     20050225 15:45:39 PST
 * Description:   Client-side glue code for bHYPRE.IJBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 203
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_IJBuildVector.h"
#include "bHYPRE_IJBuildVector_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stddef.h>
#include "sidl_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.h"
#endif

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
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
bHYPRE_IJBuildVector_addRef(
  bHYPRE_IJBuildVector self)
{
  (*self->d_epv->f_addRef)(
    self->d_object);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
bHYPRE_IJBuildVector_deleteRef(
  bHYPRE_IJBuildVector self)
{
  (*self->d_epv->f_deleteRef)(
    self->d_object);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_IJBuildVector_isSame(
  bHYPRE_IJBuildVector self,
  /*in*/ sidl_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
    iobj);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

sidl_BaseInterface
bHYPRE_IJBuildVector_queryInt(
  bHYPRE_IJBuildVector self,
  /*in*/ const char* name)
{
  return (*self->d_epv->f_queryInt)(
    self->d_object,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

sidl_bool
bHYPRE_IJBuildVector_isType(
  bHYPRE_IJBuildVector self,
  /*in*/ const char* name)
{
  return (*self->d_epv->f_isType)(
    self->d_object,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
bHYPRE_IJBuildVector_getClassInfo(
  bHYPRE_IJBuildVector self)
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object);
}

/*
 * Set the MPI Communicator.
 * 
 */

int32_t
bHYPRE_IJBuildVector_SetCommunicator(
  bHYPRE_IJBuildVector self,
  /*in*/ void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self->d_object,
    mpi_comm);
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

int32_t
bHYPRE_IJBuildVector_Initialize(
  bHYPRE_IJBuildVector self)
{
  return (*self->d_epv->f_Initialize)(
    self->d_object);
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

int32_t
bHYPRE_IJBuildVector_Assemble(
  bHYPRE_IJBuildVector self)
{
  return (*self->d_epv->f_Assemble)(
    self->d_object);
}

/*
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

int32_t
bHYPRE_IJBuildVector_GetObject(
  bHYPRE_IJBuildVector self,
  /*out*/ sidl_BaseInterface* A)
{
  return (*self->d_epv->f_GetObject)(
    self->d_object,
    A);
}

/*
 * Set the local range for a vector object.  Each process owns
 * some unique consecutive range of vector unknowns, indicated
 * by the global indices {\tt jlower} and {\tt jupper}.  The
 * data is required to be such that the value of {\tt jlower} on
 * any process $p$ be exactly one more than the value of {\tt
 * jupper} on process $p-1$.  Note that the first index of the
 * global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 * 
 */

int32_t
bHYPRE_IJBuildVector_SetLocalRange(
  bHYPRE_IJBuildVector self,
  /*in*/ int32_t jlower,
  /*in*/ int32_t jupper)
{
  return (*self->d_epv->f_SetLocalRange)(
    self->d_object,
    jlower,
    jupper);
}

/*
 * Sets values in vector.  The arrays {\tt values} and {\tt
 * indices} are of dimension {\tt nvalues} and contain the
 * vector values to be set and the corresponding global vector
 * indices, respectively.  Erases any previous values at the
 * specified locations and replaces them with new ones.
 * 
 * Not collective.
 * 
 */

int32_t
bHYPRE_IJBuildVector_SetValues(
  bHYPRE_IJBuildVector self,
  /*in*/ int32_t nvalues,
  /*in*/ struct sidl_int__array* indices,
  /*in*/ struct sidl_double__array* values)
{
  return (*self->d_epv->f_SetValues)(
    self->d_object,
    nvalues,
    indices,
    values);
}

/*
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */

int32_t
bHYPRE_IJBuildVector_AddToValues(
  bHYPRE_IJBuildVector self,
  /*in*/ int32_t nvalues,
  /*in*/ struct sidl_int__array* indices,
  /*in*/ struct sidl_double__array* values)
{
  return (*self->d_epv->f_AddToValues)(
    self->d_object,
    nvalues,
    indices,
    values);
}

/*
 * Returns range of the part of the vector owned by this
 * processor.
 * 
 */

int32_t
bHYPRE_IJBuildVector_GetLocalRange(
  bHYPRE_IJBuildVector self,
  /*out*/ int32_t* jlower,
  /*out*/ int32_t* jupper)
{
  return (*self->d_epv->f_GetLocalRange)(
    self->d_object,
    jlower,
    jupper);
}

/*
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 */

int32_t
bHYPRE_IJBuildVector_GetValues(
  bHYPRE_IJBuildVector self,
  /*in*/ int32_t nvalues,
  /*in*/ struct sidl_int__array* indices,
  /*inout*/ struct sidl_double__array** values)
{
  return (*self->d_epv->f_GetValues)(
    self->d_object,
    nvalues,
    indices,
    values);
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

int32_t
bHYPRE_IJBuildVector_Print(
  bHYPRE_IJBuildVector self,
  /*in*/ const char* filename)
{
  return (*self->d_epv->f_Print)(
    self->d_object,
    filename);
}

/*
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 * 
 */

int32_t
bHYPRE_IJBuildVector_Read(
  bHYPRE_IJBuildVector self,
  /*in*/ const char* filename,
  /*in*/ void* comm)
{
  return (*self->d_epv->f_Read)(
    self->d_object,
    filename,
    comm);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__cast(
  void* obj)
{
  bHYPRE_IJBuildVector cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_IJBuildVector) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.IJBuildVector");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_IJBuildVector__cast2(
  void* obj,
  const char* type)
{
  void* cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type);
  }

  return cast;
}
struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_IJBuildVector__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_IJBuildVector__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_IJBuildVector__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_create1dInit(
  int32_t len, 
  bHYPRE_IJBuildVector* data)
{
  return (struct 
    bHYPRE_IJBuildVector__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_IJBuildVector__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_IJBuildVector__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_borrow(
  bHYPRE_IJBuildVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_IJBuildVector__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_smartCopy(
  struct bHYPRE_IJBuildVector__array *array)
{
  return (struct bHYPRE_IJBuildVector__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_IJBuildVector__array_addRef(
  struct bHYPRE_IJBuildVector__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_IJBuildVector__array_deleteRef(
  struct bHYPRE_IJBuildVector__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__array_get1(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1)
{
  return (bHYPRE_IJBuildVector)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__array_get2(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_IJBuildVector)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__array_get3(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_IJBuildVector)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__array_get4(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_IJBuildVector)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__array_get5(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_IJBuildVector)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__array_get6(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_IJBuildVector)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__array_get7(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_IJBuildVector)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_IJBuildVector
bHYPRE_IJBuildVector__array_get(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t indices[])
{
  return (bHYPRE_IJBuildVector)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_IJBuildVector__array_set1(
  struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  bHYPRE_IJBuildVector const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildVector__array_set2(
  struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IJBuildVector const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildVector__array_set3(
  struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IJBuildVector const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildVector__array_set4(
  struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IJBuildVector const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildVector__array_set5(
  struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IJBuildVector const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildVector__array_set6(
  struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IJBuildVector const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildVector__array_set7(
  struct bHYPRE_IJBuildVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IJBuildVector const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildVector__array_set(
  struct bHYPRE_IJBuildVector__array* array,
  const int32_t indices[],
  bHYPRE_IJBuildVector const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_IJBuildVector__array_dimen(
  const struct bHYPRE_IJBuildVector__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_IJBuildVector__array_lower(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJBuildVector__array_upper(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJBuildVector__array_length(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJBuildVector__array_stride(
  const struct bHYPRE_IJBuildVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_IJBuildVector__array_isColumnOrder(
  const struct bHYPRE_IJBuildVector__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_IJBuildVector__array_isRowOrder(
  const struct bHYPRE_IJBuildVector__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_IJBuildVector__array_copy(
  const struct bHYPRE_IJBuildVector__array* src,
  struct bHYPRE_IJBuildVector__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_slice(
  struct bHYPRE_IJBuildVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_IJBuildVector__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_IJBuildVector__array*
bHYPRE_IJBuildVector__array_ensure(
  struct bHYPRE_IJBuildVector__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_IJBuildVector__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

