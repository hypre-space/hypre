/*
 * File:          bHYPRE_IJBuildMatrix_Stub.c
 * Symbol:        bHYPRE.IJBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:42 PST
 * Description:   Client-side glue code for bHYPRE.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 85
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_IJBuildMatrix.h"
#include "bHYPRE_IJBuildMatrix_IOR.h"
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
bHYPRE_IJBuildMatrix_addRef(
  bHYPRE_IJBuildMatrix self)
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
bHYPRE_IJBuildMatrix_deleteRef(
  bHYPRE_IJBuildMatrix self)
{
  (*self->d_epv->f_deleteRef)(
    self->d_object);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_IJBuildMatrix_isSame(
  bHYPRE_IJBuildMatrix self,
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
bHYPRE_IJBuildMatrix_queryInt(
  bHYPRE_IJBuildMatrix self,
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
bHYPRE_IJBuildMatrix_isType(
  bHYPRE_IJBuildMatrix self,
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
bHYPRE_IJBuildMatrix_getClassInfo(
  bHYPRE_IJBuildMatrix self)
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object);
}

/*
 * Set the MPI Communicator.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_SetCommunicator(
  bHYPRE_IJBuildMatrix self,
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
bHYPRE_IJBuildMatrix_Initialize(
  bHYPRE_IJBuildMatrix self)
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
bHYPRE_IJBuildMatrix_Assemble(
  bHYPRE_IJBuildMatrix self)
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
bHYPRE_IJBuildMatrix_GetObject(
  bHYPRE_IJBuildMatrix self,
  /*out*/ sidl_BaseInterface* A)
{
  return (*self->d_epv->f_GetObject)(
    self->d_object,
    A);
}

/*
 * Set the local range for a matrix object.  Each process owns
 * some unique consecutive range of rows, indicated by the
 * global row indices {\tt ilower} and {\tt iupper}.  The row
 * data is required to be such that the value of {\tt ilower} on
 * any process $p$ be exactly one more than the value of {\tt
 * iupper} on process $p-1$.  Note that the first row of the
 * global matrix may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically
 * should match {\tt ilower} and {\tt iupper}, respectively.
 * For rectangular matrices, {\tt jlower} and {\tt jupper}
 * should define a partitioning of the columns.  This
 * partitioning must be used for any vector $v$ that will be
 * used in matrix-vector products with the rectangular matrix.
 * The matrix data structure may use {\tt jlower} and {\tt
 * jupper} to store the diagonal blocks (rectangular in general)
 * of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_SetLocalRange(
  bHYPRE_IJBuildMatrix self,
  /*in*/ int32_t ilower,
  /*in*/ int32_t iupper,
  /*in*/ int32_t jlower,
  /*in*/ int32_t jupper)
{
  return (*self->d_epv->f_SetLocalRange)(
    self->d_object,
    ilower,
    iupper,
    jlower,
    jupper);
}

/*
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  Erases any
 * previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_SetValues(
  bHYPRE_IJBuildMatrix self,
  /*in*/ int32_t nrows,
  /*in*/ struct sidl_int__array* ncols,
  /*in*/ struct sidl_int__array* rows,
  /*in*/ struct sidl_int__array* cols,
  /*in*/ struct sidl_double__array* values)
{
  return (*self->d_epv->f_SetValues)(
    self->d_object,
    nrows,
    ncols,
    rows,
    cols,
    values);
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_AddToValues(
  bHYPRE_IJBuildMatrix self,
  /*in*/ int32_t nrows,
  /*in*/ struct sidl_int__array* ncols,
  /*in*/ struct sidl_int__array* rows,
  /*in*/ struct sidl_int__array* cols,
  /*in*/ struct sidl_double__array* values)
{
  return (*self->d_epv->f_AddToValues)(
    self->d_object,
    nrows,
    ncols,
    rows,
    cols,
    values);
}

/*
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_GetLocalRange(
  bHYPRE_IJBuildMatrix self,
  /*out*/ int32_t* ilower,
  /*out*/ int32_t* iupper,
  /*out*/ int32_t* jlower,
  /*out*/ int32_t* jupper)
{
  return (*self->d_epv->f_GetLocalRange)(
    self->d_object,
    ilower,
    iupper,
    jlower,
    jupper);
}

/*
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_GetRowCounts(
  bHYPRE_IJBuildMatrix self,
  /*in*/ int32_t nrows,
  /*in*/ struct sidl_int__array* rows,
  /*inout*/ struct sidl_int__array** ncols)
{
  return (*self->d_epv->f_GetRowCounts)(
    self->d_object,
    nrows,
    rows,
    ncols);
}

/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_GetValues(
  bHYPRE_IJBuildMatrix self,
  /*in*/ int32_t nrows,
  /*in*/ struct sidl_int__array* ncols,
  /*in*/ struct sidl_int__array* rows,
  /*in*/ struct sidl_int__array* cols,
  /*inout*/ struct sidl_double__array** values)
{
  return (*self->d_epv->f_GetValues)(
    self->d_object,
    nrows,
    ncols,
    rows,
    cols,
    values);
}

/*
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_SetRowSizes(
  bHYPRE_IJBuildMatrix self,
  /*in*/ struct sidl_int__array* sizes)
{
  return (*self->d_epv->f_SetRowSizes)(
    self->d_object,
    sizes);
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_Print(
  bHYPRE_IJBuildMatrix self,
  /*in*/ const char* filename)
{
  return (*self->d_epv->f_Print)(
    self->d_object,
    filename);
}

/*
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */

int32_t
bHYPRE_IJBuildMatrix_Read(
  bHYPRE_IJBuildMatrix self,
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

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__cast(
  void* obj)
{
  bHYPRE_IJBuildMatrix cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_IJBuildMatrix) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.IJBuildMatrix");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_IJBuildMatrix__cast2(
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
struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_IJBuildMatrix__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_IJBuildMatrix__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_IJBuildMatrix__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_create1dInit(
  int32_t len, 
  bHYPRE_IJBuildMatrix* data)
{
  return (struct 
    bHYPRE_IJBuildMatrix__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_IJBuildMatrix__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_IJBuildMatrix__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_borrow(
  bHYPRE_IJBuildMatrix* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_IJBuildMatrix__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_smartCopy(
  struct bHYPRE_IJBuildMatrix__array *array)
{
  return (struct bHYPRE_IJBuildMatrix__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_IJBuildMatrix__array_addRef(
  struct bHYPRE_IJBuildMatrix__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_IJBuildMatrix__array_deleteRef(
  struct bHYPRE_IJBuildMatrix__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get1(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1)
{
  return (bHYPRE_IJBuildMatrix)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get2(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_IJBuildMatrix)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get3(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_IJBuildMatrix)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get4(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_IJBuildMatrix)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get5(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_IJBuildMatrix)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get6(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_IJBuildMatrix)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get7(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_IJBuildMatrix)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_IJBuildMatrix
bHYPRE_IJBuildMatrix__array_get(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t indices[])
{
  return (bHYPRE_IJBuildMatrix)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_IJBuildMatrix__array_set1(
  struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  bHYPRE_IJBuildMatrix const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildMatrix__array_set2(
  struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IJBuildMatrix const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildMatrix__array_set3(
  struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IJBuildMatrix const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildMatrix__array_set4(
  struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IJBuildMatrix const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildMatrix__array_set5(
  struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IJBuildMatrix const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildMatrix__array_set6(
  struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IJBuildMatrix const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildMatrix__array_set7(
  struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IJBuildMatrix const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJBuildMatrix__array_set(
  struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t indices[],
  bHYPRE_IJBuildMatrix const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_IJBuildMatrix__array_dimen(
  const struct bHYPRE_IJBuildMatrix__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_IJBuildMatrix__array_lower(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJBuildMatrix__array_upper(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJBuildMatrix__array_length(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJBuildMatrix__array_stride(
  const struct bHYPRE_IJBuildMatrix__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_IJBuildMatrix__array_isColumnOrder(
  const struct bHYPRE_IJBuildMatrix__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_IJBuildMatrix__array_isRowOrder(
  const struct bHYPRE_IJBuildMatrix__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_IJBuildMatrix__array_copy(
  const struct bHYPRE_IJBuildMatrix__array* src,
  struct bHYPRE_IJBuildMatrix__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_slice(
  struct bHYPRE_IJBuildMatrix__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_IJBuildMatrix__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_IJBuildMatrix__array*
bHYPRE_IJBuildMatrix__array_ensure(
  struct bHYPRE_IJBuildMatrix__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_IJBuildMatrix__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

