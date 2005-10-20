/*
 * File:          bHYPRE_IJMatrixView_Stub.c
 * Symbol:        bHYPRE.IJMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.IJMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "bHYPRE_IJMatrixView.h"
#include "bHYPRE_IJMatrixView_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stddef.h>
#include <string.h>
#include "sidl_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.h"
#endif

/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct bHYPRE_IJMatrixView__object* 
  bHYPRE_IJMatrixView__remoteConnect(const char* url, sidl_BaseInterface *_ex);
static struct bHYPRE_IJMatrixView__object* 
  bHYPRE_IJMatrixView__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__connect(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_IJMatrixView__remoteConnect(url, _ex);
}

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
bHYPRE_IJMatrixView_addRef(
  /* in */ bHYPRE_IJMatrixView self)
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
bHYPRE_IJMatrixView_deleteRef(
  /* in */ bHYPRE_IJMatrixView self)
{
  (*self->d_epv->f_deleteRef)(
    self->d_object);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_IJMatrixView_isSame(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ sidl_BaseInterface iobj)
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
bHYPRE_IJMatrixView_queryInt(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* name)
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
bHYPRE_IJMatrixView_isType(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_isType)(
    self->d_object,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
bHYPRE_IJMatrixView_getClassInfo(
  /* in */ bHYPRE_IJMatrixView self)
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object);
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */

int32_t
bHYPRE_IJMatrixView_SetCommunicator(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
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
bHYPRE_IJMatrixView_Initialize(
  /* in */ bHYPRE_IJMatrixView self)
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
bHYPRE_IJMatrixView_Assemble(
  /* in */ bHYPRE_IJMatrixView self)
{
  return (*self->d_epv->f_Assemble)(
    self->d_object);
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
bHYPRE_IJMatrixView_SetLocalRange(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper)
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
 * directly to the column entries in {\tt cols}.  The last argument
 * is the size of the cols and values arrays, i.e. the total number
 * of nonzeros being provided, i.e. the sum of all values in ncols.
 * This functin erases any previous values at the specified locations and
 * replaces them with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */

int32_t
bHYPRE_IJMatrixView_SetValues(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros)
{
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1]; 
  struct sidl_int__array ncols_real;
  struct sidl_int__array*ncols_tmp = &ncols_real;
  int32_t rows_lower[1], rows_upper[1], rows_stride[1]; 
  struct sidl_int__array rows_real;
  struct sidl_int__array*rows_tmp = &rows_real;
  int32_t cols_lower[1], cols_upper[1], cols_stride[1]; 
  struct sidl_int__array cols_real;
  struct sidl_int__array*cols_tmp = &cols_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper,
    ncols_stride);
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_SetValues)(
    self->d_object,
    ncols_tmp,
    rows_tmp,
    cols_tmp,
    values_tmp);
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
bHYPRE_IJMatrixView_AddToValues(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros)
{
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1]; 
  struct sidl_int__array ncols_real;
  struct sidl_int__array*ncols_tmp = &ncols_real;
  int32_t rows_lower[1], rows_upper[1], rows_stride[1]; 
  struct sidl_int__array rows_real;
  struct sidl_int__array*rows_tmp = &rows_real;
  int32_t cols_lower[1], cols_upper[1], cols_stride[1]; 
  struct sidl_int__array cols_real;
  struct sidl_int__array*cols_tmp = &cols_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper,
    ncols_stride);
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_AddToValues)(
    self->d_object,
    ncols_tmp,
    rows_tmp,
    cols_tmp,
    values_tmp);
}

/*
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 */

int32_t
bHYPRE_IJMatrixView_GetLocalRange(
  /* in */ bHYPRE_IJMatrixView self,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper)
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
bHYPRE_IJMatrixView_GetRowCounts(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* rows,
  /* inout rarray[nrows] */ int32_t* ncols)
{
  int32_t rows_lower[1], rows_upper[1], rows_stride[1]; 
  struct sidl_int__array rows_real;
  struct sidl_int__array*rows_tmp = &rows_real;
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1]; 
  struct sidl_int__array ncols_real;
  struct sidl_int__array*ncols_tmp = &ncols_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper,
    ncols_stride);
  return (*self->d_epv->f_GetRowCounts)(
    self->d_object,
    rows_tmp,
    &ncols_tmp);
}

/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */

int32_t
bHYPRE_IJMatrixView_GetValues(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* inout rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros)
{
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1]; 
  struct sidl_int__array ncols_real;
  struct sidl_int__array*ncols_tmp = &ncols_real;
  int32_t rows_lower[1], rows_upper[1], rows_stride[1]; 
  struct sidl_int__array rows_real;
  struct sidl_int__array*rows_tmp = &rows_real;
  int32_t cols_lower[1], cols_upper[1], cols_stride[1]; 
  struct sidl_int__array cols_real;
  struct sidl_int__array*cols_tmp = &cols_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper,
    ncols_stride);
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_GetValues)(
    self->d_object,
    ncols_tmp,
    rows_tmp,
    cols_tmp,
    &values_tmp);
}

/*
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */

int32_t
bHYPRE_IJMatrixView_SetRowSizes(
  /* in */ bHYPRE_IJMatrixView self,
  /* in rarray[nrows] */ int32_t* sizes,
  /* in */ int32_t nrows)
{
  int32_t sizes_lower[1], sizes_upper[1], sizes_stride[1]; 
  struct sidl_int__array sizes_real;
  struct sidl_int__array*sizes_tmp = &sizes_real;
  sizes_upper[0] = nrows-1;
  sidl_int__array_init(sizes, sizes_tmp, 1, sizes_lower, sizes_upper,
    sizes_stride);
  return (*self->d_epv->f_SetRowSizes)(
    self->d_object,
    sizes_tmp);
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

int32_t
bHYPRE_IJMatrixView_Print(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* filename)
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
bHYPRE_IJMatrixView_Read(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm)
{
  return (*self->d_epv->f_Read)(
    self->d_object,
    filename,
    comm);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__cast(
  void* obj)
{
  bHYPRE_IJMatrixView cast = NULL;

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.IJMatrixView",
      (void*)bHYPRE_IJMatrixView__IHConnect);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_IJMatrixView) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.IJMatrixView");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_IJMatrixView__cast2(
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
/*
 * Select and execute a method by name
 */

void
bHYPRE_IJMatrixView__exec(
  /* in */ bHYPRE_IJMatrixView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs)
{
  (*self->d_epv->f__exec)(
  self->d_object,
  methodName,
  inArgs,
  outArgs);
}

/*
 * Get the URL of the Implementation of this object (for RMI)
 */

char*
bHYPRE_IJMatrixView__getURL(
  /* in */ bHYPRE_IJMatrixView self)
{
  return (*self->d_epv->f__getURL)(
  self->d_object);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_IJMatrixView__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_IJMatrixView__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_IJMatrixView__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_create1dInit(
  int32_t len, 
  bHYPRE_IJMatrixView* data)
{
  return (struct 
    bHYPRE_IJMatrixView__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_IJMatrixView__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_IJMatrixView__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_borrow(
  bHYPRE_IJMatrixView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_IJMatrixView__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_smartCopy(
  struct bHYPRE_IJMatrixView__array *array)
{
  return (struct bHYPRE_IJMatrixView__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_IJMatrixView__array_addRef(
  struct bHYPRE_IJMatrixView__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_IJMatrixView__array_deleteRef(
  struct bHYPRE_IJMatrixView__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get1(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1)
{
  return (bHYPRE_IJMatrixView)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get2(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_IJMatrixView)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get3(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_IJMatrixView)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get4(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_IJMatrixView)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get5(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_IJMatrixView)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get6(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_IJMatrixView)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get7(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_IJMatrixView)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_IJMatrixView
bHYPRE_IJMatrixView__array_get(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t indices[])
{
  return (bHYPRE_IJMatrixView)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_IJMatrixView__array_set1(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  bHYPRE_IJMatrixView const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJMatrixView__array_set2(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IJMatrixView const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJMatrixView__array_set3(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IJMatrixView const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJMatrixView__array_set4(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IJMatrixView const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJMatrixView__array_set5(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IJMatrixView const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJMatrixView__array_set6(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IJMatrixView const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJMatrixView__array_set7(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IJMatrixView const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_IJMatrixView__array_set(
  struct bHYPRE_IJMatrixView__array* array,
  const int32_t indices[],
  bHYPRE_IJMatrixView const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_IJMatrixView__array_dimen(
  const struct bHYPRE_IJMatrixView__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_IJMatrixView__array_lower(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJMatrixView__array_upper(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJMatrixView__array_length(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_IJMatrixView__array_stride(
  const struct bHYPRE_IJMatrixView__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_IJMatrixView__array_isColumnOrder(
  const struct bHYPRE_IJMatrixView__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_IJMatrixView__array_isRowOrder(
  const struct bHYPRE_IJMatrixView__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_IJMatrixView__array_copy(
  const struct bHYPRE_IJMatrixView__array* src,
  struct bHYPRE_IJMatrixView__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_slice(
  struct bHYPRE_IJMatrixView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_IJMatrixView__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_IJMatrixView__array*
bHYPRE_IJMatrixView__array_ensure(
  struct bHYPRE_IJMatrixView__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_IJMatrixView__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

#include <stdlib.h>
#include <string.h>
#include "sidl_rmi_ProtocolFactory.h"
#include "sidl_rmi_InstanceHandle.h"
#include "sidl_rmi_Invocation.h"
#include "sidl_rmi_Response.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t bHYPRE__IJMatrixView__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE__IJMatrixView__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE__IJMatrixView__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE__IJMatrixView__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct bHYPRE__IJMatrixView__epv s_rem_epv__bhypre__ijmatrixview;

static struct bHYPRE_IJMatrixView__epv s_rem_epv__bhypre_ijmatrixview;

static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

static struct bHYPRE_ProblemDefinition__epv s_rem_epv__bhypre_problemdefinition;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE__IJMatrixView__cast(
struct bHYPRE__IJMatrixView__object* self,
const char* name)
{
  void* cast = NULL;

  struct bHYPRE__IJMatrixView__object* s0;
   s0 =                               self;

  if (!strcmp(name, "bHYPRE._IJMatrixView")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.IJMatrixView")) {
    cast = (void*) &s0->d_bhypre_ijmatrixview;
  } else if (!strcmp(name, "bHYPRE.MatrixVectorView")) {
    cast = (void*) &s0->d_bhypre_matrixvectorview;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s0->d_sidl_baseinterface;
  }
  else if ((*self->d_epv->f_isType)(self,name)) {
    void* (*func)(sidl_rmi_InstanceHandle) = 
      (void* (*)(sidl_rmi_InstanceHandle)) 
      sidl_rmi_ConnectRegistry_getConnect(name);
    cast =  (*func)((sidl_rmi_InstanceHandle)self->d_data);
  }

  return cast;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE__IJMatrixView__delete(
  struct bHYPRE__IJMatrixView__object* self)
{
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE__IJMatrixView__getURL(
  struct bHYPRE__IJMatrixView__object* self)
{
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getURL(conn, &_ex);
  }
  return NULL;
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_bHYPRE__IJMatrixView__exec(
  struct bHYPRE__IJMatrixView__object* self,
  const char* methodName,
  sidl_io_Deserializer inArgs,
  sidl_io_Serializer outArgs)
{
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE__IJMatrixView_addRef(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE__IJMatrixView_deleteRef(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "deleteRef", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_bHYPRE__IJMatrixView_isSame(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in */ struct sidl_BaseInterface__object* iobj)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:queryInt */
static struct sidl_BaseInterface__object*
remote_bHYPRE__IJMatrixView_queryInt(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in */ const char* name)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_bHYPRE__IJMatrixView_isType(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in */ const char* name)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "isType", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  sidl_bool _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_bHYPRE__IJMatrixView_getClassInfo(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE__IJMatrixView_SetCommunicator(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetCommunicator", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "mpi_comm",
    bHYPRE_MPICommunicator__getURL(mpi_comm), _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:Initialize */
static int32_t
remote_bHYPRE__IJMatrixView_Initialize(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Initialize", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:Assemble */
static int32_t
remote_bHYPRE__IJMatrixView_Assemble(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Assemble", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetLocalRange */
static int32_t
remote_bHYPRE__IJMatrixView_SetLocalRange(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetLocalRange", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "ilower", ilower, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "iupper", iupper, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "jlower", jlower, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "jupper", jupper, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetValues */
static int32_t
remote_bHYPRE__IJMatrixView_SetValues(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* in rarray[nnonzeros] */ struct sidl_double__array* values)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetValues", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:AddToValues */
static int32_t
remote_bHYPRE__IJMatrixView_AddToValues(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* in rarray[nnonzeros] */ struct sidl_double__array* values)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "AddToValues", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:GetLocalRange */
static int32_t
remote_bHYPRE__IJMatrixView_GetLocalRange(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetLocalRange", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "ilower", ilower, _ex2);
  sidl_rmi_Response_unpackInt( _rsvp, "iupper", iupper, _ex2);
  sidl_rmi_Response_unpackInt( _rsvp, "jlower", jlower, _ex2);
  sidl_rmi_Response_unpackInt( _rsvp, "jupper", jupper, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:GetRowCounts */
static int32_t
remote_bHYPRE__IJMatrixView_GetRowCounts(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* inout rarray[nrows] */ struct sidl_int__array** ncols)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetRowCounts", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:GetValues */
static int32_t
remote_bHYPRE__IJMatrixView_GetValues(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* inout rarray[nnonzeros] */ struct sidl_double__array** values)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetValues", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetRowSizes */
static int32_t
remote_bHYPRE__IJMatrixView_SetRowSizes(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in rarray[nrows] */ struct sidl_int__array* sizes)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetRowSizes", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:Print */
static int32_t
remote_bHYPRE__IJMatrixView_Print(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in */ const char* filename)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Print", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "filename", filename, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:Read */
static int32_t
remote_bHYPRE__IJMatrixView_Read(
  /* in */ struct bHYPRE__IJMatrixView__object* self /* TLD */,
  /* in */ const char* filename,
  /* in */ struct bHYPRE_MPICommunicator__object* comm)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Read", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "filename", filename, _ex2);
  sidl_rmi_Invocation_packString( _inv, "comm",
    bHYPRE_MPICommunicator__getURL(comm), _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE__IJMatrixView__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE__IJMatrixView__epv*     epv = &s_rem_epv__bhypre__ijmatrixview;
  struct bHYPRE_IJMatrixView__epv*      e0  = &s_rem_epv__bhypre_ijmatrixview;
  struct bHYPRE_MatrixVectorView__epv*  e1  = 
    &s_rem_epv__bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__epv* e2  = 
    &s_rem_epv__bhypre_problemdefinition;
  struct sidl_BaseInterface__epv*       e3  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast                = remote_bHYPRE__IJMatrixView__cast;
  epv->f__delete              = remote_bHYPRE__IJMatrixView__delete;
  epv->f__exec                = remote_bHYPRE__IJMatrixView__exec;
  epv->f__getURL              = remote_bHYPRE__IJMatrixView__getURL;
  epv->f__ctor                = NULL;
  epv->f__dtor                = NULL;
  epv->f_addRef               = remote_bHYPRE__IJMatrixView_addRef;
  epv->f_deleteRef            = remote_bHYPRE__IJMatrixView_deleteRef;
  epv->f_isSame               = remote_bHYPRE__IJMatrixView_isSame;
  epv->f_queryInt             = remote_bHYPRE__IJMatrixView_queryInt;
  epv->f_isType               = remote_bHYPRE__IJMatrixView_isType;
  epv->f_getClassInfo         = remote_bHYPRE__IJMatrixView_getClassInfo;
  epv->f_SetCommunicator      = remote_bHYPRE__IJMatrixView_SetCommunicator;
  epv->f_Initialize           = remote_bHYPRE__IJMatrixView_Initialize;
  epv->f_Assemble             = remote_bHYPRE__IJMatrixView_Assemble;
  epv->f_SetLocalRange        = remote_bHYPRE__IJMatrixView_SetLocalRange;
  epv->f_SetValues            = remote_bHYPRE__IJMatrixView_SetValues;
  epv->f_AddToValues          = remote_bHYPRE__IJMatrixView_AddToValues;
  epv->f_GetLocalRange        = remote_bHYPRE__IJMatrixView_GetLocalRange;
  epv->f_GetRowCounts         = remote_bHYPRE__IJMatrixView_GetRowCounts;
  epv->f_GetValues            = remote_bHYPRE__IJMatrixView_GetValues;
  epv->f_SetRowSizes          = remote_bHYPRE__IJMatrixView_SetRowSizes;
  epv->f_Print                = remote_bHYPRE__IJMatrixView_Print;
  epv->f_Read                 = remote_bHYPRE__IJMatrixView_Read;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f__exec           = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_SetLocalRange   = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_SetLocalRange;
  e0->f_SetValues       = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_SetValues;
  e0->f_AddToValues     = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_AddToValues;
  e0->f_GetLocalRange   = (int32_t (*)(void*,int32_t*,int32_t*,int32_t*,
    int32_t*)) epv->f_GetLocalRange;
  e0->f_GetRowCounts    = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array**)) epv->f_GetRowCounts;
  e0->f_GetValues       = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,
    struct sidl_double__array**)) epv->f_GetValues;
  e0->f_SetRowSizes     = (int32_t (*)(void*,
    struct sidl_int__array*)) epv->f_SetRowSizes;
  e0->f_Print           = (int32_t (*)(void*,const char*)) epv->f_Print;
  e0->f_Read            = (int32_t (*)(void*,const char*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_Read;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f__exec           = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e1->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;

  e2->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*)) epv->f__delete;
  e2->f__exec           = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e2->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;

  e3->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete      = (void (*)(void*)) epv->f__delete;
  e3->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__remoteConnect(const char *url, sidl_BaseInterface *_ex)
{
  struct bHYPRE__IJMatrixView__object* self;

  struct bHYPRE__IJMatrixView__object* s0;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_connectInstance(url, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE__IJMatrixView__object*) malloc(
      sizeof(struct bHYPRE__IJMatrixView__object));

   s0 =                               self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE__IJMatrixView__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_bhypre_ijmatrixview.d_epv    = &s_rem_epv__bhypre_ijmatrixview;
  s0->d_bhypre_ijmatrixview.d_object = (void*) self;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre__ijmatrixview;

  self->d_data = (void*) instance;

  return bHYPRE_IJMatrixView__cast(self);
}
/* Create an instance that uses an already existing InstanceHandel to connect 
  to an existing remote object. */
static struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE__IJMatrixView__object* self;

  struct bHYPRE__IJMatrixView__object* s0;

  self =
    (struct bHYPRE__IJMatrixView__object*) malloc(
      sizeof(struct bHYPRE__IJMatrixView__object));

   s0 =                               self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE__IJMatrixView__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_bhypre_ijmatrixview.d_epv    = &s_rem_epv__bhypre_ijmatrixview;
  s0->d_bhypre_ijmatrixview.d_object = (void*) self;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre__ijmatrixview;

  self->d_data = (void*) instance;

  sidl_rmi_InstanceHandle_addRef(instance);
  return bHYPRE_IJMatrixView__cast(self);
}
