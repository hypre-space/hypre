/*
 * File:          Hypre_ParCSRMatrix_Stub.c
 * Symbol:        Hypre.ParCSRMatrix-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030210 16:05:40 PST
 * Generated:     20030210 16:05:45 PST
 * Description:   Client-side glue code for Hypre.ParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 433
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_ParCSRMatrix.h"
#include "Hypre_ParCSRMatrix_IOR.h"
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

static const struct Hypre_ParCSRMatrix__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct Hypre_ParCSRMatrix__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _ior = Hypre_ParCSRMatrix__externals();
#else
  const struct Hypre_ParCSRMatrix__external*(*dll_f)(void) =
    (const struct Hypre_ParCSRMatrix__external*(*)(void)) 
      SIDL_Loader_lookupSymbol(
      "Hypre_ParCSRMatrix__externals");
  _ior = (dll_f ? (*dll_f)() : NULL);
  if (!_ior) {
    fputs("Babel: unable to load the implementation for Hypre.ParCSRMatrix; please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
#endif
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Constructor function for the class.
 */

Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__create()
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
Hypre_ParCSRMatrix_addRef(
  Hypre_ParCSRMatrix self)
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
Hypre_ParCSRMatrix_deleteRef(
  Hypre_ParCSRMatrix self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
Hypre_ParCSRMatrix_isSame(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_queryInt(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_isType(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_getClassInfo(
  Hypre_ParCSRMatrix self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Method:  Get_IJBuildMatrix[]
 */

Hypre_IJBuildMatrix
Hypre_ParCSRMatrix_Get_IJBuildMatrix(
  Hypre_ParCSRMatrix self)
{
  return (*self->d_epv->f_Get_IJBuildMatrix)(
    self);
}

/*
 * Method:  SetCommunicator[]
 */

int32_t
Hypre_ParCSRMatrix_SetCommunicator(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_GetDoubleValue(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_GetIntValue(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_SetDoubleParameter(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_SetIntParameter(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_SetStringParameter(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_SetIntArrayParameter(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_SetDoubleArrayParameter(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_Setup(
  Hypre_ParCSRMatrix self,
  Hypre_Vector b,
  Hypre_Vector x)
{
  return (*self->d_epv->f_Setup)(
    self,
    b,
    x);
}

/*
 * Method:  Apply[]
 */

int32_t
Hypre_ParCSRMatrix_Apply(
  Hypre_ParCSRMatrix self,
  Hypre_Vector b,
  Hypre_Vector* x)
{
  return (*self->d_epv->f_Apply)(
    self,
    b,
    x);
}

/*
 * Method:  GetRow[]
 */

int32_t
Hypre_ParCSRMatrix_GetRow(
  Hypre_ParCSRMatrix self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  return (*self->d_epv->f_GetRow)(
    self,
    row,
    size,
    col_ind,
    values);
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

int32_t
Hypre_ParCSRMatrix_Initialize(
  Hypre_ParCSRMatrix self)
{
  return (*self->d_epv->f_Initialize)(
    self);
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
Hypre_ParCSRMatrix_Assemble(
  Hypre_ParCSRMatrix self)
{
  return (*self->d_epv->f_Assemble)(
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
Hypre_ParCSRMatrix_GetObject(
  Hypre_ParCSRMatrix self,
  SIDL_BaseInterface* A)
{
  return (*self->d_epv->f_GetObject)(
    self,
    A);
}

/*
 * Create a matrix object.  Each process owns some unique consecutive
 * range of rows, indicated by the global row indices {\tt ilower} and
 * {\tt iupper}.  The row data is required to be such that the value
 * of {\tt ilower} on any process $p$ be exactly one more than the
 * value of {\tt iupper} on process $p-1$.  Note that the first row of
 * the global matrix may start with any integer value.  In particular,
 * one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically should
 * match {\tt ilower} and {\tt iupper}, respectively.  For rectangular
 * matrices, {\tt jlower} and {\tt jupper} should define a
 * partitioning of the columns.  This partitioning must be used for
 * any vector $v$ that will be used in matrix-vector products with the
 * rectangular matrix.  The matrix data structure may use {\tt jlower}
 * and {\tt jupper} to store the diagonal blocks (rectangular in
 * general) of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 * 
 */

int32_t
Hypre_ParCSRMatrix_Create(
  Hypre_ParCSRMatrix self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper)
{
  return (*self->d_epv->f_Create)(
    self,
    ilower,
    iupper,
    jlower,
    jupper);
}

/*
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt ncols}
 * and {\tt rows} are of dimension {\tt nrows} and contain the number
 * of columns in each row and the row indices, respectively.  The
 * array {\tt cols} contains the column indices for each of the {\tt
 * rows}, and is ordered by rows.  The data in the {\tt values} array
 * corresponds directly to the column entries in {\tt cols}.  Erases
 * any previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before, inserts a
 * new one.
 * 
 * Not collective.
 * 
 * 
 */

int32_t
Hypre_ParCSRMatrix_SetValues(
  Hypre_ParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_SetValues)(
    self,
    nrows,
    ncols,
    rows,
    cols,
    values);
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details are
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value there
 * before, inserts a new one.
 * 
 * Not collective.
 * 
 * 
 */

int32_t
Hypre_ParCSRMatrix_AddToValues(
  Hypre_ParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_AddToValues)(
    self,
    nrows,
    ncols,
    rows,
    cols,
    values);
}

/*
 * (Optional) Set the max number of nonzeros to expect in each row.
 * The array {\tt sizes} contains estimated sizes for each row on this
 * process.  This call can significantly improve the efficiency of
 * matrix construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 * DEVELOPER NOTES: None.
 * 
 */

int32_t
Hypre_ParCSRMatrix_SetRowSizes(
  Hypre_ParCSRMatrix self,
  struct SIDL_int__array* sizes)
{
  return (*self->d_epv->f_SetRowSizes)(
    self,
    sizes);
}

/*
 * (Optional) Set the max number of nonzeros to expect in each row of
 * the diagonal and off-diagonal blocks.  The diagonal block is the
 * submatrix whose column numbers correspond to rows owned by this
 * process, and the off-diagonal block is everything else.  The arrays
 * {\tt diag\_sizes} and {\tt offdiag\_sizes} contain estimated sizes
 * for each row of the diagonal and off-diagonal blocks, respectively.
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 * 
 */

int32_t
Hypre_ParCSRMatrix_SetDiagOffdSizes(
  Hypre_ParCSRMatrix self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes)
{
  return (*self->d_epv->f_SetDiagOffdSizes)(
    self,
    diag_sizes,
    offdiag_sizes);
}

/*
 * Read the matrix from file.  This is mainly for debugging purposes.
 * 
 */

int32_t
Hypre_ParCSRMatrix_Read(
  Hypre_ParCSRMatrix self,
  const char* filename,
  void* comm)
{
  return (*self->d_epv->f_Read)(
    self,
    filename,
    comm);
}

/*
 * Print the matrix to file.  This is mainly for debugging purposes.
 * 
 */

int32_t
Hypre_ParCSRMatrix_Print(
  Hypre_ParCSRMatrix self,
  const char* filename)
{
  return (*self->d_epv->f_Print)(
    self,
    filename);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__cast(
  void* obj)
{
  Hypre_ParCSRMatrix cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_ParCSRMatrix) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.ParCSRMatrix");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_ParCSRMatrix__cast2(
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
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_createCol(int32_t        dimen,
                                    const int32_t lower[],
                                    const int32_t upper[])
{
  return (struct 
    Hypre_ParCSRMatrix__array*)SIDL_interface__array_createCol(dimen, lower,
    upper);
}

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_createRow(int32_t        dimen,
                                    const int32_t lower[],
                                    const int32_t upper[])
{
  return (struct 
    Hypre_ParCSRMatrix__array*)SIDL_interface__array_createRow(dimen, lower,
    upper);
}

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_create1d(int32_t len)
{
  return (struct Hypre_ParCSRMatrix__array*)SIDL_interface__array_create1d(len);
}

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_create2dCol(int32_t m, int32_t n)
{
  return (struct Hypre_ParCSRMatrix__array*)SIDL_interface__array_create2dCol(m,
    n);
}

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_create2dRow(int32_t m, int32_t n)
{
  return (struct Hypre_ParCSRMatrix__array*)SIDL_interface__array_create2dRow(m,
    n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_borrow(Hypre_ParCSRMatrix*firstElement,
                                 int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct Hypre_ParCSRMatrix__array*)SIDL_interface__array_borrow(
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
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_smartCopy(struct Hypre_ParCSRMatrix__array *array)
{
  return (struct Hypre_ParCSRMatrix__array*)
    SIDL_interface__array_smartCopy((struct SIDL_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
Hypre_ParCSRMatrix__array_addRef(struct Hypre_ParCSRMatrix__array* array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array *)array);
}

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
Hypre_ParCSRMatrix__array_deleteRef(struct Hypre_ParCSRMatrix__array* array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__array_get1(const struct Hypre_ParCSRMatrix__array* array,
                               const int32_t i1)
{
  return (Hypre_ParCSRMatrix)
    SIDL_interface__array_get1((const struct SIDL_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__array_get2(const struct Hypre_ParCSRMatrix__array* array,
                               const int32_t i1,
                               const int32_t i2)
{
  return (Hypre_ParCSRMatrix)
    SIDL_interface__array_get2((const struct SIDL_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__array_get3(const struct Hypre_ParCSRMatrix__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3)
{
  return (Hypre_ParCSRMatrix)
    SIDL_interface__array_get3((const struct SIDL_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__array_get4(const struct Hypre_ParCSRMatrix__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               const int32_t i4)
{
  return (Hypre_ParCSRMatrix)
    SIDL_interface__array_get4((const struct SIDL_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__array_get(const struct Hypre_ParCSRMatrix__array* array,
                              const int32_t indices[])
{
  return (Hypre_ParCSRMatrix)
    SIDL_interface__array_get((const struct SIDL_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
Hypre_ParCSRMatrix__array_set1(struct Hypre_ParCSRMatrix__array* array,
                               const int32_t i1,
                               Hypre_ParCSRMatrix const value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)array
  , i1, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
Hypre_ParCSRMatrix__array_set2(struct Hypre_ParCSRMatrix__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               Hypre_ParCSRMatrix const value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)array
  , i1, i2, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
Hypre_ParCSRMatrix__array_set3(struct Hypre_ParCSRMatrix__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               Hypre_ParCSRMatrix const value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)array
  , i1, i2, i3, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
Hypre_ParCSRMatrix__array_set4(struct Hypre_ParCSRMatrix__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               const int32_t i4,
                               Hypre_ParCSRMatrix const value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)array
  , i1, i2, i3, i4, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
Hypre_ParCSRMatrix__array_set(struct Hypre_ParCSRMatrix__array* array,
                              const int32_t indices[],
                              Hypre_ParCSRMatrix const value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)array, indices,
    (struct SIDL_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
Hypre_ParCSRMatrix__array_dimen(const struct Hypre_ParCSRMatrix__array* array)
{
  return SIDL_interface__array_dimen((struct SIDL_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_ParCSRMatrix__array_lower(const struct Hypre_ParCSRMatrix__array* array,
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
Hypre_ParCSRMatrix__array_upper(const struct Hypre_ParCSRMatrix__array* array,
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
Hypre_ParCSRMatrix__array_stride(const struct Hypre_ParCSRMatrix__array* array,
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
Hypre_ParCSRMatrix__array_isColumnOrder(const struct Hypre_ParCSRMatrix__array* 
  array)
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
Hypre_ParCSRMatrix__array_isRowOrder(const struct Hypre_ParCSRMatrix__array* 
  array)
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
Hypre_ParCSRMatrix__array_copy(const struct Hypre_ParCSRMatrix__array* src,
                                     struct Hypre_ParCSRMatrix__array* dest)
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
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_ensure(struct Hypre_ParCSRMatrix__array* src,
                                 int32_t dimen,
int     ordering)
{
  return (struct Hypre_ParCSRMatrix__array*)
    SIDL_interface__array_ensure((struct SIDL_interface__array *)src, dimen,
      ordering);
}

