/*
 * File:          Hypre_IJParCSRMatrix.h
 * Symbol:        Hypre.IJParCSRMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:17 PST
 * Generated:     20030306 17:05:20 PST
 * Description:   Client-side glue code for Hypre.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 799
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_IJParCSRMatrix_h
#define included_Hypre_IJParCSRMatrix_h

/**
 * Symbol "Hypre.IJParCSRMatrix" (version 0.1.7)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */
struct Hypre_IJParCSRMatrix__object;
struct Hypre_IJParCSRMatrix__array;
typedef struct Hypre_IJParCSRMatrix__object* Hypre_IJParCSRMatrix;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
Hypre_IJParCSRMatrix
Hypre_IJParCSRMatrix__create(void);

/**
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
Hypre_IJParCSRMatrix_addRef(
  Hypre_IJParCSRMatrix self);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_IJParCSRMatrix_deleteRef(
  Hypre_IJParCSRMatrix self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_IJParCSRMatrix_isSame(
  Hypre_IJParCSRMatrix self,
  SIDL_BaseInterface iobj);

/**
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
SIDL_BaseInterface
Hypre_IJParCSRMatrix_queryInt(
  Hypre_IJParCSRMatrix self,
  const char* name);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_IJParCSRMatrix_isType(
  Hypre_IJParCSRMatrix self,
  const char* name);

/**
 * Return the meta-data about the class implementing this interface.
 */
SIDL_ClassInfo
Hypre_IJParCSRMatrix_getClassInfo(
  Hypre_IJParCSRMatrix self);

/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_SetDiagOffdSizes(
  Hypre_IJParCSRMatrix self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_SetCommunicator(
  Hypre_IJParCSRMatrix self,
  void* mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_SetIntParameter(
  Hypre_IJParCSRMatrix self,
  const char* name,
  int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_SetDoubleParameter(
  Hypre_IJParCSRMatrix self,
  const char* name,
  double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_SetStringParameter(
  Hypre_IJParCSRMatrix self,
  const char* name,
  const char* value);

/**
 * Set the int array parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_SetIntArrayParameter(
  Hypre_IJParCSRMatrix self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the double array parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_SetDoubleArrayParameter(
  Hypre_IJParCSRMatrix self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_GetIntValue(
  Hypre_IJParCSRMatrix self,
  const char* name,
  int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_GetDoubleValue(
  Hypre_IJParCSRMatrix self,
  const char* name,
  double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_Setup(
  Hypre_IJParCSRMatrix self,
  Hypre_Vector b,
  Hypre_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_Apply(
  Hypre_IJParCSRMatrix self,
  Hypre_Vector b,
  Hypre_Vector* x);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_Initialize(
  Hypre_IJParCSRMatrix self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_Assemble(
  Hypre_IJParCSRMatrix self);

/**
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_GetObject(
  Hypre_IJParCSRMatrix self,
  SIDL_BaseInterface* A);

/**
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
 * RDF: Changed name from 'Create' (x)
 * 
 */
int32_t
Hypre_IJParCSRMatrix_SetLocalRange(
  Hypre_IJParCSRMatrix self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper);

/**
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
Hypre_IJParCSRMatrix_SetValues(
  Hypre_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values);

/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_AddToValues(
  Hypre_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values);

/**
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 * RDF: New (x)
 * 
 */
int32_t
Hypre_IJParCSRMatrix_GetLocalRange(
  Hypre_IJParCSRMatrix self,
  int32_t* ilower,
  int32_t* iupper,
  int32_t* jlower,
  int32_t* jupper);

/**
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 * RDF: New (x)
 * 
 */
int32_t
Hypre_IJParCSRMatrix_GetRowCounts(
  Hypre_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* rows,
  struct SIDL_int__array** ncols);

/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 * RDF: New (x)
 * 
 */
int32_t
Hypre_IJParCSRMatrix_GetValues(
  Hypre_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array** values);

/**
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
Hypre_IJParCSRMatrix_SetRowSizes(
  Hypre_IJParCSRMatrix self,
  struct SIDL_int__array* sizes);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_Print(
  Hypre_IJParCSRMatrix self,
  const char* filename);

/**
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_Read(
  Hypre_IJParCSRMatrix self,
  const char* filename,
  void* comm);

/**
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */
int32_t
Hypre_IJParCSRMatrix_GetRow(
  Hypre_IJParCSRMatrix self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_IJParCSRMatrix
Hypre_IJParCSRMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_IJParCSRMatrix__cast2(
  void* obj,
  const char* type);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_IJParCSRMatrix__array*
Hypre_IJParCSRMatrix__array_createCol(int32_t        dimen,
                                      const int32_t lower[],
                                      const int32_t upper[]);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_IJParCSRMatrix__array*
Hypre_IJParCSRMatrix__array_createRow(int32_t        dimen,
                                      const int32_t lower[],
                                      const int32_t upper[]);

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_IJParCSRMatrix__array*
Hypre_IJParCSRMatrix__array_create1d(int32_t len);

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_IJParCSRMatrix__array*
Hypre_IJParCSRMatrix__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_IJParCSRMatrix__array*
Hypre_IJParCSRMatrix__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct Hypre_IJParCSRMatrix__array*
Hypre_IJParCSRMatrix__array_borrow(Hypre_IJParCSRMatrix*firstElement,
                                   int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct Hypre_IJParCSRMatrix__array*
Hypre_IJParCSRMatrix__array_smartCopy(struct Hypre_IJParCSRMatrix__array 
  *array);

/**
 * Increment the array's internal reference count by one.
 */
void
Hypre_IJParCSRMatrix__array_addRef(struct Hypre_IJParCSRMatrix__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
Hypre_IJParCSRMatrix__array_deleteRef(struct Hypre_IJParCSRMatrix__array* 
  array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
Hypre_IJParCSRMatrix
Hypre_IJParCSRMatrix__array_get1(const struct Hypre_IJParCSRMatrix__array* 
  array,
                                 const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
Hypre_IJParCSRMatrix
Hypre_IJParCSRMatrix__array_get2(const struct Hypre_IJParCSRMatrix__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
Hypre_IJParCSRMatrix
Hypre_IJParCSRMatrix__array_get3(const struct Hypre_IJParCSRMatrix__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
Hypre_IJParCSRMatrix
Hypre_IJParCSRMatrix__array_get4(const struct Hypre_IJParCSRMatrix__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 const int32_t i4);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
Hypre_IJParCSRMatrix
Hypre_IJParCSRMatrix__array_get(const struct Hypre_IJParCSRMatrix__array* array,
                                const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
Hypre_IJParCSRMatrix__array_set1(struct Hypre_IJParCSRMatrix__array* array,
                                 const int32_t i1,
                                 Hypre_IJParCSRMatrix const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
Hypre_IJParCSRMatrix__array_set2(struct Hypre_IJParCSRMatrix__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 Hypre_IJParCSRMatrix const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
Hypre_IJParCSRMatrix__array_set3(struct Hypre_IJParCSRMatrix__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 Hypre_IJParCSRMatrix const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
Hypre_IJParCSRMatrix__array_set4(struct Hypre_IJParCSRMatrix__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 const int32_t i4,
                                 Hypre_IJParCSRMatrix const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
Hypre_IJParCSRMatrix__array_set(struct Hypre_IJParCSRMatrix__array* array,
                                const int32_t indices[],
                                Hypre_IJParCSRMatrix const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
Hypre_IJParCSRMatrix__array_dimen(const struct Hypre_IJParCSRMatrix__array* 
  array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_IJParCSRMatrix__array_lower(const struct Hypre_IJParCSRMatrix__array* 
  array,
                                  const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_IJParCSRMatrix__array_upper(const struct Hypre_IJParCSRMatrix__array* 
  array,
                                  const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_IJParCSRMatrix__array_stride(const struct Hypre_IJParCSRMatrix__array* 
  array,
                                   const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_IJParCSRMatrix__array_isColumnOrder(const struct 
  Hypre_IJParCSRMatrix__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_IJParCSRMatrix__array_isRowOrder(const struct 
  Hypre_IJParCSRMatrix__array* array);

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
Hypre_IJParCSRMatrix__array_copy(const struct Hypre_IJParCSRMatrix__array* src,
                                       struct Hypre_IJParCSRMatrix__array* 
  dest);

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
struct Hypre_IJParCSRMatrix__array*
Hypre_IJParCSRMatrix__array_ensure(struct Hypre_IJParCSRMatrix__array* src,
                                   int32_t dimen,
int     ordering);

#ifdef __cplusplus
}
#endif
#endif
