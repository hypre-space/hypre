/*
 * File:          Hypre_Pilut.h
 * Symbol:        Hypre.Pilut-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:14 PST
 * Description:   Client-side glue code for Hypre.Pilut
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1242
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_Pilut_h
#define included_Hypre_Pilut_h

/**
 * Symbol "Hypre.Pilut" (version 0.1.7)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 */
struct Hypre_Pilut__object;
struct Hypre_Pilut__array;
typedef struct Hypre_Pilut__object* Hypre_Pilut;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
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
Hypre_Pilut
Hypre_Pilut__create(void);

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
Hypre_Pilut_addRef(
  Hypre_Pilut self);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_Pilut_deleteRef(
  Hypre_Pilut self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_Pilut_isSame(
  Hypre_Pilut self,
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
Hypre_Pilut_queryInt(
  Hypre_Pilut self,
  const char* name);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_Pilut_isType(
  Hypre_Pilut self,
  const char* name);

/**
 * Return the meta-data about the class implementing this interface.
 */
SIDL_ClassInfo
Hypre_Pilut_getClassInfo(
  Hypre_Pilut self);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
Hypre_Pilut_SetCommunicator(
  Hypre_Pilut self,
  void* mpi_comm);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_Pilut_SetIntParameter(
  Hypre_Pilut self,
  const char* name,
  int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_Pilut_SetDoubleParameter(
  Hypre_Pilut self,
  const char* name,
  double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_Pilut_SetStringParameter(
  Hypre_Pilut self,
  const char* name,
  const char* value);

/**
 * Set the int array parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_Pilut_SetIntArrayParameter(
  Hypre_Pilut self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Set the double array parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_Pilut_SetDoubleArrayParameter(
  Hypre_Pilut self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_Pilut_GetIntValue(
  Hypre_Pilut self,
  const char* name,
  int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
Hypre_Pilut_GetDoubleValue(
  Hypre_Pilut self,
  const char* name,
  double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
Hypre_Pilut_Setup(
  Hypre_Pilut self,
  Hypre_Vector b,
  Hypre_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
Hypre_Pilut_Apply(
  Hypre_Pilut self,
  Hypre_Vector b,
  Hypre_Vector* x);

/**
 * Set the operator for the linear system being solved.
 * 
 */
int32_t
Hypre_Pilut_SetOperator(
  Hypre_Pilut self,
  Hypre_Operator A);

/**
 * (Optional) Set the convergence tolerance.
 * 
 * RDF: New
 * 
 */
int32_t
Hypre_Pilut_SetTolerance(
  Hypre_Pilut self,
  double tolerance);

/**
 * (Optional) Set maximum number of iterations.
 * 
 * RDF: New
 * 
 */
int32_t
Hypre_Pilut_SetMaxIterations(
  Hypre_Pilut self,
  int32_t max_iterations);

/**
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */
int32_t
Hypre_Pilut_SetLogging(
  Hypre_Pilut self,
  int32_t level);

/**
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */
int32_t
Hypre_Pilut_SetPrintLevel(
  Hypre_Pilut self,
  int32_t level);

/**
 * (Optional) Return the number of iterations taken.
 * 
 * RDF: New
 * 
 */
int32_t
Hypre_Pilut_GetNumIterations(
  Hypre_Pilut self,
  int32_t* num_iterations);

/**
 * (Optional) Return the norm of the relative residual.
 * 
 * RDF: New
 * 
 */
int32_t
Hypre_Pilut_GetRelResidualNorm(
  Hypre_Pilut self,
  double* norm);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_Pilut
Hypre_Pilut__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_Pilut__cast2(
  void* obj,
  const char* type);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_Pilut__array*
Hypre_Pilut__array_createCol(int32_t        dimen,
                             const int32_t lower[],
                             const int32_t upper[]);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_Pilut__array*
Hypre_Pilut__array_createRow(int32_t        dimen,
                             const int32_t lower[],
                             const int32_t upper[]);

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_Pilut__array*
Hypre_Pilut__array_create1d(int32_t len);

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_Pilut__array*
Hypre_Pilut__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_Pilut__array*
Hypre_Pilut__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct Hypre_Pilut__array*
Hypre_Pilut__array_borrow(Hypre_Pilut*firstElement,
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
struct Hypre_Pilut__array*
Hypre_Pilut__array_smartCopy(struct Hypre_Pilut__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
Hypre_Pilut__array_addRef(struct Hypre_Pilut__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
Hypre_Pilut__array_deleteRef(struct Hypre_Pilut__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
Hypre_Pilut
Hypre_Pilut__array_get1(const struct Hypre_Pilut__array* array,
                        const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
Hypre_Pilut
Hypre_Pilut__array_get2(const struct Hypre_Pilut__array* array,
                        const int32_t i1,
                        const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
Hypre_Pilut
Hypre_Pilut__array_get3(const struct Hypre_Pilut__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
Hypre_Pilut
Hypre_Pilut__array_get4(const struct Hypre_Pilut__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        const int32_t i4);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
Hypre_Pilut
Hypre_Pilut__array_get(const struct Hypre_Pilut__array* array,
                       const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
Hypre_Pilut__array_set1(struct Hypre_Pilut__array* array,
                        const int32_t i1,
                        Hypre_Pilut const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
Hypre_Pilut__array_set2(struct Hypre_Pilut__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        Hypre_Pilut const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
Hypre_Pilut__array_set3(struct Hypre_Pilut__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        Hypre_Pilut const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
Hypre_Pilut__array_set4(struct Hypre_Pilut__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        const int32_t i4,
                        Hypre_Pilut const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
Hypre_Pilut__array_set(struct Hypre_Pilut__array* array,
                       const int32_t indices[],
                       Hypre_Pilut const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
Hypre_Pilut__array_dimen(const struct Hypre_Pilut__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_Pilut__array_lower(const struct Hypre_Pilut__array* array,
                         const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_Pilut__array_upper(const struct Hypre_Pilut__array* array,
                         const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_Pilut__array_stride(const struct Hypre_Pilut__array* array,
                          const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_Pilut__array_isColumnOrder(const struct Hypre_Pilut__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_Pilut__array_isRowOrder(const struct Hypre_Pilut__array* array);

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
Hypre_Pilut__array_copy(const struct Hypre_Pilut__array* src,
                              struct Hypre_Pilut__array* dest);

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
struct Hypre_Pilut__array*
Hypre_Pilut__array_ensure(struct Hypre_Pilut__array* src,
                          int32_t dimen,
int     ordering);

#ifdef __cplusplus
}
#endif
#endif
