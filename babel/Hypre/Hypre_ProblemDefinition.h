/*
 * File:          Hypre_ProblemDefinition.h
 * Symbol:        Hypre.ProblemDefinition-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:00 PST
 * Generated:     20030121 14:39:07 PST
 * Description:   Client-side glue code for Hypre.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 87
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_ProblemDefinition_h
#define included_Hypre_ProblemDefinition_h

/**
 * Symbol "Hypre.ProblemDefinition" (version 0.1.6)
 * 
 * <p>The purpose of a ProblemDefinition is to:</p>
 * <ul>
 * <li>present the user with a particular view of how to define
 *     a problem</li>
 * <li>construct and return a "problem object"</li>
 * </ul>
 * 
 * <p>A "problem object" is an intentionally vague term that corresponds
 * to any useful object used to define a problem.  Prime examples are:</p>
 * <ul>
 * <li>a LinearOperator object, i.e., something with a matvec</li>
 * <li>a MatrixAccess object, i.e., something with a getrow</li>
 * <li>a Vector, i.e., something with a dot, axpy, ...</li>
 * </ul>
 * 
 * <p>Note that the terms "Initialize" and "Assemble" are reserved here
 * for defining problem objects through a particular user interface.</p>
 */
struct Hypre_ProblemDefinition__object;
struct Hypre_ProblemDefinition__array;
typedef struct Hypre_ProblemDefinition__object* Hypre_ProblemDefinition;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
Hypre_ProblemDefinition_addRef(
  Hypre_ProblemDefinition self);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_ProblemDefinition_deleteRef(
  Hypre_ProblemDefinition self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_ProblemDefinition_isSame(
  Hypre_ProblemDefinition self,
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
Hypre_ProblemDefinition_queryInt(
  Hypre_ProblemDefinition self,
  const char* name);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_ProblemDefinition_isType(
  Hypre_ProblemDefinition self,
  const char* name);

/**
 * Method:  SetCommunicator[]
 */
int32_t
Hypre_ProblemDefinition_SetCommunicator(
  Hypre_ProblemDefinition self,
  void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */
int32_t
Hypre_ProblemDefinition_Initialize(
  Hypre_ProblemDefinition self);

/**
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */
int32_t
Hypre_ProblemDefinition_Assemble(
  Hypre_ProblemDefinition self);

/**
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
Hypre_ProblemDefinition_GetObject(
  Hypre_ProblemDefinition self,
  SIDL_BaseInterface* A);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_ProblemDefinition
Hypre_ProblemDefinition__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_ProblemDefinition__cast2(
  void* obj,
  const char* type);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_createCol(int32_t        dimen,
                                         const int32_t lower[],
                                         const int32_t upper[]);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_createRow(int32_t        dimen,
                                         const int32_t lower[],
                                         const int32_t upper[]);

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_create1d(int32_t len);

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_borrow(Hypre_ProblemDefinition*firstElement,
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
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_smartCopy(struct Hypre_ProblemDefinition__array 
  *array);

/**
 * Increment the array's internal reference count by one.
 */
void
Hypre_ProblemDefinition__array_addRef(struct Hypre_ProblemDefinition__array* 
  array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
Hypre_ProblemDefinition__array_deleteRef(struct Hypre_ProblemDefinition__array* 
  array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
Hypre_ProblemDefinition
Hypre_ProblemDefinition__array_get1(const struct 
  Hypre_ProblemDefinition__array* array,
                                    const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
Hypre_ProblemDefinition
Hypre_ProblemDefinition__array_get2(const struct 
  Hypre_ProblemDefinition__array* array,
                                    const int32_t i1,
                                    const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
Hypre_ProblemDefinition
Hypre_ProblemDefinition__array_get3(const struct 
  Hypre_ProblemDefinition__array* array,
                                    const int32_t i1,
                                    const int32_t i2,
                                    const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
Hypre_ProblemDefinition
Hypre_ProblemDefinition__array_get4(const struct 
  Hypre_ProblemDefinition__array* array,
                                    const int32_t i1,
                                    const int32_t i2,
                                    const int32_t i3,
                                    const int32_t i4);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
Hypre_ProblemDefinition
Hypre_ProblemDefinition__array_get(const struct Hypre_ProblemDefinition__array* 
  array,
                                   const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
Hypre_ProblemDefinition__array_set1(struct Hypre_ProblemDefinition__array* 
  array,
                                    const int32_t i1,
                                    Hypre_ProblemDefinition const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
Hypre_ProblemDefinition__array_set2(struct Hypre_ProblemDefinition__array* 
  array,
                                    const int32_t i1,
                                    const int32_t i2,
                                    Hypre_ProblemDefinition const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
Hypre_ProblemDefinition__array_set3(struct Hypre_ProblemDefinition__array* 
  array,
                                    const int32_t i1,
                                    const int32_t i2,
                                    const int32_t i3,
                                    Hypre_ProblemDefinition const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
Hypre_ProblemDefinition__array_set4(struct Hypre_ProblemDefinition__array* 
  array,
                                    const int32_t i1,
                                    const int32_t i2,
                                    const int32_t i3,
                                    const int32_t i4,
                                    Hypre_ProblemDefinition const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
Hypre_ProblemDefinition__array_set(struct Hypre_ProblemDefinition__array* array,
                                   const int32_t indices[],
                                   Hypre_ProblemDefinition const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
Hypre_ProblemDefinition__array_dimen(const struct 
  Hypre_ProblemDefinition__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_ProblemDefinition__array_lower(const struct 
  Hypre_ProblemDefinition__array* array,
                                     const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_ProblemDefinition__array_upper(const struct 
  Hypre_ProblemDefinition__array* array,
                                     const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_ProblemDefinition__array_stride(const struct 
  Hypre_ProblemDefinition__array* array,
                                      const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_ProblemDefinition__array_isColumnOrder(const struct 
  Hypre_ProblemDefinition__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_ProblemDefinition__array_isRowOrder(const struct 
  Hypre_ProblemDefinition__array* array);

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
Hypre_ProblemDefinition__array_copy(const struct 
  Hypre_ProblemDefinition__array* src,
                                          struct 
  Hypre_ProblemDefinition__array* dest);

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
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_ensure(struct Hypre_ProblemDefinition__array* 
  src,
                                      int32_t dimen,
int     ordering);

#ifdef __cplusplus
}
#endif
#endif
