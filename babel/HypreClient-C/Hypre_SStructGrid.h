/*
 * File:          Hypre_SStructGrid.h
 * Symbol:        Hypre.SStructGrid-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:18 PST
 * Generated:     20030306 17:05:20 PST
 * Description:   Client-side glue code for Hypre.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 914
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_SStructGrid_h
#define included_Hypre_SStructGrid_h

/**
 * Symbol "Hypre.SStructGrid" (version 0.1.7)
 * 
 * The semi-structured grid class.
 * 
 */
struct Hypre_SStructGrid__object;
struct Hypre_SStructGrid__array;
typedef struct Hypre_SStructGrid__object* Hypre_SStructGrid;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_SStructVariable_h
#include "Hypre_SStructVariable.h"
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
Hypre_SStructGrid
Hypre_SStructGrid__create(void);

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
Hypre_SStructGrid_addRef(
  Hypre_SStructGrid self);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_SStructGrid_deleteRef(
  Hypre_SStructGrid self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_SStructGrid_isSame(
  Hypre_SStructGrid self,
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
Hypre_SStructGrid_queryInt(
  Hypre_SStructGrid self,
  const char* name);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_SStructGrid_isType(
  Hypre_SStructGrid self,
  const char* name);

/**
 * Return the meta-data about the class implementing this interface.
 */
SIDL_ClassInfo
Hypre_SStructGrid_getClassInfo(
  Hypre_SStructGrid self);

/**
 * Set the number of dimensions {\tt ndim} and the number of
 * structured parts {\tt nparts}.
 * 
 */
int32_t
Hypre_SStructGrid_SetNumDimParts(
  Hypre_SStructGrid self,
  int32_t ndim,
  int32_t nparts);

/**
 * Set the extents for a box on a structured part of the grid.
 * 
 */
int32_t
Hypre_SStructGrid_SetExtents(
  Hypre_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper);

/**
 * Describe the variables that live on a structured part of the
 * grid.
 * 
 */
int32_t
Hypre_SStructGrid_SetVariable(
  Hypre_SStructGrid self,
  int32_t part,
  int32_t var,
  enum Hypre_SStructVariable__enum vartype);

/**
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 * 
 */
int32_t
Hypre_SStructGrid_AddVariable(
  Hypre_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  enum Hypre_SStructVariable__enum vartype);

/**
 * Describe how regions just outside of a part relate to other
 * parts.  This is done a box at a time.
 * 
 * The indexes {\tt ilower} and {\tt iupper} map directly to the
 * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
 * it is required that indexes increase from {\tt ilower} to
 * {\tt iupper}, indexes may increase and/or decrease from {\tt
 * nbor\_ilower} to {\tt nbor\_iupper}.
 * 
 * The {\tt index\_map} describes the mapping of indexes 0, 1,
 * and 2 on part {\tt part} to the corresponding indexes on part
 * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
 * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
 * and 0 on part {\tt nbor\_part}, respectively.
 * 
 * NOTE: All parts related to each other via this routine must
 * have an identical list of variables and variable types.  For
 * example, if part 0 has only two variables on it, a cell
 * centered variable and a node centered variable, and we
 * declare part 1 to be a neighbor of part 0, then part 1 must
 * also have only two variables on it, and they must be of type
 * cell and node.
 * 
 */
int32_t
Hypre_SStructGrid_SetNeighborBox(
  Hypre_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t nbor_part,
  struct SIDL_int__array* nbor_ilower,
  struct SIDL_int__array* nbor_iupper,
  struct SIDL_int__array* index_map);

/**
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables
 * minus one.  Each process owns some unique consecutive range
 * of variables, defined by {\tt ilower} and {\tt iupper}.
 * 
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 * 
 */
int32_t
Hypre_SStructGrid_AddUnstructuredPart(
  Hypre_SStructGrid self,
  int32_t ilower,
  int32_t iupper);

/**
 * (Optional) Set periodic for a particular part.
 * 
 */
int32_t
Hypre_SStructGrid_SetPeriodic(
  Hypre_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* periodic);

/**
 * Setting ghost in the sgrids.
 * 
 */
int32_t
Hypre_SStructGrid_SetNumGhost(
  Hypre_SStructGrid self,
  struct SIDL_int__array* num_ghost);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_SStructGrid
Hypre_SStructGrid__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_SStructGrid__cast2(
  void* obj,
  const char* type);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_SStructGrid__array*
Hypre_SStructGrid__array_createCol(int32_t        dimen,
                                   const int32_t lower[],
                                   const int32_t upper[]);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_SStructGrid__array*
Hypre_SStructGrid__array_createRow(int32_t        dimen,
                                   const int32_t lower[],
                                   const int32_t upper[]);

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_SStructGrid__array*
Hypre_SStructGrid__array_create1d(int32_t len);

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_SStructGrid__array*
Hypre_SStructGrid__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct Hypre_SStructGrid__array*
Hypre_SStructGrid__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct Hypre_SStructGrid__array*
Hypre_SStructGrid__array_borrow(Hypre_SStructGrid*firstElement,
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
struct Hypre_SStructGrid__array*
Hypre_SStructGrid__array_smartCopy(struct Hypre_SStructGrid__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
Hypre_SStructGrid__array_addRef(struct Hypre_SStructGrid__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
Hypre_SStructGrid__array_deleteRef(struct Hypre_SStructGrid__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
Hypre_SStructGrid
Hypre_SStructGrid__array_get1(const struct Hypre_SStructGrid__array* array,
                              const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
Hypre_SStructGrid
Hypre_SStructGrid__array_get2(const struct Hypre_SStructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
Hypre_SStructGrid
Hypre_SStructGrid__array_get3(const struct Hypre_SStructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
Hypre_SStructGrid
Hypre_SStructGrid__array_get4(const struct Hypre_SStructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              const int32_t i3,
                              const int32_t i4);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
Hypre_SStructGrid
Hypre_SStructGrid__array_get(const struct Hypre_SStructGrid__array* array,
                             const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
Hypre_SStructGrid__array_set1(struct Hypre_SStructGrid__array* array,
                              const int32_t i1,
                              Hypre_SStructGrid const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
Hypre_SStructGrid__array_set2(struct Hypre_SStructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              Hypre_SStructGrid const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
Hypre_SStructGrid__array_set3(struct Hypre_SStructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              const int32_t i3,
                              Hypre_SStructGrid const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
Hypre_SStructGrid__array_set4(struct Hypre_SStructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              const int32_t i3,
                              const int32_t i4,
                              Hypre_SStructGrid const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
Hypre_SStructGrid__array_set(struct Hypre_SStructGrid__array* array,
                             const int32_t indices[],
                             Hypre_SStructGrid const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
Hypre_SStructGrid__array_dimen(const struct Hypre_SStructGrid__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_SStructGrid__array_lower(const struct Hypre_SStructGrid__array* array,
                               const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_SStructGrid__array_upper(const struct Hypre_SStructGrid__array* array,
                               const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
Hypre_SStructGrid__array_stride(const struct Hypre_SStructGrid__array* array,
                                const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_SStructGrid__array_isColumnOrder(const struct Hypre_SStructGrid__array* 
  array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
Hypre_SStructGrid__array_isRowOrder(const struct Hypre_SStructGrid__array* 
  array);

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
Hypre_SStructGrid__array_copy(const struct Hypre_SStructGrid__array* src,
                                    struct Hypre_SStructGrid__array* dest);

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
struct Hypre_SStructGrid__array*
Hypre_SStructGrid__array_ensure(struct Hypre_SStructGrid__array* src,
                                int32_t dimen,
int     ordering);

#ifdef __cplusplus
}
#endif
#endif
