/*
 * File:          bHYPRE_SStructVariable_Stub.c
 * Symbol:        bHYPRE.SStructVariable-v1.0.0
 * Symbol Type:   enumeration
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:34 PST
 * Generated:     20030320 16:52:41 PST
 * Description:   Client-side glue code for bHYPRE.SStructVariable
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 888
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructVariable.h"
#ifndef included_SIDL_int_IOR_h
#include "SIDL_int_IOR.h"
#endif
#include <stddef.h>

/**
 * Create a dense array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 */
struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_createCol(int32_t        dimen,
                                        const int32_t lower[],
                                        const int32_t upper[])
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_createCol(dimen,
    lower, upper);
}

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 */
struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_createRow(int32_t        dimen,
                                        const int32_t lower[],
                                        const int32_t upper[])
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_createRow(dimen,
    lower, upper);
}

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 */
struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create1d(int32_t len)
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_create1d(len);
}

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 */
struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create2dCol(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_create2dCol(m,
    n);
}

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 */
struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_create2dRow(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_create2dRow(m,
    n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * The borrowed data must be a pointer to int32_t.
 */
struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_borrow(int32_t *firstElement,
                                     int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct bHYPRE_SStructVariable__array*)SIDL_int__array_borrow(
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
struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_smartCopy(struct bHYPRE_SStructVariable__array 
  *array)
{
  return (struct bHYPRE_SStructVariable__array*)
    SIDL_int__array_smartCopy((struct SIDL_int__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
bHYPRE_SStructVariable__array_addRef(struct bHYPRE_SStructVariable__array* 
  array)
{
  SIDL_int__array_addRef((struct SIDL_int__array *)array);
}

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 */
void
bHYPRE_SStructVariable__array_deleteRef(struct bHYPRE_SStructVariable__array* 
  array)
{
  SIDL_int__array_deleteRef((struct SIDL_int__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get1(const struct bHYPRE_SStructVariable__array* 
  array,
                                   const int32_t i1)
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get1((const struct SIDL_int__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get2(const struct bHYPRE_SStructVariable__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2)
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get2((const struct SIDL_int__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get3(const struct bHYPRE_SStructVariable__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3)
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get3((const struct SIDL_int__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get4(const struct bHYPRE_SStructVariable__array* 
  array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   const int32_t i4)
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get4((const struct SIDL_int__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
enum bHYPRE_SStructVariable__enum
bHYPRE_SStructVariable__array_get(const struct bHYPRE_SStructVariable__array* 
  array,
                                  const int32_t indices[])
{
  return (enum bHYPRE_SStructVariable__enum)
    SIDL_int__array_get((const struct SIDL_int__array *)array, indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
bHYPRE_SStructVariable__array_set1(struct bHYPRE_SStructVariable__array* array,
                                   const int32_t i1,
                                   enum bHYPRE_SStructVariable__enum const 
  value)
{
  SIDL_int__array_set1((struct SIDL_int__array *)array
  , i1, (int32_t)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
bHYPRE_SStructVariable__array_set2(struct bHYPRE_SStructVariable__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   enum bHYPRE_SStructVariable__enum const 
  value)
{
  SIDL_int__array_set2((struct SIDL_int__array *)array
  , i1, i2, (int32_t)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
bHYPRE_SStructVariable__array_set3(struct bHYPRE_SStructVariable__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   enum bHYPRE_SStructVariable__enum const 
  value)
{
  SIDL_int__array_set3((struct SIDL_int__array *)array
  , i1, i2, i3, (int32_t)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
bHYPRE_SStructVariable__array_set4(struct bHYPRE_SStructVariable__array* array,
                                   const int32_t i1,
                                   const int32_t i2,
                                   const int32_t i3,
                                   const int32_t i4,
                                   enum bHYPRE_SStructVariable__enum const 
  value)
{
  SIDL_int__array_set4((struct SIDL_int__array *)array
  , i1, i2, i3, i4, (int32_t)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
bHYPRE_SStructVariable__array_set(struct bHYPRE_SStructVariable__array* array,
                                  const int32_t indices[],
                                  enum bHYPRE_SStructVariable__enum const value)
{
  SIDL_int__array_set((struct SIDL_int__array *)array, indices, (int32_t)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
bHYPRE_SStructVariable__array_dimen(const struct bHYPRE_SStructVariable__array* 
  array)
{
  return SIDL_int__array_dimen((struct SIDL_int__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
bHYPRE_SStructVariable__array_lower(const struct bHYPRE_SStructVariable__array* 
  array,
                                    const int32_t ind)
{
  return SIDL_int__array_lower((struct SIDL_int__array *)array, ind);
}

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
bHYPRE_SStructVariable__array_upper(const struct bHYPRE_SStructVariable__array* 
  array,
                                    const int32_t ind)
{
  return SIDL_int__array_upper((struct SIDL_int__array *)array, ind);
}

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
bHYPRE_SStructVariable__array_stride(const struct 
  bHYPRE_SStructVariable__array* array,
                                     const int32_t ind)
{
  return SIDL_int__array_stride((struct SIDL_int__array *)array, ind);
}

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
bHYPRE_SStructVariable__array_isColumnOrder(const struct 
  bHYPRE_SStructVariable__array* array)
{
  return SIDL_int__array_isColumnOrder((struct SIDL_int__array *)array);
}

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
bHYPRE_SStructVariable__array_isRowOrder(const struct 
  bHYPRE_SStructVariable__array* array)
{
  return SIDL_int__array_isRowOrder((struct SIDL_int__array *)array);
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
bHYPRE_SStructVariable__array_copy(const struct bHYPRE_SStructVariable__array* 
  src,
                                         struct bHYPRE_SStructVariable__array* 
  dest)
{
  SIDL_int__array_copy((const struct SIDL_int__array *)src,
                       (struct SIDL_int__array *)dest);
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
struct bHYPRE_SStructVariable__array*
bHYPRE_SStructVariable__array_ensure(struct bHYPRE_SStructVariable__array* src,
                                     int32_t dimen,
int     ordering)
{
  return (struct bHYPRE_SStructVariable__array*)
    SIDL_int__array_ensure((struct SIDL_int__array *)src, dimen, ordering);
}

