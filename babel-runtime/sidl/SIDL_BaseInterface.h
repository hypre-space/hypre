/*
 * File:          SIDL_BaseInterface.h
 * Symbol:        SIDL.BaseInterface-v0.8.1
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for SIDL.BaseInterface
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 */

#ifndef included_SIDL_BaseInterface_h
#define included_SIDL_BaseInterface_h

/**
 * Symbol "SIDL.BaseInterface" (version 0.8.1)
 * 
 * Every interface in <code>SIDL</code> implicitly inherits
 * from <code>BaseInterface</code>, and it is implemented
 * by <code>BaseClass</code> below.
 */
struct SIDL_BaseInterface__object;
struct SIDL_BaseInterface__array;
typedef struct SIDL_BaseInterface__object* SIDL_BaseInterface;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
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
SIDL_BaseInterface_addRef(
  SIDL_BaseInterface self);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
SIDL_BaseInterface_deleteRef(
  SIDL_BaseInterface self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
SIDL_BaseInterface_isSame(
  SIDL_BaseInterface self,
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
SIDL_BaseInterface_queryInt(
  SIDL_BaseInterface self,
  const char* name);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
SIDL_BaseInterface_isType(
  SIDL_BaseInterface self,
  const char* name);

/**
 * Cast method for interface and class type conversions.
 */
SIDL_BaseInterface
SIDL_BaseInterface__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
SIDL_BaseInterface__cast2(
  void* obj,
  const char* type);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_BaseInterface__array*
SIDL_BaseInterface__array_createCol(int32_t        dimen,
                                    const int32_t lower[],
                                    const int32_t upper[]);

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_BaseInterface__array*
SIDL_BaseInterface__array_createRow(int32_t        dimen,
                                    const int32_t lower[],
                                    const int32_t upper[]);

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_BaseInterface__array*
SIDL_BaseInterface__array_create1d(int32_t len);

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_BaseInterface__array*
SIDL_BaseInterface__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_BaseInterface__array*
SIDL_BaseInterface__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct SIDL_BaseInterface__array*
SIDL_BaseInterface__array_borrow(SIDL_BaseInterface*firstElement,
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
struct SIDL_BaseInterface__array*
SIDL_BaseInterface__array_smartCopy(struct SIDL_BaseInterface__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
SIDL_BaseInterface__array_addRef(struct SIDL_BaseInterface__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
SIDL_BaseInterface__array_deleteRef(struct SIDL_BaseInterface__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
SIDL_BaseInterface
SIDL_BaseInterface__array_get1(const struct SIDL_BaseInterface__array* array,
                               const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
SIDL_BaseInterface
SIDL_BaseInterface__array_get2(const struct SIDL_BaseInterface__array* array,
                               const int32_t i1,
                               const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
SIDL_BaseInterface
SIDL_BaseInterface__array_get3(const struct SIDL_BaseInterface__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
SIDL_BaseInterface
SIDL_BaseInterface__array_get4(const struct SIDL_BaseInterface__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               const int32_t i4);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
SIDL_BaseInterface
SIDL_BaseInterface__array_get(const struct SIDL_BaseInterface__array* array,
                              const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
SIDL_BaseInterface__array_set1(struct SIDL_BaseInterface__array* array,
                               const int32_t i1,
                               SIDL_BaseInterface const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
SIDL_BaseInterface__array_set2(struct SIDL_BaseInterface__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               SIDL_BaseInterface const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
SIDL_BaseInterface__array_set3(struct SIDL_BaseInterface__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               SIDL_BaseInterface const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
SIDL_BaseInterface__array_set4(struct SIDL_BaseInterface__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               const int32_t i4,
                               SIDL_BaseInterface const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
SIDL_BaseInterface__array_set(struct SIDL_BaseInterface__array* array,
                              const int32_t indices[],
                              SIDL_BaseInterface const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
SIDL_BaseInterface__array_dimen(const struct SIDL_BaseInterface__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
SIDL_BaseInterface__array_lower(const struct SIDL_BaseInterface__array* array,
                                const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
SIDL_BaseInterface__array_upper(const struct SIDL_BaseInterface__array* array,
                                const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
SIDL_BaseInterface__array_stride(const struct SIDL_BaseInterface__array* array,
                                 const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
SIDL_BaseInterface__array_isColumnOrder(const struct SIDL_BaseInterface__array* 
  array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
SIDL_BaseInterface__array_isRowOrder(const struct SIDL_BaseInterface__array* 
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
SIDL_BaseInterface__array_copy(const struct SIDL_BaseInterface__array* src,
                                     struct SIDL_BaseInterface__array* dest);

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
struct SIDL_BaseInterface__array*
SIDL_BaseInterface__array_ensure(struct SIDL_BaseInterface__array* src,
                                 int32_t dimen,
int     ordering);

#ifdef __cplusplus
}
#endif
#endif
