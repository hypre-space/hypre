/*
 * File:          SIDL_Loader_Stub.c
 * Symbol:        SIDL.Loader-v0.8.2
 * Symbol Type:   class
 * Babel Version: 0.8.4
 * Release:       $Name: V1-9-0b $
 * Revision:      @(#) $Id: SIDL_Loader_Stub.c,v 1.4 2003/04/07 21:44:31 painter Exp $
 * Description:   Client-side glue code for SIDL.Loader
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
 * babel-version = 0.8.4
 */

#include "SIDL_Loader.h"
#include "SIDL_Loader_IOR.h"
#ifndef included_SIDL_interface_IOR_h
#include "SIDL_interface_IOR.h"
#endif
#include <stddef.h>
#include "SIDL_BaseInterface_IOR.h"

/*
 * Hold pointer to IOR functions.
 */

static const struct SIDL_Loader__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct SIDL_Loader__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
  _ior = SIDL_Loader__externals();
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Hold pointer to static entry point vector
 */

static const struct SIDL_Loader__sepv *_sepv = NULL;
/*
 * Return pointer to static functions.
 */

#define _getSEPV() (_sepv ? _sepv : (_sepv = (*(_getIOR()->getStaticEPV))()))

/*
 * Constructor function for the class.
 */

SIDL_Loader
SIDL_Loader__create()
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
SIDL_Loader_addRef(
  SIDL_Loader self)
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
SIDL_Loader_deleteRef(
  SIDL_Loader self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
SIDL_Loader_isSame(
  SIDL_Loader self,
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
SIDL_Loader_queryInt(
  SIDL_Loader self,
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
SIDL_Loader_isType(
  SIDL_Loader self,
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
SIDL_Loader_getClassInfo(
  SIDL_Loader self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 */

void
SIDL_Loader_setSearchPath(
  const char* path_name)
{
  (_getSEPV()->f_setSearchPath)(
    path_name);
}

/*
 * Return the current search path.  If the search path has not been
 * set, then the search path will be taken from environment variable
 * SIDL_DLL_PATH.
 */

char*
SIDL_Loader_getSearchPath(
  void)
{
  return (_getSEPV()->f_getSearchPath)(
    );
}

/*
 * Append the specified path fragment to the beginning of the
 * current search path.  If the search path has not yet been set
 * by a call to <code>setSearchPath</code>, then this fragment will
 * be appended to the path in environment variable SIDL_DLL_PATH.
 */

void
SIDL_Loader_addSearchPath(
  const char* path_fragment)
{
  (_getSEPV()->f_addSearchPath)(
    path_fragment);
}

/*
 * Load the specified library if it has not already been loaded.
 * The URI format is defined in class <code>DLL</code>.  The search
 * path is not searched to resolve the library name.
 */

SIDL_bool
SIDL_Loader_loadLibrary(
  const char* uri)
{
  return (_getSEPV()->f_loadLibrary)(
    uri);
}

/*
 * Append the specified DLL to the beginning of the list of already
 * loaded DLLs.
 */

void
SIDL_Loader_addDLL(
  SIDL_DLL dll)
{
  (_getSEPV()->f_addDLL)(
    dll);
}

/*
 * Unload all dynamic link libraries.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */

void
SIDL_Loader_unloadLibraries(
  void)
{
  (_getSEPV()->f_unloadLibraries)(
    );
}

/*
 * Look up the secified symbol name.  If the symbol name cannot be
 * found in one of the already loaded libraries, then the method will
 * search through the library search path.  A null is returned if the
 * symbol could not be resolved.
 */

void*
SIDL_Loader_lookupSymbol(
  const char* linker_name)
{
  return (_getSEPV()->f_lookupSymbol)(
    linker_name);
}

/*
 * Create an instance of the specified SIDL class.  If the class
 * constructor cannot be found in one of the already loaded libraries,
 * then the method will search through the library search path.  A null
 * object is returned if the symbol could not be resolved.
 */

SIDL_BaseClass
SIDL_Loader_createClass(
  const char* sidl_name)
{
  return (_getSEPV()->f_createClass)(
    sidl_name);
}

/*
 * Cast method for interface and class type conversions.
 */

SIDL_Loader
SIDL_Loader__cast(
  void* obj)
{
  SIDL_Loader cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (SIDL_Loader) (*base->d_epv->f__cast)(
      base->d_object,
      "SIDL.Loader");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
SIDL_Loader__cast2(
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
struct SIDL_Loader__array*
SIDL_Loader__array_createCol(int32_t        dimen,
                             const int32_t lower[],
                             const int32_t upper[])
{
  return (struct SIDL_Loader__array*)SIDL_interface__array_createCol(dimen,
    lower, upper);
}

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_Loader__array*
SIDL_Loader__array_createRow(int32_t        dimen,
                             const int32_t lower[],
                             const int32_t upper[])
{
  return (struct SIDL_Loader__array*)SIDL_interface__array_createRow(dimen,
    lower, upper);
}

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_Loader__array*
SIDL_Loader__array_create1d(int32_t len)
{
  return (struct SIDL_Loader__array*)SIDL_interface__array_create1d(len);
}

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_Loader__array*
SIDL_Loader__array_create2dCol(int32_t m, int32_t n)
{
  return (struct SIDL_Loader__array*)SIDL_interface__array_create2dCol(m, n);
}

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct SIDL_Loader__array*
SIDL_Loader__array_create2dRow(int32_t m, int32_t n)
{
  return (struct SIDL_Loader__array*)SIDL_interface__array_create2dRow(m, n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct SIDL_Loader__array*
SIDL_Loader__array_borrow(SIDL_Loader*firstElement,
                          int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct SIDL_Loader__array*)SIDL_interface__array_borrow(
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
struct SIDL_Loader__array*
SIDL_Loader__array_smartCopy(struct SIDL_Loader__array *array)
{
  return (struct SIDL_Loader__array*)
    SIDL_interface__array_smartCopy((struct SIDL_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
SIDL_Loader__array_addRef(struct SIDL_Loader__array* array)
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
SIDL_Loader__array_deleteRef(struct SIDL_Loader__array* array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
SIDL_Loader
SIDL_Loader__array_get1(const struct SIDL_Loader__array* array,
                        const int32_t i1)
{
  return (SIDL_Loader)
    SIDL_interface__array_get1((const struct SIDL_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
SIDL_Loader
SIDL_Loader__array_get2(const struct SIDL_Loader__array* array,
                        const int32_t i1,
                        const int32_t i2)
{
  return (SIDL_Loader)
    SIDL_interface__array_get2((const struct SIDL_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
SIDL_Loader
SIDL_Loader__array_get3(const struct SIDL_Loader__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3)
{
  return (SIDL_Loader)
    SIDL_interface__array_get3((const struct SIDL_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
SIDL_Loader
SIDL_Loader__array_get4(const struct SIDL_Loader__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        const int32_t i4)
{
  return (SIDL_Loader)
    SIDL_interface__array_get4((const struct SIDL_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
SIDL_Loader
SIDL_Loader__array_get(const struct SIDL_Loader__array* array,
                       const int32_t indices[])
{
  return (SIDL_Loader)
    SIDL_interface__array_get((const struct SIDL_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
SIDL_Loader__array_set1(struct SIDL_Loader__array* array,
                        const int32_t i1,
                        SIDL_Loader const value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)array
  , i1, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
SIDL_Loader__array_set2(struct SIDL_Loader__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        SIDL_Loader const value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)array
  , i1, i2, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
SIDL_Loader__array_set3(struct SIDL_Loader__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        SIDL_Loader const value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)array
  , i1, i2, i3, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
SIDL_Loader__array_set4(struct SIDL_Loader__array* array,
                        const int32_t i1,
                        const int32_t i2,
                        const int32_t i3,
                        const int32_t i4,
                        SIDL_Loader const value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)array
  , i1, i2, i3, i4, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
SIDL_Loader__array_set(struct SIDL_Loader__array* array,
                       const int32_t indices[],
                       SIDL_Loader const value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)array, indices,
    (struct SIDL_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
SIDL_Loader__array_dimen(const struct SIDL_Loader__array* array)
{
  return SIDL_interface__array_dimen((struct SIDL_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
SIDL_Loader__array_lower(const struct SIDL_Loader__array* array,
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
SIDL_Loader__array_upper(const struct SIDL_Loader__array* array,
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
SIDL_Loader__array_stride(const struct SIDL_Loader__array* array,
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
SIDL_Loader__array_isColumnOrder(const struct SIDL_Loader__array* array)
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
SIDL_Loader__array_isRowOrder(const struct SIDL_Loader__array* array)
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
SIDL_Loader__array_copy(const struct SIDL_Loader__array* src,
                              struct SIDL_Loader__array* dest)
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
struct SIDL_Loader__array*
SIDL_Loader__array_ensure(struct SIDL_Loader__array* src,
                          int32_t dimen,
                          int     ordering)
{
  return (struct SIDL_Loader__array*)
    SIDL_interface__array_ensure((struct SIDL_interface__array *)src, dimen,
      ordering);
}

