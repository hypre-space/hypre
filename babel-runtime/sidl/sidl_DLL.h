/*
 * File:          sidl_DLL.h
 * Symbol:        sidl.DLL-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for sidl.DLL
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
 */

#ifndef included_sidl_DLL_h
#define included_sidl_DLL_h

/**
 * Symbol "sidl.DLL" (version 0.9.15)
 * 
 * The <code>DLL</code> class encapsulates access to a single
 * dynamically linked library.  DLLs are loaded at run-time using
 * the <code>loadLibrary</code> method and later unloaded using
 * <code>unloadLibrary</code>.  Symbols in a loaded library are
 * resolved to an opaque pointer by method <code>lookupSymbol</code>.
 * Class instances are created by <code>createClass</code>.
 */
struct sidl_DLL__object;
struct sidl_DLL__array;
typedef struct sidl_DLL__object* sidl_DLL;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif

#ifndef included_sidl_rmi_Call_h
#include "sidl_rmi_Call.h"
#endif
#ifndef included_sidl_rmi_Return_h
#include "sidl_rmi_Return.h"
#endif
#ifdef SIDL_C_HAS_INLINE
#ifndef included_sidl_DLL_IOR_h
#include "sidl_DLL_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct sidl_DLL__object*
sidl_DLL__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
sidl_DLL
sidl_DLL__createRemote(const char * url, sidl_BaseInterface *_ex);


/**
 * RMI connector function for the class.(addrefs)
 */
sidl_DLL
sidl_DLL__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Load a dynamic link library using the specified URI.  The
 * URI may be of the form "main:", "lib:", "file:", "ftp:", or
 * "http:".  A URI that starts with any other protocol string
 * is assumed to be a file name.  The "main:" URI creates a
 * library that allows access to global symbols in the running
 * program's main address space.  The "lib:X" URI converts the
 * library "X" into a platform-specific name (e.g., libX.so) and
 * loads that library.  The "file:" URI opens the DLL from the
 * specified file path.  The "ftp:" and "http:" URIs copy the
 * specified library from the remote site into a local temporary
 * file and open that file.  This method returns true if the
 * DLL was loaded successfully and false otherwise.  Note that
 * the "ftp:" and "http:" protocols are valid only if the W3C
 * WWW library is available.
 * 
 * @param uri          the URI to load. This can be a .la file
 * (a metadata file produced by libtool) or
 * a shared library binary (i.e., .so,
 * .dll or whatever is appropriate for your
 * OS)
 * @param loadGlobally <code>true</code> means that the shared
 * library symbols will be loaded into the
 * global namespace; <code>false</code> 
 * means they will be loaded into a 
 * private namespace. Some operating systems
 * may not be able to honor the value presented
 * here.
 * @param loadLazy     <code>true</code> instructs the loader to
 * that symbols can be resolved as needed (lazy)
 * instead of requiring everything to be resolved
 * now (at load time).
 */
SIDL_C_INLINE_DECL
sidl_bool
sidl_DLL_loadLibrary(
  /* in */ sidl_DLL self,
  /* in */ const char* uri,
  /* in */ sidl_bool loadGlobally,
  /* in */ sidl_bool loadLazy,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_loadLibrary)(
    self,
    uri,
    loadGlobally,
    loadLazy,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Get the library name.  This is the name used to load the
 * library in <code>loadLibrary</code> except that all file names
 * contain the "file:" protocol.
 */
SIDL_C_INLINE_DECL
char*
sidl_DLL_getName(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getName)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Return true if the library was loaded into the global namespace.
 */
SIDL_C_INLINE_DECL
sidl_bool
sidl_DLL_isGlobal(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isGlobal)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Return true if the library was loaded using lazy symbol resolution.
 */
SIDL_C_INLINE_DECL
sidl_bool
sidl_DLL_isLazy(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isLazy)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Unload the dynamic link library.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */
SIDL_C_INLINE_DECL
void
sidl_DLL_unloadLibrary(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unloadLibrary)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Lookup a symbol from the DLL and return the associated pointer.
 * A null value is returned if the name does not exist.
 */
SIDL_C_INLINE_DECL
void*
sidl_DLL_lookupSymbol(
  /* in */ sidl_DLL self,
  /* in */ const char* linker_name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_lookupSymbol)(
    self,
    linker_name,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Create an instance of the sidl class.  If the class constructor
 * is not defined in this DLL, then return null.
 */
SIDL_C_INLINE_DECL
sidl_BaseClass
sidl_DLL_createClass(
  /* in */ sidl_DLL self,
  /* in */ const char* sidl_name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_createClass)(
    self,
    sidl_name,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
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
SIDL_C_INLINE_DECL
void
sidl_DLL_addRef(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_addRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
SIDL_C_INLINE_DECL
void
sidl_DLL_deleteRef(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_deleteRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_C_INLINE_DECL
sidl_bool
sidl_DLL_isSame(
  /* in */ sidl_DLL self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_C_INLINE_DECL
sidl_bool
sidl_DLL_isType(
  /* in */ sidl_DLL self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isType)(
    self,
    name,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Return the meta-data about the class implementing this interface.
 */
SIDL_C_INLINE_DECL
sidl_ClassInfo
sidl_DLL_getClassInfo(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getClassInfo)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct sidl_DLL__object*
sidl_DLL__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
sidl_DLL__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
sidl_DLL__exec(
  /* in */ sidl_DLL self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__exec)(
    self,
    methodName,
    inArgs,
    outArgs,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * Get the URL of the Implementation of this object (for RMI)
 */
SIDL_C_INLINE_DECL
char*
sidl_DLL__getURL(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__getURL)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * On a remote object, addrefs the remote instance.
 */
SIDL_C_INLINE_DECL
void
sidl_DLL__raddRef(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__raddRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
SIDL_C_INLINE_DECL
sidl_bool
sidl_DLL__isRemote(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__isRemote)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
sidl_DLL__isLocal(
  /* in */ sidl_DLL self,
  /* out */ sidl_BaseInterface *_ex);
/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DLL__array*
sidl_DLL__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DLL__array*
sidl_DLL__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DLL__array*
sidl_DLL__array_create1d(int32_t len);

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidl_DLL__array*
sidl_DLL__array_create1dInit(
  int32_t len, 
  sidl_DLL* data);

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DLL__array*
sidl_DLL__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DLL__array*
sidl_DLL__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidl_DLL__array*
sidl_DLL__array_borrow(
  sidl_DLL* firstElement,
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
struct sidl_DLL__array*
sidl_DLL__array_smartCopy(
  struct sidl_DLL__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
sidl_DLL__array_addRef(
  struct sidl_DLL__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
sidl_DLL__array_deleteRef(
  struct sidl_DLL__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidl_DLL
sidl_DLL__array_get1(
  const struct sidl_DLL__array* array,
  const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidl_DLL
sidl_DLL__array_get2(
  const struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidl_DLL
sidl_DLL__array_get3(
  const struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidl_DLL
sidl_DLL__array_get4(
  const struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidl_DLL
sidl_DLL__array_get5(
  const struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidl_DLL
sidl_DLL__array_get6(
  const struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidl_DLL
sidl_DLL__array_get7(
  const struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidl_DLL
sidl_DLL__array_get(
  const struct sidl_DLL__array* array,
  const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidl_DLL__array_set1(
  struct sidl_DLL__array* array,
  const int32_t i1,
  sidl_DLL const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidl_DLL__array_set2(
  struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  sidl_DLL const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidl_DLL__array_set3(
  struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidl_DLL const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidl_DLL__array_set4(
  struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidl_DLL const value);

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidl_DLL__array_set5(
  struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidl_DLL const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidl_DLL__array_set6(
  struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidl_DLL const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidl_DLL__array_set7(
  struct sidl_DLL__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidl_DLL const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidl_DLL__array_set(
  struct sidl_DLL__array* array,
  const int32_t indices[],
  sidl_DLL const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidl_DLL__array_dimen(
  const struct sidl_DLL__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_DLL__array_lower(
  const struct sidl_DLL__array* array,
  const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_DLL__array_upper(
  const struct sidl_DLL__array* array,
  const int32_t ind);

/**
 * Return the length of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_DLL__array_length(
  const struct sidl_DLL__array* array,
  const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_DLL__array_stride(
  const struct sidl_DLL__array* array,
  const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidl_DLL__array_isColumnOrder(
  const struct sidl_DLL__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidl_DLL__array_isRowOrder(
  const struct sidl_DLL__array* array);

/**
 * Create a sub-array of another array. This resulting
 * array shares data with the original array. The new
 * array can be of the same dimension or potentially
 * less assuming the original array has dimension
 * greater than 1.  If you are removing dimension,
 * indicate the dimensions to remove by setting
 * numElem[i] to zero for any dimension i wthat should
 * go away in the new array.  The meaning of each
 * argument is covered below.
 * 
 * src       the array to be created will be a subset
 *           of this array. If this argument is NULL,
 *           NULL will be returned. The array returned
 *           borrows data from src, so modifying src or
 *           the returned array will modify both
 *           arrays.
 * 
 * dimen     this argument must be greater than zero
 *           and less than or equal to the dimension of
 *           src. An illegal value will cause a NULL
 *           return value.
 * 
 * numElem   this specifies how many elements from src
 *           should be taken in each dimension. A zero
 *           entry indicates that the dimension should
 *           not appear in the new array.  This
 *           argument should be an array with an entry
 *           for each dimension of src.  Passing NULL
 *           here will cause NULL to be returned.  If
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           greater than upper[i] for src or if
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           less than lower[i] for src, NULL will be
 *           returned.
 * 
 * srcStart  this array holds the coordinates of the
 *           first element of the new array. If this
 *           argument is NULL, the first element of src
 *           will be the first element of the new
 *           array. If non-NULL, this argument should
 *           be an array with an entry for each
 *           dimension of src.  If srcStart[i] is less
 *           than lower[i] for the array src, NULL will
 *           be returned.
 * 
 * srcStride this array lets you specify the stride
 *           between elements in each dimension of
 *           src. This stride is relative to the
 *           coordinate system of the src array. If
 *           this argument is NULL, the stride is taken
 *           to be one in each dimension.  If non-NULL,
 *           this argument should be an array with an
 *           entry for each dimension of src.
 * 
 * newLower  this argument is like lower in a create
 *           method. It sets the coordinates for the
 *           first element in the new array.  If this
 *           argument is NULL, the values indicated by
 *           srcStart will be used. If non-NULL, this
 *           should be an array with dimen elements.
 */
struct sidl_DLL__array*
sidl_DLL__array_slice(
  struct sidl_DLL__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

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
sidl_DLL__array_copy(
  const struct sidl_DLL__array* src,
  struct sidl_DLL__array* dest);

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
 * defined in enum sidl_array_ordering
 * (e.g. sidl_general_order, sidl_column_major_order, or
 * sidl_row_major_order). If you specify
 * sidl_general_order, this routine will only check the
 * dimension because any matrix is sidl_general_order.
 * 
 * The caller assumes ownership of the returned reference
 * unless it's NULL.
 */
struct sidl_DLL__array*
sidl_DLL__array_ensure(
  struct sidl_DLL__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak sidl_DLL__connectI

#pragma weak sidl_DLL__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct sidl_DLL__object*
sidl_DLL__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct sidl_DLL__object*
sidl_DLL__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
