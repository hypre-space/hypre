/*
 * File:          bHYPRE_SStructGrid_Stub.c
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:42 PST
 * Generated:     20030314 14:22:44 PST
 * Description:   Client-side glue code for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 892
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructGrid.h"
#include "bHYPRE_SStructGrid_IOR.h"
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

static const struct bHYPRE_SStructGrid__external *_ior = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_SStructGrid__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _ior = bHYPRE_SStructGrid__externals();
#else
  const struct bHYPRE_SStructGrid__external*(*dll_f)(void) =
    (const struct bHYPRE_SStructGrid__external*(*)(void)) 
      SIDL_Loader_lookupSymbol(
      "bHYPRE_SStructGrid__externals");
  _ior = (dll_f ? (*dll_f)() : NULL);
  if (!_ior) {
    fputs("Babel: unable to load the implementation for bHYPRE.SStructGrid; please set SIDL_DLL_PATH\n", stderr);
    exit(-1);
  }
#endif
  return _ior;
}

#define _getIOR() (_ior ? _ior : _loadIOR())

/*
 * Constructor function for the class.
 */

bHYPRE_SStructGrid
bHYPRE_SStructGrid__create()
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
bHYPRE_SStructGrid_addRef(
  bHYPRE_SStructGrid self)
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
bHYPRE_SStructGrid_deleteRef(
  bHYPRE_SStructGrid self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
bHYPRE_SStructGrid_isSame(
  bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_queryInt(
  bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_isType(
  bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_getClassInfo(
  bHYPRE_SStructGrid self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Set the number of dimensions {\tt ndim} and the number of
 * structured parts {\tt nparts}.
 * 
 */

int32_t
bHYPRE_SStructGrid_SetNumDimParts(
  bHYPRE_SStructGrid self,
  int32_t ndim,
  int32_t nparts)
{
  return (*self->d_epv->f_SetNumDimParts)(
    self,
    ndim,
    nparts);
}

/*
 * Set the extents for a box on a structured part of the grid.
 * 
 */

int32_t
bHYPRE_SStructGrid_SetExtents(
  bHYPRE_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper)
{
  return (*self->d_epv->f_SetExtents)(
    self,
    part,
    ilower,
    iupper);
}

/*
 * Describe the variables that live on a structured part of the
 * grid.
 * 
 */

int32_t
bHYPRE_SStructGrid_SetVariable(
  bHYPRE_SStructGrid self,
  int32_t part,
  int32_t var,
  enum bHYPRE_SStructVariable__enum vartype)
{
  return (*self->d_epv->f_SetVariable)(
    self,
    part,
    var,
    vartype);
}

/*
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 * 
 */

int32_t
bHYPRE_SStructGrid_AddVariable(
  bHYPRE_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  enum bHYPRE_SStructVariable__enum vartype)
{
  return (*self->d_epv->f_AddVariable)(
    self,
    part,
    index,
    var,
    vartype);
}

/*
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
bHYPRE_SStructGrid_SetNeighborBox(
  bHYPRE_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t nbor_part,
  struct SIDL_int__array* nbor_ilower,
  struct SIDL_int__array* nbor_iupper,
  struct SIDL_int__array* index_map)
{
  return (*self->d_epv->f_SetNeighborBox)(
    self,
    part,
    ilower,
    iupper,
    nbor_part,
    nbor_ilower,
    nbor_iupper,
    index_map);
}

/*
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
bHYPRE_SStructGrid_AddUnstructuredPart(
  bHYPRE_SStructGrid self,
  int32_t ilower,
  int32_t iupper)
{
  return (*self->d_epv->f_AddUnstructuredPart)(
    self,
    ilower,
    iupper);
}

/*
 * (Optional) Set periodic for a particular part.
 * 
 */

int32_t
bHYPRE_SStructGrid_SetPeriodic(
  bHYPRE_SStructGrid self,
  int32_t part,
  struct SIDL_int__array* periodic)
{
  return (*self->d_epv->f_SetPeriodic)(
    self,
    part,
    periodic);
}

/*
 * Setting ghost in the sgrids.
 * 
 */

int32_t
bHYPRE_SStructGrid_SetNumGhost(
  bHYPRE_SStructGrid self,
  struct SIDL_int__array* num_ghost)
{
  return (*self->d_epv->f_SetNumGhost)(
    self,
    num_ghost);
}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_SStructGrid
bHYPRE_SStructGrid__cast(
  void* obj)
{
  bHYPRE_SStructGrid cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (bHYPRE_SStructGrid) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructGrid");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_SStructGrid__cast2(
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
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_createCol(int32_t        dimen,
                                    const int32_t lower[],
                                    const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructGrid__array*)SIDL_interface__array_createCol(dimen, lower,
    upper);
}

/**
 * Create a dense array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_createRow(int32_t        dimen,
                                    const int32_t lower[],
                                    const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructGrid__array*)SIDL_interface__array_createRow(dimen, lower,
    upper);
}

/**
 * Create a dense one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create1d(int32_t len)
{
  return (struct bHYPRE_SStructGrid__array*)SIDL_interface__array_create1d(len);
}

/**
 * Create a dense two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create2dCol(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructGrid__array*)SIDL_interface__array_create2dCol(m,
    n);
}

/**
 * Create a dense two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create2dRow(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructGrid__array*)SIDL_interface__array_create2dRow(m,
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
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_borrow(bHYPRE_SStructGrid*firstElement,
                                 int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[])
{
  return (struct bHYPRE_SStructGrid__array*)SIDL_interface__array_borrow(
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
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_smartCopy(struct bHYPRE_SStructGrid__array *array)
{
  return (struct bHYPRE_SStructGrid__array*)
    SIDL_interface__array_smartCopy((struct SIDL_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
bHYPRE_SStructGrid__array_addRef(struct bHYPRE_SStructGrid__array* array)
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
bHYPRE_SStructGrid__array_deleteRef(struct bHYPRE_SStructGrid__array* array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get1(const struct bHYPRE_SStructGrid__array* array,
                               const int32_t i1)
{
  return (bHYPRE_SStructGrid)
    SIDL_interface__array_get1((const struct SIDL_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get2(const struct bHYPRE_SStructGrid__array* array,
                               const int32_t i1,
                               const int32_t i2)
{
  return (bHYPRE_SStructGrid)
    SIDL_interface__array_get2((const struct SIDL_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get3(const struct bHYPRE_SStructGrid__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3)
{
  return (bHYPRE_SStructGrid)
    SIDL_interface__array_get3((const struct SIDL_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get4(const struct bHYPRE_SStructGrid__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               const int32_t i4)
{
  return (bHYPRE_SStructGrid)
    SIDL_interface__array_get4((const struct SIDL_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get(const struct bHYPRE_SStructGrid__array* array,
                              const int32_t indices[])
{
  return (bHYPRE_SStructGrid)
    SIDL_interface__array_get((const struct SIDL_interface__array *)array,
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
bHYPRE_SStructGrid__array_set1(struct bHYPRE_SStructGrid__array* array,
                               const int32_t i1,
                               bHYPRE_SStructGrid const value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)array
  , i1, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
bHYPRE_SStructGrid__array_set2(struct bHYPRE_SStructGrid__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               bHYPRE_SStructGrid const value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)array
  , i1, i2, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
bHYPRE_SStructGrid__array_set3(struct bHYPRE_SStructGrid__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               bHYPRE_SStructGrid const value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)array
  , i1, i2, i3, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
bHYPRE_SStructGrid__array_set4(struct bHYPRE_SStructGrid__array* array,
                               const int32_t i1,
                               const int32_t i2,
                               const int32_t i3,
                               const int32_t i4,
                               bHYPRE_SStructGrid const value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)array
  , i1, i2, i3, i4, (struct SIDL_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
bHYPRE_SStructGrid__array_set(struct bHYPRE_SStructGrid__array* array,
                              const int32_t indices[],
                              bHYPRE_SStructGrid const value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)array, indices,
    (struct SIDL_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
bHYPRE_SStructGrid__array_dimen(const struct bHYPRE_SStructGrid__array* array)
{
  return SIDL_interface__array_dimen((struct SIDL_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range is from 0 to dimen-1.
 */
int32_t
bHYPRE_SStructGrid__array_lower(const struct bHYPRE_SStructGrid__array* array,
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
bHYPRE_SStructGrid__array_upper(const struct bHYPRE_SStructGrid__array* array,
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
bHYPRE_SStructGrid__array_stride(const struct bHYPRE_SStructGrid__array* array,
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
bHYPRE_SStructGrid__array_isColumnOrder(const struct bHYPRE_SStructGrid__array* 
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
bHYPRE_SStructGrid__array_isRowOrder(const struct bHYPRE_SStructGrid__array* 
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
bHYPRE_SStructGrid__array_copy(const struct bHYPRE_SStructGrid__array* src,
                                     struct bHYPRE_SStructGrid__array* dest)
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
struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_ensure(struct bHYPRE_SStructGrid__array* src,
                                 int32_t dimen,
int     ordering)
{
  return (struct bHYPRE_SStructGrid__array*)
    SIDL_interface__array_ensure((struct SIDL_interface__array *)src, dimen,
      ordering);
}

