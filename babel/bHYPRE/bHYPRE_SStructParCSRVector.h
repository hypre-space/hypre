/*
 * File:          bHYPRE_SStructParCSRVector.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:42 PST
 * Description:   Client-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 842
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructParCSRVector_h
#define included_bHYPRE_SStructParCSRVector_h

/**
 * Symbol "bHYPRE.SStructParCSRVector" (version 1.0.0)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_SStructParCSRVector__object;
struct bHYPRE_SStructParCSRVector__array;
typedef struct bHYPRE_SStructParCSRVector__object* bHYPRE_SStructParCSRVector;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__create(void);

void
bHYPRE_SStructParCSRVector_addRef(
  /*in*/ bHYPRE_SStructParCSRVector self);

void
bHYPRE_SStructParCSRVector_deleteRef(
  /*in*/ bHYPRE_SStructParCSRVector self);

sidl_bool
bHYPRE_SStructParCSRVector_isSame(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructParCSRVector_queryInt(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ const char* name);

sidl_bool
bHYPRE_SStructParCSRVector_isType(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ const char* name);

sidl_ClassInfo
bHYPRE_SStructParCSRVector_getClassInfo(
  /*in*/ bHYPRE_SStructParCSRVector self);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Clear(
  /*in*/ bHYPRE_SStructParCSRVector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Copy(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ bHYPRE_Vector x);

/**
 * Create an {\tt x} compatible with {\tt self}.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Clone(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*out*/ bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Scale(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Dot(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ bHYPRE_Vector x,
  /*out*/ double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Axpy(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ double a,
  /*in*/ bHYPRE_Vector x);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_SetCommunicator(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Initialize(
  /*in*/ bHYPRE_SStructParCSRVector self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Assemble(
  /*in*/ bHYPRE_SStructParCSRVector self);

/**
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_GetObject(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*out*/ sidl_BaseInterface* A);

/**
 * Set the vector grid.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_SetGrid(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ bHYPRE_SStructGrid grid);

/**
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_SetValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* value);

/**
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_SetBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* values);

/**
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_AddToValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* value);

/**
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_AddToBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ struct sidl_double__array* values);

/**
 * Gather vector data before calling {\tt GetValues}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Gather(
  /*in*/ bHYPRE_SStructParCSRVector self);

/**
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_GetValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*out*/ double* value);

/**
 * Get vector coefficients a box at a time.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_GetBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*inout*/ struct sidl_double__array** values);

/**
 * Set the vector to be complex.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_SetComplex(
  /*in*/ bHYPRE_SStructParCSRVector self);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Print(
  /*in*/ bHYPRE_SStructParCSRVector self,
  /*in*/ const char* filename,
  /*in*/ int32_t all);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructParCSRVector__cast2(
  void* obj,
  const char* type);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1d(int32_t len);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructParCSRVector* data);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_borrow(
  bHYPRE_SStructParCSRVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_smartCopy(
  struct bHYPRE_SStructParCSRVector__array *array);

void
bHYPRE_SStructParCSRVector__array_addRef(
  struct bHYPRE_SStructParCSRVector__array* array);

void
bHYPRE_SStructParCSRVector__array_deleteRef(
  struct bHYPRE_SStructParCSRVector__array* array);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get1(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get2(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get3(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get4(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get5(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get6(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get7(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructParCSRVector__array_set1(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set2(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set3(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set4(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set5(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set6(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set7(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t indices[],
  bHYPRE_SStructParCSRVector const value);

int32_t
bHYPRE_SStructParCSRVector__array_dimen(
  const struct bHYPRE_SStructParCSRVector__array* array);

int32_t
bHYPRE_SStructParCSRVector__array_lower(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRVector__array_upper(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRVector__array_length(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRVector__array_stride(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind);

int
bHYPRE_SStructParCSRVector__array_isColumnOrder(
  const struct bHYPRE_SStructParCSRVector__array* array);

int
bHYPRE_SStructParCSRVector__array_isRowOrder(
  const struct bHYPRE_SStructParCSRVector__array* array);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_slice(
  struct bHYPRE_SStructParCSRVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructParCSRVector__array_copy(
  const struct bHYPRE_SStructParCSRVector__array* src,
  struct bHYPRE_SStructParCSRVector__array* dest);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_ensure(
  struct bHYPRE_SStructParCSRVector__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
