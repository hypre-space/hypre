/*
 * File:          bHYPRE_SStructParCSRVector.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:26 PST
 * Description:   Client-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 837
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

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
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
  bHYPRE_SStructParCSRVector self);

void
bHYPRE_SStructParCSRVector_deleteRef(
  bHYPRE_SStructParCSRVector self);

SIDL_bool
bHYPRE_SStructParCSRVector_isSame(
  bHYPRE_SStructParCSRVector self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_SStructParCSRVector_queryInt(
  bHYPRE_SStructParCSRVector self,
  const char* name);

SIDL_bool
bHYPRE_SStructParCSRVector_isType(
  bHYPRE_SStructParCSRVector self,
  const char* name);

SIDL_ClassInfo
bHYPRE_SStructParCSRVector_getClassInfo(
  bHYPRE_SStructParCSRVector self);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Clear(
  bHYPRE_SStructParCSRVector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Copy(
  bHYPRE_SStructParCSRVector self,
  bHYPRE_Vector x);

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
  bHYPRE_SStructParCSRVector self,
  bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Scale(
  bHYPRE_SStructParCSRVector self,
  double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Dot(
  bHYPRE_SStructParCSRVector self,
  bHYPRE_Vector x,
  double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Axpy(
  bHYPRE_SStructParCSRVector self,
  double a,
  bHYPRE_Vector x);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_SetCommunicator(
  bHYPRE_SStructParCSRVector self,
  void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Initialize(
  bHYPRE_SStructParCSRVector self);

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
  bHYPRE_SStructParCSRVector self);

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
bHYPRE_SStructParCSRVector_GetObject(
  bHYPRE_SStructParCSRVector self,
  SIDL_BaseInterface* A);

/**
 * Set the vector grid.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_SetGrid(
  bHYPRE_SStructParCSRVector self,
  bHYPRE_SStructGrid grid);

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
  bHYPRE_SStructParCSRVector self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value);

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
  bHYPRE_SStructParCSRVector self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values);

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
  bHYPRE_SStructParCSRVector self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value);

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
  bHYPRE_SStructParCSRVector self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values);

/**
 * Gather vector data before calling {\tt GetValues}.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Gather(
  bHYPRE_SStructParCSRVector self);

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
  bHYPRE_SStructParCSRVector self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  double* value);

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
  bHYPRE_SStructParCSRVector self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array** values);

/**
 * Set the vector to be complex.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_SetComplex(
  bHYPRE_SStructParCSRVector self);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructParCSRVector_Print(
  bHYPRE_SStructParCSRVector self,
  const char* filename,
  int32_t all);

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
bHYPRE_SStructParCSRVector__array_createCol(int32_t        dimen,
                                            const int32_t lower[],
                                            const int32_t upper[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createRow(int32_t        dimen,
                                            const int32_t lower[],
                                            const int32_t upper[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1d(int32_t len);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_borrow(
  bHYPRE_SStructParCSRVector*firstElement,
                                         int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_smartCopy(struct 
  bHYPRE_SStructParCSRVector__array *array);

void
bHYPRE_SStructParCSRVector__array_addRef(struct 
  bHYPRE_SStructParCSRVector__array* array);

void
bHYPRE_SStructParCSRVector__array_deleteRef(struct 
  bHYPRE_SStructParCSRVector__array* array);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get1(const struct 
  bHYPRE_SStructParCSRVector__array* array,
                                       const int32_t i1);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get2(const struct 
  bHYPRE_SStructParCSRVector__array* array,
                                       const int32_t i1,
                                       const int32_t i2);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get3(const struct 
  bHYPRE_SStructParCSRVector__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       const int32_t i3);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get4(const struct 
  bHYPRE_SStructParCSRVector__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       const int32_t i3,
                                       const int32_t i4);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get(const struct 
  bHYPRE_SStructParCSRVector__array* array,
                                      const int32_t indices[]);

void
bHYPRE_SStructParCSRVector__array_set1(struct 
  bHYPRE_SStructParCSRVector__array* array,
                                       const int32_t i1,
                                       bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set2(struct 
  bHYPRE_SStructParCSRVector__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set3(struct 
  bHYPRE_SStructParCSRVector__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       const int32_t i3,
                                       bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set4(struct 
  bHYPRE_SStructParCSRVector__array* array,
                                       const int32_t i1,
                                       const int32_t i2,
                                       const int32_t i3,
                                       const int32_t i4,
                                       bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set(struct bHYPRE_SStructParCSRVector__array* 
  array,
                                      const int32_t indices[],
                                      bHYPRE_SStructParCSRVector const value);

int32_t
bHYPRE_SStructParCSRVector__array_dimen(const struct 
  bHYPRE_SStructParCSRVector__array* array);

int32_t
bHYPRE_SStructParCSRVector__array_lower(const struct 
  bHYPRE_SStructParCSRVector__array* array,
                                        const int32_t ind);

int32_t
bHYPRE_SStructParCSRVector__array_upper(const struct 
  bHYPRE_SStructParCSRVector__array* array,
                                        const int32_t ind);

int32_t
bHYPRE_SStructParCSRVector__array_stride(const struct 
  bHYPRE_SStructParCSRVector__array* array,
                                         const int32_t ind);

int
bHYPRE_SStructParCSRVector__array_isColumnOrder(const struct 
  bHYPRE_SStructParCSRVector__array* array);

int
bHYPRE_SStructParCSRVector__array_isRowOrder(const struct 
  bHYPRE_SStructParCSRVector__array* array);

void
bHYPRE_SStructParCSRVector__array_slice(const struct 
  bHYPRE_SStructParCSRVector__array* src,
                                              int32_t        dimen,
                                              const int32_t  numElem[],
                                              const int32_t  *srcStart,
                                              const int32_t  *srcStride,
                                              const int32_t  *newStart);

void
bHYPRE_SStructParCSRVector__array_copy(const struct 
  bHYPRE_SStructParCSRVector__array* src,
                                             struct 
  bHYPRE_SStructParCSRVector__array* dest);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_ensure(struct 
  bHYPRE_SStructParCSRVector__array* src,
                                         int32_t dimen,
                                         int     ordering);

#ifdef __cplusplus
}
#endif
#endif
