/*
 * File:          bHYPRE_SStructVector.h
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:35 PST
 * Generated:     20030401 14:47:41 PST
 * Description:   Client-side glue code for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1074
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructVector_h
#define included_bHYPRE_SStructVector_h

/**
 * Symbol "bHYPRE.SStructVector" (version 1.0.0)
 * 
 * The semi-structured grid vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_SStructVector__object;
struct bHYPRE_SStructVector__array;
typedef struct bHYPRE_SStructVector__object* bHYPRE_SStructVector;

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
bHYPRE_SStructVector
bHYPRE_SStructVector__create(void);

void
bHYPRE_SStructVector_addRef(
  bHYPRE_SStructVector self);

void
bHYPRE_SStructVector_deleteRef(
  bHYPRE_SStructVector self);

SIDL_bool
bHYPRE_SStructVector_isSame(
  bHYPRE_SStructVector self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_SStructVector_queryInt(
  bHYPRE_SStructVector self,
  const char* name);

SIDL_bool
bHYPRE_SStructVector_isType(
  bHYPRE_SStructVector self,
  const char* name);

SIDL_ClassInfo
bHYPRE_SStructVector_getClassInfo(
  bHYPRE_SStructVector self);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_SStructVector_Clear(
  bHYPRE_SStructVector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_SStructVector_Copy(
  bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Clone(
  bHYPRE_SStructVector self,
  bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_SStructVector_Scale(
  bHYPRE_SStructVector self,
  double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructVector_Dot(
  bHYPRE_SStructVector self,
  bHYPRE_Vector x,
  double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_SStructVector_Axpy(
  bHYPRE_SStructVector self,
  double a,
  bHYPRE_Vector x);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_SStructVector_SetCommunicator(
  bHYPRE_SStructVector self,
  void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_SStructVector_Initialize(
  bHYPRE_SStructVector self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_SStructVector_Assemble(
  bHYPRE_SStructVector self);

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
bHYPRE_SStructVector_GetObject(
  bHYPRE_SStructVector self,
  SIDL_BaseInterface* A);

/**
 * Set the vector grid.
 * 
 */
int32_t
bHYPRE_SStructVector_SetGrid(
  bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_SetValues(
  bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_SetBoxValues(
  bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_AddToValues(
  bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_AddToBoxValues(
  bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Gather(
  bHYPRE_SStructVector self);

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
bHYPRE_SStructVector_GetValues(
  bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_GetBoxValues(
  bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_SetComplex(
  bHYPRE_SStructVector self);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructVector_Print(
  bHYPRE_SStructVector self,
  const char* filename,
  int32_t all);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_SStructVector
bHYPRE_SStructVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructVector__cast2(
  void* obj,
  const char* type);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_createCol(int32_t        dimen,
                                      const int32_t lower[],
                                      const int32_t upper[]);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_createRow(int32_t        dimen,
                                      const int32_t lower[],
                                      const int32_t upper[]);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_create1d(int32_t len);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_borrow(bHYPRE_SStructVector*firstElement,
                                   int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_smartCopy(struct bHYPRE_SStructVector__array 
  *array);

void
bHYPRE_SStructVector__array_addRef(struct bHYPRE_SStructVector__array* array);

void
bHYPRE_SStructVector__array_deleteRef(struct bHYPRE_SStructVector__array* 
  array);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get1(const struct bHYPRE_SStructVector__array* 
  array,
                                 const int32_t i1);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get2(const struct bHYPRE_SStructVector__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get3(const struct bHYPRE_SStructVector__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get4(const struct bHYPRE_SStructVector__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 const int32_t i4);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get(const struct bHYPRE_SStructVector__array* array,
                                const int32_t indices[]);

void
bHYPRE_SStructVector__array_set1(struct bHYPRE_SStructVector__array* array,
                                 const int32_t i1,
                                 bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set2(struct bHYPRE_SStructVector__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set3(struct bHYPRE_SStructVector__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set4(struct bHYPRE_SStructVector__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 const int32_t i4,
                                 bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set(struct bHYPRE_SStructVector__array* array,
                                const int32_t indices[],
                                bHYPRE_SStructVector const value);

int32_t
bHYPRE_SStructVector__array_dimen(const struct bHYPRE_SStructVector__array* 
  array);

int32_t
bHYPRE_SStructVector__array_lower(const struct bHYPRE_SStructVector__array* 
  array,
                                  const int32_t ind);

int32_t
bHYPRE_SStructVector__array_upper(const struct bHYPRE_SStructVector__array* 
  array,
                                  const int32_t ind);

int32_t
bHYPRE_SStructVector__array_stride(const struct bHYPRE_SStructVector__array* 
  array,
                                   const int32_t ind);

int
bHYPRE_SStructVector__array_isColumnOrder(const struct 
  bHYPRE_SStructVector__array* array);

int
bHYPRE_SStructVector__array_isRowOrder(const struct 
  bHYPRE_SStructVector__array* array);

void
bHYPRE_SStructVector__array_slice(const struct bHYPRE_SStructVector__array* src,
                                        int32_t        dimen,
                                        const int32_t  numElem[],
                                        const int32_t  *srcStart,
                                        const int32_t  *srcStride,
                                        const int32_t  *newStart);

void
bHYPRE_SStructVector__array_copy(const struct bHYPRE_SStructVector__array* src,
                                       struct bHYPRE_SStructVector__array* 
  dest);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_ensure(struct bHYPRE_SStructVector__array* src,
                                   int32_t dimen,
                                   int     ordering);

#ifdef __cplusplus
}
#endif
#endif
