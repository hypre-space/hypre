/*
 * File:          bHYPRE_SStructBuildVector.h
 * Symbol:        bHYPRE.SStructBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:41 PST
 * Description:   Client-side glue code for bHYPRE.SStructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 418
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructBuildVector_h
#define included_bHYPRE_SStructBuildVector_h

/**
 * Symbol "bHYPRE.SStructBuildVector" (version 1.0.0)
 */
struct bHYPRE_SStructBuildVector__object;
struct bHYPRE_SStructBuildVector__array;
typedef struct bHYPRE_SStructBuildVector__object* bHYPRE_SStructBuildVector;

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

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructBuildVector_addRef(
  bHYPRE_SStructBuildVector self);

void
bHYPRE_SStructBuildVector_deleteRef(
  bHYPRE_SStructBuildVector self);

SIDL_bool
bHYPRE_SStructBuildVector_isSame(
  bHYPRE_SStructBuildVector self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_SStructBuildVector_queryInt(
  bHYPRE_SStructBuildVector self,
  const char* name);

SIDL_bool
bHYPRE_SStructBuildVector_isType(
  bHYPRE_SStructBuildVector self,
  const char* name);

SIDL_ClassInfo
bHYPRE_SStructBuildVector_getClassInfo(
  bHYPRE_SStructBuildVector self);

int32_t
bHYPRE_SStructBuildVector_SetCommunicator(
  bHYPRE_SStructBuildVector self,
  void* mpi_comm);

int32_t
bHYPRE_SStructBuildVector_Initialize(
  bHYPRE_SStructBuildVector self);

int32_t
bHYPRE_SStructBuildVector_Assemble(
  bHYPRE_SStructBuildVector self);

int32_t
bHYPRE_SStructBuildVector_GetObject(
  bHYPRE_SStructBuildVector self,
  SIDL_BaseInterface* A);

/**
 * Set the vector grid.
 * 
 */
int32_t
bHYPRE_SStructBuildVector_SetGrid(
  bHYPRE_SStructBuildVector self,
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
bHYPRE_SStructBuildVector_SetValues(
  bHYPRE_SStructBuildVector self,
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
bHYPRE_SStructBuildVector_SetBoxValues(
  bHYPRE_SStructBuildVector self,
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
bHYPRE_SStructBuildVector_AddToValues(
  bHYPRE_SStructBuildVector self,
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
bHYPRE_SStructBuildVector_AddToBoxValues(
  bHYPRE_SStructBuildVector self,
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
bHYPRE_SStructBuildVector_Gather(
  bHYPRE_SStructBuildVector self);

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
bHYPRE_SStructBuildVector_GetValues(
  bHYPRE_SStructBuildVector self,
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
bHYPRE_SStructBuildVector_GetBoxValues(
  bHYPRE_SStructBuildVector self,
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
bHYPRE_SStructBuildVector_SetComplex(
  bHYPRE_SStructBuildVector self);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructBuildVector_Print(
  bHYPRE_SStructBuildVector self,
  const char* filename,
  int32_t all);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_SStructBuildVector
bHYPRE_SStructBuildVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructBuildVector__cast2(
  void* obj,
  const char* type);

struct bHYPRE_SStructBuildVector__array*
bHYPRE_SStructBuildVector__array_createCol(int32_t        dimen,
                                           const int32_t lower[],
                                           const int32_t upper[]);

struct bHYPRE_SStructBuildVector__array*
bHYPRE_SStructBuildVector__array_createRow(int32_t        dimen,
                                           const int32_t lower[],
                                           const int32_t upper[]);

struct bHYPRE_SStructBuildVector__array*
bHYPRE_SStructBuildVector__array_create1d(int32_t len);

struct bHYPRE_SStructBuildVector__array*
bHYPRE_SStructBuildVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructBuildVector__array*
bHYPRE_SStructBuildVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructBuildVector__array*
bHYPRE_SStructBuildVector__array_borrow(bHYPRE_SStructBuildVector*firstElement,
                                        int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_SStructBuildVector__array*
bHYPRE_SStructBuildVector__array_smartCopy(struct 
  bHYPRE_SStructBuildVector__array *array);

void
bHYPRE_SStructBuildVector__array_addRef(struct 
  bHYPRE_SStructBuildVector__array* array);

void
bHYPRE_SStructBuildVector__array_deleteRef(struct 
  bHYPRE_SStructBuildVector__array* array);

bHYPRE_SStructBuildVector
bHYPRE_SStructBuildVector__array_get1(const struct 
  bHYPRE_SStructBuildVector__array* array,
                                      const int32_t i1);

bHYPRE_SStructBuildVector
bHYPRE_SStructBuildVector__array_get2(const struct 
  bHYPRE_SStructBuildVector__array* array,
                                      const int32_t i1,
                                      const int32_t i2);

bHYPRE_SStructBuildVector
bHYPRE_SStructBuildVector__array_get3(const struct 
  bHYPRE_SStructBuildVector__array* array,
                                      const int32_t i1,
                                      const int32_t i2,
                                      const int32_t i3);

bHYPRE_SStructBuildVector
bHYPRE_SStructBuildVector__array_get4(const struct 
  bHYPRE_SStructBuildVector__array* array,
                                      const int32_t i1,
                                      const int32_t i2,
                                      const int32_t i3,
                                      const int32_t i4);

bHYPRE_SStructBuildVector
bHYPRE_SStructBuildVector__array_get(const struct 
  bHYPRE_SStructBuildVector__array* array,
                                     const int32_t indices[]);

void
bHYPRE_SStructBuildVector__array_set1(struct bHYPRE_SStructBuildVector__array* 
  array,
                                      const int32_t i1,
                                      bHYPRE_SStructBuildVector const value);

void
bHYPRE_SStructBuildVector__array_set2(struct bHYPRE_SStructBuildVector__array* 
  array,
                                      const int32_t i1,
                                      const int32_t i2,
                                      bHYPRE_SStructBuildVector const value);

void
bHYPRE_SStructBuildVector__array_set3(struct bHYPRE_SStructBuildVector__array* 
  array,
                                      const int32_t i1,
                                      const int32_t i2,
                                      const int32_t i3,
                                      bHYPRE_SStructBuildVector const value);

void
bHYPRE_SStructBuildVector__array_set4(struct bHYPRE_SStructBuildVector__array* 
  array,
                                      const int32_t i1,
                                      const int32_t i2,
                                      const int32_t i3,
                                      const int32_t i4,
                                      bHYPRE_SStructBuildVector const value);

void
bHYPRE_SStructBuildVector__array_set(struct bHYPRE_SStructBuildVector__array* 
  array,
                                     const int32_t indices[],
                                     bHYPRE_SStructBuildVector const value);

int32_t
bHYPRE_SStructBuildVector__array_dimen(const struct 
  bHYPRE_SStructBuildVector__array* array);

int32_t
bHYPRE_SStructBuildVector__array_lower(const struct 
  bHYPRE_SStructBuildVector__array* array,
                                       const int32_t ind);

int32_t
bHYPRE_SStructBuildVector__array_upper(const struct 
  bHYPRE_SStructBuildVector__array* array,
                                       const int32_t ind);

int32_t
bHYPRE_SStructBuildVector__array_stride(const struct 
  bHYPRE_SStructBuildVector__array* array,
                                        const int32_t ind);

int
bHYPRE_SStructBuildVector__array_isColumnOrder(const struct 
  bHYPRE_SStructBuildVector__array* array);

int
bHYPRE_SStructBuildVector__array_isRowOrder(const struct 
  bHYPRE_SStructBuildVector__array* array);

void
bHYPRE_SStructBuildVector__array_slice(const struct 
  bHYPRE_SStructBuildVector__array* src,
                                             int32_t        dimen,
                                             const int32_t  numElem[],
                                             const int32_t  *srcStart,
                                             const int32_t  *srcStride,
                                             const int32_t  *newStart);

void
bHYPRE_SStructBuildVector__array_copy(const struct 
  bHYPRE_SStructBuildVector__array* src,
                                            struct 
  bHYPRE_SStructBuildVector__array* dest);

struct bHYPRE_SStructBuildVector__array*
bHYPRE_SStructBuildVector__array_ensure(struct 
  bHYPRE_SStructBuildVector__array* src,
                                        int32_t dimen,
                                        int     ordering);

#ifdef __cplusplus
}
#endif
#endif
