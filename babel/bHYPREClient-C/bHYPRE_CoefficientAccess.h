/*
 * File:          bHYPRE_CoefficientAccess.h
 * Symbol:        bHYPRE.CoefficientAccess-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:42 PST
 * Description:   Client-side glue code for bHYPRE.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 766
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_CoefficientAccess_h
#define included_bHYPRE_CoefficientAccess_h

/**
 * Symbol "bHYPRE.CoefficientAccess" (version 1.0.0)
 */
struct bHYPRE_CoefficientAccess__object;
struct bHYPRE_CoefficientAccess__array;
typedef struct bHYPRE_CoefficientAccess__object* bHYPRE_CoefficientAccess;

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

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_CoefficientAccess_addRef(
  bHYPRE_CoefficientAccess self);

void
bHYPRE_CoefficientAccess_deleteRef(
  bHYPRE_CoefficientAccess self);

SIDL_bool
bHYPRE_CoefficientAccess_isSame(
  bHYPRE_CoefficientAccess self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_CoefficientAccess_queryInt(
  bHYPRE_CoefficientAccess self,
  const char* name);

SIDL_bool
bHYPRE_CoefficientAccess_isType(
  bHYPRE_CoefficientAccess self,
  const char* name);

SIDL_ClassInfo
bHYPRE_CoefficientAccess_getClassInfo(
  bHYPRE_CoefficientAccess self);

/**
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */
int32_t
bHYPRE_CoefficientAccess_GetRow(
  bHYPRE_CoefficientAccess self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_CoefficientAccess__cast2(
  void* obj,
  const char* type);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_createCol(int32_t        dimen,
                                          const int32_t lower[],
                                          const int32_t upper[]);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_createRow(int32_t        dimen,
                                          const int32_t lower[],
                                          const int32_t upper[]);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_create1d(int32_t len);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_borrow(bHYPRE_CoefficientAccess*firstElement,
                                       int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_smartCopy(struct 
  bHYPRE_CoefficientAccess__array *array);

void
bHYPRE_CoefficientAccess__array_addRef(struct bHYPRE_CoefficientAccess__array* 
  array);

void
bHYPRE_CoefficientAccess__array_deleteRef(struct 
  bHYPRE_CoefficientAccess__array* array);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get1(const struct 
  bHYPRE_CoefficientAccess__array* array,
                                     const int32_t i1);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get2(const struct 
  bHYPRE_CoefficientAccess__array* array,
                                     const int32_t i1,
                                     const int32_t i2);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get3(const struct 
  bHYPRE_CoefficientAccess__array* array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     const int32_t i3);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get4(const struct 
  bHYPRE_CoefficientAccess__array* array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     const int32_t i3,
                                     const int32_t i4);

bHYPRE_CoefficientAccess
bHYPRE_CoefficientAccess__array_get(const struct 
  bHYPRE_CoefficientAccess__array* array,
                                    const int32_t indices[]);

void
bHYPRE_CoefficientAccess__array_set1(struct bHYPRE_CoefficientAccess__array* 
  array,
                                     const int32_t i1,
                                     bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set2(struct bHYPRE_CoefficientAccess__array* 
  array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set3(struct bHYPRE_CoefficientAccess__array* 
  array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     const int32_t i3,
                                     bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set4(struct bHYPRE_CoefficientAccess__array* 
  array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     const int32_t i3,
                                     const int32_t i4,
                                     bHYPRE_CoefficientAccess const value);

void
bHYPRE_CoefficientAccess__array_set(struct bHYPRE_CoefficientAccess__array* 
  array,
                                    const int32_t indices[],
                                    bHYPRE_CoefficientAccess const value);

int32_t
bHYPRE_CoefficientAccess__array_dimen(const struct 
  bHYPRE_CoefficientAccess__array* array);

int32_t
bHYPRE_CoefficientAccess__array_lower(const struct 
  bHYPRE_CoefficientAccess__array* array,
                                      const int32_t ind);

int32_t
bHYPRE_CoefficientAccess__array_upper(const struct 
  bHYPRE_CoefficientAccess__array* array,
                                      const int32_t ind);

int32_t
bHYPRE_CoefficientAccess__array_stride(const struct 
  bHYPRE_CoefficientAccess__array* array,
                                       const int32_t ind);

int
bHYPRE_CoefficientAccess__array_isColumnOrder(const struct 
  bHYPRE_CoefficientAccess__array* array);

int
bHYPRE_CoefficientAccess__array_isRowOrder(const struct 
  bHYPRE_CoefficientAccess__array* array);

void
bHYPRE_CoefficientAccess__array_slice(const struct 
  bHYPRE_CoefficientAccess__array* src,
                                            int32_t        dimen,
                                            const int32_t  numElem[],
                                            const int32_t  *srcStart,
                                            const int32_t  *srcStride,
                                            const int32_t  *newStart);

void
bHYPRE_CoefficientAccess__array_copy(const struct 
  bHYPRE_CoefficientAccess__array* src,
                                           struct 
  bHYPRE_CoefficientAccess__array* dest);

struct bHYPRE_CoefficientAccess__array*
bHYPRE_CoefficientAccess__array_ensure(struct bHYPRE_CoefficientAccess__array* 
  src,
                                       int32_t dimen,
                                       int     ordering);

#ifdef __cplusplus
}
#endif
#endif
