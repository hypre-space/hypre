/*
 * File:          bHYPRE_Vector.h
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:27 PST
 * Description:   Client-side glue code for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 667
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_Vector_h
#define included_bHYPRE_Vector_h

/**
 * Symbol "bHYPRE.Vector" (version 1.0.0)
 */
struct bHYPRE_Vector__object;
struct bHYPRE_Vector__array;
typedef struct bHYPRE_Vector__object* bHYPRE_Vector;

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
bHYPRE_Vector_addRef(
  bHYPRE_Vector self);

void
bHYPRE_Vector_deleteRef(
  bHYPRE_Vector self);

SIDL_bool
bHYPRE_Vector_isSame(
  bHYPRE_Vector self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_Vector_queryInt(
  bHYPRE_Vector self,
  const char* name);

SIDL_bool
bHYPRE_Vector_isType(
  bHYPRE_Vector self,
  const char* name);

SIDL_ClassInfo
bHYPRE_Vector_getClassInfo(
  bHYPRE_Vector self);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_Vector_Clear(
  bHYPRE_Vector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_Vector_Copy(
  bHYPRE_Vector self,
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
bHYPRE_Vector_Clone(
  bHYPRE_Vector self,
  bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_Vector_Scale(
  bHYPRE_Vector self,
  double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_Vector_Dot(
  bHYPRE_Vector self,
  bHYPRE_Vector x,
  double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_Vector_Axpy(
  bHYPRE_Vector self,
  double a,
  bHYPRE_Vector x);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_Vector
bHYPRE_Vector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_Vector__cast2(
  void* obj,
  const char* type);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_createCol(int32_t        dimen,
                               const int32_t lower[],
                               const int32_t upper[]);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_createRow(int32_t        dimen,
                               const int32_t lower[],
                               const int32_t upper[]);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create1d(int32_t len);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_borrow(bHYPRE_Vector*firstElement,
                            int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_smartCopy(struct bHYPRE_Vector__array *array);

void
bHYPRE_Vector__array_addRef(struct bHYPRE_Vector__array* array);

void
bHYPRE_Vector__array_deleteRef(struct bHYPRE_Vector__array* array);

bHYPRE_Vector
bHYPRE_Vector__array_get1(const struct bHYPRE_Vector__array* array,
                          const int32_t i1);

bHYPRE_Vector
bHYPRE_Vector__array_get2(const struct bHYPRE_Vector__array* array,
                          const int32_t i1,
                          const int32_t i2);

bHYPRE_Vector
bHYPRE_Vector__array_get3(const struct bHYPRE_Vector__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          const int32_t i3);

bHYPRE_Vector
bHYPRE_Vector__array_get4(const struct bHYPRE_Vector__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          const int32_t i3,
                          const int32_t i4);

bHYPRE_Vector
bHYPRE_Vector__array_get(const struct bHYPRE_Vector__array* array,
                         const int32_t indices[]);

void
bHYPRE_Vector__array_set1(struct bHYPRE_Vector__array* array,
                          const int32_t i1,
                          bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set2(struct bHYPRE_Vector__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set3(struct bHYPRE_Vector__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          const int32_t i3,
                          bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set4(struct bHYPRE_Vector__array* array,
                          const int32_t i1,
                          const int32_t i2,
                          const int32_t i3,
                          const int32_t i4,
                          bHYPRE_Vector const value);

void
bHYPRE_Vector__array_set(struct bHYPRE_Vector__array* array,
                         const int32_t indices[],
                         bHYPRE_Vector const value);

int32_t
bHYPRE_Vector__array_dimen(const struct bHYPRE_Vector__array* array);

int32_t
bHYPRE_Vector__array_lower(const struct bHYPRE_Vector__array* array,
                           const int32_t ind);

int32_t
bHYPRE_Vector__array_upper(const struct bHYPRE_Vector__array* array,
                           const int32_t ind);

int32_t
bHYPRE_Vector__array_stride(const struct bHYPRE_Vector__array* array,
                            const int32_t ind);

int
bHYPRE_Vector__array_isColumnOrder(const struct bHYPRE_Vector__array* array);

int
bHYPRE_Vector__array_isRowOrder(const struct bHYPRE_Vector__array* array);

void
bHYPRE_Vector__array_slice(const struct bHYPRE_Vector__array* src,
                                 int32_t        dimen,
                                 const int32_t  numElem[],
                                 const int32_t  *srcStart,
                                 const int32_t  *srcStride,
                                 const int32_t  *newStart);

void
bHYPRE_Vector__array_copy(const struct bHYPRE_Vector__array* src,
                                struct bHYPRE_Vector__array* dest);

struct bHYPRE_Vector__array*
bHYPRE_Vector__array_ensure(struct bHYPRE_Vector__array* src,
                            int32_t dimen,
                            int     ordering);

#ifdef __cplusplus
}
#endif
#endif
