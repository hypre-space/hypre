/*
 * File:          bHYPRE_SStructStencil.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:29 PST
 * Description:   Client-side glue code for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1001
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructStencil_h
#define included_bHYPRE_SStructStencil_h

/**
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 * 
 */
struct bHYPRE_SStructStencil__object;
struct bHYPRE_SStructStencil__array;
typedef struct bHYPRE_SStructStencil__object* bHYPRE_SStructStencil;

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

/**
 * Constructor function for the class.
 */
bHYPRE_SStructStencil
bHYPRE_SStructStencil__create(void);

void
bHYPRE_SStructStencil_addRef(
  bHYPRE_SStructStencil self);

void
bHYPRE_SStructStencil_deleteRef(
  bHYPRE_SStructStencil self);

SIDL_bool
bHYPRE_SStructStencil_isSame(
  bHYPRE_SStructStencil self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_SStructStencil_queryInt(
  bHYPRE_SStructStencil self,
  const char* name);

SIDL_bool
bHYPRE_SStructStencil_isType(
  bHYPRE_SStructStencil self,
  const char* name);

SIDL_ClassInfo
bHYPRE_SStructStencil_getClassInfo(
  bHYPRE_SStructStencil self);

/**
 * Set the number of spatial dimensions and stencil entries.
 * 
 */
int32_t
bHYPRE_SStructStencil_SetNumDimSize(
  bHYPRE_SStructStencil self,
  int32_t ndim,
  int32_t size);

/**
 * Set a stencil entry.
 * 
 */
int32_t
bHYPRE_SStructStencil_SetEntry(
  bHYPRE_SStructStencil self,
  int32_t entry,
  struct SIDL_int__array* offset,
  int32_t var);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_SStructStencil
bHYPRE_SStructStencil__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructStencil__cast2(
  void* obj,
  const char* type);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_createCol(int32_t        dimen,
                                       const int32_t lower[],
                                       const int32_t upper[]);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_createRow(int32_t        dimen,
                                       const int32_t lower[],
                                       const int32_t upper[]);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create1d(int32_t len);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_borrow(bHYPRE_SStructStencil*firstElement,
                                    int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_smartCopy(struct bHYPRE_SStructStencil__array 
  *array);

void
bHYPRE_SStructStencil__array_addRef(struct bHYPRE_SStructStencil__array* array);

void
bHYPRE_SStructStencil__array_deleteRef(struct bHYPRE_SStructStencil__array* 
  array);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get1(const struct bHYPRE_SStructStencil__array* 
  array,
                                  const int32_t i1);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get2(const struct bHYPRE_SStructStencil__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get3(const struct bHYPRE_SStructStencil__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get4(const struct bHYPRE_SStructStencil__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  const int32_t i4);

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get(const struct bHYPRE_SStructStencil__array* 
  array,
                                 const int32_t indices[]);

void
bHYPRE_SStructStencil__array_set1(struct bHYPRE_SStructStencil__array* array,
                                  const int32_t i1,
                                  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set2(struct bHYPRE_SStructStencil__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set3(struct bHYPRE_SStructStencil__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set4(struct bHYPRE_SStructStencil__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  const int32_t i4,
                                  bHYPRE_SStructStencil const value);

void
bHYPRE_SStructStencil__array_set(struct bHYPRE_SStructStencil__array* array,
                                 const int32_t indices[],
                                 bHYPRE_SStructStencil const value);

int32_t
bHYPRE_SStructStencil__array_dimen(const struct bHYPRE_SStructStencil__array* 
  array);

int32_t
bHYPRE_SStructStencil__array_lower(const struct bHYPRE_SStructStencil__array* 
  array,
                                   const int32_t ind);

int32_t
bHYPRE_SStructStencil__array_upper(const struct bHYPRE_SStructStencil__array* 
  array,
                                   const int32_t ind);

int32_t
bHYPRE_SStructStencil__array_stride(const struct bHYPRE_SStructStencil__array* 
  array,
                                    const int32_t ind);

int
bHYPRE_SStructStencil__array_isColumnOrder(const struct 
  bHYPRE_SStructStencil__array* array);

int
bHYPRE_SStructStencil__array_isRowOrder(const struct 
  bHYPRE_SStructStencil__array* array);

void
bHYPRE_SStructStencil__array_slice(const struct bHYPRE_SStructStencil__array* 
  src,
                                         int32_t        dimen,
                                         const int32_t  numElem[],
                                         const int32_t  *srcStart,
                                         const int32_t  *srcStride,
                                         const int32_t  *newStart);

void
bHYPRE_SStructStencil__array_copy(const struct bHYPRE_SStructStencil__array* 
  src,
                                        struct bHYPRE_SStructStencil__array* 
  dest);

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_ensure(struct bHYPRE_SStructStencil__array* src,
                                    int32_t dimen,
                                    int     ordering);

#ifdef __cplusplus
}
#endif
#endif
