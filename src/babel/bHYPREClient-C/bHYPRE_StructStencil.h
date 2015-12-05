/*
 * File:          bHYPRE_StructStencil.h
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:35 PST
 * Generated:     20030401 14:47:43 PST
 * Description:   Client-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1088
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructStencil_h
#define included_bHYPRE_StructStencil_h

/**
 * Symbol "bHYPRE.StructStencil" (version 1.0.0)
 * 
 * Define a structured stencil for a structured problem
 * description.  More than one implementation is not envisioned,
 * thus the decision has been made to make this a class rather than
 * an interface.
 * 
 */
struct bHYPRE_StructStencil__object;
struct bHYPRE_StructStencil__array;
typedef struct bHYPRE_StructStencil__object* bHYPRE_StructStencil;

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
bHYPRE_StructStencil
bHYPRE_StructStencil__create(void);

void
bHYPRE_StructStencil_addRef(
  bHYPRE_StructStencil self);

void
bHYPRE_StructStencil_deleteRef(
  bHYPRE_StructStencil self);

SIDL_bool
bHYPRE_StructStencil_isSame(
  bHYPRE_StructStencil self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_StructStencil_queryInt(
  bHYPRE_StructStencil self,
  const char* name);

SIDL_bool
bHYPRE_StructStencil_isType(
  bHYPRE_StructStencil self,
  const char* name);

SIDL_ClassInfo
bHYPRE_StructStencil_getClassInfo(
  bHYPRE_StructStencil self);

/**
 * Method:  SetDimension[]
 */
int32_t
bHYPRE_StructStencil_SetDimension(
  bHYPRE_StructStencil self,
  int32_t dim);

/**
 * Method:  SetSize[]
 */
int32_t
bHYPRE_StructStencil_SetSize(
  bHYPRE_StructStencil self,
  int32_t size);

/**
 * Method:  SetElement[]
 */
int32_t
bHYPRE_StructStencil_SetElement(
  bHYPRE_StructStencil self,
  int32_t index,
  struct SIDL_int__array* offset);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_StructStencil
bHYPRE_StructStencil__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructStencil__cast2(
  void* obj,
  const char* type);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_createCol(int32_t        dimen,
                                      const int32_t lower[],
                                      const int32_t upper[]);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_createRow(int32_t        dimen,
                                      const int32_t lower[],
                                      const int32_t upper[]);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create1d(int32_t len);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_borrow(bHYPRE_StructStencil*firstElement,
                                   int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_smartCopy(struct bHYPRE_StructStencil__array 
  *array);

void
bHYPRE_StructStencil__array_addRef(struct bHYPRE_StructStencil__array* array);

void
bHYPRE_StructStencil__array_deleteRef(struct bHYPRE_StructStencil__array* 
  array);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get1(const struct bHYPRE_StructStencil__array* 
  array,
                                 const int32_t i1);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get2(const struct bHYPRE_StructStencil__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get3(const struct bHYPRE_StructStencil__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get4(const struct bHYPRE_StructStencil__array* 
  array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 const int32_t i4);

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get(const struct bHYPRE_StructStencil__array* array,
                                const int32_t indices[]);

void
bHYPRE_StructStencil__array_set1(struct bHYPRE_StructStencil__array* array,
                                 const int32_t i1,
                                 bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set2(struct bHYPRE_StructStencil__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set3(struct bHYPRE_StructStencil__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set4(struct bHYPRE_StructStencil__array* array,
                                 const int32_t i1,
                                 const int32_t i2,
                                 const int32_t i3,
                                 const int32_t i4,
                                 bHYPRE_StructStencil const value);

void
bHYPRE_StructStencil__array_set(struct bHYPRE_StructStencil__array* array,
                                const int32_t indices[],
                                bHYPRE_StructStencil const value);

int32_t
bHYPRE_StructStencil__array_dimen(const struct bHYPRE_StructStencil__array* 
  array);

int32_t
bHYPRE_StructStencil__array_lower(const struct bHYPRE_StructStencil__array* 
  array,
                                  const int32_t ind);

int32_t
bHYPRE_StructStencil__array_upper(const struct bHYPRE_StructStencil__array* 
  array,
                                  const int32_t ind);

int32_t
bHYPRE_StructStencil__array_stride(const struct bHYPRE_StructStencil__array* 
  array,
                                   const int32_t ind);

int
bHYPRE_StructStencil__array_isColumnOrder(const struct 
  bHYPRE_StructStencil__array* array);

int
bHYPRE_StructStencil__array_isRowOrder(const struct 
  bHYPRE_StructStencil__array* array);

void
bHYPRE_StructStencil__array_slice(const struct bHYPRE_StructStencil__array* src,
                                        int32_t        dimen,
                                        const int32_t  numElem[],
                                        const int32_t  *srcStart,
                                        const int32_t  *srcStride,
                                        const int32_t  *newStart);

void
bHYPRE_StructStencil__array_copy(const struct bHYPRE_StructStencil__array* src,
                                       struct bHYPRE_StructStencil__array* 
  dest);

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_ensure(struct bHYPRE_StructStencil__array* src,
                                   int32_t dimen,
                                   int     ordering);

#ifdef __cplusplus
}
#endif
#endif
