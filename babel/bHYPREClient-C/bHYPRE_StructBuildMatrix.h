/*
 * File:          bHYPRE_StructBuildMatrix.h
 * Symbol:        bHYPRE.StructBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:45 PST
 * Generated:     20050317 11:17:48 PST
 * Description:   Client-side glue code for bHYPRE.StructBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 543
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructBuildMatrix_h
#define included_bHYPRE_StructBuildMatrix_h

/**
 * Symbol "bHYPRE.StructBuildMatrix" (version 1.0.0)
 */
struct bHYPRE_StructBuildMatrix__object;
struct bHYPRE_StructBuildMatrix__array;
typedef struct bHYPRE_StructBuildMatrix__object* bHYPRE_StructBuildMatrix;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_bHYPRE_StructStencil_h
#include "bHYPRE_StructStencil.h"
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

void
bHYPRE_StructBuildMatrix_addRef(
  /*in*/ bHYPRE_StructBuildMatrix self);

void
bHYPRE_StructBuildMatrix_deleteRef(
  /*in*/ bHYPRE_StructBuildMatrix self);

sidl_bool
bHYPRE_StructBuildMatrix_isSame(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructBuildMatrix_queryInt(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ const char* name);

sidl_bool
bHYPRE_StructBuildMatrix_isType(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ const char* name);

sidl_ClassInfo
bHYPRE_StructBuildMatrix_getClassInfo(
  /*in*/ bHYPRE_StructBuildMatrix self);

int32_t
bHYPRE_StructBuildMatrix_SetCommunicator(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ void* mpi_comm);

int32_t
bHYPRE_StructBuildMatrix_Initialize(
  /*in*/ bHYPRE_StructBuildMatrix self);

int32_t
bHYPRE_StructBuildMatrix_Assemble(
  /*in*/ bHYPRE_StructBuildMatrix self);

int32_t
bHYPRE_StructBuildMatrix_GetObject(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*out*/ sidl_BaseInterface* A);

/**
 * Method:  SetGrid[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetGrid(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ bHYPRE_StructGrid grid);

/**
 * Method:  SetStencil[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetStencil(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ bHYPRE_StructStencil stencil);

/**
 * Method:  SetValues[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetValues(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t num_stencil_indices,
  /*in*/ struct sidl_int__array* stencil_indices,
  /*in*/ struct sidl_double__array* values);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetBoxValues(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t num_stencil_indices,
  /*in*/ struct sidl_int__array* stencil_indices,
  /*in*/ struct sidl_double__array* values);

/**
 * Method:  SetNumGhost[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetNumGhost(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ struct sidl_int__array* num_ghost);

/**
 * Method:  SetSymmetric[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetSymmetric(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ int32_t symmetric);

/**
 * Method:  SetConstantEntries[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetConstantEntries(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ int32_t num_stencil_constant_points,
  /*in*/ struct sidl_int__array* stencil_constant_points);

/**
 * Method:  SetConstantValues[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetConstantValues(
  /*in*/ bHYPRE_StructBuildMatrix self,
  /*in*/ int32_t num_stencil_indices,
  /*in*/ struct sidl_int__array* stencil_indices,
  /*in*/ struct sidl_double__array* values);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructBuildMatrix__cast2(
  void* obj,
  const char* type);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_create1d(int32_t len);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_create1dInit(
  int32_t len, 
  bHYPRE_StructBuildMatrix* data);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_borrow(
  bHYPRE_StructBuildMatrix* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_smartCopy(
  struct bHYPRE_StructBuildMatrix__array *array);

void
bHYPRE_StructBuildMatrix__array_addRef(
  struct bHYPRE_StructBuildMatrix__array* array);

void
bHYPRE_StructBuildMatrix__array_deleteRef(
  struct bHYPRE_StructBuildMatrix__array* array);

bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__array_get1(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1);

bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__array_get2(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__array_get3(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__array_get4(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__array_get5(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__array_get6(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__array_get7(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__array_get(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t indices[]);

void
bHYPRE_StructBuildMatrix__array_set1(
  struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  bHYPRE_StructBuildMatrix const value);

void
bHYPRE_StructBuildMatrix__array_set2(
  struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructBuildMatrix const value);

void
bHYPRE_StructBuildMatrix__array_set3(
  struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructBuildMatrix const value);

void
bHYPRE_StructBuildMatrix__array_set4(
  struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructBuildMatrix const value);

void
bHYPRE_StructBuildMatrix__array_set5(
  struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructBuildMatrix const value);

void
bHYPRE_StructBuildMatrix__array_set6(
  struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructBuildMatrix const value);

void
bHYPRE_StructBuildMatrix__array_set7(
  struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructBuildMatrix const value);

void
bHYPRE_StructBuildMatrix__array_set(
  struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t indices[],
  bHYPRE_StructBuildMatrix const value);

int32_t
bHYPRE_StructBuildMatrix__array_dimen(
  const struct bHYPRE_StructBuildMatrix__array* array);

int32_t
bHYPRE_StructBuildMatrix__array_lower(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructBuildMatrix__array_upper(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructBuildMatrix__array_length(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructBuildMatrix__array_stride(
  const struct bHYPRE_StructBuildMatrix__array* array,
  const int32_t ind);

int
bHYPRE_StructBuildMatrix__array_isColumnOrder(
  const struct bHYPRE_StructBuildMatrix__array* array);

int
bHYPRE_StructBuildMatrix__array_isRowOrder(
  const struct bHYPRE_StructBuildMatrix__array* array);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_slice(
  struct bHYPRE_StructBuildMatrix__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructBuildMatrix__array_copy(
  const struct bHYPRE_StructBuildMatrix__array* src,
  struct bHYPRE_StructBuildMatrix__array* dest);

struct bHYPRE_StructBuildMatrix__array*
bHYPRE_StructBuildMatrix__array_ensure(
  struct bHYPRE_StructBuildMatrix__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
