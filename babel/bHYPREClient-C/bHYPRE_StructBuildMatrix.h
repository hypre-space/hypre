/*
 * File:          bHYPRE_StructBuildMatrix.h
 * Symbol:        bHYPRE.StructBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.StructBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
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

#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/**
 * RMI connector function for the class.
 */
bHYPRE_StructBuildMatrix
bHYPRE_StructBuildMatrix__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_StructBuildMatrix_addRef(
  /* in */ bHYPRE_StructBuildMatrix self);

void
bHYPRE_StructBuildMatrix_deleteRef(
  /* in */ bHYPRE_StructBuildMatrix self);

sidl_bool
bHYPRE_StructBuildMatrix_isSame(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_StructBuildMatrix_queryInt(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ const char* name);

sidl_bool
bHYPRE_StructBuildMatrix_isType(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_StructBuildMatrix_getClassInfo(
  /* in */ bHYPRE_StructBuildMatrix self);

int32_t
bHYPRE_StructBuildMatrix_SetCommunicator(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ void* mpi_comm);

int32_t
bHYPRE_StructBuildMatrix_Initialize(
  /* in */ bHYPRE_StructBuildMatrix self);

int32_t
bHYPRE_StructBuildMatrix_Assemble(
  /* in */ bHYPRE_StructBuildMatrix self);

int32_t
bHYPRE_StructBuildMatrix_GetObject(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* out */ sidl_BaseInterface* A);

/**
 * Method:  SetGrid[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetGrid(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ bHYPRE_StructGrid grid);

/**
 * Method:  SetStencil[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetStencil(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ bHYPRE_StructStencil stencil);

/**
 * Method:  SetValues[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetValues(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values);

/**
 * Method:  SetBoxValues[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetBoxValues(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues);

/**
 * Method:  SetNumGhost[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetNumGhost(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2);

/**
 * Method:  SetSymmetric[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetSymmetric(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ int32_t symmetric);

/**
 * Method:  SetConstantEntries[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetConstantEntries(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ int32_t num_stencil_constant_points,
  /* in rarray[num_stencil_constant_points] */ int32_t* 
    stencil_constant_points);

/**
 * Method:  SetConstantValues[]
 */
int32_t
bHYPRE_StructBuildMatrix_SetConstantValues(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructBuildMatrix__object*
bHYPRE_StructBuildMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructBuildMatrix__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_StructBuildMatrix__exec(
  /* in */ bHYPRE_StructBuildMatrix self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_StructBuildMatrix__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_StructBuildMatrix__getURL(
  /* in */ bHYPRE_StructBuildMatrix self);
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
