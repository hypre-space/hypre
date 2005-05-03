/*
 * File:          bHYPRE_SStructBuildMatrix.h
 * Symbol:        bHYPRE.SStructBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * Description:   Client-side glue code for bHYPRE.SStructBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 */

#ifndef included_bHYPRE_SStructBuildMatrix_h
#define included_bHYPRE_SStructBuildMatrix_h

/**
 * Symbol "bHYPRE.SStructBuildMatrix" (version 1.0.0)
 */
struct bHYPRE_SStructBuildMatrix__object;
struct bHYPRE_SStructBuildMatrix__array;
typedef struct bHYPRE_SStructBuildMatrix__object* bHYPRE_SStructBuildMatrix;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
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
bHYPRE_SStructBuildMatrix_addRef(
  /*in*/ bHYPRE_SStructBuildMatrix self);

void
bHYPRE_SStructBuildMatrix_deleteRef(
  /*in*/ bHYPRE_SStructBuildMatrix self);

sidl_bool
bHYPRE_SStructBuildMatrix_isSame(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructBuildMatrix_queryInt(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ const char* name);

sidl_bool
bHYPRE_SStructBuildMatrix_isType(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ const char* name);

sidl_ClassInfo
bHYPRE_SStructBuildMatrix_getClassInfo(
  /*in*/ bHYPRE_SStructBuildMatrix self);

int32_t
bHYPRE_SStructBuildMatrix_SetCommunicator(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ void* mpi_comm);

int32_t
bHYPRE_SStructBuildMatrix_Initialize(
  /*in*/ bHYPRE_SStructBuildMatrix self);

int32_t
bHYPRE_SStructBuildMatrix_Assemble(
  /*in*/ bHYPRE_SStructBuildMatrix self);

int32_t
bHYPRE_SStructBuildMatrix_GetObject(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*out*/ sidl_BaseInterface* A);

/**
 * Set the matrix graph.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_SetGraph(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ bHYPRE_SStructGraph graph);

/**
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_SetValues(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values);

/**
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_SetBoxValues(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values);

/**
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_AddToValues(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values);

/**
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_AddToBoxValues(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t var,
  /*in*/ int32_t nentries,
  /*in*/ struct sidl_int__array* entries,
  /*in*/ struct sidl_double__array* values);

/**
 * Define symmetry properties for the stencil entries in the
 * matrix.  The boolean argument {\tt symmetric} is applied to
 * stencil entries on part {\tt part} that couple variable {\tt
 * var} to variable {\tt to\_var}.  A value of -1 may be used
 * for {\tt part}, {\tt var}, or {\tt to\_var} to specify
 * ``all''.  For example, if {\tt part} and {\tt to\_var} are
 * set to -1, then the boolean is applied to stencil entries on
 * all parts that couple variable {\tt var} to all other
 * variables.
 * 
 * By default, matrices are assumed to be nonsymmetric.
 * Significant storage savings can be made if the matrix is
 * symmetric.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_SetSymmetric(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ int32_t part,
  /*in*/ int32_t var,
  /*in*/ int32_t to_var,
  /*in*/ int32_t symmetric);

/**
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_SetNSSymmetric(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ int32_t symmetric);

/**
 * Set the matrix to be complex.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_SetComplex(
  /*in*/ bHYPRE_SStructBuildMatrix self);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructBuildMatrix_Print(
  /*in*/ bHYPRE_SStructBuildMatrix self,
  /*in*/ const char* filename,
  /*in*/ int32_t all);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructBuildMatrix__cast2(
  void* obj,
  const char* type);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_create1d(int32_t len);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructBuildMatrix* data);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_borrow(
  bHYPRE_SStructBuildMatrix* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_smartCopy(
  struct bHYPRE_SStructBuildMatrix__array *array);

void
bHYPRE_SStructBuildMatrix__array_addRef(
  struct bHYPRE_SStructBuildMatrix__array* array);

void
bHYPRE_SStructBuildMatrix__array_deleteRef(
  struct bHYPRE_SStructBuildMatrix__array* array);

bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__array_get1(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1);

bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__array_get2(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__array_get3(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__array_get4(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__array_get5(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__array_get6(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__array_get7(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructBuildMatrix
bHYPRE_SStructBuildMatrix__array_get(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructBuildMatrix__array_set1(
  struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  bHYPRE_SStructBuildMatrix const value);

void
bHYPRE_SStructBuildMatrix__array_set2(
  struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructBuildMatrix const value);

void
bHYPRE_SStructBuildMatrix__array_set3(
  struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructBuildMatrix const value);

void
bHYPRE_SStructBuildMatrix__array_set4(
  struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructBuildMatrix const value);

void
bHYPRE_SStructBuildMatrix__array_set5(
  struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructBuildMatrix const value);

void
bHYPRE_SStructBuildMatrix__array_set6(
  struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructBuildMatrix const value);

void
bHYPRE_SStructBuildMatrix__array_set7(
  struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructBuildMatrix const value);

void
bHYPRE_SStructBuildMatrix__array_set(
  struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t indices[],
  bHYPRE_SStructBuildMatrix const value);

int32_t
bHYPRE_SStructBuildMatrix__array_dimen(
  const struct bHYPRE_SStructBuildMatrix__array* array);

int32_t
bHYPRE_SStructBuildMatrix__array_lower(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructBuildMatrix__array_upper(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructBuildMatrix__array_length(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructBuildMatrix__array_stride(
  const struct bHYPRE_SStructBuildMatrix__array* array,
  const int32_t ind);

int
bHYPRE_SStructBuildMatrix__array_isColumnOrder(
  const struct bHYPRE_SStructBuildMatrix__array* array);

int
bHYPRE_SStructBuildMatrix__array_isRowOrder(
  const struct bHYPRE_SStructBuildMatrix__array* array);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_slice(
  struct bHYPRE_SStructBuildMatrix__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructBuildMatrix__array_copy(
  const struct bHYPRE_SStructBuildMatrix__array* src,
  struct bHYPRE_SStructBuildMatrix__array* dest);

struct bHYPRE_SStructBuildMatrix__array*
bHYPRE_SStructBuildMatrix__array_ensure(
  struct bHYPRE_SStructBuildMatrix__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
