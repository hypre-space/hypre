/*
 * File:          bHYPRE_SStructGrid.h
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:39 PST
 * Description:   Client-side glue code for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 904
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructGrid_h
#define included_bHYPRE_SStructGrid_h

/**
 * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
 * 
 * The semi-structured grid class.
 * 
 */
struct bHYPRE_SStructGrid__object;
struct bHYPRE_SStructGrid__array;
typedef struct bHYPRE_SStructGrid__object* bHYPRE_SStructGrid;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructVariable_h
#include "bHYPRE_SStructVariable.h"
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

/**
 * Constructor function for the class.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__create(void);

void
bHYPRE_SStructGrid_addRef(
  /*in*/ bHYPRE_SStructGrid self);

void
bHYPRE_SStructGrid_deleteRef(
  /*in*/ bHYPRE_SStructGrid self);

sidl_bool
bHYPRE_SStructGrid_isSame(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructGrid_queryInt(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ const char* name);

sidl_bool
bHYPRE_SStructGrid_isType(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ const char* name);

sidl_ClassInfo
bHYPRE_SStructGrid_getClassInfo(
  /*in*/ bHYPRE_SStructGrid self);

/**
 * Set the number of dimensions {\tt ndim} and the number of
 * structured parts {\tt nparts}.
 * 
 */
int32_t
bHYPRE_SStructGrid_SetNumDimParts(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ int32_t ndim,
  /*in*/ int32_t nparts);

/**
 * Set the extents for a box on a structured part of the grid.
 * 
 */
int32_t
bHYPRE_SStructGrid_SetExtents(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper);

/**
 * Describe the variables that live on a structured part of the
 * grid.
 * 
 */
int32_t
bHYPRE_SStructGrid_SetVariable(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ int32_t part,
  /*in*/ int32_t var,
  /*in*/ enum bHYPRE_SStructVariable__enum vartype);

/**
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 * 
 */
int32_t
bHYPRE_SStructGrid_AddVariable(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* index,
  /*in*/ int32_t var,
  /*in*/ enum bHYPRE_SStructVariable__enum vartype);

/**
 * Describe how regions just outside of a part relate to other
 * parts.  This is done a box at a time.
 * 
 * The indexes {\tt ilower} and {\tt iupper} map directly to the
 * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
 * it is required that indexes increase from {\tt ilower} to
 * {\tt iupper}, indexes may increase and/or decrease from {\tt
 * nbor\_ilower} to {\tt nbor\_iupper}.
 * 
 * The {\tt index\_map} describes the mapping of indexes 0, 1,
 * and 2 on part {\tt part} to the corresponding indexes on part
 * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
 * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
 * and 0 on part {\tt nbor\_part}, respectively.
 * 
 * NOTE: All parts related to each other via this routine must
 * have an identical list of variables and variable types.  For
 * example, if part 0 has only two variables on it, a cell
 * centered variable and a node centered variable, and we
 * declare part 1 to be a neighbor of part 0, then part 1 must
 * also have only two variables on it, and they must be of type
 * cell and node.
 * 
 */
int32_t
bHYPRE_SStructGrid_SetNeighborBox(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* ilower,
  /*in*/ struct sidl_int__array* iupper,
  /*in*/ int32_t nbor_part,
  /*in*/ struct sidl_int__array* nbor_ilower,
  /*in*/ struct sidl_int__array* nbor_iupper,
  /*in*/ struct sidl_int__array* index_map);

/**
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables
 * minus one.  Each process owns some unique consecutive range
 * of variables, defined by {\tt ilower} and {\tt iupper}.
 * 
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 * 
 */
int32_t
bHYPRE_SStructGrid_AddUnstructuredPart(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ int32_t ilower,
  /*in*/ int32_t iupper);

/**
 * (Optional) Set periodic for a particular part.
 * 
 */
int32_t
bHYPRE_SStructGrid_SetPeriodic(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ int32_t part,
  /*in*/ struct sidl_int__array* periodic);

/**
 * Setting ghost in the sgrids.
 * 
 */
int32_t
bHYPRE_SStructGrid_SetNumGhost(
  /*in*/ bHYPRE_SStructGrid self,
  /*in*/ struct sidl_int__array* num_ghost);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructGrid__cast2(
  void* obj,
  const char* type);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create1d(int32_t len);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructGrid* data);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_borrow(
  bHYPRE_SStructGrid* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_smartCopy(
  struct bHYPRE_SStructGrid__array *array);

void
bHYPRE_SStructGrid__array_addRef(
  struct bHYPRE_SStructGrid__array* array);

void
bHYPRE_SStructGrid__array_deleteRef(
  struct bHYPRE_SStructGrid__array* array);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get1(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get2(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get3(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get4(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get5(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get6(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get7(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructGrid__array_set1(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set2(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set3(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set4(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set5(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set6(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set7(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructGrid const value);

void
bHYPRE_SStructGrid__array_set(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t indices[],
  bHYPRE_SStructGrid const value);

int32_t
bHYPRE_SStructGrid__array_dimen(
  const struct bHYPRE_SStructGrid__array* array);

int32_t
bHYPRE_SStructGrid__array_lower(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGrid__array_upper(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGrid__array_length(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGrid__array_stride(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind);

int
bHYPRE_SStructGrid__array_isColumnOrder(
  const struct bHYPRE_SStructGrid__array* array);

int
bHYPRE_SStructGrid__array_isRowOrder(
  const struct bHYPRE_SStructGrid__array* array);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_slice(
  struct bHYPRE_SStructGrid__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructGrid__array_copy(
  const struct bHYPRE_SStructGrid__array* src,
  struct bHYPRE_SStructGrid__array* dest);

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_ensure(
  struct bHYPRE_SStructGrid__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
