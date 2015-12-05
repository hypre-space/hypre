/*
 * File:          bHYPRE_SStructGraph.h
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:34 PST
 * Generated:     20030401 14:47:42 PST
 * Description:   Client-side glue code for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1022
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_SStructGraph_h
#define included_bHYPRE_SStructGraph_h

/**
 * Symbol "bHYPRE.SStructGraph" (version 1.0.0)
 * 
 * The semi-structured grid graph class.
 * 
 */
struct bHYPRE_SStructGraph__object;
struct bHYPRE_SStructGraph__array;
typedef struct bHYPRE_SStructGraph__object* bHYPRE_SStructGraph;

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
#ifndef included_bHYPRE_SStructStencil_h
#include "bHYPRE_SStructStencil.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
bHYPRE_SStructGraph
bHYPRE_SStructGraph__create(void);

void
bHYPRE_SStructGraph_addRef(
  bHYPRE_SStructGraph self);

void
bHYPRE_SStructGraph_deleteRef(
  bHYPRE_SStructGraph self);

SIDL_bool
bHYPRE_SStructGraph_isSame(
  bHYPRE_SStructGraph self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_SStructGraph_queryInt(
  bHYPRE_SStructGraph self,
  const char* name);

SIDL_bool
bHYPRE_SStructGraph_isType(
  bHYPRE_SStructGraph self,
  const char* name);

SIDL_ClassInfo
bHYPRE_SStructGraph_getClassInfo(
  bHYPRE_SStructGraph self);

/**
 * Set the grid.
 * 
 */
int32_t
bHYPRE_SStructGraph_SetGrid(
  bHYPRE_SStructGraph self,
  bHYPRE_SStructGrid grid);

/**
 * Set the stencil for a variable on a structured part of the
 * grid.
 * 
 */
int32_t
bHYPRE_SStructGraph_SetStencil(
  bHYPRE_SStructGraph self,
  int32_t part,
  int32_t var,
  bHYPRE_SStructStencil stencil);

/**
 * Add a non-stencil graph entry at a particular index.  This
 * graph entry is appended to the existing graph entries, and is
 * referenced as such.
 * 
 * NOTE: Users are required to set graph entries on all
 * processes that own the associated variables.  This means that
 * some data will be multiply defined.
 * 
 */
int32_t
bHYPRE_SStructGraph_AddEntries(
  bHYPRE_SStructGraph self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  int32_t to_part,
  struct SIDL_int__array* to_index,
  int32_t to_var);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_SStructGraph
bHYPRE_SStructGraph__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructGraph__cast2(
  void* obj,
  const char* type);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_createCol(int32_t        dimen,
                                     const int32_t lower[],
                                     const int32_t upper[]);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_createRow(int32_t        dimen,
                                     const int32_t lower[],
                                     const int32_t upper[]);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create1d(int32_t len);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_borrow(bHYPRE_SStructGraph*firstElement,
                                  int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_smartCopy(struct bHYPRE_SStructGraph__array *array);

void
bHYPRE_SStructGraph__array_addRef(struct bHYPRE_SStructGraph__array* array);

void
bHYPRE_SStructGraph__array_deleteRef(struct bHYPRE_SStructGraph__array* array);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get1(const struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get2(const struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get3(const struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get4(const struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                const int32_t i4);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get(const struct bHYPRE_SStructGraph__array* array,
                               const int32_t indices[]);

void
bHYPRE_SStructGraph__array_set1(struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set2(struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set3(struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set4(struct bHYPRE_SStructGraph__array* array,
                                const int32_t i1,
                                const int32_t i2,
                                const int32_t i3,
                                const int32_t i4,
                                bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set(struct bHYPRE_SStructGraph__array* array,
                               const int32_t indices[],
                               bHYPRE_SStructGraph const value);

int32_t
bHYPRE_SStructGraph__array_dimen(const struct bHYPRE_SStructGraph__array* 
  array);

int32_t
bHYPRE_SStructGraph__array_lower(const struct bHYPRE_SStructGraph__array* array,
                                 const int32_t ind);

int32_t
bHYPRE_SStructGraph__array_upper(const struct bHYPRE_SStructGraph__array* array,
                                 const int32_t ind);

int32_t
bHYPRE_SStructGraph__array_stride(const struct bHYPRE_SStructGraph__array* 
  array,
                                  const int32_t ind);

int
bHYPRE_SStructGraph__array_isColumnOrder(const struct 
  bHYPRE_SStructGraph__array* array);

int
bHYPRE_SStructGraph__array_isRowOrder(const struct bHYPRE_SStructGraph__array* 
  array);

void
bHYPRE_SStructGraph__array_slice(const struct bHYPRE_SStructGraph__array* src,
                                       int32_t        dimen,
                                       const int32_t  numElem[],
                                       const int32_t  *srcStart,
                                       const int32_t  *srcStride,
                                       const int32_t  *newStart);

void
bHYPRE_SStructGraph__array_copy(const struct bHYPRE_SStructGraph__array* src,
                                      struct bHYPRE_SStructGraph__array* dest);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_ensure(struct bHYPRE_SStructGraph__array* src,
                                  int32_t dimen,
                                  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
