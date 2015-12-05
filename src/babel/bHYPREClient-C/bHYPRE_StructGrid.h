/*
 * File:          bHYPRE_StructGrid.h
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:35 PST
 * Generated:     20030401 14:47:43 PST
 * Description:   Client-side glue code for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 1101
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_StructGrid_h
#define included_bHYPRE_StructGrid_h

/**
 * Symbol "bHYPRE.StructGrid" (version 1.0.0)
 * 
 * Define a structured grid class.
 * 
 */
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructGrid__array;
typedef struct bHYPRE_StructGrid__object* bHYPRE_StructGrid;

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
bHYPRE_StructGrid
bHYPRE_StructGrid__create(void);

void
bHYPRE_StructGrid_addRef(
  bHYPRE_StructGrid self);

void
bHYPRE_StructGrid_deleteRef(
  bHYPRE_StructGrid self);

SIDL_bool
bHYPRE_StructGrid_isSame(
  bHYPRE_StructGrid self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_StructGrid_queryInt(
  bHYPRE_StructGrid self,
  const char* name);

SIDL_bool
bHYPRE_StructGrid_isType(
  bHYPRE_StructGrid self,
  const char* name);

SIDL_ClassInfo
bHYPRE_StructGrid_getClassInfo(
  bHYPRE_StructGrid self);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_StructGrid_SetCommunicator(
  bHYPRE_StructGrid self,
  void* mpi_comm);

/**
 * Method:  SetDimension[]
 */
int32_t
bHYPRE_StructGrid_SetDimension(
  bHYPRE_StructGrid self,
  int32_t dim);

/**
 * Method:  SetExtents[]
 */
int32_t
bHYPRE_StructGrid_SetExtents(
  bHYPRE_StructGrid self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper);

/**
 * Method:  SetPeriodic[]
 */
int32_t
bHYPRE_StructGrid_SetPeriodic(
  bHYPRE_StructGrid self,
  struct SIDL_int__array* periodic);

/**
 * Method:  Assemble[]
 */
int32_t
bHYPRE_StructGrid_Assemble(
  bHYPRE_StructGrid self);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_StructGrid
bHYPRE_StructGrid__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructGrid__cast2(
  void* obj,
  const char* type);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_createCol(int32_t        dimen,
                                   const int32_t lower[],
                                   const int32_t upper[]);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_createRow(int32_t        dimen,
                                   const int32_t lower[],
                                   const int32_t upper[]);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create1d(int32_t len);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_borrow(bHYPRE_StructGrid*firstElement,
                                int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_smartCopy(struct bHYPRE_StructGrid__array *array);

void
bHYPRE_StructGrid__array_addRef(struct bHYPRE_StructGrid__array* array);

void
bHYPRE_StructGrid__array_deleteRef(struct bHYPRE_StructGrid__array* array);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get1(const struct bHYPRE_StructGrid__array* array,
                              const int32_t i1);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get2(const struct bHYPRE_StructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get3(const struct bHYPRE_StructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              const int32_t i3);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get4(const struct bHYPRE_StructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              const int32_t i3,
                              const int32_t i4);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get(const struct bHYPRE_StructGrid__array* array,
                             const int32_t indices[]);

void
bHYPRE_StructGrid__array_set1(struct bHYPRE_StructGrid__array* array,
                              const int32_t i1,
                              bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set2(struct bHYPRE_StructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set3(struct bHYPRE_StructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              const int32_t i3,
                              bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set4(struct bHYPRE_StructGrid__array* array,
                              const int32_t i1,
                              const int32_t i2,
                              const int32_t i3,
                              const int32_t i4,
                              bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set(struct bHYPRE_StructGrid__array* array,
                             const int32_t indices[],
                             bHYPRE_StructGrid const value);

int32_t
bHYPRE_StructGrid__array_dimen(const struct bHYPRE_StructGrid__array* array);

int32_t
bHYPRE_StructGrid__array_lower(const struct bHYPRE_StructGrid__array* array,
                               const int32_t ind);

int32_t
bHYPRE_StructGrid__array_upper(const struct bHYPRE_StructGrid__array* array,
                               const int32_t ind);

int32_t
bHYPRE_StructGrid__array_stride(const struct bHYPRE_StructGrid__array* array,
                                const int32_t ind);

int
bHYPRE_StructGrid__array_isColumnOrder(const struct bHYPRE_StructGrid__array* 
  array);

int
bHYPRE_StructGrid__array_isRowOrder(const struct bHYPRE_StructGrid__array* 
  array);

void
bHYPRE_StructGrid__array_slice(const struct bHYPRE_StructGrid__array* src,
                                     int32_t        dimen,
                                     const int32_t  numElem[],
                                     const int32_t  *srcStart,
                                     const int32_t  *srcStride,
                                     const int32_t  *newStart);

void
bHYPRE_StructGrid__array_copy(const struct bHYPRE_StructGrid__array* src,
                                    struct bHYPRE_StructGrid__array* dest);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_ensure(struct bHYPRE_StructGrid__array* src,
                                int32_t dimen,
                                int     ordering);

#ifdef __cplusplus
}
#endif
#endif
