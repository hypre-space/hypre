/*
 * File:          bHYPRE_SStructGraph.h
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
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

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_bHYPRE_SStructStencil_h
#include "bHYPRE_SStructStencil.h"
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
 * Constructor function for the class.
 */
struct bHYPRE_SStructGraph__object*
bHYPRE_SStructGraph__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructGraph
bHYPRE_SStructGraph__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_SStructGraph
bHYPRE_SStructGraph__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_SStructGraph_addRef(
  /* in */ bHYPRE_SStructGraph self);

void
bHYPRE_SStructGraph_deleteRef(
  /* in */ bHYPRE_SStructGraph self);

sidl_bool
bHYPRE_SStructGraph_isSame(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructGraph_queryInt(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStructGraph_isType(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStructGraph_getClassInfo(
  /* in */ bHYPRE_SStructGraph self);

/**
 * Method:  Create[]
 */
bHYPRE_SStructGraph
bHYPRE_SStructGraph_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid);

/**
 * Set the grid and communicator.
 * DEPRECATED, use Create:
 * 
 */
int32_t
bHYPRE_SStructGraph_SetCommGrid(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid);

/**
 * Set the stencil for a variable on a structured part of the
 * grid.
 * 
 */
int32_t
bHYPRE_SStructGraph_SetStencil(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ bHYPRE_SStructStencil stencil);

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
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t to_part,
  /* in */ int32_t* to_index,
  /* in */ int32_t to_var);

/**
 * Method:  SetObjectType[]
 */
int32_t
bHYPRE_SStructGraph_SetObjectType(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t type);

/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
bHYPRE_SStructGraph_SetCommunicator(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_SStructGraph_Initialize(
  /* in */ bHYPRE_SStructGraph self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_SStructGraph_Assemble(
  /* in */ bHYPRE_SStructGraph self);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructGraph__object*
bHYPRE_SStructGraph__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructGraph__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStructGraph__exec(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_SStructGraph__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStructGraph__getURL(
  /* in */ bHYPRE_SStructGraph self);
struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create1d(int32_t len);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructGraph* data);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_borrow(
  bHYPRE_SStructGraph* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_smartCopy(
  struct bHYPRE_SStructGraph__array *array);

void
bHYPRE_SStructGraph__array_addRef(
  struct bHYPRE_SStructGraph__array* array);

void
bHYPRE_SStructGraph__array_deleteRef(
  struct bHYPRE_SStructGraph__array* array);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get1(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t i1);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get2(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get3(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get4(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get5(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get6(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get7(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructGraph
bHYPRE_SStructGraph__array_get(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructGraph__array_set1(
  struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set2(
  struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set3(
  struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set4(
  struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set5(
  struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set6(
  struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set7(
  struct bHYPRE_SStructGraph__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructGraph const value);

void
bHYPRE_SStructGraph__array_set(
  struct bHYPRE_SStructGraph__array* array,
  const int32_t indices[],
  bHYPRE_SStructGraph const value);

int32_t
bHYPRE_SStructGraph__array_dimen(
  const struct bHYPRE_SStructGraph__array* array);

int32_t
bHYPRE_SStructGraph__array_lower(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGraph__array_upper(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGraph__array_length(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructGraph__array_stride(
  const struct bHYPRE_SStructGraph__array* array,
  const int32_t ind);

int
bHYPRE_SStructGraph__array_isColumnOrder(
  const struct bHYPRE_SStructGraph__array* array);

int
bHYPRE_SStructGraph__array_isRowOrder(
  const struct bHYPRE_SStructGraph__array* array);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_slice(
  struct bHYPRE_SStructGraph__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructGraph__array_copy(
  const struct bHYPRE_SStructGraph__array* src,
  struct bHYPRE_SStructGraph__array* dest);

struct bHYPRE_SStructGraph__array*
bHYPRE_SStructGraph__array_ensure(
  struct bHYPRE_SStructGraph__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
