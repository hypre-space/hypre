/*
 * File:          bHYPRE_IJVectorView.h
 * Symbol:        bHYPRE.IJVectorView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.IJVectorView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_IJVectorView_h
#define included_bHYPRE_IJVectorView_h

/**
 * Symbol "bHYPRE.IJVectorView" (version 1.0.0)
 */
struct bHYPRE_IJVectorView__object;
struct bHYPRE_IJVectorView__array;
typedef struct bHYPRE_IJVectorView__object* bHYPRE_IJVectorView;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
bHYPRE_IJVectorView
bHYPRE_IJVectorView__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_IJVectorView_addRef(
  /* in */ bHYPRE_IJVectorView self);

void
bHYPRE_IJVectorView_deleteRef(
  /* in */ bHYPRE_IJVectorView self);

sidl_bool
bHYPRE_IJVectorView_isSame(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_IJVectorView_queryInt(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ const char* name);

sidl_bool
bHYPRE_IJVectorView_isType(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_IJVectorView_getClassInfo(
  /* in */ bHYPRE_IJVectorView self);

int32_t
bHYPRE_IJVectorView_SetCommunicator(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

int32_t
bHYPRE_IJVectorView_Initialize(
  /* in */ bHYPRE_IJVectorView self);

int32_t
bHYPRE_IJVectorView_Assemble(
  /* in */ bHYPRE_IJVectorView self);

/**
 * Set the local range for a vector object.  Each process owns
 * some unique consecutive range of vector unknowns, indicated
 * by the global indices {\tt jlower} and {\tt jupper}.  The
 * data is required to be such that the value of {\tt jlower} on
 * any process $p$ be exactly one more than the value of {\tt
 * jupper} on process $p-1$.  Note that the first index of the
 * global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 * 
 */
int32_t
bHYPRE_IJVectorView_SetLocalRange(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

/**
 * Sets values in vector.  The arrays {\tt values} and {\tt
 * indices} are of dimension {\tt nvalues} and contain the
 * vector values to be set and the corresponding global vector
 * indices, respectively.  Erases any previous values at the
 * specified locations and replaces them with new ones.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJVectorView_SetValues(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* in */ double* values);

/**
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJVectorView_AddToValues(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* in */ double* values);

/**
 * Returns range of the part of the vector owned by this
 * processor.
 * 
 */
int32_t
bHYPRE_IJVectorView_GetLocalRange(
  /* in */ bHYPRE_IJVectorView self,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper);

/**
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJVectorView_GetValues(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* inout */ double* values);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJVectorView_Print(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ const char* filename);

/**
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJVectorView_Read(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IJVectorView__object*
bHYPRE_IJVectorView__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IJVectorView__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_IJVectorView__exec(
  /* in */ bHYPRE_IJVectorView self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_IJVectorView__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_IJVectorView__getURL(
  /* in */ bHYPRE_IJVectorView self);
struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_create1d(int32_t len);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_create1dInit(
  int32_t len, 
  bHYPRE_IJVectorView* data);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_borrow(
  bHYPRE_IJVectorView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_smartCopy(
  struct bHYPRE_IJVectorView__array *array);

void
bHYPRE_IJVectorView__array_addRef(
  struct bHYPRE_IJVectorView__array* array);

void
bHYPRE_IJVectorView__array_deleteRef(
  struct bHYPRE_IJVectorView__array* array);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get1(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get2(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get3(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get4(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get5(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get6(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get7(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_IJVectorView
bHYPRE_IJVectorView__array_get(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t indices[]);

void
bHYPRE_IJVectorView__array_set1(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set2(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set3(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set4(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set5(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set6(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set7(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IJVectorView const value);

void
bHYPRE_IJVectorView__array_set(
  struct bHYPRE_IJVectorView__array* array,
  const int32_t indices[],
  bHYPRE_IJVectorView const value);

int32_t
bHYPRE_IJVectorView__array_dimen(
  const struct bHYPRE_IJVectorView__array* array);

int32_t
bHYPRE_IJVectorView__array_lower(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJVectorView__array_upper(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJVectorView__array_length(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJVectorView__array_stride(
  const struct bHYPRE_IJVectorView__array* array,
  const int32_t ind);

int
bHYPRE_IJVectorView__array_isColumnOrder(
  const struct bHYPRE_IJVectorView__array* array);

int
bHYPRE_IJVectorView__array_isRowOrder(
  const struct bHYPRE_IJVectorView__array* array);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_slice(
  struct bHYPRE_IJVectorView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_IJVectorView__array_copy(
  const struct bHYPRE_IJVectorView__array* src,
  struct bHYPRE_IJVectorView__array* dest);

struct bHYPRE_IJVectorView__array*
bHYPRE_IJVectorView__array_ensure(
  struct bHYPRE_IJVectorView__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
