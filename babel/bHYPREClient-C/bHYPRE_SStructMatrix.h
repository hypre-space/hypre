/*
 * File:          bHYPRE_SStructMatrix.h
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_SStructMatrix_h
#define included_bHYPRE_SStructMatrix_h

/**
 * Symbol "bHYPRE.SStructMatrix" (version 1.0.0)
 * 
 * The semi-structured grid matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_SStructMatrix__object;
struct bHYPRE_SStructMatrix__array;
typedef struct bHYPRE_SStructMatrix__object* bHYPRE_SStructMatrix;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
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
struct bHYPRE_SStructMatrix__object*
bHYPRE_SStructMatrix__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_SStructMatrix_addRef(
  /* in */ bHYPRE_SStructMatrix self);

void
bHYPRE_SStructMatrix_deleteRef(
  /* in */ bHYPRE_SStructMatrix self);

sidl_bool
bHYPRE_SStructMatrix_isSame(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_SStructMatrix_queryInt(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name);

sidl_bool
bHYPRE_SStructMatrix_isType(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_SStructMatrix_getClassInfo(
  /* in */ bHYPRE_SStructMatrix self);

/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetCommunicator(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_SStructMatrix_Initialize(
  /* in */ bHYPRE_SStructMatrix self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_SStructMatrix_Assemble(
  /* in */ bHYPRE_SStructMatrix self);

/**
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_SStructMatrix_GetObject(
  /* in */ bHYPRE_SStructMatrix self,
  /* out */ sidl_BaseInterface* A);

/**
 * Set the matrix graph.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetGraph(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_SStructGraph graph);

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
bHYPRE_SStructMatrix_SetValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ struct sidl_int__array* entries,
  /* in */ struct sidl_double__array* values);

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
bHYPRE_SStructMatrix_SetBoxValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* ilower,
  /* in */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ struct sidl_int__array* entries,
  /* in */ struct sidl_double__array* values);

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
bHYPRE_SStructMatrix_AddToValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ struct sidl_int__array* entries,
  /* in */ struct sidl_double__array* values);

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
bHYPRE_SStructMatrix_AddToBoxValues(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* ilower,
  /* in */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ struct sidl_int__array* entries,
  /* in */ struct sidl_double__array* values);

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
bHYPRE_SStructMatrix_SetSymmetric(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric);

/**
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetNSSymmetric(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ int32_t symmetric);

/**
 * Set the matrix to be complex.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetComplex(
  /* in */ bHYPRE_SStructMatrix self);

/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_SStructMatrix_Print(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* filename,
  /* in */ int32_t all);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetIntParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value);

/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetDoubleParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ double value);

/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetStringParameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ const char* value);

/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_GetIntValue(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value);

/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_GetDoubleValue(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* name,
  /* out */ double* value);

/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_Setup(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
bHYPRE_SStructMatrix_Apply(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructMatrix__object*
bHYPRE_SStructMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructMatrix__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_SStructMatrix__exec(
  /* in */ bHYPRE_SStructMatrix self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_SStructMatrix__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_SStructMatrix__getURL(
  /* in */ bHYPRE_SStructMatrix self);
struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_create1d(int32_t len);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructMatrix* data);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_borrow(
  bHYPRE_SStructMatrix* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_smartCopy(
  struct bHYPRE_SStructMatrix__array *array);

void
bHYPRE_SStructMatrix__array_addRef(
  struct bHYPRE_SStructMatrix__array* array);

void
bHYPRE_SStructMatrix__array_deleteRef(
  struct bHYPRE_SStructMatrix__array* array);

bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__array_get1(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1);

bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__array_get2(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__array_get3(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__array_get4(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__array_get5(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__array_get6(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__array_get7(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructMatrix
bHYPRE_SStructMatrix__array_get(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructMatrix__array_set1(
  struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  bHYPRE_SStructMatrix const value);

void
bHYPRE_SStructMatrix__array_set2(
  struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructMatrix const value);

void
bHYPRE_SStructMatrix__array_set3(
  struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructMatrix const value);

void
bHYPRE_SStructMatrix__array_set4(
  struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructMatrix const value);

void
bHYPRE_SStructMatrix__array_set5(
  struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructMatrix const value);

void
bHYPRE_SStructMatrix__array_set6(
  struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructMatrix const value);

void
bHYPRE_SStructMatrix__array_set7(
  struct bHYPRE_SStructMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructMatrix const value);

void
bHYPRE_SStructMatrix__array_set(
  struct bHYPRE_SStructMatrix__array* array,
  const int32_t indices[],
  bHYPRE_SStructMatrix const value);

int32_t
bHYPRE_SStructMatrix__array_dimen(
  const struct bHYPRE_SStructMatrix__array* array);

int32_t
bHYPRE_SStructMatrix__array_lower(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrix__array_upper(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrix__array_length(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructMatrix__array_stride(
  const struct bHYPRE_SStructMatrix__array* array,
  const int32_t ind);

int
bHYPRE_SStructMatrix__array_isColumnOrder(
  const struct bHYPRE_SStructMatrix__array* array);

int
bHYPRE_SStructMatrix__array_isRowOrder(
  const struct bHYPRE_SStructMatrix__array* array);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_slice(
  struct bHYPRE_SStructMatrix__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructMatrix__array_copy(
  const struct bHYPRE_SStructMatrix__array* src,
  struct bHYPRE_SStructMatrix__array* dest);

struct bHYPRE_SStructMatrix__array*
bHYPRE_SStructMatrix__array_ensure(
  struct bHYPRE_SStructMatrix__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
