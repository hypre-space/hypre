/*
 * File:          Hypre_IJBuildMatrix.h
 * Symbol:        Hypre.IJBuildMatrix-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:32 PDT
 * Description:   Client-side glue code for Hypre.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_IJBuildMatrix_h
#define included_Hypre_IJBuildMatrix_h

/**
 * Symbol "Hypre.IJBuildMatrix" (version 0.1.5)
 * 
 * 
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 * 
 * 
 */
struct Hypre_IJBuildMatrix__object;
struct Hypre_IJBuildMatrix__array;
typedef struct Hypre_IJBuildMatrix__object* Hypre_IJBuildMatrix;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details are
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value there
 * before, inserts a new one.
 * 
 * Not collective.
 * 
 * 
 */
int32_t
Hypre_IJBuildMatrix_AddToValues(
  Hypre_IJBuildMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values);

/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 * 
 */
int32_t
Hypre_IJBuildMatrix_Print(
  Hypre_IJBuildMatrix self,
  const char* filename);

/**
 * Create a matrix object.  Each process owns some unique consecutive
 * range of rows, indicated by the global row indices {\tt ilower} and
 * {\tt iupper}.  The row data is required to be such that the value
 * of {\tt ilower} on any process $p$ be exactly one more than the
 * value of {\tt iupper} on process $p-1$.  Note that the first row of
 * the global matrix may start with any integer value.  In particular,
 * one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically should
 * match {\tt ilower} and {\tt iupper}, respectively.  For rectangular
 * matrices, {\tt jlower} and {\tt jupper} should define a
 * partitioning of the columns.  This partitioning must be used for
 * any vector $v$ that will be used in matrix-vector products with the
 * rectangular matrix.  The matrix data structure may use {\tt jlower}
 * and {\tt jupper} to store the diagonal blocks (rectangular in
 * general) of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 * 
 */
int32_t
Hypre_IJBuildMatrix_Create(
  Hypre_IJBuildMatrix self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper);

/**
 * (Optional) Set the max number of nonzeros to expect in each row.
 * The array {\tt sizes} contains estimated sizes for each row on this
 * process.  This call can significantly improve the efficiency of
 * matrix construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 * DEVELOPER NOTES: None.
 * 
 */
int32_t
Hypre_IJBuildMatrix_SetRowSizes(
  Hypre_IJBuildMatrix self,
  struct SIDL_int__array* sizes);

/**
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>SIDL</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */
void
Hypre_IJBuildMatrix_addReference(
  Hypre_IJBuildMatrix self);

/**
 * Method:  SetCommunicator
 */
int32_t
Hypre_IJBuildMatrix_SetCommunicator(
  Hypre_IJBuildMatrix self,
  void* mpi_comm);

/**
 * Read the matrix from file.  This is mainly for debugging purposes.
 * 
 */
int32_t
Hypre_IJBuildMatrix_Read(
  Hypre_IJBuildMatrix self,
  const char* filename,
  void* comm);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_IJBuildMatrix_isInstanceOf(
  Hypre_IJBuildMatrix self,
  const char* name);

/**
 * The problem definition interface is a "builder" that creates an object
 * that contains the problem definition information, e.g. a matrix. To
 * perform subsequent operations with that object, it must be returned from
 * the problem definition object. "GetObject" performs this function.
 * <note>At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface. QueryInterface or Cast must
 * be used on the returned object to convert it into a known type.</note>
 * 
 * 
 */
int32_t
Hypre_IJBuildMatrix_GetObject(
  Hypre_IJBuildMatrix self,
  SIDL_BaseInterface* A);

/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt ncols}
 * and {\tt rows} are of dimension {\tt nrows} and contain the number
 * of columns in each row and the row indices, respectively.  The
 * array {\tt cols} contains the column indices for each of the {\tt
 * rows}, and is ordered by rows.  The data in the {\tt values} array
 * corresponds directly to the column entries in {\tt cols}.  Erases
 * any previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before, inserts a
 * new one.
 * 
 * Not collective.
 * 
 * 
 */
int32_t
Hypre_IJBuildMatrix_SetValues(
  Hypre_IJBuildMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values);

/**
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */
int32_t
Hypre_IJBuildMatrix_Assemble(
  Hypre_IJBuildMatrix self);

/**
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteReference</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
SIDL_BaseInterface
Hypre_IJBuildMatrix_queryInterface(
  Hypre_IJBuildMatrix self,
  const char* name);

/**
 * (Optional) Set the max number of nonzeros to expect in each row of
 * the diagonal and off-diagonal blocks.  The diagonal block is the
 * submatrix whose column numbers correspond to rows owned by this
 * process, and the off-diagonal block is everything else.  The arrays
 * {\tt diag\_sizes} and {\tt offdiag\_sizes} contain estimated sizes
 * for each row of the diagonal and off-diagonal blocks, respectively.
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 * 
 */
int32_t
Hypre_IJBuildMatrix_SetDiagOffdSizes(
  Hypre_IJBuildMatrix self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_IJBuildMatrix_deleteReference(
  Hypre_IJBuildMatrix self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_IJBuildMatrix_isSame(
  Hypre_IJBuildMatrix self,
  SIDL_BaseInterface iobj);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */
int32_t
Hypre_IJBuildMatrix_Initialize(
  Hypre_IJBuildMatrix self);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_IJBuildMatrix
Hypre_IJBuildMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_IJBuildMatrix__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_IJBuildMatrix__array*
Hypre_IJBuildMatrix__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_IJBuildMatrix__array*
Hypre_IJBuildMatrix__array_borrow(
  struct Hypre_IJBuildMatrix__object** firstElement,
  int32_t                              dimen,
  const int32_t                        lower[],
  const int32_t                        upper[],
  const int32_t                        stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_IJBuildMatrix__array_destroy(
  struct Hypre_IJBuildMatrix__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_IJBuildMatrix__array_dimen(const struct Hypre_IJBuildMatrix__array 
  *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_IJBuildMatrix__array_lower(const struct Hypre_IJBuildMatrix__array *array,
  int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_IJBuildMatrix__array_upper(const struct Hypre_IJBuildMatrix__array *array,
  int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_IJBuildMatrix__object*
Hypre_IJBuildMatrix__array_get(
  const struct Hypre_IJBuildMatrix__array* array,
  const int32_t                            indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_IJBuildMatrix__object*
Hypre_IJBuildMatrix__array_get4(
  const struct Hypre_IJBuildMatrix__array* array,
  int32_t                                  i1,
  int32_t                                  i2,
  int32_t                                  i3,
  int32_t                                  i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_IJBuildMatrix__array_set(
  struct Hypre_IJBuildMatrix__array*  array,
  const int32_t                       indices[],
  struct Hypre_IJBuildMatrix__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_IJBuildMatrix__array_set4(
  struct Hypre_IJBuildMatrix__array*  array,
  int32_t                             i1,
  int32_t                             i2,
  int32_t                             i3,
  int32_t                             i4,
  struct Hypre_IJBuildMatrix__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_IJBuildMatrix__array_get1(a,i1) \
  Hypre_IJBuildMatrix__array_get4(a,i1,0,0,0)

#define Hypre_IJBuildMatrix__array_get2(a,i1,i2) \
  Hypre_IJBuildMatrix__array_get4(a,i1,i2,0,0)

#define Hypre_IJBuildMatrix__array_get3(a,i1,i2,i3) \
  Hypre_IJBuildMatrix__array_get4(a,i1,i2,i3,0)

#define Hypre_IJBuildMatrix__array_set1(a,i1,v) \
  Hypre_IJBuildMatrix__array_set4(a,i1,0,0,0,v)

#define Hypre_IJBuildMatrix__array_set2(a,i1,i2,v) \
  Hypre_IJBuildMatrix__array_set4(a,i1,i2,0,0,v)

#define Hypre_IJBuildMatrix__array_set3(a,i1,i2,i3,v) \
  Hypre_IJBuildMatrix__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
