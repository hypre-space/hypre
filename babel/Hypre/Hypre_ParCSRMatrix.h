/*
 * File:          Hypre_ParCSRMatrix.h
 * Symbol:        Hypre.ParCSRMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:48 PDT
 * Description:   Client-side glue code for Hypre.ParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_ParCSRMatrix_h
#define included_Hypre_ParCSRMatrix_h

/**
 * Symbol "Hypre.ParCSRMatrix" (version 0.1.5)
 * 
 * A single class that implements both a build interface and an operator
 * interface. It returns itself for <code>GetConstructedObject</code>.
 */
struct Hypre_ParCSRMatrix__object;
struct Hypre_ParCSRMatrix__array;
typedef struct Hypre_ParCSRMatrix__object* Hypre_ParCSRMatrix;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__create(void);

/**
 * Method:  SetIntArrayParameter
 */
int32_t
Hypre_ParCSRMatrix_SetIntArrayParameter(
  Hypre_ParCSRMatrix self,
  const char* name,
  struct SIDL_int__array* value);

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
Hypre_ParCSRMatrix_SetRowSizes(
  Hypre_ParCSRMatrix self,
  struct SIDL_int__array* sizes);

/**
 * Method:  SetCommunicator
 */
int32_t
Hypre_ParCSRMatrix_SetCommunicator(
  Hypre_ParCSRMatrix self,
  void* mpi_comm);

/**
 * Read the matrix from file.  This is mainly for debugging purposes.
 * 
 */
int32_t
Hypre_ParCSRMatrix_Read(
  Hypre_ParCSRMatrix self,
  const char* filename,
  void* comm);

/**
 * Method:  SetStringParameter
 */
int32_t
Hypre_ParCSRMatrix_SetStringParameter(
  Hypre_ParCSRMatrix self,
  const char* name,
  const char* value);

/**
 * Method:  SetDoubleParameter
 */
int32_t
Hypre_ParCSRMatrix_SetDoubleParameter(
  Hypre_ParCSRMatrix self,
  const char* name,
  double value);

/**
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */
int32_t
Hypre_ParCSRMatrix_Assemble(
  Hypre_ParCSRMatrix self);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */
int32_t
Hypre_ParCSRMatrix_Initialize(
  Hypre_ParCSRMatrix self);

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
Hypre_ParCSRMatrix_SetDiagOffdSizes(
  Hypre_ParCSRMatrix self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_ParCSRMatrix_isSame(
  Hypre_ParCSRMatrix self,
  SIDL_BaseInterface iobj);

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
Hypre_ParCSRMatrix_addReference(
  Hypre_ParCSRMatrix self);

/**
 * Method:  SetDoubleArrayParameter
 */
int32_t
Hypre_ParCSRMatrix_SetDoubleArrayParameter(
  Hypre_ParCSRMatrix self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_ParCSRMatrix_deleteReference(
  Hypre_ParCSRMatrix self);

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
Hypre_ParCSRMatrix_AddToValues(
  Hypre_ParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values);

/**
 * Method:  Setup
 */
int32_t
Hypre_ParCSRMatrix_Setup(
  Hypre_ParCSRMatrix self,
  Hypre_Vector x,
  Hypre_Vector y);

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
Hypre_ParCSRMatrix_Create(
  Hypre_ParCSRMatrix self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper);

/**
 * Method:  SetIntParameter
 */
int32_t
Hypre_ParCSRMatrix_SetIntParameter(
  Hypre_ParCSRMatrix self,
  const char* name,
  int32_t value);

/**
 * Method:  GetIntValue
 */
int32_t
Hypre_ParCSRMatrix_GetIntValue(
  Hypre_ParCSRMatrix self,
  const char* name,
  int32_t* value);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_ParCSRMatrix_isInstanceOf(
  Hypre_ParCSRMatrix self,
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
Hypre_ParCSRMatrix_GetObject(
  Hypre_ParCSRMatrix self,
  SIDL_BaseInterface* A);

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
Hypre_ParCSRMatrix_queryInterface(
  Hypre_ParCSRMatrix self,
  const char* name);

/**
 * Method:  Apply
 */
int32_t
Hypre_ParCSRMatrix_Apply(
  Hypre_ParCSRMatrix self,
  Hypre_Vector x,
  Hypre_Vector* y);

/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 * 
 */
int32_t
Hypre_ParCSRMatrix_Print(
  Hypre_ParCSRMatrix self,
  const char* filename);

/**
 * Method:  GetDoubleValue
 */
int32_t
Hypre_ParCSRMatrix_GetDoubleValue(
  Hypre_ParCSRMatrix self,
  const char* name,
  double* value);

/**
 * Method:  GetRow
 */
int32_t
Hypre_ParCSRMatrix_GetRow(
  Hypre_ParCSRMatrix self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values);

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
Hypre_ParCSRMatrix_SetValues(
  Hypre_ParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_ParCSRMatrix
Hypre_ParCSRMatrix__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_ParCSRMatrix__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_ParCSRMatrix__array*
Hypre_ParCSRMatrix__array_borrow(
  struct Hypre_ParCSRMatrix__object** firstElement,
  int32_t                             dimen,
  const int32_t                       lower[],
  const int32_t                       upper[],
  const int32_t                       stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_ParCSRMatrix__array_destroy(
  struct Hypre_ParCSRMatrix__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_ParCSRMatrix__array_dimen(const struct Hypre_ParCSRMatrix__array *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_ParCSRMatrix__array_lower(const struct Hypre_ParCSRMatrix__array *array,
  int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_ParCSRMatrix__array_upper(const struct Hypre_ParCSRMatrix__array *array,
  int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_ParCSRMatrix__object*
Hypre_ParCSRMatrix__array_get(
  const struct Hypre_ParCSRMatrix__array* array,
  const int32_t                           indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_ParCSRMatrix__object*
Hypre_ParCSRMatrix__array_get4(
  const struct Hypre_ParCSRMatrix__array* array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_ParCSRMatrix__array_set(
  struct Hypre_ParCSRMatrix__array*  array,
  const int32_t                      indices[],
  struct Hypre_ParCSRMatrix__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_ParCSRMatrix__array_set4(
  struct Hypre_ParCSRMatrix__array*  array,
  int32_t                            i1,
  int32_t                            i2,
  int32_t                            i3,
  int32_t                            i4,
  struct Hypre_ParCSRMatrix__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_ParCSRMatrix__array_get1(a,i1) \
  Hypre_ParCSRMatrix__array_get4(a,i1,0,0,0)

#define Hypre_ParCSRMatrix__array_get2(a,i1,i2) \
  Hypre_ParCSRMatrix__array_get4(a,i1,i2,0,0)

#define Hypre_ParCSRMatrix__array_get3(a,i1,i2,i3) \
  Hypre_ParCSRMatrix__array_get4(a,i1,i2,i3,0)

#define Hypre_ParCSRMatrix__array_set1(a,i1,v) \
  Hypre_ParCSRMatrix__array_set4(a,i1,0,0,0,v)

#define Hypre_ParCSRMatrix__array_set2(a,i1,i2,v) \
  Hypre_ParCSRMatrix__array_set4(a,i1,i2,0,0,v)

#define Hypre_ParCSRMatrix__array_set3(a,i1,i2,i3,v) \
  Hypre_ParCSRMatrix__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
