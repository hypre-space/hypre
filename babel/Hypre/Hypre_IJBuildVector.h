/*
 * File:          Hypre_IJBuildVector.h
 * Symbol:        Hypre.IJBuildVector-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:32 PDT
 * Description:   Client-side glue code for Hypre.IJBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_IJBuildVector_h
#define included_Hypre_IJBuildVector_h

/**
 * Symbol "Hypre.IJBuildVector" (version 0.1.5)
 */
struct Hypre_IJBuildVector__object;
struct Hypre_IJBuildVector__array;
typedef struct Hypre_IJBuildVector__object* Hypre_IJBuildVector;

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
 * Method:  SetLocalComponents
 */
int32_t
Hypre_IJBuildVector_SetLocalComponents(
  Hypre_IJBuildVector self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values);

/**
 * Adds to values in vector.  Usage details are analogous to
 * \Ref{SetValues}.
 * 
 * Not collective.
 * 
 * 
 */
int32_t
Hypre_IJBuildVector_AddToValues(
  Hypre_IJBuildVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values);

/**
 * Create a vector object.  Each process owns some unique consecutive
 * range of vector unknowns, indicated by the global indices {\tt
 * jlower} and {\tt jupper}.  The data is required to be such that the
 * value of {\tt jlower} on any process $p$ be exactly one more than
 * the value of {\tt jupper} on process $p-1$.  Note that the first
 * index of the global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 * 
 * 
 */
int32_t
Hypre_IJBuildVector_Create(
  Hypre_IJBuildVector self,
  void* comm,
  int32_t jlower,
  int32_t jupper);

/**
 * Method:  SetPartitioning
 */
int32_t
Hypre_IJBuildVector_SetPartitioning(
  Hypre_IJBuildVector self,
  struct SIDL_int__array* partitioning);

/**
 * Method:  SetCommunicator
 */
int32_t
Hypre_IJBuildVector_SetCommunicator(
  Hypre_IJBuildVector self,
  void* mpi_comm);

/**
 * Read the vector from file.  This is mainly for debugging purposes.
 * 
 * 
 */
int32_t
Hypre_IJBuildVector_Read(
  Hypre_IJBuildVector self,
  const char* filename,
  void* comm);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_IJBuildVector_isInstanceOf(
  Hypre_IJBuildVector self,
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
Hypre_IJBuildVector_GetObject(
  Hypre_IJBuildVector self,
  SIDL_BaseInterface* A);

/**
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */
int32_t
Hypre_IJBuildVector_Assemble(
  Hypre_IJBuildVector self);

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
Hypre_IJBuildVector_queryInterface(
  Hypre_IJBuildVector self,
  const char* name);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */
int32_t
Hypre_IJBuildVector_Initialize(
  Hypre_IJBuildVector self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_IJBuildVector_isSame(
  Hypre_IJBuildVector self,
  SIDL_BaseInterface iobj);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 * 
 */
int32_t
Hypre_IJBuildVector_Print(
  Hypre_IJBuildVector self,
  const char* filename);

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
Hypre_IJBuildVector_addReference(
  Hypre_IJBuildVector self);

/**
 * Method:  SetLocalComponentsInBlock
 */
int32_t
Hypre_IJBuildVector_SetLocalComponentsInBlock(
  Hypre_IJBuildVector self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values);

/**
 * Method:  SetGlobalSize
 */
int32_t
Hypre_IJBuildVector_SetGlobalSize(
  Hypre_IJBuildVector self,
  int32_t n);

/**
 * Method:  AddtoLocalComponents
 */
int32_t
Hypre_IJBuildVector_AddtoLocalComponents(
  Hypre_IJBuildVector self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values);

/**
 * Sets values in vector.  The arrays {\tt values} and {\tt indices}
 * are of dimension {\tt nvalues} and contain the vector values to be
 * set and the corresponding global vector indices, respectively.
 * Erases any previous values at the specified locations and replaces
 * them with new ones.
 * 
 * Not collective.
 * 
 * 
 */
int32_t
Hypre_IJBuildVector_SetValues(
  Hypre_IJBuildVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values);

/**
 * Method:  AddToLocalComponentsInBlock
 */
int32_t
Hypre_IJBuildVector_AddToLocalComponentsInBlock(
  Hypre_IJBuildVector self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_IJBuildVector_deleteReference(
  Hypre_IJBuildVector self);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_IJBuildVector
Hypre_IJBuildVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_IJBuildVector__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_IJBuildVector__array*
Hypre_IJBuildVector__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_IJBuildVector__array*
Hypre_IJBuildVector__array_borrow(
  struct Hypre_IJBuildVector__object** firstElement,
  int32_t                              dimen,
  const int32_t                        lower[],
  const int32_t                        upper[],
  const int32_t                        stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_IJBuildVector__array_destroy(
  struct Hypre_IJBuildVector__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_IJBuildVector__array_dimen(const struct Hypre_IJBuildVector__array 
  *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_IJBuildVector__array_lower(const struct Hypre_IJBuildVector__array *array,
  int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_IJBuildVector__array_upper(const struct Hypre_IJBuildVector__array *array,
  int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_IJBuildVector__object*
Hypre_IJBuildVector__array_get(
  const struct Hypre_IJBuildVector__array* array,
  const int32_t                            indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_IJBuildVector__object*
Hypre_IJBuildVector__array_get4(
  const struct Hypre_IJBuildVector__array* array,
  int32_t                                  i1,
  int32_t                                  i2,
  int32_t                                  i3,
  int32_t                                  i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_IJBuildVector__array_set(
  struct Hypre_IJBuildVector__array*  array,
  const int32_t                       indices[],
  struct Hypre_IJBuildVector__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_IJBuildVector__array_set4(
  struct Hypre_IJBuildVector__array*  array,
  int32_t                             i1,
  int32_t                             i2,
  int32_t                             i3,
  int32_t                             i4,
  struct Hypre_IJBuildVector__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_IJBuildVector__array_get1(a,i1) \
  Hypre_IJBuildVector__array_get4(a,i1,0,0,0)

#define Hypre_IJBuildVector__array_get2(a,i1,i2) \
  Hypre_IJBuildVector__array_get4(a,i1,i2,0,0)

#define Hypre_IJBuildVector__array_get3(a,i1,i2,i3) \
  Hypre_IJBuildVector__array_get4(a,i1,i2,i3,0)

#define Hypre_IJBuildVector__array_set1(a,i1,v) \
  Hypre_IJBuildVector__array_set4(a,i1,0,0,0,v)

#define Hypre_IJBuildVector__array_set2(a,i1,i2,v) \
  Hypre_IJBuildVector__array_set4(a,i1,i2,0,0,v)

#define Hypre_IJBuildVector__array_set3(a,i1,i2,i3,v) \
  Hypre_IJBuildVector__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
