/*
 * File:          Hypre_ParCSRVector_Stub.c
 * Symbol:        Hypre.ParCSRVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:43 PDT
 * Description:   Client-side glue code for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_ParCSRVector.h"
#include "Hypre_ParCSRVector_IOR.h"
#include <stddef.h>
#include "SIDL_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "SIDL_Loader.h"
#endif

/*
 * Return pointer to internal IOR functions.
 */

static const struct Hypre_ParCSRVector__external* _getIOR(void)
{
  static const struct Hypre_ParCSRVector__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_ParCSRVector__externals();
#else
    const struct Hypre_ParCSRVector__external*(*dll_f)(void) =
      (const struct Hypre_ParCSRVector__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_ParCSRVector__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.ParCSRVector; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor function for the class.
 */

Hypre_ParCSRVector
Hypre_ParCSRVector__create()
{
  return (*(_getIOR()->createObject))();
}

/*
 * Method:  SetLocalComponents
 */

int32_t
Hypre_ParCSRVector_SetLocalComponents(
  Hypre_ParCSRVector self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_SetLocalComponents)(
    self,
    num_values,
    glob_vec_indices,
    value_indices,
    values);
}

/*
 * create an x compatible with y
 */

int32_t
Hypre_ParCSRVector_Clone(
  Hypre_ParCSRVector self,
  Hypre_Vector* x)
{
  return (*self->d_epv->f_Clone)(
    self,
    x);
}

/*
 * Method:  SetCommunicator
 */

int32_t
Hypre_ParCSRVector_SetCommunicator(
  Hypre_ParCSRVector self,
  void* mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
}

/*
 * Read the vector from file.  This is mainly for debugging purposes.
 * 
 * 
 */

int32_t
Hypre_ParCSRVector_Read(
  Hypre_ParCSRVector self,
  const char* filename,
  void* comm)
{
  return (*self->d_epv->f_Read)(
    self,
    filename,
    comm);
}

/*
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */

int32_t
Hypre_ParCSRVector_Assemble(
  Hypre_ParCSRVector self)
{
  return (*self->d_epv->f_Assemble)(
    self);
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

int32_t
Hypre_ParCSRVector_Initialize(
  Hypre_ParCSRVector self)
{
  return (*self->d_epv->f_Initialize)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_bool
Hypre_ParCSRVector_isSame(
  Hypre_ParCSRVector self,
  SIDL_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * y <- 0 (where y=self)
 */

int32_t
Hypre_ParCSRVector_Clear(
  Hypre_ParCSRVector self)
{
  return (*self->d_epv->f_Clear)(
    self);
}

/*
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
Hypre_ParCSRVector_addReference(
  Hypre_ParCSRVector self)
{
  (*self->d_epv->f_addReference)(
    self);
}

/*
 * y <- a*y 
 */

int32_t
Hypre_ParCSRVector_Scale(
  Hypre_ParCSRVector self,
  double a)
{
  return (*self->d_epv->f_Scale)(
    self,
    a);
}

/*
 * Method:  SetLocalComponentsInBlock
 */

int32_t
Hypre_ParCSRVector_SetLocalComponentsInBlock(
  Hypre_ParCSRVector self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_SetLocalComponentsInBlock)(
    self,
    glob_vec_index_start,
    glob_vec_index_stop,
    value_indices,
    values);
}

/*
 * d <- (y,x)
 */

int32_t
Hypre_ParCSRVector_Dot(
  Hypre_ParCSRVector self,
  Hypre_Vector x,
  double* d)
{
  return (*self->d_epv->f_Dot)(
    self,
    x,
    d);
}

/*
 * Method:  SetGlobalSize
 */

int32_t
Hypre_ParCSRVector_SetGlobalSize(
  Hypre_ParCSRVector self,
  int32_t n)
{
  return (*self->d_epv->f_SetGlobalSize)(
    self,
    n);
}

/*
 * y <- a*x + y
 */

int32_t
Hypre_ParCSRVector_Axpy(
  Hypre_ParCSRVector self,
  double a,
  Hypre_Vector x)
{
  return (*self->d_epv->f_Axpy)(
    self,
    a,
    x);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
Hypre_ParCSRVector_deleteReference(
  Hypre_ParCSRVector self)
{
  (*self->d_epv->f_deleteReference)(
    self);
}

/*
 * Adds to values in vector.  Usage details are analogous to
 * \Ref{SetValues}.
 * 
 * Not collective.
 * 
 * 
 */

int32_t
Hypre_ParCSRVector_AddToValues(
  Hypre_ParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_AddToValues)(
    self,
    nvalues,
    indices,
    values);
}

/*
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
Hypre_ParCSRVector_Create(
  Hypre_ParCSRVector self,
  void* comm,
  int32_t jlower,
  int32_t jupper)
{
  return (*self->d_epv->f_Create)(
    self,
    comm,
    jlower,
    jupper);
}

/*
 * Method:  SetPartitioning
 */

int32_t
Hypre_ParCSRVector_SetPartitioning(
  Hypre_ParCSRVector self,
  struct SIDL_int__array* partitioning)
{
  return (*self->d_epv->f_SetPartitioning)(
    self,
    partitioning);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_bool
Hypre_ParCSRVector_isInstanceOf(
  Hypre_ParCSRVector self,
  const char* name)
{
  return (*self->d_epv->f_isInstanceOf)(
    self,
    name);
}

/*
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
Hypre_ParCSRVector_GetObject(
  Hypre_ParCSRVector self,
  SIDL_BaseInterface* A)
{
  return (*self->d_epv->f_GetObject)(
    self,
    A);
}

/*
 * y <- x 
 */

int32_t
Hypre_ParCSRVector_Copy(
  Hypre_ParCSRVector self,
  Hypre_Vector x)
{
  return (*self->d_epv->f_Copy)(
    self,
    x);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteReference</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

SIDL_BaseInterface
Hypre_ParCSRVector_queryInterface(
  Hypre_ParCSRVector self,
  const char* name)
{
  return (*self->d_epv->f_queryInterface)(
    self,
    name);
}

/*
 * Print the vector to file.  This is mainly for debugging purposes.
 * 
 */

int32_t
Hypre_ParCSRVector_Print(
  Hypre_ParCSRVector self,
  const char* filename)
{
  return (*self->d_epv->f_Print)(
    self,
    filename);
}

/*
 * Method:  AddtoLocalComponents
 */

int32_t
Hypre_ParCSRVector_AddtoLocalComponents(
  Hypre_ParCSRVector self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_AddtoLocalComponents)(
    self,
    num_values,
    glob_vec_indices,
    value_indices,
    values);
}

/*
 * Method:  GetRow
 */

int32_t
Hypre_ParCSRVector_GetRow(
  Hypre_ParCSRVector self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  return (*self->d_epv->f_GetRow)(
    self,
    row,
    size,
    col_ind,
    values);
}

/*
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
Hypre_ParCSRVector_SetValues(
  Hypre_ParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_SetValues)(
    self,
    nvalues,
    indices,
    values);
}

/*
 * Method:  AddToLocalComponentsInBlock
 */

int32_t
Hypre_ParCSRVector_AddToLocalComponentsInBlock(
  Hypre_ParCSRVector self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  return (*self->d_epv->f_AddToLocalComponentsInBlock)(
    self,
    glob_vec_index_start,
    glob_vec_index_stop,
    value_indices,
    values);
}

/*
 * Cast method for interface and class type conversions.
 */

Hypre_ParCSRVector
Hypre_ParCSRVector__cast(
  void* obj)
{
  Hypre_ParCSRVector cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (Hypre_ParCSRVector) (*base->d_epv->f__cast)(
      base->d_object,
      "Hypre.ParCSRVector");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
Hypre_ParCSRVector__cast2(
  void* obj,
  const char* type)
{
  void* cast = NULL;

  if (obj != NULL) {
    SIDL_BaseInterface base = (SIDL_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type);
  }

  return cast;
}
/*
 * Create a new copy of the array.
 */

struct Hypre_ParCSRVector__array*
Hypre_ParCSRVector__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (*(_getIOR()->createArray))(dimen, lower, upper);
}

/*
 * Borrow a pointer to the array.
 */

struct Hypre_ParCSRVector__array*
Hypre_ParCSRVector__array_borrow(
  struct Hypre_ParCSRVector__object** firstElement,
  int32_t                             dimen,
  const int32_t                       lower[],
  const int32_t                       upper[],
  const int32_t                       stride[])
{
  return (*(_getIOR()->borrowArray))
    (firstElement, dimen, lower, upper, stride);
}

/*
 * Destructor for the array.
 */

void
Hypre_ParCSRVector__array_destroy(
  struct Hypre_ParCSRVector__array* array)
{
  (*(_getIOR()->destroyArray))(array);
}

/*
 * Return the dimension of the array.
 */

int32_t
Hypre_ParCSRVector__array_dimen(const struct Hypre_ParCSRVector__array *array)
{
  return (*(_getIOR()->getDimen))(array);
}

/*
 * Return array lower bounds.
 */

int32_t
Hypre_ParCSRVector__array_lower(const struct Hypre_ParCSRVector__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getLower))(array,ind);
}

/*
 * Return array upper bounds.
 */

int32_t
Hypre_ParCSRVector__array_upper(const struct Hypre_ParCSRVector__array *array,
  int32_t ind)
{
  return (*(_getIOR()->getUpper))(array,ind);
}

/*
 * Return an element of the array.
 */

struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__array_get(
  const struct Hypre_ParCSRVector__array* array,
  const int32_t                           indices[])
{
  return (*(_getIOR()->getElement))(array,indices);
}

/*
 * Return an element of the array.
 */

struct Hypre_ParCSRVector__object*
Hypre_ParCSRVector__array_get4(
  const struct Hypre_ParCSRVector__array* array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4)
{
  return (*(_getIOR()->getElement4))
    (array, i1, i2, i3, i4);
}

/*
 * Set an element of the array.
 */

void
Hypre_ParCSRVector__array_set(
  struct Hypre_ParCSRVector__array*  array,
  const int32_t                      indices[],
  struct Hypre_ParCSRVector__object* value)
{
  (*(_getIOR()->setElement))(array,indices, value);
}

/*
 * Set an element of the array.
 */

void
Hypre_ParCSRVector__array_set4(
  struct Hypre_ParCSRVector__array*  array,
  int32_t                            i1,
  int32_t                            i2,
  int32_t                            i3,
  int32_t                            i4,
  struct Hypre_ParCSRVector__object* value)
{
  (*(_getIOR()->setElement4))
    (array, i1, i2, i3, i4, value);
}
