/*
 * File:          Hypre_ParCSRVector_Impl.c
 * Symbol:        Hypre.ParCSRVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:19 PST
 * Description:   Server-side implementation for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.ParCSRVector" (version 0.1.5)
 */

#include "Hypre_ParCSRVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector__ctor"

void
impl_Hypre_ParCSRVector__ctor(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector__dtor"

void
impl_Hypre_ParCSRVector__dtor(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector._dtor) */
}

/*
 * Method:  AddToLocalComponentsInBlock
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_AddToLocalComponentsInBlock"

int32_t
impl_Hypre_ParCSRVector_AddToLocalComponentsInBlock(
  Hypre_ParCSRVector self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE 
    splicer.begin(Hypre.ParCSRVector.AddToLocalComponentsInBlock) */
  /* Insert the implementation of the AddToLocalComponentsInBlock method 
    here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.AddToLocalComponentsInBlock) 
    */
}

/*
 * Adds to values in vector.  Usage details are analogous to
 * \Ref{SetValues}.
 * 
 * Not collective.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_AddToValues"

int32_t
impl_Hypre_ParCSRVector_AddToValues(
  Hypre_ParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.AddToValues) */
}

/*
 * Method:  AddtoLocalComponents
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_AddtoLocalComponents"

int32_t
impl_Hypre_ParCSRVector_AddtoLocalComponents(
  Hypre_ParCSRVector self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.AddtoLocalComponents) */
  /* Insert the implementation of the AddtoLocalComponents method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.AddtoLocalComponents) */
}

/*
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Assemble"

int32_t
impl_Hypre_ParCSRVector_Assemble(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Assemble) */
}

/*
 * y <- a*x + y
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Axpy"

int32_t
impl_Hypre_ParCSRVector_Axpy(
  Hypre_ParCSRVector self,
  double a,
  Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Axpy) */
}

/*
 * y <- 0 (where y=self)
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Clear"

int32_t
impl_Hypre_ParCSRVector_Clear(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Clear) */
  /* Insert the implementation of the Clear method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Clear) */
}

/*
 * create an x compatible with y
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Clone"

int32_t
impl_Hypre_ParCSRVector_Clone(
  Hypre_ParCSRVector self,
  Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Clone) */
  /* Insert the implementation of the Clone method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Clone) */
}

/*
 * y <- x 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Copy"

int32_t
impl_Hypre_ParCSRVector_Copy(
  Hypre_ParCSRVector self,
  Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Copy) */
  /* Insert the implementation of the Copy method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Copy) */
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

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Create"

int32_t
impl_Hypre_ParCSRVector_Create(
  Hypre_ParCSRVector self,
  void* comm,
  int32_t jlower,
  int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Create) */
  /* Insert the implementation of the Create method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Create) */
}

/*
 * d <- (y,x)
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Dot"

int32_t
impl_Hypre_ParCSRVector_Dot(
  Hypre_ParCSRVector self,
  Hypre_Vector x,
  double* d)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Dot) */
  /* Insert the implementation of the Dot method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Dot) */
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

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_GetObject"

int32_t
impl_Hypre_ParCSRVector_GetObject(
  Hypre_ParCSRVector self,
  SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.GetObject) */
}

/*
 * Method:  GetRow
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_GetRow"

int32_t
impl_Hypre_ParCSRVector_GetRow(
  Hypre_ParCSRVector self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.GetRow) */
  /* Insert the implementation of the GetRow method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.GetRow) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Initialize"

int32_t
impl_Hypre_ParCSRVector_Initialize(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Initialize) */
}

/*
 * Print the vector to file.  This is mainly for debugging purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Print"

int32_t
impl_Hypre_ParCSRVector_Print(
  Hypre_ParCSRVector self,
  const char* filename)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Print) */
  /* Insert the implementation of the Print method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Print) */
}

/*
 * Read the vector from file.  This is mainly for debugging purposes.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Read"

int32_t
impl_Hypre_ParCSRVector_Read(
  Hypre_ParCSRVector self,
  const char* filename,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Read) */
  /* Insert the implementation of the Read method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Read) */
}

/*
 * y <- a*y 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Scale"

int32_t
impl_Hypre_ParCSRVector_Scale(
  Hypre_ParCSRVector self,
  double a)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Scale) */
  /* Insert the implementation of the Scale method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Scale) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetCommunicator"

int32_t
impl_Hypre_ParCSRVector_SetCommunicator(
  Hypre_ParCSRVector self,
  void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetCommunicator) */
}

/*
 * Method:  SetGlobalSize
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetGlobalSize"

int32_t
impl_Hypre_ParCSRVector_SetGlobalSize(
  Hypre_ParCSRVector self,
  int32_t n)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetGlobalSize) */
  /* Insert the implementation of the SetGlobalSize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetGlobalSize) */
}

/*
 * Method:  SetLocalComponents
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetLocalComponents"

int32_t
impl_Hypre_ParCSRVector_SetLocalComponents(
  Hypre_ParCSRVector self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetLocalComponents) */
  /* Insert the implementation of the SetLocalComponents method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetLocalComponents) */
}

/*
 * Method:  SetLocalComponentsInBlock
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetLocalComponentsInBlock"

int32_t
impl_Hypre_ParCSRVector_SetLocalComponentsInBlock(
  Hypre_ParCSRVector self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetLocalComponentsInBlock) 
    */
  /* Insert the implementation of the SetLocalComponentsInBlock method here... 
    */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetLocalComponentsInBlock) */
}

/*
 * Method:  SetPartitioning
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetPartitioning"

int32_t
impl_Hypre_ParCSRVector_SetPartitioning(
  Hypre_ParCSRVector self,
  struct SIDL_int__array* partitioning)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetPartitioning) */
  /* Insert the implementation of the SetPartitioning method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetPartitioning) */
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

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetValues"

int32_t
impl_Hypre_ParCSRVector_SetValues(
  Hypre_ParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetValues) */
}
