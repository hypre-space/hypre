/*
 * File:          bHYPRE_StructVector_Impl.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:31 PST
 * Description:   Server-side implementation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1129
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */

#include "bHYPRE_StructVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector__ctor"

void
impl_bHYPRE_StructVector__ctor(
  bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector__dtor"

void
impl_bHYPRE_StructVector__dtor(
  bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetCommunicator"

int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  bHYPRE_StructVector self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Initialize"

int32_t
impl_bHYPRE_StructVector_Initialize(
  bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Initialize) */
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Assemble"

int32_t
impl_bHYPRE_StructVector_Assemble(
  bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Assemble) */
}

/*
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_GetObject"

int32_t
impl_bHYPRE_StructVector_GetObject(
  bHYPRE_StructVector self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.GetObject) */
}

/*
 * Method:  SetGrid[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetGrid"

int32_t
impl_bHYPRE_StructVector_SetGrid(
  bHYPRE_StructVector self, bHYPRE_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetGrid) */
}

/*
 * Method:  SetStencil[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetStencil"

int32_t
impl_bHYPRE_StructVector_SetStencil(
  bHYPRE_StructVector self, bHYPRE_StructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetStencil) */
}

/*
 * Method:  SetValue[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetValue"

int32_t
impl_bHYPRE_StructVector_SetValue(
  bHYPRE_StructVector self, struct SIDL_int__array* grid_index, double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetValue) */
  /* Insert the implementation of the SetValue method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetValue) */
}

/*
 * Method:  SetBoxValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetBoxValues"

int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  bHYPRE_StructVector self, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetBoxValues) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Clear"

int32_t
impl_bHYPRE_StructVector_Clear(
  bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Clear) */
  /* Insert the implementation of the Clear method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Copy"

int32_t
impl_bHYPRE_StructVector_Copy(
  bHYPRE_StructVector self, bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Copy) */
  /* Insert the implementation of the Copy method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Copy) */
}

/*
 * Create an {\tt x} compatible with {\tt self}.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Clone"

int32_t
impl_bHYPRE_StructVector_Clone(
  bHYPRE_StructVector self, bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Clone) */
  /* Insert the implementation of the Clone method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Clone) */
}

/*
 * Scale {\self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Scale"

int32_t
impl_bHYPRE_StructVector_Scale(
  bHYPRE_StructVector self, double a)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Scale) */
  /* Insert the implementation of the Scale method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Dot"

int32_t
impl_bHYPRE_StructVector_Dot(
  bHYPRE_StructVector self, bHYPRE_Vector x, double* d)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Dot) */
  /* Insert the implementation of the Dot method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Axpy"

int32_t
impl_bHYPRE_StructVector_Axpy(
  bHYPRE_StructVector self, double a, bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Axpy) */
}
