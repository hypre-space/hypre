/*
 * File:          bHYPRE_SStructParCSRVector_Impl.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:43 PST
 * Description:   Server-side implementation for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 842
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructParCSRVector" (version 1.0.0)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_SStructParCSRVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector__ctor"

void
impl_bHYPRE_SStructParCSRVector__ctor(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector__dtor"

void
impl_bHYPRE_SStructParCSRVector__dtor(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._dtor) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Clear"

int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Clear) */
  /* Insert the implementation of the Clear method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Copy"

int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Copy) */
  /* Insert the implementation of the Copy method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Copy) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Clone"

int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  /*in*/ bHYPRE_SStructParCSRVector self, /*out*/ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Clone) */
  /* Insert the implementation of the Clone method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Clone) */
}

/*
 * Scale {\tt self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Scale"

int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ double a)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Scale) */
  /* Insert the implementation of the Scale method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Dot"

int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ bHYPRE_Vector x,
    /*out*/ double* d)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Dot) */
  /* Insert the implementation of the Dot method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Axpy"

int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ double a,
    /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Axpy) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetCommunicator"

int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Initialize"

int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Assemble"

int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Assemble) */
}

/*
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

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetObject"

int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  /*in*/ bHYPRE_SStructParCSRVector self, /*out*/ sidl_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetObject) */
}

/*
 * Set the vector grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetGrid"

int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ bHYPRE_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetGrid) */
}

/*
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetValues"

int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetValues) */
}

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetBoxValues"

int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetBoxValues) */
}

/*
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_AddToValues"

int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.AddToValues) */
}

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_AddToBoxValues"

int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.AddToBoxValues) */
}

/*
 * Gather vector data before calling {\tt GetValues}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Gather"

int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Gather) */
  /* Insert the implementation of the Gather method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Gather) */
}

/*
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetValues"

int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.GetValues) */
  /* Insert the implementation of the GetValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetValues) */
}

/*
 * Get vector coefficients a box at a time.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetBoxValues"

int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*inout*/ struct sidl_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.GetBoxValues) */
  /* Insert the implementation of the GetBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetBoxValues) */
}

/*
 * Set the vector to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetComplex"

int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetComplex) */
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Print"

int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ const char* filename,
    /*in*/ int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Print) */
  /* Insert the implementation of the Print method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Print) */
}
