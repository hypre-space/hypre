/*
 * File:          bHYPRE_SStructVector_Impl.c
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:44 PST
 * Description:   Server-side implementation for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1079
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructVector" (version 1.0.0)
 * 
 * The semi-structured grid vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_SStructVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector__ctor"

void
impl_bHYPRE_SStructVector__ctor(
  /*in*/ bHYPRE_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector__dtor"

void
impl_bHYPRE_SStructVector__dtor(
  /*in*/ bHYPRE_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector._dtor) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_Clear"

int32_t
impl_bHYPRE_SStructVector_Clear(
  /*in*/ bHYPRE_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Clear) */
  /* Insert the implementation of the Clear method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_Copy"

int32_t
impl_bHYPRE_SStructVector_Copy(
  /*in*/ bHYPRE_SStructVector self, /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Copy) */
  /* Insert the implementation of the Copy method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Copy) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_Clone"

int32_t
impl_bHYPRE_SStructVector_Clone(
  /*in*/ bHYPRE_SStructVector self, /*out*/ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Clone) */
  /* Insert the implementation of the Clone method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Clone) */
}

/*
 * Scale {\tt self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_Scale"

int32_t
impl_bHYPRE_SStructVector_Scale(
  /*in*/ bHYPRE_SStructVector self, /*in*/ double a)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Scale) */
  /* Insert the implementation of the Scale method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_Dot"

int32_t
impl_bHYPRE_SStructVector_Dot(
  /*in*/ bHYPRE_SStructVector self, /*in*/ bHYPRE_Vector x, /*out*/ double* d)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Dot) */
  /* Insert the implementation of the Dot method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_Axpy"

int32_t
impl_bHYPRE_SStructVector_Axpy(
  /*in*/ bHYPRE_SStructVector self, /*in*/ double a, /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Axpy) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_SetCommunicator"

int32_t
impl_bHYPRE_SStructVector_SetCommunicator(
  /*in*/ bHYPRE_SStructVector self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_Initialize"

int32_t
impl_bHYPRE_SStructVector_Initialize(
  /*in*/ bHYPRE_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_Assemble"

int32_t
impl_bHYPRE_SStructVector_Assemble(
  /*in*/ bHYPRE_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Assemble) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_GetObject"

int32_t
impl_bHYPRE_SStructVector_GetObject(
  /*in*/ bHYPRE_SStructVector self, /*out*/ sidl_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.GetObject) */
}

/*
 * Set the vector grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_SetGrid"

int32_t
impl_bHYPRE_SStructVector_SetGrid(
  /*in*/ bHYPRE_SStructVector self, /*in*/ bHYPRE_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.SetGrid) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_SetValues"

int32_t
impl_bHYPRE_SStructVector_SetValues(
  /*in*/ bHYPRE_SStructVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.SetValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_SetBoxValues"

int32_t
impl_bHYPRE_SStructVector_SetBoxValues(
  /*in*/ bHYPRE_SStructVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.SetBoxValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_AddToValues"

int32_t
impl_bHYPRE_SStructVector_AddToValues(
  /*in*/ bHYPRE_SStructVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.AddToValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_AddToBoxValues"

int32_t
impl_bHYPRE_SStructVector_AddToBoxValues(
  /*in*/ bHYPRE_SStructVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.AddToBoxValues) */
}

/*
 * Gather vector data before calling {\tt GetValues}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_Gather"

int32_t
impl_bHYPRE_SStructVector_Gather(
  /*in*/ bHYPRE_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Gather) */
  /* Insert the implementation of the Gather method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Gather) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_GetValues"

int32_t
impl_bHYPRE_SStructVector_GetValues(
  /*in*/ bHYPRE_SStructVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.GetValues) */
  /* Insert the implementation of the GetValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.GetValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructVector_GetBoxValues"

int32_t
impl_bHYPRE_SStructVector_GetBoxValues(
  /*in*/ bHYPRE_SStructVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*inout*/ struct sidl_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.GetBoxValues) */
  /* Insert the implementation of the GetBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.GetBoxValues) */
}

/*
 * Set the vector to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_SetComplex"

int32_t
impl_bHYPRE_SStructVector_SetComplex(
  /*in*/ bHYPRE_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.SetComplex) */
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructVector_Print"

int32_t
impl_bHYPRE_SStructVector_Print(
  /*in*/ bHYPRE_SStructVector self, /*in*/ const char* filename,
    /*in*/ int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructVector.Print) */
  /* Insert the implementation of the Print method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructVector.Print) */
}
