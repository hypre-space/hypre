/*
 * File:          Hypre_SStructParCSRVector_Impl.c
 * Symbol:        Hypre.SStructParCSRVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 847
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.SStructParCSRVector" (version 0.1.7)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

#include "Hypre_SStructParCSRVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector__ctor"

void
impl_Hypre_SStructParCSRVector__ctor(
  Hypre_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector__dtor"

void
impl_Hypre_SStructParCSRVector__dtor(
  Hypre_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_SetCommunicator"

int32_t
impl_Hypre_SStructParCSRVector_SetCommunicator(
  Hypre_SStructParCSRVector self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Initialize"

int32_t
impl_Hypre_SStructParCSRVector_Initialize(
  Hypre_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Initialize) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Assemble"

int32_t
impl_Hypre_SStructParCSRVector_Assemble(
  Hypre_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Assemble) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_GetObject"

int32_t
impl_Hypre_SStructParCSRVector_GetObject(
  Hypre_SStructParCSRVector self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.GetObject) */
}

/*
 * Set the vector grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_SetGrid"

int32_t
impl_Hypre_SStructParCSRVector_SetGrid(
  Hypre_SStructParCSRVector self, Hypre_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.SetGrid) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_SetValues"

int32_t
impl_Hypre_SStructParCSRVector_SetValues(
  Hypre_SStructParCSRVector self, int32_t part, struct SIDL_int__array* index,
    int32_t var, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.SetValues) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_SetBoxValues"

int32_t
impl_Hypre_SStructParCSRVector_SetBoxValues(
  Hypre_SStructParCSRVector self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.SetBoxValues) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_AddToValues"

int32_t
impl_Hypre_SStructParCSRVector_AddToValues(
  Hypre_SStructParCSRVector self, int32_t part, struct SIDL_int__array* index,
    int32_t var, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.AddToValues) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_AddToBoxValues"

int32_t
impl_Hypre_SStructParCSRVector_AddToBoxValues(
  Hypre_SStructParCSRVector self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.AddToBoxValues) */
}

/*
 * Gather vector data before calling {\tt GetValues}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Gather"

int32_t
impl_Hypre_SStructParCSRVector_Gather(
  Hypre_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Gather) */
  /* Insert the implementation of the Gather method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Gather) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_GetValues"

int32_t
impl_Hypre_SStructParCSRVector_GetValues(
  Hypre_SStructParCSRVector self, int32_t part, struct SIDL_int__array* index,
    int32_t var, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.GetValues) */
  /* Insert the implementation of the GetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.GetValues) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_GetBoxValues"

int32_t
impl_Hypre_SStructParCSRVector_GetBoxValues(
  Hypre_SStructParCSRVector self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var,
    struct SIDL_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.GetBoxValues) */
  /* Insert the implementation of the GetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.GetBoxValues) */
}

/*
 * Set the vector to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_SetComplex"

int32_t
impl_Hypre_SStructParCSRVector_SetComplex(
  Hypre_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.SetComplex) */
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Print"

int32_t
impl_Hypre_SStructParCSRVector_Print(
  Hypre_SStructParCSRVector self, const char* filename, int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Print) */
  /* Insert the implementation of the Print method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Print) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Clear"

int32_t
impl_Hypre_SStructParCSRVector_Clear(
  Hypre_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Clear) */
  /* Insert the implementation of the Clear method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Copy"

int32_t
impl_Hypre_SStructParCSRVector_Copy(
  Hypre_SStructParCSRVector self, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Copy) */
  /* Insert the implementation of the Copy method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Copy) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Clone"

int32_t
impl_Hypre_SStructParCSRVector_Clone(
  Hypre_SStructParCSRVector self, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Clone) */
  /* Insert the implementation of the Clone method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Clone) */
}

/*
 * Scale {\self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Scale"

int32_t
impl_Hypre_SStructParCSRVector_Scale(
  Hypre_SStructParCSRVector self, double a)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Scale) */
  /* Insert the implementation of the Scale method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Dot"

int32_t
impl_Hypre_SStructParCSRVector_Dot(
  Hypre_SStructParCSRVector self, Hypre_Vector x, double* d)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Dot) */
  /* Insert the implementation of the Dot method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRVector_Axpy"

int32_t
impl_Hypre_SStructParCSRVector_Axpy(
  Hypre_SStructParCSRVector self, double a, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRVector.Axpy) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
