/*
 * File:          Hypre_SStructVector_Impl.c
 * Symbol:        Hypre.SStructVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1084
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.SStructVector" (version 0.1.7)
 * 
 * The semi-structured grid vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

#include "Hypre_SStructVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.SStructVector._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector__ctor"

void
impl_Hypre_SStructVector__ctor(
  Hypre_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector__dtor"

void
impl_Hypre_SStructVector__dtor(
  Hypre_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_SetCommunicator"

int32_t
impl_Hypre_SStructVector_SetCommunicator(
  Hypre_SStructVector self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_Initialize"

int32_t
impl_Hypre_SStructVector_Initialize(
  Hypre_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Initialize) */
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
#define __FUNC__ "impl_Hypre_SStructVector_Assemble"

int32_t
impl_Hypre_SStructVector_Assemble(
  Hypre_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Assemble) */
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
#define __FUNC__ "impl_Hypre_SStructVector_GetObject"

int32_t
impl_Hypre_SStructVector_GetObject(
  Hypre_SStructVector self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.GetObject) */
}

/*
 * Set the vector grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_SetGrid"

int32_t
impl_Hypre_SStructVector_SetGrid(
  Hypre_SStructVector self, Hypre_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.SetGrid) */
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
#define __FUNC__ "impl_Hypre_SStructVector_SetValues"

int32_t
impl_Hypre_SStructVector_SetValues(
  Hypre_SStructVector self, int32_t part, struct SIDL_int__array* index,
    int32_t var, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.SetValues) */
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
#define __FUNC__ "impl_Hypre_SStructVector_SetBoxValues"

int32_t
impl_Hypre_SStructVector_SetBoxValues(
  Hypre_SStructVector self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.SetBoxValues) */
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
#define __FUNC__ "impl_Hypre_SStructVector_AddToValues"

int32_t
impl_Hypre_SStructVector_AddToValues(
  Hypre_SStructVector self, int32_t part, struct SIDL_int__array* index,
    int32_t var, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.AddToValues) */
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
#define __FUNC__ "impl_Hypre_SStructVector_AddToBoxValues"

int32_t
impl_Hypre_SStructVector_AddToBoxValues(
  Hypre_SStructVector self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.AddToBoxValues) */
}

/*
 * Gather vector data before calling {\tt GetValues}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_Gather"

int32_t
impl_Hypre_SStructVector_Gather(
  Hypre_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Gather) */
  /* Insert the implementation of the Gather method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Gather) */
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
#define __FUNC__ "impl_Hypre_SStructVector_GetValues"

int32_t
impl_Hypre_SStructVector_GetValues(
  Hypre_SStructVector self, int32_t part, struct SIDL_int__array* index,
    int32_t var, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.GetValues) */
  /* Insert the implementation of the GetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.GetValues) */
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
#define __FUNC__ "impl_Hypre_SStructVector_GetBoxValues"

int32_t
impl_Hypre_SStructVector_GetBoxValues(
  Hypre_SStructVector self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var,
    struct SIDL_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.GetBoxValues) */
  /* Insert the implementation of the GetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.GetBoxValues) */
}

/*
 * Set the vector to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_SetComplex"

int32_t
impl_Hypre_SStructVector_SetComplex(
  Hypre_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.SetComplex) */
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_Print"

int32_t
impl_Hypre_SStructVector_Print(
  Hypre_SStructVector self, const char* filename, int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Print) */
  /* Insert the implementation of the Print method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Print) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_Clear"

int32_t
impl_Hypre_SStructVector_Clear(
  Hypre_SStructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Clear) */
  /* Insert the implementation of the Clear method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_Copy"

int32_t
impl_Hypre_SStructVector_Copy(
  Hypre_SStructVector self, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Copy) */
  /* Insert the implementation of the Copy method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Copy) */
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
#define __FUNC__ "impl_Hypre_SStructVector_Clone"

int32_t
impl_Hypre_SStructVector_Clone(
  Hypre_SStructVector self, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Clone) */
  /* Insert the implementation of the Clone method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Clone) */
}

/*
 * Scale {\self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_Scale"

int32_t
impl_Hypre_SStructVector_Scale(
  Hypre_SStructVector self, double a)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Scale) */
  /* Insert the implementation of the Scale method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_Dot"

int32_t
impl_Hypre_SStructVector_Dot(
  Hypre_SStructVector self, Hypre_Vector x, double* d)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Dot) */
  /* Insert the implementation of the Dot method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructVector_Axpy"

int32_t
impl_Hypre_SStructVector_Axpy(
  Hypre_SStructVector self, double a, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructVector.Axpy) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
