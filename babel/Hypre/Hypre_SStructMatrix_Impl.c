/*
 * File:          Hypre_SStructMatrix_Impl.c
 * Symbol:        Hypre.SStructMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1072
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.SStructMatrix" (version 0.1.7)
 * 
 * The semi-structured grid matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */

#include "Hypre_SStructMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix__ctor"

void
impl_Hypre_SStructMatrix__ctor(
  Hypre_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix__dtor"

void
impl_Hypre_SStructMatrix__dtor(
  Hypre_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetCommunicator"

int32_t
impl_Hypre_SStructMatrix_SetCommunicator(
  Hypre_SStructMatrix self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetIntParameter"

int32_t
impl_Hypre_SStructMatrix_SetIntParameter(
  Hypre_SStructMatrix self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetDoubleParameter"

int32_t
impl_Hypre_SStructMatrix_SetDoubleParameter(
  Hypre_SStructMatrix self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetStringParameter"

int32_t
impl_Hypre_SStructMatrix_SetStringParameter(
  Hypre_SStructMatrix self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetStringParameter) */
}

/*
 * Set the int array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetIntArrayParameter"

int32_t
impl_Hypre_SStructMatrix_SetIntArrayParameter(
  Hypre_SStructMatrix self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetIntArrayParameter) */
}

/*
 * Set the double array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetDoubleArrayParameter"

int32_t
impl_Hypre_SStructMatrix_SetDoubleArrayParameter(
  Hypre_SStructMatrix self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetDoubleArrayParameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_GetIntValue"

int32_t
impl_Hypre_SStructMatrix_GetIntValue(
  Hypre_SStructMatrix self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_GetDoubleValue"

int32_t
impl_Hypre_SStructMatrix_GetDoubleValue(
  Hypre_SStructMatrix self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_Setup"

int32_t
impl_Hypre_SStructMatrix_Setup(
  Hypre_SStructMatrix self, Hypre_Vector b, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_Apply"

int32_t
impl_Hypre_SStructMatrix_Apply(
  Hypre_SStructMatrix self, Hypre_Vector b, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.Apply) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_Initialize"

int32_t
impl_Hypre_SStructMatrix_Initialize(
  Hypre_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.Initialize) */
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
#define __FUNC__ "impl_Hypre_SStructMatrix_Assemble"

int32_t
impl_Hypre_SStructMatrix_Assemble(
  Hypre_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.Assemble) */
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
#define __FUNC__ "impl_Hypre_SStructMatrix_GetObject"

int32_t
impl_Hypre_SStructMatrix_GetObject(
  Hypre_SStructMatrix self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.GetObject) */
}

/*
 * Set the matrix graph.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetGraph"

int32_t
impl_Hypre_SStructMatrix_SetGraph(
  Hypre_SStructMatrix self, Hypre_SStructGraph graph)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetGraph) */
  /* Insert the implementation of the SetGraph method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetGraph) */
}

/*
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetValues"

int32_t
impl_Hypre_SStructMatrix_SetValues(
  Hypre_SStructMatrix self, int32_t part, struct SIDL_int__array* index,
    int32_t var, int32_t nentries, struct SIDL_int__array* entries,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetValues) */
}

/*
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetBoxValues"

int32_t
impl_Hypre_SStructMatrix_SetBoxValues(
  Hypre_SStructMatrix self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var, int32_t nentries,
    struct SIDL_int__array* entries, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetBoxValues) */
}

/*
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_AddToValues"

int32_t
impl_Hypre_SStructMatrix_AddToValues(
  Hypre_SStructMatrix self, int32_t part, struct SIDL_int__array* index,
    int32_t var, int32_t nentries, struct SIDL_int__array* entries,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.AddToValues) */
}

/*
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_AddToBoxValues"

int32_t
impl_Hypre_SStructMatrix_AddToBoxValues(
  Hypre_SStructMatrix self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var, int32_t nentries,
    struct SIDL_int__array* entries, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.AddToBoxValues) */
}

/*
 * Define symmetry properties for the stencil entries in the
 * matrix.  The boolean argument {\tt symmetric} is applied to
 * stencil entries on part {\tt part} that couple variable {\tt
 * var} to variable {\tt to\_var}.  A value of -1 may be used
 * for {\tt part}, {\tt var}, or {\tt to\_var} to specify
 * ``all''.  For example, if {\tt part} and {\tt to\_var} are
 * set to -1, then the boolean is applied to stencil entries on
 * all parts that couple variable {\tt var} to all other
 * variables.
 * 
 * By default, matrices are assumed to be nonsymmetric.
 * Significant storage savings can be made if the matrix is
 * symmetric.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetSymmetric"

int32_t
impl_Hypre_SStructMatrix_SetSymmetric(
  Hypre_SStructMatrix self, int32_t part, int32_t var, int32_t to_var,
    int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetSymmetric) */
}

/*
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetNSSymmetric"

int32_t
impl_Hypre_SStructMatrix_SetNSSymmetric(
  Hypre_SStructMatrix self, int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetNSSymmetric) */
  /* Insert the implementation of the SetNSSymmetric method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetNSSymmetric) */
}

/*
 * Set the matrix to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_SetComplex"

int32_t
impl_Hypre_SStructMatrix_SetComplex(
  Hypre_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.SetComplex) */
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructMatrix_Print"

int32_t
impl_Hypre_SStructMatrix_Print(
  Hypre_SStructMatrix self, const char* filename, int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructMatrix.Print) */
  /* Insert the implementation of the Print method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructMatrix.Print) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
