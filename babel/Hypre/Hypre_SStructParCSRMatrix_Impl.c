/*
 * File:          Hypre_SStructParCSRMatrix_Impl.c
 * Symbol:        Hypre.SStructParCSRMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 837
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.SStructParCSRMatrix" (version 0.1.7)
 * 
 * The SStructParCSR matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */

#include "Hypre_SStructParCSRMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix__ctor"

void
impl_Hypre_SStructParCSRMatrix__ctor(
  Hypre_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix__dtor"

void
impl_Hypre_SStructParCSRMatrix__dtor(
  Hypre_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetCommunicator"

int32_t
impl_Hypre_SStructParCSRMatrix_SetCommunicator(
  Hypre_SStructParCSRMatrix self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetIntParameter"

int32_t
impl_Hypre_SStructParCSRMatrix_SetIntParameter(
  Hypre_SStructParCSRMatrix self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetDoubleParameter"

int32_t
impl_Hypre_SStructParCSRMatrix_SetDoubleParameter(
  Hypre_SStructParCSRMatrix self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetDoubleParameter) 
    */
  /* Insert the implementation of the SetDoubleParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetStringParameter"

int32_t
impl_Hypre_SStructParCSRMatrix_SetStringParameter(
  Hypre_SStructParCSRMatrix self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetStringParameter) 
    */
  /* Insert the implementation of the SetStringParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetStringParameter) */
}

/*
 * Set the int array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetIntArrayParameter"

int32_t
impl_Hypre_SStructParCSRMatrix_SetIntArrayParameter(
  Hypre_SStructParCSRMatrix self, const char* name,
    struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE 
    splicer.begin(Hypre.SStructParCSRMatrix.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetIntArrayParameter) 
    */
}

/*
 * Set the double array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetDoubleArrayParameter"

int32_t
impl_Hypre_SStructParCSRMatrix_SetDoubleArrayParameter(
  Hypre_SStructParCSRMatrix self, const char* name,
    struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE 
    splicer.begin(Hypre.SStructParCSRMatrix.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
  /* DO-NOT-DELETE 
    splicer.end(Hypre.SStructParCSRMatrix.SetDoubleArrayParameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_GetIntValue"

int32_t
impl_Hypre_SStructParCSRMatrix_GetIntValue(
  Hypre_SStructParCSRMatrix self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_GetDoubleValue"

int32_t
impl_Hypre_SStructParCSRMatrix_GetDoubleValue(
  Hypre_SStructParCSRMatrix self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_Setup"

int32_t
impl_Hypre_SStructParCSRMatrix_Setup(
  Hypre_SStructParCSRMatrix self, Hypre_Vector b, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_Apply"

int32_t
impl_Hypre_SStructParCSRMatrix_Apply(
  Hypre_SStructParCSRMatrix self, Hypre_Vector b, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.Apply) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_Initialize"

int32_t
impl_Hypre_SStructParCSRMatrix_Initialize(
  Hypre_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.Initialize) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_Assemble"

int32_t
impl_Hypre_SStructParCSRMatrix_Assemble(
  Hypre_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.Assemble) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_GetObject"

int32_t
impl_Hypre_SStructParCSRMatrix_GetObject(
  Hypre_SStructParCSRMatrix self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.GetObject) */
}

/*
 * Set the matrix graph.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetGraph"

int32_t
impl_Hypre_SStructParCSRMatrix_SetGraph(
  Hypre_SStructParCSRMatrix self, Hypre_SStructGraph graph)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetGraph) */
  /* Insert the implementation of the SetGraph method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetGraph) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetValues"

int32_t
impl_Hypre_SStructParCSRMatrix_SetValues(
  Hypre_SStructParCSRMatrix self, int32_t part, struct SIDL_int__array* index,
    int32_t var, int32_t nentries, struct SIDL_int__array* entries,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetValues) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetBoxValues"

int32_t
impl_Hypre_SStructParCSRMatrix_SetBoxValues(
  Hypre_SStructParCSRMatrix self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var, int32_t nentries,
    struct SIDL_int__array* entries, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetBoxValues) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_AddToValues"

int32_t
impl_Hypre_SStructParCSRMatrix_AddToValues(
  Hypre_SStructParCSRMatrix self, int32_t part, struct SIDL_int__array* index,
    int32_t var, int32_t nentries, struct SIDL_int__array* entries,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.AddToValues) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_AddToBoxValues"

int32_t
impl_Hypre_SStructParCSRMatrix_AddToBoxValues(
  Hypre_SStructParCSRMatrix self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var, int32_t nentries,
    struct SIDL_int__array* entries, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.AddToBoxValues) */
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
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetSymmetric"

int32_t
impl_Hypre_SStructParCSRMatrix_SetSymmetric(
  Hypre_SStructParCSRMatrix self, int32_t part, int32_t var, int32_t to_var,
    int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetSymmetric) */
}

/*
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetNSSymmetric"

int32_t
impl_Hypre_SStructParCSRMatrix_SetNSSymmetric(
  Hypre_SStructParCSRMatrix self, int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetNSSymmetric) */
  /* Insert the implementation of the SetNSSymmetric method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetNSSymmetric) */
}

/*
 * Set the matrix to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_SetComplex"

int32_t
impl_Hypre_SStructParCSRMatrix_SetComplex(
  Hypre_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.SetComplex) */
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructParCSRMatrix_Print"

int32_t
impl_Hypre_SStructParCSRMatrix_Print(
  Hypre_SStructParCSRMatrix self, const char* filename, int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructParCSRMatrix.Print) */
  /* Insert the implementation of the Print method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructParCSRMatrix.Print) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
