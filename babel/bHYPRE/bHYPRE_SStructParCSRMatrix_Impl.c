/*
 * File:          bHYPRE_SStructParCSRMatrix_Impl.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:30 PST
 * Description:   Server-side implementation for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 827
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructParCSRMatrix" (version 1.0.0)
 * 
 * The SStructParCSR matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_SStructParCSRMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix__ctor"

void
impl_bHYPRE_SStructParCSRMatrix__ctor(
  bHYPRE_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix__dtor"

void
impl_bHYPRE_SStructParCSRMatrix__dtor(
  bHYPRE_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetCommunicator"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  bHYPRE_SStructParCSRMatrix self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntParameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  bHYPRE_SStructParCSRMatrix self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  bHYPRE_SStructParCSRMatrix self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleParameter) 
    */
  /* Insert the implementation of the SetDoubleParameter method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetStringParameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  bHYPRE_SStructParCSRMatrix self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetStringParameter) 
    */
  /* Insert the implementation of the SetStringParameter method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  bHYPRE_SStructParCSRMatrix self, const char* name,
    struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE 
    splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE 
    splicer.end(bHYPRE.SStructParCSRMatrix.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  bHYPRE_SStructParCSRMatrix self, const char* name,
    struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE 
    splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE 
    splicer.end(bHYPRE.SStructParCSRMatrix.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  bHYPRE_SStructParCSRMatrix self, const char* name,
    struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE 
    splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE 
    splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  bHYPRE_SStructParCSRMatrix self, const char* name,
    struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE 
    splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE 
    splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetIntValue"

int32_t
impl_bHYPRE_SStructParCSRMatrix_GetIntValue(
  bHYPRE_SStructParCSRMatrix self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue"

int32_t
impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  bHYPRE_SStructParCSRMatrix self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Setup"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Setup(
  bHYPRE_SStructParCSRMatrix self, bHYPRE_Vector b, bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Apply"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Apply(
  bHYPRE_SStructParCSRMatrix self, bHYPRE_Vector b, bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Apply) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Initialize"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Initialize(
  bHYPRE_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Assemble"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Assemble(
  bHYPRE_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Assemble) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetObject"

int32_t
impl_bHYPRE_SStructParCSRMatrix_GetObject(
  bHYPRE_SStructParCSRMatrix self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetObject) */
}

/*
 * Set the matrix graph.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetGraph"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetGraph(
  bHYPRE_SStructParCSRMatrix self, bHYPRE_SStructGraph graph)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetGraph) */
  /* Insert the implementation of the SetGraph method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetGraph) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetValues"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetValues(
  bHYPRE_SStructParCSRMatrix self, int32_t part, struct SIDL_int__array* index,
    int32_t var, int32_t nentries, struct SIDL_int__array* entries,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetBoxValues"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  bHYPRE_SStructParCSRMatrix self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var, int32_t nentries,
    struct SIDL_int__array* entries, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetBoxValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_AddToValues"

int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToValues(
  bHYPRE_SStructParCSRMatrix self, int32_t part, struct SIDL_int__array* index,
    int32_t var, int32_t nentries, struct SIDL_int__array* entries,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.AddToValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues"

int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  bHYPRE_SStructParCSRMatrix self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t var, int32_t nentries,
    struct SIDL_int__array* entries, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.AddToBoxValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetSymmetric"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  bHYPRE_SStructParCSRMatrix self, int32_t part, int32_t var, int32_t to_var,
    int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetSymmetric) */
}

/*
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  bHYPRE_SStructParCSRMatrix self, int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetNSSymmetric) */
  /* Insert the implementation of the SetNSSymmetric method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetNSSymmetric) */
}

/*
 * Set the matrix to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetComplex"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetComplex(
  bHYPRE_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetComplex) */
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Print"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Print(
  bHYPRE_SStructParCSRMatrix self, const char* filename, int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Print) */
  /* Insert the implementation of the Print method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Print) */
}
