/*
 * File:          bHYPRE_SStructMatrix_Impl.c
 * Symbol:        bHYPRE.SStructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:40 PST
 * Description:   Server-side implementation for bHYPRE.SStructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1062
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructMatrix" (version 1.0.0)
 * 
 * The semi-structured grid matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_SStructMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix__ctor"

void
impl_bHYPRE_SStructMatrix__ctor(
  /*in*/ bHYPRE_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix__dtor"

void
impl_bHYPRE_SStructMatrix__dtor(
  /*in*/ bHYPRE_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetCommunicator"

int32_t
impl_bHYPRE_SStructMatrix_SetCommunicator(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetIntParameter"

int32_t
impl_bHYPRE_SStructMatrix_SetIntParameter(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*in*/ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetDoubleParameter"

int32_t
impl_bHYPRE_SStructMatrix_SetDoubleParameter(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*in*/ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetStringParameter"

int32_t
impl_bHYPRE_SStructMatrix_SetStringParameter(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*in*/ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetIntArray1Parameter"

int32_t
impl_bHYPRE_SStructMatrix_SetIntArray1Parameter(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetIntArray2Parameter"

int32_t
impl_bHYPRE_SStructMatrix_SetIntArray2Parameter(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetDoubleArray1Parameter) 
    */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_SStructMatrix_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetDoubleArray2Parameter) 
    */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_GetIntValue"

int32_t
impl_bHYPRE_SStructMatrix_GetIntValue(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*out*/ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_GetDoubleValue"

int32_t
impl_bHYPRE_SStructMatrix_GetDoubleValue(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* name,
    /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_Setup"

int32_t
impl_bHYPRE_SStructMatrix_Setup(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ bHYPRE_Vector b,
    /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_Apply"

int32_t
impl_bHYPRE_SStructMatrix_Apply(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ bHYPRE_Vector b,
    /*inout*/ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.Apply) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_Initialize"

int32_t
impl_bHYPRE_SStructMatrix_Initialize(
  /*in*/ bHYPRE_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_SStructMatrix_Assemble"

int32_t
impl_bHYPRE_SStructMatrix_Assemble(
  /*in*/ bHYPRE_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.Assemble) */
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
#define __FUNC__ "impl_bHYPRE_SStructMatrix_GetObject"

int32_t
impl_bHYPRE_SStructMatrix_GetObject(
  /*in*/ bHYPRE_SStructMatrix self, /*out*/ sidl_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.GetObject) */
}

/*
 * Set the matrix graph.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetGraph"

int32_t
impl_bHYPRE_SStructMatrix_SetGraph(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ bHYPRE_SStructGraph graph)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetGraph) */
  /* Insert the implementation of the SetGraph method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetGraph) */
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
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetValues"

int32_t
impl_bHYPRE_SStructMatrix_SetValues(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ int32_t nentries, /*in*/ struct sidl_int__array* entries,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetBoxValues"

int32_t
impl_bHYPRE_SStructMatrix_SetBoxValues(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ int32_t nentries, /*in*/ struct sidl_int__array* entries,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetBoxValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructMatrix_AddToValues"

int32_t
impl_bHYPRE_SStructMatrix_AddToValues(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ int32_t nentries, /*in*/ struct sidl_int__array* entries,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.AddToValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructMatrix_AddToBoxValues"

int32_t
impl_bHYPRE_SStructMatrix_AddToBoxValues(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ int32_t nentries, /*in*/ struct sidl_int__array* entries,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.AddToBoxValues) */
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
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetSymmetric"

int32_t
impl_bHYPRE_SStructMatrix_SetSymmetric(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ int32_t part, /*in*/ int32_t var,
    /*in*/ int32_t to_var, /*in*/ int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetSymmetric) */
}

/*
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetNSSymmetric"

int32_t
impl_bHYPRE_SStructMatrix_SetNSSymmetric(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetNSSymmetric) */
  /* Insert the implementation of the SetNSSymmetric method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetNSSymmetric) */
}

/*
 * Set the matrix to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_SetComplex"

int32_t
impl_bHYPRE_SStructMatrix_SetComplex(
  /*in*/ bHYPRE_SStructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.SetComplex) */
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructMatrix_Print"

int32_t
impl_bHYPRE_SStructMatrix_Print(
  /*in*/ bHYPRE_SStructMatrix self, /*in*/ const char* filename,
    /*in*/ int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructMatrix.Print) */
  /* Insert the implementation of the Print method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructMatrix.Print) */
}
