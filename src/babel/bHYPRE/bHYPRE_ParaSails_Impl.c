/*
 * File:          bHYPRE_ParaSails_Impl.c
 * Symbol:        bHYPRE.ParaSails-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.ParaSails
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.ParaSails" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * ParaSails requires an IJParCSR matrix
 */

#include "bHYPRE_ParaSails_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._includes) */
/* Put additional includes or other arbitrary code here... */



#include "hypre_babel_exception_handler.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParaSails__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._load) */
  /* Insert-Code-Here {bHYPRE.ParaSails._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParaSails__ctor(
  /* in */ bHYPRE_ParaSails self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* Note: user calls of __create() are DEPRECATED, _Create also calls this function */

   struct bHYPRE_ParaSails__data * data;
   data = hypre_CTAlloc( struct bHYPRE_ParaSails__data, 1 );
   data -> comm = (MPI_Comm) NULL;
   data -> solver = (HYPRE_Solver) NULL;
   data -> matrix = (bHYPRE_IJParCSRMatrix) NULL;
   /* set any other data components here */
   bHYPRE_ParaSails__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParaSails__ctor2(
  /* in */ bHYPRE_ParaSails self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._ctor2) */
    /* Insert-Code-Here {bHYPRE.ParaSails._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParaSails__dtor(
  /* in */ bHYPRE_ParaSails self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   if ( data->matrix != (bHYPRE_IJParCSRMatrix) NULL )
      bHYPRE_IJParCSRMatrix_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   ierr += HYPRE_ParaSailsDestroy( data->solver );
   hypre_assert( ierr== 0 );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._dtor) */
  }
}

/*
 *  This function is the preferred way to create a ParaSails solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_ParaSails
impl_bHYPRE_ParaSails_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_IJParCSRMatrix A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.Create) */
  /* Insert-Code-Here {bHYPRE.ParaSails.Create} (Create method) */

   int ierr = 0;
   HYPRE_Solver dummy;
   HYPRE_Solver * Hsolver = &dummy;
   bHYPRE_ParaSails solver = bHYPRE_ParaSails__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_ParaSails__data * data = bHYPRE_ParaSails__get_data( solver );

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   ierr += HYPRE_ParCSRParaSailsCreate( (data->comm), Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   data->matrix = A;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetOperator(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_ParaSails__data * data;
   bHYPRE_IJParCSRMatrix Amat;

   Amat = (bHYPRE_IJParCSRMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.IJParCSRMatrix", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( Amat!=NULL );

   data = bHYPRE_ParaSails__get_data( self );
   data->matrix = Amat;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex );

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetTolerance(
  /* in */ bHYPRE_ParaSails self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetMaxIterations(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetMaxIterations) */
  }
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetLogging(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   /* This function should be called before Setup.  Log level changes
    * may require allocation or freeing of arrays, which is presently
    * only done there.  It may be possible to support log_level
    * changes at other times, but there is little need.  */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   ierr += HYPRE_ParaSailsSetLogging( solver, level );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetLogging) */
  }
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetPrintLevel(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_GetNumIterations(
  /* in */ bHYPRE_ParaSails self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_GetRelResidualNorm(
  /* in */ bHYPRE_ParaSails self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetCommunicator(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED  Use _Create */

   int ierr = 0;
   HYPRE_Solver dummy;
   HYPRE_Solver * solver = &dummy;

   struct bHYPRE_ParaSails__data * data = bHYPRE_ParaSails__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   ierr += HYPRE_ParCSRParaSailsCreate( (data->comm), solver );
   data -> solver = *solver;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParaSails_Destroy(
  /* in */ bHYPRE_ParaSails self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.Destroy) */
    /* Insert-Code-Here {bHYPRE.ParaSails.Destroy} (Destroy method) */
     bHYPRE_ParaSails_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetIntParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Nlevels")==0 )
   {
      ierr += HYPRE_ParaSailsSetNlevels( solver, value );
   }
   else if ( strcmp(name,"Sym")==0 )
   {
      ierr += HYPRE_ParaSailsSetSym( solver, value );
   }
   else if ( strcmp(name,"Reuse")==0 )
   {
      ierr += HYPRE_ParaSailsSetReuse( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_ParaSailsSetLogging( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetDoubleParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Thresh")==0 )
   {
      ierr += HYPRE_ParaSailsSetThresh( solver, value );
   }
   else if (strcmp(name,"Loadbal")==0 )
   {
      ierr += HYPRE_ParaSailsSetLoadbal( solver, value );
   }
   else if (strcmp(name,"Filter")==0 )
   {
      ierr += HYPRE_ParaSailsSetFilter( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetStringParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetIntArray1Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetIntArray2Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetDoubleArray1Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetDoubleArray2Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_GetIntValue(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Nlevels")==0 )
   {
      ierr += HYPRE_ParaSailsGetNlevels( solver, value );
   }
   else if ( strcmp(name,"Sym")==0 )
   {
      ierr += HYPRE_ParaSailsGetSym( solver, value );
   }
   else if ( strcmp(name,"Reuse")==0 )
   {
      ierr += HYPRE_ParaSailsGetReuse( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_ParaSailsGetLogging( solver, value );
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_GetDoubleValue(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Thresh")==0 )
   {
      ierr += HYPRE_ParaSailsGetThresh( solver, value );
   }
   else if (strcmp(name,"Loadbal")==0 )
   {
      ierr += HYPRE_ParaSailsGetLoadbal( solver, value );
   }
   else if (strcmp(name,"Filter")==0 )
   {
      ierr += HYPRE_ParaSailsGetFilter( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_Setup(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.Setup) */
  /* Insert the implementation of the Setup method here... */

   /* SetCommunicator or Create should have been called by now, to trigger the
      solver's Create call. */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;

   bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector",_ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bHYPREP_b!=NULL );

   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   bHYPREP_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( x, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bHYPREP_x!=NULL );

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   ierr += HYPRE_ParaSailsSetup( solver, bHYPREP_A, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_Apply(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;

   bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bHYPREP_b!=NULL );

   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   if ( *x==NULL )
   {
      /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or hypre_assert(x
       * has the right size) */
      bHYPRE_Vector_Clone( b, x, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Vector_Clear( *x, _ex ); SIDL_CHECK(*_ex);
   }
   bHYPREP_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( *x, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bHYPREP_x!=NULL );

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_ParaSailsSolve( solver, bHYPREP_A, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_ApplyAdjoint(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.ParaSails.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_IJParCSRMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_ParaSails_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_IJParCSRMatrix__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParaSails_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParaSails_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_ParaSails__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_ParaSails(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_ParaSails__connectI(url, ar, _ex);
}
struct bHYPRE_ParaSails__object* 
  impl_bHYPRE_ParaSails_fcast_bHYPRE_ParaSails(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_ParaSails__cast(bi, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_ParaSails_fcast_bHYPRE_Solver(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_ParaSails_fcast_bHYPRE_Vector(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_ParaSails_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParaSails_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_ParaSails_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_ParaSails_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
