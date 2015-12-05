/*
 * File:          bHYPRE_StructSMG_Impl.c
 * Symbol:        bHYPRE.StructSMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.StructSMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructSMG" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects
 * using the {\tt \_\_cast} methods.
 * 
 * The StructSMG solver requires a Struct matrix.
 */

#include "bHYPRE_StructSMG_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._includes) */
/* Put additional includes or other arbitrary code here... */



#include "hypre_babel_exception_handler.h"
#include "bHYPRE_StructMatrix.h"
#include "bHYPRE_StructMatrix_Impl.h"
#include "bHYPRE_StructVector.h"
#include "bHYPRE_StructVector_Impl.h"
#include "_hypre_struct_ls.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructSMG__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._load) */
  /* Insert-Code-Here {bHYPRE.StructSMG._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructSMG__ctor(
  /* in */ bHYPRE_StructSMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._ctor) */
  /* Insert the implementation of the constructor method here... */

   /*  How to make and use a StructSMG solver:
       First call this constructor, through bHYPRE_StructSMG__create
       Then call SetCommunicator, SetOperator, and set whatever parameters you
       want (e.g. logging).
       Then call Setup.  Finally you can call Apply.
       Destroy the solver by calling deleteRef.
    */

   /* Note: user calls of __create() are DEPRECATED, _Create also calls this function */

   struct bHYPRE_StructSMG__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructSMG__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> solver = (HYPRE_StructSolver) NULL;
   data -> matrix = (bHYPRE_StructMatrix) NULL;
   bHYPRE_StructSMG__set_data( self, data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructSMG__ctor2(
  /* in */ bHYPRE_StructSMG self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._ctor2) */
    /* Insert-Code-Here {bHYPRE.StructSMG._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructSMG__dtor(
  /* in */ bHYPRE_StructSMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructSMG__data * data;
   data = bHYPRE_StructSMG__get_data( self );
   ierr += HYPRE_StructSMGDestroy( data->solver );
   bHYPRE_StructMatrix_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._dtor) */
  }
}

/*
 *  This function is the preferred way to create a Struct SMG solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructSMG
impl_bHYPRE_StructSMG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructMatrix A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.Create) */
  /* Insert-Code-Here {bHYPRE.StructSMG.Create} (Create method) */

   int ierr = 0;
   HYPRE_StructSolver dummy;
   HYPRE_StructSolver * Hsolver = &dummy;
   bHYPRE_StructSMG solver = bHYPRE_StructSMG__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_StructSMG__data * data = bHYPRE_StructSMG__get_data( solver );

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   ierr += HYPRE_StructSMGCreate( (data->comm), Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   data->matrix = A;
   bHYPRE_StructMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetOperator(
  /* in */ bHYPRE_StructSMG self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   /* not normally needed, as an operator is provided in Create */
   int ierr = 0;
   struct bHYPRE_StructSMG__data * data;
   bHYPRE_StructMatrix Amat;

   Amat = (bHYPRE_StructMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.StructMatrix", _ex ); SIDL_CHECK(*_ex);
   if ( Amat==NULL ) hypre_assert( "Unrecognized operator type."==(char *)A );

   data = bHYPRE_StructSMG__get_data( self );
   data->matrix = Amat;
   bHYPRE_StructMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetTolerance(
  /* in */ bHYPRE_StructSMG self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructSMGSetTol( solver, tolerance );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetMaxIterations(
  /* in */ bHYPRE_StructSMG self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructSMGSetMaxIter( solver, max_iterations );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_StructSMG_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetLogging(
  /* in */ bHYPRE_StructSMG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructSMGSetLogging( solver, level );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_StructSMG_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetPrintLevel(
  /* in */ bHYPRE_StructSMG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructSMGSetPrintLevel( solver, level );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_GetNumIterations(
  /* in */ bHYPRE_StructSMG self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructSMGGetNumIterations( solver, num_iterations );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_GetRelResidualNorm(
  /* in */ bHYPRE_StructSMG self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructSMGGetFinalRelativeResidualNorm( solver, norm );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetCommunicator(
  /* in */ bHYPRE_StructSMG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED   Use Create */

   int ierr = 0;
   HYPRE_StructSolver dummy;
   HYPRE_StructSolver * solver = &dummy;
   struct bHYPRE_StructSMG__data * data = bHYPRE_StructSMG__get_data( self );

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   ierr += HYPRE_StructSMGCreate( (data->comm), solver );
   data -> solver = *solver;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructSMG_Destroy(
  /* in */ bHYPRE_StructSMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.Destroy) */
    /* Insert-Code-Here {bHYPRE.StructSMG.Destroy} (Destroy method) */
     bHYPRE_StructSMG_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetIntParameter(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"MemoryUse")==0 || strcmp(name,"memory use")==0 )
   {
      ierr += HYPRE_StructSMGSetMemoryUse( solver, value );      
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      ierr += HYPRE_StructSMGSetMaxIter( solver, value );
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"relative change test")==0 )
   {
      ierr += HYPRE_StructSMGSetRelChange( solver, value );
   }
   else if ( strcmp(name,"NonZeroGuess")==0 || strcmp(name,"nonzero guess")==0 )
   {
      if ( value==0 )
      {
         ierr += HYPRE_StructSMGSetZeroGuess( solver );
      }
      else if ( value==1 )
      {
         ierr += HYPRE_StructSMGSetNonZeroGuess( solver );
      }
      else
      {
         ierr += HYPRE_StructSMGSetNonZeroGuess( solver );
         ++ierr;
      }
   }
   else if ( strcmp(name,"ZeroGuess")==0 || strcmp(name,"zero guess")==0 )
   {
      if ( value==0 )
      {
         ierr += HYPRE_StructSMGSetNonZeroGuess( solver );
      }
      else if ( value==1 )
      {
         ierr += HYPRE_StructSMGSetZeroGuess( solver );
      }
      else
      {
         ierr += HYPRE_StructSMGSetZeroGuess( solver );
         ++ierr;
      }
   }
   else if ( strcmp(name,"NumPreRelax")==0 || strcmp(name,"num prerelax")==0 )
   {
      ierr += HYPRE_StructSMGSetNumPreRelax( solver, value );
   }
   else if ( strcmp(name,"NumPostRelax")==0 || strcmp(name,"num postrelax")==0 )
   {
      ierr += HYPRE_StructSMGSetNumPostRelax( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_StructSMGSetLogging( solver, value );
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      ierr += HYPRE_StructSMGSetPrintLevel( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetDoubleParameter(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 )
   {
      ierr += HYPRE_StructSMGSetTol( solver, value );      
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetStringParameter(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1; /* not implemented */
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetIntArray1Parameter(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetIntArray2Parameter(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_GetIntValue(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"NumIterations")==0 )
   {
      ierr = HYPRE_StructSMGGetNumIterations( solver, value );
   }
   else if ( strcmp(name,"MemoryUse")==0 || strcmp(name,"memory use")==0 )
   {
      ierr += HYPRE_StructSMGGetMemoryUse( solver, value );      
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      ierr += HYPRE_StructSMGGetMaxIter( solver, value );
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"relative change test")==0 )
   {
      ierr += HYPRE_StructSMGGetRelChange( solver, value );
   }
   else if ( strcmp(name,"NonZeroGuess")==0 || strcmp(name,"nonzero guess")==0 )
   {
      ierr += HYPRE_StructSMGGetZeroGuess( solver, value );
      if ( *value==0 )
         *value = 1;
      else if ( *value==1 )
         *value = 0;
      else
         ++ierr;
   }
   else if ( strcmp(name,"ZeroGuess")==0 || strcmp(name,"zero guess")==0 )
   {
      ierr += HYPRE_StructSMGGetZeroGuess( solver, value );
   }
   else if ( strcmp(name,"NumPreRelax")==0 || strcmp(name,"num prerelax")==0 )
   {
      ierr += HYPRE_StructSMGGetNumPreRelax( solver, value );
   }
   else if ( strcmp(name,"NumPostRelax")==0 || strcmp(name,"num postrelax")==0 )
   {
      ierr += HYPRE_StructSMGGetNumPostRelax( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_StructSMGGetLogging( solver, value );
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      ierr += HYPRE_StructSMGGetPrintLevel( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_GetDoubleValue(
  /* in */ bHYPRE_StructSMG self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 )
   {
      ierr = HYPRE_StructSMGGetFinalRelativeResidualNorm( solver, value );
   }
   else if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 )
   {
      ierr += HYPRE_StructSMGGetTol( solver, value );      
   }
   else
   {
      ierr = 1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_Setup(
  /* in */ bHYPRE_StructSMG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;
   struct bHYPRE_StructMatrix__data * dataA;
   struct bHYPRE_StructVector__data * datab, * datax;
   bHYPRE_StructMatrix A;
   HYPRE_StructMatrix HA;
   bHYPRE_StructVector bHYPREP_b, bHYPREP_x;
   HYPRE_StructVector Hb, Hx;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;
   A = data->matrix;
   dataA = bHYPRE_StructMatrix__get_data( A );
   HA = dataA -> matrix;

   bHYPREP_b = (bHYPRE_StructVector) bHYPRE_Vector__cast2( b, "bHYPRE.StructVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)x );

   datab = bHYPRE_StructVector__get_data( bHYPREP_b );
   Hb = datab -> vec;

   bHYPREP_x = (bHYPRE_StructVector) bHYPRE_Vector__cast2( x, "bHYPRE.StructVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)(x) );

   datax = bHYPRE_StructVector__get_data( bHYPREP_x );
   Hx = datax -> vec;

   ierr += HYPRE_StructSMGSetup( solver, HA, Hb, Hx );

   bHYPRE_StructVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_StructVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_Apply(
  /* in */ bHYPRE_StructSMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;
   struct bHYPRE_StructMatrix__data * dataA;
   struct bHYPRE_StructVector__data * datab, * datax;
   bHYPRE_StructMatrix A;
   HYPRE_StructMatrix HA;
   bHYPRE_StructVector bHYPREP_b, bHYPREP_x;
   HYPRE_StructVector Hb, Hx;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;
   A = data->matrix;
   dataA = bHYPRE_StructMatrix__get_data( A );
   HA = dataA -> matrix;

   bHYPREP_b = (bHYPRE_StructVector) bHYPRE_Vector__cast2( b, "bHYPRE.StructVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)x );

   datab = bHYPRE_StructVector__get_data( bHYPREP_b );
   Hb = datab -> vec;

   if ( *x==NULL )
   {
      /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or hypre_assert(x
       * has the right size) */
      bHYPRE_Vector_Clone( b, x, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Vector_Clear( *x, _ex ); SIDL_CHECK(*_ex);
   }
   bHYPREP_x = (bHYPRE_StructVector) bHYPRE_Vector__cast2( *x, "bHYPRE.StructVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)(*x) );

   datax = bHYPRE_StructVector__get_data( bHYPREP_x );
   Hx = datax -> vec;

   ierr += HYPRE_StructSMGSolve( solver, HA, Hb, Hx );

   bHYPRE_StructVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_StructVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructSMG_ApplyAdjoint(
  /* in */ bHYPRE_StructSMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.StructSMG.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructSMG_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructSMG_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_StructSMG_fconnect_bHYPRE_Operator(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_StructSMG_fcast_bHYPRE_Operator(
  void* bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_StructSMG_fconnect_bHYPRE_Solver(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_StructSMG_fcast_bHYPRE_Solver(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructSMG_fconnect_bHYPRE_StructMatrix(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructSMG_fcast_bHYPRE_StructMatrix(void* bi, sidl_BaseInterface* 
  _ex) {
  return bHYPRE_StructMatrix__cast(bi, _ex);
}
struct bHYPRE_StructSMG__object* 
  impl_bHYPRE_StructSMG_fconnect_bHYPRE_StructSMG(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_StructSMG__connectI(url, ar, _ex);
}
struct bHYPRE_StructSMG__object* impl_bHYPRE_StructSMG_fcast_bHYPRE_StructSMG(
  void* bi, sidl_BaseInterface* _ex) {
  return bHYPRE_StructSMG__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_StructSMG_fconnect_bHYPRE_Vector(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_StructSMG_fcast_bHYPRE_Vector(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_StructSMG_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_StructSMG_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructSMG_fconnect_sidl_BaseInterface(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructSMG_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_StructSMG_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_StructSMG_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructSMG_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructSMG_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}

