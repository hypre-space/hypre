/*
 * File:          bHYPRE_HGMRES_Impl.c
 * Symbol:        bHYPRE.HGMRES-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.HGMRES
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.HGMRES" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * The regular GMRES solver calls Babel-interface matrix and vector functions.
 * The HGMRES solver calls HYPRE interface functions.
 * The regular solver will work with any consistent matrix, vector, and
 * preconditioner classes.  The HGMRES solver will work with the more common ones.
 * 
 * The HGMRES solver checks whether the matrix, vectors, and preconditioner
 * are of known types, and will not work with any other types.
 * Presently, the recognized data types are:
 * matrix, vector: IJParCSRMatrix, IJParCSRVector
 * preconditioner: BoomerAMG, ParCSRDiagScale
 */

#include "bHYPRE_HGMRES_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES._includes) */
/* Insert-Code-Here {bHYPRE.HGMRES._includes} (includes and arbitrary code) */


#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_BoomerAMG.h"
#include "bHYPRE_BoomerAMG_Impl.h"
#include "bHYPRE_ParaSails.h"
#include "bHYPRE_ParaSails_Impl.h"
#include "bHYPRE_ParCSRDiagScale.h"
#include "bHYPRE_ParCSRDiagScale_Impl.h"
#include "bHYPRE_PCG_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
#include "bHYPRE_IdentitySolver.h"

#include "hypre_babel_exception_handler.h"
/*#include "mpi.h"*/

/* This function should be used to initialize the parameter cache
 * in the bHYPRE_HGMRES__data object. */
int impl_bHYPRE_HGMRES_Copy_Parameters_from_HYPRE_struct( bHYPRE_HGMRES self )
{
   /* Parameters are copied only if they have nonsense values which tell
      us that the user has not set them. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_HGMRES__data * data;

   data = bHYPRE_HGMRES__get_data( self );
   hypre_assert( data->solver != NULL );
   solver = data->solver;

   /* double parameters: */
   if ( data->tol == -1.234 )
      ierr += HYPRE_GMRESGetTol( solver, &(data->tol) );

   /* int parameters: */
   if ( data->k_dim == -1234 )
      ierr += HYPRE_GMRESGetKDim( solver, &(data->k_dim) );
   if ( data->max_iter == -1234 )
      ierr += HYPRE_GMRESGetMaxIter( solver, &(data->max_iter) );
   if ( data->min_iter == -1234 )
      ierr += HYPRE_GMRESGetMinIter( solver, &(data->min_iter) );
   if ( data->rel_change == -1234 )
      ierr += HYPRE_GMRESGetRelChange( solver, &(data->rel_change) );
   if ( data->stop_crit == -1234 )
      ierr += HYPRE_GMRESGetStopCrit( solver, &(data->stop_crit) );
   if ( data->printlevel == -1234)
      ierr += HYPRE_GMRESGetPrintLevel( solver, &(data->printlevel) );
   if ( data->log_level == -1234 )
      ierr += HYPRE_GMRESGetLogging( solver, &(data->log_level) );

   return ierr;
}

int impl_bHYPRE_HGMRES_Copy_Parameters_to_HYPRE_struct( bHYPRE_HGMRES self )
/* Copy parameter cache from the bHYPRE_HGMRES__data object into the
 * HYPRE_Solver object */
{
   /* Parameters are left at their HYPRE defaults if they have bHYPRE nonsense
      values which tell us that the user has not set them. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_HGMRES__data * data;

   data = bHYPRE_HGMRES__get_data( self );
   hypre_assert( data->solver != NULL );
   solver = data->solver;

   /* double parameters: */
   ierr += HYPRE_GMRESSetTol( solver, data->tol );

   /* int parameters: */
   if ( data->k_dim != -1234 )
      ierr += HYPRE_GMRESSetKDim( solver, data->k_dim );
   if ( data->max_iter != -1234 )
      ierr += HYPRE_GMRESSetMaxIter( solver, data->max_iter );
   if ( data->min_iter != -1234 )
      ierr += HYPRE_GMRESSetMinIter( solver, data->min_iter );
   if ( data->rel_change != -1234 )
      ierr += HYPRE_GMRESSetRelChange( solver, data->rel_change );
   if ( data->stop_crit != -1234 )
      ierr += HYPRE_GMRESSetStopCrit( solver, data->stop_crit );
   if ( data->printlevel != -1234 )
      ierr += HYPRE_GMRESSetPrintLevel( solver, data->printlevel );
   if ( data->log_level != -1234 )
      ierr += HYPRE_GMRESSetLogging( solver, data->log_level );

   return ierr;
}

/* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_HGMRES__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES._load) */
  /* Insert-Code-Here {bHYPRE.HGMRES._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_HGMRES__ctor(
  /* in */ bHYPRE_HGMRES self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES._ctor) */
  /* Insert-Code-Here {bHYPRE.HGMRES._ctor} (constructor method) */

   /* Note: user calls of __create() are DEPRECATED, _Create also calls this function */

   struct bHYPRE_HGMRES__data * data;
   data = hypre_CTAlloc( struct bHYPRE_HGMRES__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> solver = NULL;
   data -> matrix = NULL;
   data -> vector_type = NULL;
   /* We would like to call HYPRE_<vector type>GMRESCreate at this
    * point, but it's impossible until we know the vector type.
    * That's needed because the C-language Krylov solvers need to be
    * told exactly what functions to call.  If we were to switch to a
    * Babel-based GMRES solver, we would be able to use generic
    * function names; hence we could really initialize GMRES here. */

   /* default values (copied from gmres.c; better to get them by
    * function calls)...*/
/*
   data -> tol        = 1.0e-06;
   data -> k_dim      = 5;
   data -> min_iter   = 0;
   data -> max_iter   = 1000;
   data -> rel_change = 0;
   data -> stop_crit  = 0;*/ /* rel. residual norm */
   /* initial nonsense values, later we should get good values
    * either by user calls or out of the HYPRE object...*/
   data -> tol        = -1.234;
   data -> k_dim      = -1234;
   data -> min_iter   = -1234;
   data -> max_iter   = -1234;
   data -> rel_change = -1234;
   data -> stop_crit  = -1234; /* rel. residual norm */

   data -> bprecond = (bHYPRE_Solver)NULL;

   /* set any other data components here */

   bHYPRE_HGMRES__set_data( self, data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_HGMRES__ctor2(
  /* in */ bHYPRE_HGMRES self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES._ctor2) */
    /* Insert-Code-Here {bHYPRE.HGMRES._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_HGMRES__dtor(
  /* in */ bHYPRE_HGMRES self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES._dtor) */
  /* Insert-Code-Here {bHYPRE.HGMRES._dtor} (destructor method) */

   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( self );

   if ( data->vector_type == "ParVector" )
   {
      ierr += HYPRE_ParCSRGMRESDestroy( data->solver );
   }
   /* To Do: support more vector types */
   else
   {
      /* Unsupported vector type.  We're unlikely to reach this point. */
      ierr++;
   }
   bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   /* delete any nontrivial data components here */
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES._dtor) */
  }
}

/*
 *  This function is the preferred way to create a HGMRES solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_HGMRES
impl_bHYPRE_HGMRES_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.Create) */
  /* Insert-Code-Here {bHYPRE.HGMRES.Create} (Create method) */

   /* HYPRE_ParCSRGMRESCreate or HYPRE_StructGMRESCreate or ... cannot be
      called until later because we don't know the vector type yet */

   bHYPRE_HGMRES solver = bHYPRE_HGMRES__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( solver );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   return solver;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.Create) */
  }
}

/*
 * Set the preconditioner.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetPreconditioner"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetPreconditioner(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Solver s,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetPreconditioner) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetPreconditioner} (SetPreconditioner method) */

   int ierr = 0;
   char * precond_name;
   HYPRE_Solver * solverprecond;
   struct bHYPRE_HGMRES__data * dataself;
   struct bHYPRE_BoomerAMG__data * AMG_dataprecond;
   bHYPRE_BoomerAMG AMG_s;
   struct bHYPRE_ParaSails__data * PS_dataprecond;
   bHYPRE_ParaSails PS_s;
   HYPRE_PtrToSolverFcn precond, precond_setup; /* functions */
   bHYPRE_ParCSRDiagScale PCDS_s;
   bHYPRE_IdentitySolver Id_s;

   dataself = bHYPRE_HGMRES__get_data( self );
   dataself -> bprecond = s;

   AMG_s = (bHYPRE_BoomerAMG) bHYPRE_Solver__cast2( s, "bHYPRE.BoomerAMG", _ex ); SIDL_CHECK(*_ex);
   PS_s = (bHYPRE_ParaSails) bHYPRE_Solver__cast2( s, "bHYPRE.ParaSails", _ex ); SIDL_CHECK(*_ex);
   PCDS_s = (bHYPRE_ParCSRDiagScale) bHYPRE_Solver__cast2( s, "bHYPRE.ParCSRDiagScale", _ex );
   SIDL_CHECK(*_ex);
   Id_s = (bHYPRE_IdentitySolver) bHYPRE_Solver__cast2( s, "bHYPRE.IdentitySolver", _ex );
   SIDL_CHECK(*_ex);
   bHYPRE_Solver_deleteRef( s, _ex ); SIDL_CHECK(*_ex);
   if ( AMG_s )
   {
      precond_name = "BoomerAMG";
      AMG_dataprecond = bHYPRE_BoomerAMG__get_data( AMG_s );
      solverprecond = &AMG_dataprecond->solver;
      hypre_assert( solverprecond != NULL );
      precond = (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup;
   }
   else if ( PS_s)
   {
      precond_name = "ParaSails";
      PS_dataprecond = bHYPRE_ParaSails__get_data( PS_s );
      solverprecond = &PS_dataprecond->solver;
      hypre_assert( solverprecond != NULL );
      precond = (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup;
   }
   else if ( PCDS_s )
   {
      precond_name = "ParCSRDiagScale";
      solverprecond = (HYPRE_Solver *) hypre_CTAlloc( double, 1 );
      /* ... HYPRE diagonal scaling needs no solver object, but we
       * must provide a HYPRE_Solver object.  It will be totally
       * ignored. */
      precond = (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale;
      precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup;
   }
   else if ( Id_s )
   {
      /* s is an IdentitySolver, a dummy which just "solves" the identity matrix */
      precond_name = "IdentitySolver";
      /* The right thing at this point is to check for vector type, then specify
         the right hypre identity solver (_really_ the right thing is to use the
         babel-level identity solver, but that requires PCG to be implemented at the
         babel level). */
      precond = (HYPRE_PtrToSolverFcn) hypre_ParKrylovIdentity;
      precond_setup = (HYPRE_PtrToSolverFcn) hypre_ParKrylovIdentitySetup;
   }
   /* put other preconditioner types here */
   else
   {
      hypre_assert( "GMRES_SetPreconditioner cannot recognize preconditioner"==0 );
   }

   /* We can't actually set the HYPRE preconditioner, because that
    * requires knowing what the solver object is - but that requires
    * knowing its data type but _that_ requires knowing the kind of
    * matrix and vectors we'll need; not known until Apply is called.
    * So save the information in the bHYPRE data structure, and stick
    * it in HYPRE later... */
   dataself->precond_name = precond_name;
   dataself->precond = precond;
   dataself->precond_setup = precond_setup;
   dataself->solverprecond = solverprecond;
   /* For an example call, see test/IJ_linear_solvers.c, line 1686.
    * The four arguments are: self's (solver) data; and, for the
    * preconditioner: solver function, setup function, data */

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetPreconditioner) */
  }
}

/*
 * Method:  GetPreconditioner[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_GetPreconditioner"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_GetPreconditioner(
  /* in */ bHYPRE_HGMRES self,
  /* out */ bHYPRE_Solver* s,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.GetPreconditioner) */
  /* Insert-Code-Here {bHYPRE.HGMRES.GetPreconditioner} (GetPreconditioner method) */

   int ierr = 0;
   struct bHYPRE_HGMRES__data * dataself = bHYPRE_HGMRES__get_data( self );
   *s = dataself -> bprecond ;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.GetPreconditioner) */
  }
}

/*
 * Method:  Clone[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_Clone"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_Clone(
  /* in */ bHYPRE_HGMRES self,
  /* out */ bHYPRE_PreconditionedSolver* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.Clone) */
  /* Insert-Code-Here {bHYPRE.HGMRES.Clone} (Clone method) */
   return 1;
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.Clone) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetOperator(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetOperator) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetOperator} (SetOperator method) */

   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;

   data = bHYPRE_HGMRES__get_data( self );
   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetTolerance(
  /* in */ bHYPRE_HGMRES self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetTolerance} (SetTolerance method) */

   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( self );

   data -> tol = tolerance;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetMaxIterations(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetMaxIterations} (SetMaxIterations method) */

   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( self );

   data -> max_iter = max_iterations;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_HGMRES_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetLogging(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetLogging) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetLogging} (SetLogging method) */

   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( self );

   data -> log_level = level;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_HGMRES_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetPrintLevel(
  /* in */ bHYPRE_HGMRES self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetPrintLevel} (SetPrintLevel method) */

   /* The normal way to implement this function would be to call the
    * corresponding HYPRE function to set the print level.  That can't
    * always be done because the HYPRE struct may not exist.  The
    * HYPRE struct may not exist because it can't be created until we
    * know the vector type - and that is not known until Apply is
    * first called.  So what we do is save the print level in a cache
    * belonging to this Babel interface, and copy it into the HYPRE
    * struct once Apply is called.  */
   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( self );

   data -> printlevel = level;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_GetNumIterations(
  /* in */ bHYPRE_HGMRES self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.HGMRES.GetNumIterations} (GetNumIterations method) */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_HGMRES__data * data;

   data = bHYPRE_HGMRES__get_data( self );
   hypre_assert( data->solver != NULL );
   solver = data->solver;

   ierr += HYPRE_GMRESGetNumIterations( solver, num_iterations );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_GetRelResidualNorm(
  /* in */ bHYPRE_HGMRES self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.HGMRES.GetRelResidualNorm} (GetRelResidualNorm method) */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_HGMRES__data * data;

   data = bHYPRE_HGMRES__get_data( self );
   hypre_assert( data->solver != NULL );
   solver = data->solver;

   ierr += HYPRE_GMRESGetFinalRelativeResidualNorm( solver, norm );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetCommunicator(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetCommunicator} (SetCommunicator method) */

   /* DEPRECATED  Use Create */

   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_HGMRES_Destroy(
  /* in */ bHYPRE_HGMRES self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.Destroy) */
    /* Insert-Code-Here {bHYPRE.HGMRES.Destroy} (Destroy method) */
     bHYPRE_HGMRES_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetIntParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetIntParameter} (SetIntParameter method) */

   /* The normal way to implement this function would be to call the
    * corresponding HYPRE function to set the parameter.  That can't
    * always be done because the HYPRE struct may not exist.  The
    * HYPRE struct may not exist because it can't be created until we
    * know the vector type - and that is not known until Apply is
    * first called.  So what we do is save the parameter in a cache
    * belonging to this Babel interface, and copy it into the HYPRE
    * struct once Apply is called.   (The copy into the HYPRE struct is
    * also done in Setup) */
   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( self );

   if ( strcmp(name,"KDim")==0 )
   {
      data -> k_dim = value;
   }
   else if ( strcmp(name,"MinIter")==0 )
   {
      data -> min_iter = value;
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      data -> max_iter = value;
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"relative change test")==0 )
   {
      data -> rel_change = value;
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      data -> log_level = value;
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      data -> printlevel = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetDoubleParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetDoubleParameter} (SetDoubleParameter method) */

   /* The normal way to implement this function would be to call the
    * corresponding HYPRE function to set the parameter.  That can't
    * always be done because the HYPRE struct may not exist.  The
    * HYPRE struct may not exist because it can't be created until we
    * know the vector type - and that is not known until Apply is
    * first called.  So what we do is save the parameter in a cache
    * belonging to this Babel interface, and copy it into the HYPRE
    * struct once Apply is called.  */
   int ierr = 0;
   struct bHYPRE_HGMRES__data * data;
   data = bHYPRE_HGMRES__get_data( self );

   if ( strcmp(name,"Tolerance")==0 || strcmp(name,"Tol")==0 )
   {
      data -> tol = value;
   }
   else
   {
      ierr = 1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetStringParameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetStringParameter} (SetStringParameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetIntArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetIntArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetDoubleArray1Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_SetDoubleArray2Parameter(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.HGMRES.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_GetIntValue(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.HGMRES.GetIntValue} (GetIntValue method) */

   /* A return value of -1234 means that the parameter has not been
      set yet.  In that case an error flag will be returned too. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_HGMRES__data * data;

   data = bHYPRE_HGMRES__get_data( self );
   hypre_assert( data->solver != NULL );
   solver = data->solver;

   /* The underlying HYPRE PCG object has actually been created if & only if
      data->vector_type is non-null.  If so, make sure our local parameter cache
      if up-to-date.  */
   if ( data -> vector_type != NULL )
      ierr += impl_bHYPRE_HGMRES_Copy_Parameters_from_HYPRE_struct( self );

   if ( strcmp(name,"NumIterations")==0 )
   {
      ierr += HYPRE_GMRESGetNumIterations( solver, value );
   }
   if ( strcmp(name,"Converged")==0 )
   {
      ierr += HYPRE_GMRESGetConverged( solver, value );
   }
   else if ( strcmp(name,"KDim")==0 )
   {
      *value = data -> k_dim;
   }
   else if ( strcmp(name,"MinIter")==0 )
   {
      *value = data -> min_iter;
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      *value = data -> max_iter;
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"relative change test")==0 )
   {
      *value = data -> rel_change;
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      *value = data -> log_level;
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      *value = data -> printlevel;
   }
   else
   {
      ierr = 1;
   }

   if ( *value == -1234 ) ++ierr;
   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_GetDoubleValue(
  /* in */ bHYPRE_HGMRES self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.HGMRES.GetDoubleValue} (GetDoubleValue method) */

   /* A return value of -1234 means that the parameter has not been
      set yet.  In that case an error flag will be returned too. */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_HGMRES__data * data;

   data = bHYPRE_HGMRES__get_data( self );
   hypre_assert( data->solver != NULL );
   solver = data->solver;

   /* The underlying HYPRE PCG object has actually been created if & only if
      data->vector_type is non-null.  If so, make sure our local parameter cache
      if up-to-date.  */
   if ( data -> vector_type != NULL )
      ierr += impl_bHYPRE_HGMRES_Copy_Parameters_from_HYPRE_struct( self );

   if ( strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 )
   {
      ierr += HYPRE_GMRESGetFinalRelativeResidualNorm( solver, value );
   }
   else if ( strcmp(name,"Tolerance")==0 || strcmp(name,"Tol")==0 )
   {
      *value = data -> tol;
   }
   else
   {
      ierr = 1;
   }

   if ( *value == -1.234 ) ++ierr;
   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_Setup(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.Setup) */
  /* Insert-Code-Here {bHYPRE.HGMRES.Setup} (Setup method) */

   int ierr=0;
   MPI_Comm comm;
   HYPRE_Solver solver;
   HYPRE_Solver * psolver = &solver; /* will get a real value later */
   struct bHYPRE_HGMRES__data * data;
   bHYPRE_Operator mat;
   HYPRE_Matrix HYPRE_A;
   bHYPRE_IJParCSRMatrix bHYPREP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   HYPRE_Vector HYPRE_x, HYPRE_b;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = bHYPRE_HGMRES__get_data( self );
   comm = data->comm;
   /* SetCommunicator should have been called earlier */
   hypre_assert( comm != MPI_COMM_NULL );
   mat = data->matrix;
   /* SetOperator should have been called earlier */
   hypre_assert( mat != NULL );

   if ( data -> vector_type == NULL )
   {
      /* This is the first time this Babel GMRES object has seen a
       * vector.  So we are ready to create the bHYPRE GMRES object. */
      if ( bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex) )
      {
         SIDL_CHECK(*_ex);
         bHYPRE_Vector_deleteRef( b, _ex ); SIDL_CHECK(*_ex);
         data -> vector_type = "ParVector";
         HYPRE_ParCSRGMRESCreate( comm, psolver );
         hypre_assert( solver != NULL );
         data -> solver = *psolver;
      }
      /* Add more vector types here */
      else
      {
         SIDL_CHECK(*_ex);
         hypre_assert( "only IJParCSRVector supported by GMRES"==0 );
      }
      bHYPRE_HGMRES__set_data( self, data );
      ierr += impl_bHYPRE_HGMRES_Copy_Parameters_from_HYPRE_struct( self );
   }
   else
   {
      solver = data->solver;
      hypre_assert( solver != NULL );
   }
   /* The SetParameter functions set parameters in the local
    * Babel-interface struct, "data".  That is because the HYPRE
    * struct (where they are actually used) may not exist yet when the
    * functions are called.  At this point we finally know the HYPRE
    * struct exists, so we copy the parameters to it. */
   ierr += impl_bHYPRE_HGMRES_Copy_Parameters_to_HYPRE_struct( self );
   if ( data->vector_type == "ParVector" )
   {
      bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex );
      SIDL_CHECK(*_ex);
      datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
      ij_b = datab -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
      bb = (HYPRE_ParVector) objectb;
      HYPRE_b = (HYPRE_Vector) bb;

      bHYPREP_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( x, "bHYPRE.IJParCSRVector", _ex );
      SIDL_CHECK(*_ex);
      datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
      ij_x = datax -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
      xx = (HYPRE_ParVector) objectx;
      HYPRE_x = (HYPRE_Vector) xx;

      bHYPREP_A = (bHYPRE_IJParCSRMatrix) bHYPRE_Operator__cast2( mat, "bHYPRE.IJParCSRMatrix", _ex );
      SIDL_CHECK(*_ex);
      hypre_assert( bHYPREP_A != NULL );
      dataA = bHYPRE_IJParCSRMatrix__get_data( bHYPREP_A );
      ij_A = dataA -> ij_A;
      ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
      AA = (HYPRE_ParCSRMatrix) objectA;
      HYPRE_A = (HYPRE_Matrix) AA;
   }
   else
   {
      hypre_assert( "only IJParCSRVector supported by GMRES"==0 );
   }
      
   ierr += HYPRE_GMRESSetPrecond( solver, data->precond, data->precond_setup,
                                  *(data->solverprecond) );
   HYPRE_GMRESSetup( solver, HYPRE_A, HYPRE_b, HYPRE_x );

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_Apply(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.Apply) */
  /* Insert-Code-Here {bHYPRE.HGMRES.Apply} (Apply method) */

   /* In the long run, the solver should be implemented right here,
    * calling the appropriate bHYPRE functions.  But for now we are
    * calling the existing HYPRE solver.  Advantages: don't want to
    * have two versions of the same GMRES solver lying around.
    * Disadvantage: we have to cache user-supplied parameters until
    * the Apply call, where we make the GMRES object and really set
    * the parameters - messy and unnatural. */
   int ierr=0;
   MPI_Comm comm;
   HYPRE_Solver solver;
   HYPRE_Solver * psolver = &solver; /* will get a real value later */
   struct bHYPRE_HGMRES__data * data;
   bHYPRE_Operator mat;
   HYPRE_Matrix HYPRE_A;
   bHYPRE_IJParCSRMatrix bHYPREP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   HYPRE_Vector HYPRE_x, HYPRE_b;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = bHYPRE_HGMRES__get_data( self );
   comm = data->comm;
   /* SetCommunicator should have been called earlier */
   hypre_assert( comm != MPI_COMM_NULL );
   mat = data->matrix;
   /* SetOperator should have been called earlier */
   hypre_assert( mat != NULL );

   if ( data -> vector_type == NULL )
   {
      /* This is the first time this Babel GMRES object has seen a
       * vector.  So we are ready to create the bHYPRE GMRES object. */
      if ( bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex) )
      {
         SIDL_CHECK(*_ex);
         bHYPRE_Vector_deleteRef( b, _ex ); SIDL_CHECK(*_ex);
         data -> vector_type = "ParVector";
         HYPRE_ParCSRGMRESCreate( comm, psolver );
         hypre_assert( solver != NULL );
         data -> solver = *psolver;
      }
      /* Add more vector types here */
      else
      {
         SIDL_CHECK(*_ex);
         hypre_assert( "only IJParCSRVector supported by GMRES"==0 );
      }
      bHYPRE_HGMRES__set_data( self, data );
      ierr += impl_bHYPRE_HGMRES_Copy_Parameters_from_HYPRE_struct( self );
   }
   else
   {
      solver = data->solver;
      hypre_assert( solver != NULL );
   }
   /* The SetParameter functions set parameters in the local
    * Babel-interface struct, "data".  That is because the HYPRE
    * struct (where they are actually used) may not exist yet when the
    * functions are called.  At this point we finally know the HYPRE
    * struct exists, so we copy the parameters to it. */
   ierr += impl_bHYPRE_HGMRES_Copy_Parameters_to_HYPRE_struct( self );
   if ( data->vector_type == "ParVector" )
   {
      bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex );
      SIDL_CHECK(*_ex);
      datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
      ij_b = datab -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
      bb = (HYPRE_ParVector) objectb;
      HYPRE_b = (HYPRE_Vector) bb;

      bHYPREP_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( *x, "bHYPRE.IJParCSRVector", _ex );
      SIDL_CHECK(*_ex);
      datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
      ij_x = datax -> ij_b;
      ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
      xx = (HYPRE_ParVector) objectx;
      HYPRE_x = (HYPRE_Vector) xx;

      bHYPREP_A = (bHYPRE_IJParCSRMatrix) bHYPRE_Operator__cast2( mat, "bHYPRE.IJParCSRMatrix", _ex );
      SIDL_CHECK(*_ex);
      hypre_assert( bHYPREP_A != NULL );
      dataA = bHYPRE_IJParCSRMatrix__get_data( bHYPREP_A );
      ij_A = dataA -> ij_A;
      ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
      AA = (HYPRE_ParCSRMatrix) objectA;
      HYPRE_A = (HYPRE_Matrix) AA;
   }
   else
   {
      hypre_assert( "only IJParCSRVector supported by GMRES"==0 );
   }
      
   ierr += HYPRE_GMRESSetPrecond( solver, data->precond, data->precond_setup,
                                  *(data->solverprecond) );

   HYPRE_GMRESSolve( solver, HYPRE_A, HYPRE_b, HYPRE_x );

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_HGMRES_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_HGMRES_ApplyAdjoint(
  /* in */ bHYPRE_HGMRES self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.HGMRES.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.HGMRES.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

    /* DO-NOT-DELETE splicer.end(bHYPRE.HGMRES.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_HGMRES__object* impl_bHYPRE_HGMRES_fconnect_bHYPRE_HGMRES(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_HGMRES__connectI(url, ar, _ex);
}
struct bHYPRE_HGMRES__object* impl_bHYPRE_HGMRES_fcast_bHYPRE_HGMRES(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_HGMRES__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_MPICommunicator(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_HGMRES_fcast_bHYPRE_MPICommunicator(void* bi, sidl_BaseInterface* 
  _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_HGMRES_fconnect_bHYPRE_Operator(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_HGMRES_fcast_bHYPRE_Operator(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_HGMRES_fconnect_bHYPRE_PreconditionedSolver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_PreconditionedSolver__connectI(url, ar, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_HGMRES_fcast_bHYPRE_PreconditionedSolver(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_PreconditionedSolver__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_HGMRES_fconnect_bHYPRE_Solver(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_HGMRES_fcast_bHYPRE_Solver(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_HGMRES_fconnect_bHYPRE_Vector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_HGMRES_fcast_bHYPRE_Vector(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_HGMRES_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_HGMRES_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* impl_bHYPRE_HGMRES_fcast_sidl_BaseInterface(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_HGMRES_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_HGMRES_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_HGMRES_fconnect_sidl_RuntimeException(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_HGMRES_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
