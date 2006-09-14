/*
 * File:          bHYPRE_CGNR_Impl.c
 * Symbol:        bHYPRE.CGNR-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.CGNR
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.CGNR" (version 1.0.0)
 * 
 * Objects of this type can be cast to PreconditionedSolver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * CGNR solver calls Babel-interface functions
 */

#include "bHYPRE_CGNR_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR._includes) */
/* Insert-Code-Here {bHYPRE.CGNR._includes} (includes and arbitrary code) */

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "bHYPRE_MPICommunicator_Impl.h"
#include "bHYPRE_IdentitySolver_Impl.h"
#include "bHYPRE_MatrixVectorView.h"
#include <math.h>
#include <assert.h>
#include "hypre_babel_exception_handler.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.CGNR._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_CGNR__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR._load) */
  /* Insert-Code-Here {bHYPRE.CGNR._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_CGNR__ctor(
  /* in */ bHYPRE_CGNR self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR._ctor) */
  /* Insert-Code-Here {bHYPRE.CGNR._ctor} (constructor method) */

   struct bHYPRE_CGNR__data * data;
   data = hypre_CTAlloc( struct bHYPRE_CGNR__data, 1 );


   /* additional log info (logged when `logging' > 0) */

   data -> mpicomm      = bHYPRE_MPICommunicator_CreateC( (void *)MPI_COMM_NULL, _ex ); SIDL_CHECK(*_ex);

   data -> matrix       = (bHYPRE_Operator)NULL;
   data -> precond      = (bHYPRE_Solver)NULL;

   data -> tol           = 1.0e-06;
   data -> rel_residual_norm = 0.0;
   data -> min_iter      = 0;
   data -> max_iter      = 1000;
   data -> stop_crit     = 0;
   data -> num_iterations = 0;
   data -> converged     = 0;

   data -> p            = (bHYPRE_Vector)NULL;
   data -> q            = (bHYPRE_Vector)NULL;
   data -> r            = (bHYPRE_Vector)NULL;
   data -> t            = (bHYPRE_Vector)NULL;

   data -> print_level   = 0;
   data -> logging       = 0;
   data -> norms         = NULL;
   data -> log_file_name = NULL;

   bHYPRE_CGNR__set_data( self, data );

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_CGNR__ctor2(
  /* in */ bHYPRE_CGNR self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR._ctor2) */
    /* Insert-Code-Here {bHYPRE.CGNR._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_CGNR__dtor(
  /* in */ bHYPRE_CGNR self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR._dtor) */
  /* Insert-Code-Here {bHYPRE.CGNR._dtor} (destructor method) */

   struct bHYPRE_CGNR__data * data;
   data = bHYPRE_CGNR__get_data( self );

   if (data)
   {
      if ( (data -> norms) != NULL )
      {
         hypre_TFree( data -> norms );
         data -> norms = NULL;
      } 

      if ( data -> p != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->p, _ex ); SIDL_CHECK(*_ex);
      if ( data -> q != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->q, _ex ); SIDL_CHECK(*_ex);
      if ( data -> r != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->r, _ex ); SIDL_CHECK(*_ex);
      if ( data -> t != (bHYPRE_Vector)NULL )
         bHYPRE_Vector_deleteRef( data->t, _ex ); SIDL_CHECK(*_ex);
      if ( data -> matrix != (bHYPRE_Operator)NULL )
         bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
      if ( data -> precond != (bHYPRE_Solver)NULL )
         bHYPRE_Solver_deleteRef( data->precond , _ex); SIDL_CHECK(*_ex);

      hypre_TFree( data );
   }

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_CGNR
impl_bHYPRE_CGNR_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.Create) */
  /* Insert-Code-Here {bHYPRE.CGNR.Create} (Create method) */

   bHYPRE_CGNR solver = bHYPRE_CGNR__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( solver );
   bHYPRE_IdentitySolver Id  = bHYPRE_IdentitySolver_Create( mpi_comm, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_Solver IdS = bHYPRE_Solver__cast( Id, _ex ); SIDL_CHECK(*_ex);

   if (data->mpicomm) {
      bHYPRE_MPICommunicator_deleteRef( data->mpicomm, _ex ); SIDL_CHECK(*_ex);
   }
   data->mpicomm = mpi_comm;
   if( data->matrix != (bHYPRE_Operator)NULL )
      bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex );

   data->precond = IdS;

   bHYPRE_IdentitySolver_deleteRef( Id, _ex ); SIDL_CHECK(*_ex);
   /* ...Create and cast created 2 references, we're keeping only one (data->precond) */

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.Create) */
  }
}

/*
 * Set the preconditioner.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetPreconditioner"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetPreconditioner(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Solver s,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetPreconditioner) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetPreconditioner} (SetPreconditioner method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );
   if( data->precond != (bHYPRE_Solver)NULL )
      bHYPRE_Solver_deleteRef( data->precond, _ex ); SIDL_CHECK(*_ex);

   data->precond = s;
   bHYPRE_Solver_addRef( data->precond, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetPreconditioner) */
  }
}

/*
 * Method:  GetPreconditioner[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_GetPreconditioner"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_GetPreconditioner(
  /* in */ bHYPRE_CGNR self,
  /* out */ bHYPRE_Solver* s,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.GetPreconditioner) */
  /* Insert-Code-Here {bHYPRE.CGNR.GetPreconditioner} (GetPreconditioner method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   *s = data->precond;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.GetPreconditioner) */
  }
}

/*
 * Method:  Clone[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_Clone"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_Clone(
  /* in */ bHYPRE_CGNR self,
  /* out */ bHYPRE_PreconditionedSolver* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.Clone) */
  /* Insert-Code-Here {bHYPRE.CGNR.Clone} (Clone method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );
   struct bHYPRE_CGNR__data * datax;
   bHYPRE_CGNR CGNR_x;

   CGNR_x = bHYPRE_CGNR_Create( data->mpicomm, data->matrix, _ex ); SIDL_CHECK(*_ex);

   /* Copy most data members.
      The preconditioner copy will be a shallow copy (just the pointer);
      it is likely to be replaced later.
      But don't copy anything created in Setup (p,q,r,t,norms,log_file_name).
      The user will call Setup on x, later
      Also don't copy the end-of-solve diagnostics (converged,num_iterations,
      rel_residual_norm) */

   datax = bHYPRE_CGNR__get_data( CGNR_x );
   datax->tol               = data->tol;
   datax->min_iter          = data->min_iter;
   datax->max_iter          = data->max_iter;
   datax->stop_crit         = data->stop_crit;
   datax->print_level       = data->print_level;
   datax->logging           = data->logging;

   bHYPRE_CGNR_SetPreconditioner( CGNR_x, data->precond, _ex ); SIDL_CHECK(*_ex);

   *x = bHYPRE_PreconditionedSolver__cast( CGNR_x, _ex ); SIDL_CHECK(*_ex);

   bHYPRE_CGNR_deleteRef( CGNR_x,_ex ); SIDL_CHECK(*_ex);
   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.Clone) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetOperator(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetOperator) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetOperator} (SetOperator method) */

   /* DEPRECATED  the second argument in Create does the same thing */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data;

   data = bHYPRE_CGNR__get_data( self );
   if( data->matrix != (bHYPRE_Operator)NULL )
   { bHYPRE_Operator_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);}

   data->matrix = A;
   bHYPRE_Operator_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetTolerance(
  /* in */ bHYPRE_CGNR self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetTolerance} (SetTolerance method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   data -> tol = tolerance;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetMaxIterations(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetMaxIterations} (SetMaxIterations method) */
   /* SetIntParameter will do this same job, ans SetIntParameter is the only way
      to set MinIterations */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   data -> max_iter = max_iterations;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_CGNR_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetLogging(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetLogging) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetLogging} (SetLogging method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   data -> logging = level;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_CGNR_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetPrintLevel(
  /* in */ bHYPRE_CGNR self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetPrintLevel} (SetPrintLevel method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   data -> print_level = level;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_GetNumIterations(
  /* in */ bHYPRE_CGNR self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.CGNR.GetNumIterations} (GetNumIterations method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   *num_iterations = data->num_iterations;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_GetRelResidualNorm(
  /* in */ bHYPRE_CGNR self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.CGNR.GetRelResidualNorm} (GetRelResidualNorm method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   *norm = data->rel_residual_norm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetCommunicator(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetCommunicator} (SetCommunicator method) */
   return 1;  /* DEPRECATED and will never be implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_CGNR_Destroy(
  /* in */ bHYPRE_CGNR self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.Destroy) */
    /* Insert-Code-Here {bHYPRE.CGNR.Destroy} (Destroy method) */
     bHYPRE_CGNR_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetIntParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetIntParameter} (SetIntParameter method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   if ( strcmp(name,"MinIter")==0 || strcmp(name,"MinIterations")==0 )
   {
      data -> min_iter = value;
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      data -> max_iter = value;
   }
   else if ( strcmp(name,"StopCrit")==0 || strcmp(name,"stopping criterion")==0 )
   {
      data -> stop_crit = value;
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      data -> logging = value;
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      data -> print_level = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetDoubleParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetDoubleParameter} (SetDoubleParameter method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   if ( strcmp(name,"Tolerance")==0  || strcmp(name,"Tol")==0 )
   {
      data -> tol = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetStringParameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetStringParameter} (SetStringParameter method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   if ( strcmp(name,"LogFileName")==0  || strcmp(name,"log file name")==0 )
   {
      data -> log_file_name = value;
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetIntArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetIntArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetDoubleArray1Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_SetDoubleArray2Parameter(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.CGNR.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_GetIntValue(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.CGNR.GetIntValue} (GetIntValue method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   if ( strcmp(name,"NumIterations")==0 )
   {
      *value = data -> num_iterations;
   }
   else if ( strcmp(name,"Converged")==0 )
   {
      *value = data -> converged;
   }
   else if ( strcmp(name,"MinIter")==0 || strcmp(name,"MinIterations")==0 )
   {
      *value = data -> max_iter;
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      *value = data -> max_iter;
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      *value = data -> logging;
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      *value = data -> print_level;
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_GetDoubleValue(
  /* in */ bHYPRE_CGNR self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.CGNR.GetDoubleValue} (GetDoubleValue method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );

   if ( strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 )
   {
      *value = data -> rel_residual_norm;
   }
   else if ( strcmp(name,"Tolerance")==0  || strcmp(name,"Tol")==0 )
   {
      *value = data -> tol;
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_Setup(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.Setup) */
  /* Insert-Code-Here {bHYPRE.CGNR.Setup} (Setup method) */

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );
   int   max_iter = data->max_iter;
   bHYPRE_MatrixVectorView Vp, Vq, Vr, Vt;

   /* Setup should not be called more than once. */
   hypre_assert( data->p == (bHYPRE_Vector)NULL );
   hypre_assert( data->q == (bHYPRE_Vector)NULL );
   hypre_assert( data->r == (bHYPRE_Vector)NULL );
   hypre_assert( data->t == (bHYPRE_Vector)NULL );

   ierr += bHYPRE_Vector_Clone( x, &(data->p), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( x, &(data->q), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( b, &(data->r), _ex ); SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clone( b, &(data->t), _ex ); SIDL_CHECK(*_ex);
   Vp = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->p, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex)
   Vq = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->q, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex)
   Vr = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->r, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex)
   Vt = (bHYPRE_MatrixVectorView) bHYPRE_Vector__cast2( data->t, "bHYPRE.MatrixVectorView", _ex );
   SIDL_CHECK(*_ex)
   if ( Vp )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vp, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_MatrixVectorView_deleteRef( Vp, _ex ); SIDL_CHECK(*_ex);
   }
   if ( Vq )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vq, _ex );
      bHYPRE_MatrixVectorView_deleteRef( Vq, _ex ); SIDL_CHECK(*_ex);
   }
   if ( Vr )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vr, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_MatrixVectorView_deleteRef( Vr, _ex ); SIDL_CHECK(*_ex);
   }
   if ( Vt )
   {
      ierr += bHYPRE_MatrixVectorView_Assemble( Vt, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_MatrixVectorView_deleteRef( Vt, _ex ); SIDL_CHECK(*_ex);
   }

   ierr += bHYPRE_Solver_Setup( data->precond, b, x, _ex ); SIDL_CHECK(*_ex);

   if ( data->logging>0  || data->print_level>0 ) 
   {  /* arrays needed for logging */
      if ( data->norms != NULL )
         hypre_TFree( data->norms );
      data->norms = hypre_CTAlloc( double, max_iter + 1 );
   }

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_Apply(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.Apply) */
  /* Insert-Code-Here {bHYPRE.CGNR.Apply} (Apply method) */

/*--------------------------------------------------------------------------
 * hypre_CGNRSolve: apply CG to (AC)^TACy = (AC)^Tb, x = Cy
 *--------------------------------------------------------------------------*/

   int ierr = 0;
   struct bHYPRE_CGNR__data * data = bHYPRE_CGNR__get_data( self );
   bHYPRE_Operator A = data->matrix;
   double          tol          = data -> tol;
   /* not used int             min_iter     = data -> min_iter;*/
   int             max_iter     = data -> max_iter;
   int             stop_crit    = data -> stop_crit;
   bHYPRE_Vector   p            = data -> p;
   bHYPRE_Vector   q            = data -> q;
   bHYPRE_Vector   r            = data -> r;
   bHYPRE_Vector   t            = data -> t;
   bHYPRE_Solver   precond      = data -> precond;
   /* not used int             print_level  = data -> print_level;*/
   int             logging      = data -> logging;
   double         *norms        = data -> norms;
   bHYPRE_MPICommunicator  bmpicomm = data -> mpicomm;
   MPI_Comm        comm;
                
   double          alpha, beta;
   double          gamma, gamma_old;
   double          bi_prod, i_prod, eps;
   double          ieee_check = 0.;

   int             i = 0;
   int             my_id, num_procs;
   int             x_not_set = 1;

   comm = bHYPRE_MPICommunicator__get_data(bmpicomm)->mpi_comm;
   MPI_Comm_size( comm, &num_procs );
   MPI_Comm_rank( comm, &my_id );

   /*-----------------------------------------------------------------------
    * Start cgnr solve
    *-----------------------------------------------------------------------*/

   if (logging > 1 && my_id == 0)
   {
/* not used yet      log_file_name = (cgnr_data -> log_file_name); */
      printf("Iters       ||r||_2      conv.rate  ||r||_2/||b||_2\n");
      printf("-----    ------------    ---------  ------------ \n");
   }

   /* compute eps */
   ierr += bHYPRE_Vector_Dot( b, b, &bi_prod, _ex ); SIDL_CHECK(*_ex);

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (bi_prod != 0.) ieee_check = bi_prod/bi_prod; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0)
      {
        printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        printf("ERROR -- hypre_CGNRSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied b.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   if (stop_crit) 
      eps = tol*tol; /* absolute residual norm */
   else
      eps = (tol*tol)*bi_prod; /* relative residual norm */

   /* Check to see if the rhs vector b is zero */
   if (bi_prod == 0.0)
   {
      /* Set x equal to zero and return */
      ierr += bHYPRE_Vector_Copy( *x, b, _ex ); SIDL_CHECK(*_ex);
      if (logging > 0)
      {
         norms[0]     = 0.0;
      }
      data -> converged = 1;
      ierr = 0;
      return ierr;
   }

   /* r = b - Ax */
   /* >>> It would be better define a matvec operation which directly calls the
      HYPRE matvec, so we can do this in fewer primitive operations.
      However, the cost savings would be negligible. */
   ierr += bHYPRE_Operator_Apply( A, *x, &r, _ex );  /* r = Ax */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Axpy( r, -1.0, b, _ex );   /* r = r - b = Ax - b */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Scale( r, -1.0, _ex );     /* r = -r = b - Ax */
   SIDL_CHECK(*_ex);

   /* Set initial residual norm */
   if (logging > 0)
   {
      ierr += bHYPRE_Vector_Dot( r, r, &(norms[0]), _ex ); SIDL_CHECK(*_ex);
      norms[0] = sqrt( norms[0] );

      /* Since it is does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (norms[0] != 0.) ieee_check = norms[0]/norms[0]; /* INF -> NaN conversion */
      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (logging > 0)
         {
           printf("\n\nERROR detected by Hypre ...  BEGIN\n");
           printf("ERROR -- hypre_CGNRSolve: INFs and/or NaNs detected in input.\n");
           printf("User probably placed non-numerics in supplied A or x_0.\n");
           printf("Returning error flag += 101.  Program not terminated.\n");
           printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         ierr += 101;
         return ierr;
      }
   }

   /* t = C^T*A^T*r */
   ierr += bHYPRE_Operator_ApplyAdjoint( A, r, &q, _ex ); /* q = A' r */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Vector_Clear( t, _ex ); /* t = 0 */ /* not needed ? */
   SIDL_CHECK(*_ex);
   ierr += bHYPRE_Solver_ApplyAdjoint( precond, q, &t, _ex ); /* Ap' t = q */
   SIDL_CHECK(*_ex);
   /* ... where ' denotes transpose and Ap is matrix solved by the preconditioner */

   /* p = r */
   ierr += bHYPRE_Vector_Copy( p, r, _ex ); SIDL_CHECK(*_ex);

   /* gamma = <t,t> */
   ierr += bHYPRE_Vector_Dot( t, t, &gamma, _ex ); SIDL_CHECK(*_ex);
   hypre_assert( ierr==1 );

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (gamma != 0.) ieee_check = gamma/gamma; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0)
      {
        printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        printf("ERROR -- hypre_CGNRSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied A or x_0.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   /* Here begins the main loop... */
   while ((i+1) <= max_iter)
   {
      i++;

      /* q = A*C*p */
      ierr += bHYPRE_Vector_Clear( t, _ex ); /* not needed ? */
      SIDL_CHECK(*_ex);
      ierr += bHYPRE_Solver_Apply( precond, p, &t, _ex ); /* Ap t = p, t = C p */
      SIDL_CHECK(*_ex);
      ierr += bHYPRE_Operator_Apply( A, t, &q, _ex );  /* q = A t = A C p */
      SIDL_CHECK(*_ex);

      /* alpha = gamma / <q,q> */
      ierr += bHYPRE_Vector_Dot( q, q, &alpha, _ex ); /* alpha = <q,q> */
      SIDL_CHECK(*_ex);
      alpha = gamma / alpha;

      gamma_old = gamma;

      /* x = x + alpha*p */
      ierr += bHYPRE_Vector_Axpy( *x, alpha, p, _ex ); SIDL_CHECK(*_ex);

      /* r = r - alpha*q */
      ierr += bHYPRE_Vector_Axpy( r, -alpha, q, _ex ); SIDL_CHECK(*_ex);
	 
      /* t = C^T*A^T*r */
      ierr += bHYPRE_Operator_ApplyAdjoint( A, r, &q, _ex ); /* q = A' r */
      SIDL_CHECK(*_ex);
      ierr += bHYPRE_Vector_Clear( t, _ex ); /* t = 0 */ /* not needed ? */
      SIDL_CHECK(*_ex);
      ierr += bHYPRE_Solver_ApplyAdjoint( precond, q, &t, _ex ); /* Ap' t = q, t = C' q */
      SIDL_CHECK(*_ex);

      /* gamma = <t,t> */
      ierr += bHYPRE_Vector_Dot( t, t, &gamma, _ex ); SIDL_CHECK(*_ex);

      /* set i_prod for convergence test */
      ierr += bHYPRE_Vector_Dot( r, r, &i_prod, _ex ); SIDL_CHECK(*_ex);

      /* log norm info */
      if (logging > 0)
      {
         norms[i]     = sqrt(i_prod);
         if (logging > 1 && my_id == 0)
         {
            printf("% 5d    %e    %f   %e\n", i, norms[i], norms[i]/ 
                   norms[i-1], norms[i]/bi_prod );
         }
      }

      /* check for convergence */
      if (i_prod < eps)
      {
         /*-----------------------------------------------------------------
          * Generate solution q = Cx
          *-----------------------------------------------------------------*/
         ierr += bHYPRE_Vector_Clear( q, _ex ); /* not needed ? */
         SIDL_CHECK(*_ex);
         ierr += bHYPRE_Solver_Apply( precond, *x, &q, _ex ); /* Ap q = x */
         SIDL_CHECK(*_ex);
         /* r = b - Aq */
         ierr += bHYPRE_Operator_Apply( A, q, &r, _ex ); /* r = A q */
         SIDL_CHECK(*_ex);
         ierr += bHYPRE_Vector_Axpy( r, -1.0, b, _ex );  /* r = r - b = A q - b */
         SIDL_CHECK(*_ex);
         ierr += bHYPRE_Vector_Scale( r, -1.0, _ex );    /* r = -r = b - A q */
         SIDL_CHECK(*_ex);
         ierr += bHYPRE_Vector_Dot( r, r, &i_prod, _ex ); /* i_prod = <r,r> */
         SIDL_CHECK(*_ex);
         if (i_prod < eps) 
         {
            data -> converged = 1;
            ierr += bHYPRE_Vector_Copy( *x, q, _ex ); /* x = q */
            SIDL_CHECK(*_ex);
	    x_not_set = 0;
	    break;
         }
      }

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = t + beta p */
      ierr += bHYPRE_Vector_Scale( p, beta, _ex ); /* p = beta * p */
      SIDL_CHECK(*_ex);
      ierr += bHYPRE_Vector_Axpy( p, 1.0, t, _ex ); /* p = p + t */
      SIDL_CHECK(*_ex);
   }
   /* end of main loop */
   hypre_assert( ierr==0 );

  /*-----------------------------------------------------------------
   * Generate solution x = Cx
   *-----------------------------------------------------------------*/
   if (x_not_set)
   {
      ierr += bHYPRE_Vector_Copy( q, *x, _ex ); /* q = x */
      SIDL_CHECK(*_ex);
      ierr += bHYPRE_Vector_Clear( *x, _ex );   /* x = 0 */ /* not needed ? */
      SIDL_CHECK(*_ex);
      ierr += bHYPRE_Solver_Apply( precond, q, x, _ex ); /* Ap x = q */
      SIDL_CHECK(*_ex);
   }

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   bi_prod = sqrt(bi_prod);

   if (logging > 1 && my_id == 0)
   {
      printf("\n\n");
   }

   data -> num_iterations = i;
   data -> rel_residual_norm = norms[i]/bi_prod;

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_CGNR_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_CGNR_ApplyAdjoint(
  /* in */ bHYPRE_CGNR self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.CGNR.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.CGNR.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

  /* DO-NOT-DELETE splicer.end(bHYPRE.CGNR.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_CGNR__object* impl_bHYPRE_CGNR_fconnect_bHYPRE_CGNR(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_CGNR__connectI(url, ar, _ex);
}
struct bHYPRE_CGNR__object* impl_bHYPRE_CGNR_fcast_bHYPRE_CGNR(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_CGNR__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_CGNR_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_CGNR_fconnect_bHYPRE_Operator(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_CGNR_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_CGNR_fconnect_bHYPRE_PreconditionedSolver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_PreconditionedSolver__connectI(url, ar, _ex);
}
struct bHYPRE_PreconditionedSolver__object* 
  impl_bHYPRE_CGNR_fcast_bHYPRE_PreconditionedSolver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_PreconditionedSolver__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_CGNR_fconnect_bHYPRE_Solver(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_CGNR_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_CGNR_fconnect_bHYPRE_Vector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_CGNR_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_CGNR_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_CGNR_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_CGNR_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_CGNR_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_CGNR_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_CGNR_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_CGNR_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
