/*
 * File:          bHYPRE_SStructSplit_Impl.c
 * Symbol:        bHYPRE.SStructSplit-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.SStructSplit
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructSplit" (version 1.0.0)
 * 
 * 
 * Documentation goes here.
 * 
 * The SStructSplit solver requires a SStruct matrix.
 */

#include "bHYPRE_SStructSplit_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._includes) */
/* Insert-Code-Here {bHYPRE.SStructSplit._includes} (includes and arbitrary code) */

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

#include <assert.h>
#include "hypre_babel_exception_handler.h"
#include "bHYPRE_SStructMatrix.h"
#include "bHYPRE_SStructMatrix_Impl.h"
#include "bHYPRE_SStructVector.h"
#include "bHYPRE_SStructVector_Impl.h"
#include "HYPRE_sstruct_ls.h"
#include "sstruct_ls.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructSplit__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._load) */
  /* Insert-Code-Here {bHYPRE.SStructSplit._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructSplit__ctor(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._ctor) */
  /* Insert-Code-Here {bHYPRE.SStructSplit._ctor} (constructor method) */

   struct bHYPRE_SStructSplit__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructSplit__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> solver = (HYPRE_SStructSolver) NULL;
   data -> matrix = (bHYPRE_SStructMatrix) NULL;
   bHYPRE_SStructSplit__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructSplit__ctor2(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._ctor2) */
    /* Insert-Code-Here {bHYPRE.SStructSplit._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructSplit__dtor(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._dtor) */
  /* Insert-Code-Here {bHYPRE.SStructSplit._dtor} (destructor method) */

   int ierr = 0;
   struct bHYPRE_SStructSplit__data * data;
   data = bHYPRE_SStructSplit__get_data( self );
   ierr += HYPRE_SStructSplitDestroy( data->solver );
   bHYPRE_SStructMatrix_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_SStructSplit
impl_bHYPRE_SStructSplit_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.Create) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.Create} (Create method) */

   int ierr = 0;
   bHYPRE_SStructSplit solver = bHYPRE_SStructSplit__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_SStructSplit__data * data = bHYPRE_SStructSplit__get_data( solver );
   HYPRE_SStructSolver dummy;
   HYPRE_SStructSolver * Hsolver = &dummy;
   bHYPRE_SStructMatrix Amat;

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   ierr += HYPRE_SStructSplitCreate( (data->comm), Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   Amat = (bHYPRE_SStructMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.SStructMatrix", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( Amat!=NULL );

   data->matrix = Amat;
   bHYPRE_SStructMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_SStructMatrix_deleteRef( Amat, _ex ); SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetOperator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetOperator) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetOperator} (SetOperator method) */

   int ierr = 0;
   struct bHYPRE_SStructSplit__data * data;
   bHYPRE_SStructMatrix Amat;

   Amat = (bHYPRE_SStructMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.SStructMatrix", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( Amat!=NULL );

   data = bHYPRE_SStructSplit__get_data( self );
   data->matrix = Amat;
   bHYPRE_SStructMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_SStructMatrix_deleteRef( Amat, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetTolerance(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetTolerance} (SetTolerance method) */

   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;

   ierr = HYPRE_SStructSplitSetTol( solver, tolerance );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetMaxIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetMaxIterations} (SetMaxIterations method) */

   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;

   ierr = HYPRE_SStructSplitSetMaxIter( solver, max_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetLogging(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetLogging) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetLogging} (SetLogging method) */

  /* ignored by HYPRE_SStructSplit, but it sets Logging to 0 for solvers it calls */
   if ( level==0 ) return 0;
   else return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetPrintLevel(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetPrintLevel} (SetPrintLevel method) */

  /* ignored by HYPRE_SStructSplit, but it sets PrintLevel to 0 for solvers it calls */
   if ( level==0 ) return 0;
   else return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_GetNumIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.GetNumIterations} (GetNumIterations method) */

   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;

   ierr = HYPRE_SStructSplitGetNumIterations( solver, num_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_GetRelResidualNorm(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.GetRelResidualNorm} (GetRelResidualNorm method) */

   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;

   ierr = HYPRE_SStructSplitGetFinalRelativeResidualNorm( solver, norm );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetCommunicator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetCommunicator} (SetCommunicator method) */
   return 1; /* deprecated and will never be implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetCommunicator) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetIntParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetIntParameter} (SetIntParameter method) */


   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      ierr += bHYPRE_SStructSplit_SetMaxIterations( self, value, _ex ); SIDL_CHECK(*_ex);
   }
   else if ( strcmp(name,"NonZeroGuess")==0 || strcmp(name,"nonzero guess")==0 )
   {
      if ( value==0 )
      {
         ierr += HYPRE_SStructSplitSetZeroGuess( solver );
      }
      else if ( value==1 )
      {
         ierr += HYPRE_SStructSplitSetNonZeroGuess( solver );
      }
      else
      {
         ierr += HYPRE_SStructSplitSetNonZeroGuess( solver );
         ++ierr;
      }
   }
   else if ( strcmp(name,"ZeroGuess")==0 || strcmp(name,"zero guess")==0 )
   {
      if ( value==0 )
      {
         ierr += HYPRE_SStructSplitSetNonZeroGuess( solver );
      }
      else if ( value==1 )
      {
         ierr += HYPRE_SStructSplitSetZeroGuess( solver );
      }
      else
      {
         ierr += HYPRE_SStructSplitSetZeroGuess( solver );
         ++ierr;
      }
   }
   else
   {
      ierr=1;
   }

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetDoubleParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetDoubleParameter} (SetDoubleParameter method) */

   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 )
   {
      ierr += bHYPRE_SStructSplit_SetTolerance( self, value, _ex ); SIDL_CHECK(*_ex);
   }
   else
   {
      ierr=1;
   }

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetStringParameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
   /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetStringParameter) */
   /* Insert-Code-Here {bHYPRE.SStructSplit.SetStringParameter} (SetStringParameter method) */


   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"StructSolver")==0 || strcmp(name,"Struct Solver")==0 )
   {
      /* It would make sense to pass the solver object as an actual solver.  But
         that would require us to Babelize the whole "Split" algorithm.  The HYPRE
         Split solver sets all sub-solver parameters; it has no need for any more
         than a solver id. */

      if ( strcmp(value,"HYPRE_SMG")==0 || strcmp(value,"SMG")==0 )
      {
         ierr += HYPRE_SStructSplitSetStructSolver( solver, HYPRE_SMG );
         printf("***** SMG\n");
      }
      else if ( strcmp(value,"HYPRE_PFMG")==0 || strcmp(value,"PFMG")==0 )
      {
         ierr += HYPRE_SStructSplitSetStructSolver( solver, HYPRE_PFMG );
         printf("***** PFMG\n");
      }
      else
         ++ierr;   /* no other solvers supported */
   }
   else
   {
      ierr=1;
   }

   return ierr;

   /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_GetIntValue(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.GetIntValue} (GetIntValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_GetDoubleValue(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.GetDoubleValue} (GetDoubleValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_Setup(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.Setup) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.Setup} (Setup method) */

   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;
   struct bHYPRE_SStructMatrix__data * dataA;
   struct bHYPRE_SStructVector__data * datab, * datax;
   bHYPRE_SStructMatrix A;
   HYPRE_SStructMatrix HA;
   bHYPRE_SStructVector bHb, bHx;
   HYPRE_SStructVector Hb, Hx;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;
   A = data->matrix;
   dataA = bHYPRE_SStructMatrix__get_data( A );
   HA = dataA -> matrix;

   bHb = (bHYPRE_SStructVector) bHYPRE_Vector__cast2(b, "bHYPRE.SStructVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bHb!=NULL );

   datab = bHYPRE_SStructVector__get_data( bHb );
   Hb = datab -> vec;

   bHx = (bHYPRE_SStructVector) bHYPRE_Vector__cast2( x, "bHYPRE.SStructVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bHx!=NULL );

   datax = bHYPRE_SStructVector__get_data( bHx );
   Hx = datax -> vec;

   ierr += HYPRE_SStructSplitSetup( solver, HA, Hb, Hx );

   bHYPRE_SStructVector_deleteRef( bHb, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_SStructVector_deleteRef( bHx, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_Apply(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.Apply) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.Apply} (Apply method) */

   int ierr = 0;
   HYPRE_SStructSolver solver;
   struct bHYPRE_SStructSplit__data * data;
   struct bHYPRE_SStructMatrix__data * dataA;
   struct bHYPRE_SStructVector__data * datab, * datax;
   bHYPRE_SStructMatrix A;
   HYPRE_SStructMatrix HA;
   bHYPRE_SStructVector bHb, bHx;
   HYPRE_SStructVector Hb, Hx;

   data = bHYPRE_SStructSplit__get_data( self );
   solver = data->solver;
   A = data->matrix;
   dataA = bHYPRE_SStructMatrix__get_data( A );
   HA = dataA -> matrix;

   bHb = (bHYPRE_SStructVector) bHYPRE_Vector__cast2( b, "bHYPRE.SStructVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bHb!=NULL );

   datab = bHYPRE_SStructVector__get_data( bHb );
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

   bHx = (bHYPRE_SStructVector) bHYPRE_Vector__cast2( *x, "bHYPRE.SStructVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bHx!=NULL );

   datax = bHYPRE_SStructVector__get_data( bHx );
   Hx = datax -> vec;

   ierr += HYPRE_SStructSplitSolve( solver, HA, Hb, Hx );

   bHYPRE_SStructVector_deleteRef( bHb, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_SStructVector_deleteRef( bHx, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_ApplyAdjoint(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.ApplyAdjoint} (ApplyAdjoint method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructSplit_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructSplit_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_SStructSplit__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_SStructSplit(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructSplit__connectI(url, ar, _ex);
}
struct bHYPRE_SStructSplit__object* 
  impl_bHYPRE_SStructSplit_fcast_bHYPRE_SStructSplit(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructSplit__cast(bi, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructSplit_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructSplit_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructSplit_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructSplit_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructSplit_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructSplit_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
