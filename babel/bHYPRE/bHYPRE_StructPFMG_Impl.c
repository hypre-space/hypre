/*
 * File:          bHYPRE_StructPFMG_Impl.c
 * Symbol:        bHYPRE.StructPFMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.StructPFMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructPFMG" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * The StructPFMG solver requires a Struct matrix.
 */

#include "bHYPRE_StructPFMG_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._includes) */
/* Put additional includes or other arbitrary code here... */

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
#include "bHYPRE_StructMatrix.h"
#include "bHYPRE_StructMatrix_Impl.h"
#include "bHYPRE_StructVector.h"
#include "bHYPRE_StructVector_Impl.h"
#include "HYPRE_struct_ls.h"
#include "struct_ls.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructPFMG__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._load) */
  /* Insert-Code-Here {bHYPRE.StructPFMG._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructPFMG__ctor(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_StructPFMG__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructPFMG__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> solver = (HYPRE_StructSolver) NULL;
   data -> matrix = (bHYPRE_StructMatrix) NULL;
   bHYPRE_StructPFMG__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructPFMG__ctor2(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._ctor2) */
    /* Insert-Code-Here {bHYPRE.StructPFMG._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructPFMG__dtor(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructPFMG__data * data;
   data = bHYPRE_StructPFMG__get_data( self );
   ierr += HYPRE_StructPFMGDestroy( data->solver );
   bHYPRE_StructMatrix_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructPFMG
impl_bHYPRE_StructPFMG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructMatrix A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.Create) */
  /* Insert-Code-Here {bHYPRE.StructPFMG.Create} (Create method) */

   int ierr = 0;
   bHYPRE_StructPFMG solver = bHYPRE_StructPFMG__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_StructPFMG__data * data = bHYPRE_StructPFMG__get_data( solver );
   HYPRE_StructSolver dummy;
   HYPRE_StructSolver * Hsolver = &dummy;

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   ierr += HYPRE_StructPFMGCreate( (data->comm), Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   data->matrix = A;
   bHYPRE_StructMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetOperator(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_StructPFMG__data * data;
   bHYPRE_StructMatrix Amat;

   Amat = (bHYPRE_StructMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.StructMatrix", _ex );
   SIDL_CHECK(*_ex);
   if ( Amat==NULL) hypre_assert( "Unrecognized operator type."==(char *)A );

   data = bHYPRE_StructPFMG__get_data( self );
   data->matrix = Amat;
   bHYPRE_StructMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetTolerance(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructPFMGSetTol( solver, tolerance );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetMaxIterations(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructPFMGSetMaxIter( solver, max_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetLogging(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructPFMGSetLogging( solver, level );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetPrintLevel(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
 
   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructPFMGSetPrintLevel( solver, level );

   return ierr;

 /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_GetNumIterations(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructPFMGGetNumIterations( solver, num_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_GetRelResidualNorm(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructPFMGGetFinalRelativeResidualNorm( solver, norm );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetCommunicator(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   HYPRE_StructSolver dummy;
   HYPRE_StructSolver * solver = &dummy;
   struct bHYPRE_StructPFMG__data * data = bHYPRE_StructPFMG__get_data( self );

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   if ( data->solver == NULL )
   {
      ierr += HYPRE_StructPFMGCreate( (data->comm), solver );
      data -> solver = *solver;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructPFMG_Destroy(
  /* in */ bHYPRE_StructPFMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.Destroy) */
    /* Insert-Code-Here {bHYPRE.StructPFMG.Destroy} (Destroy method) */
     bHYPRE_StructPFMG_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetIntParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      ierr += HYPRE_StructPFMGSetMaxIter( solver, value );
   }
   else if ( strcmp(name,"MaxLevels")==0 )
   {
      ierr += HYPRE_StructPFMGSetMaxLevels( solver, value );
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"relative change test")==0 )
   {
      ierr += HYPRE_StructPFMGSetRelChange( solver, value );
   }
   else if ( strcmp(name,"NonZeroGuess")==0 || strcmp(name,"nonzero guess")==0 )
   {
      if ( value==0 )
      {
         ierr += HYPRE_StructPFMGSetZeroGuess( solver );
      }
      else if ( value==1 )
      {
         ierr += HYPRE_StructPFMGSetNonZeroGuess( solver );
      }
      else
      {
         ierr += HYPRE_StructPFMGSetNonZeroGuess( solver );
         ++ierr;
      }
   }
   else if ( strcmp(name,"ZeroGuess")==0 || strcmp(name,"zero guess")==0 )
   {
      if ( value==0 )
      {
         ierr += HYPRE_StructPFMGSetNonZeroGuess( solver );
      }
      else if ( value==1 )
      {
         ierr += HYPRE_StructPFMGSetZeroGuess( solver );
      }
      else
      {
         ierr += HYPRE_StructPFMGSetZeroGuess( solver );
         ++ierr;
      }
   }
   else if ( strcmp(name,"RelaxType")==0 )
   {
      ierr += HYPRE_StructPFMGSetRelaxType( solver, value );
   }
   else if ( strcmp(name,"RapType")==0 || strcmp(name,"rap type")==0 )
   {
      ierr += HYPRE_StructPFMGSetRAPType( solver, value );
   }
   else if ( strcmp(name,"NumPreRelax")==0 || strcmp(name,"num prerelax")==0 )
   {
      ierr += HYPRE_StructPFMGSetNumPreRelax( solver, value );
   }
   else if ( strcmp(name,"NumPostRelax")==0 || strcmp(name,"num postrelax")==0 )
   {
      ierr += HYPRE_StructPFMGSetNumPostRelax( solver, value );
   }
   else if ( strcmp(name,"SkipRelax")==0 || strcmp(name,"skip relax")==0 )
   {
      ierr += HYPRE_StructPFMGSetSkipRelax( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_StructPFMGSetLogging( solver, value );
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      ierr += HYPRE_StructPFMGSetPrintLevel( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetDoubleParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 )
   {
      ierr += HYPRE_StructPFMGSetTol( solver, value );      
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetStringParameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetIntArray1Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetIntArray2Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Dxyz")==0 )
   {
      ierr += HYPRE_StructPFMGSetDxyz( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_GetIntValue(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"NumIterations")==0 )
   {
      ierr = HYPRE_StructPFMGGetNumIterations( solver, value );
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      ierr += HYPRE_StructPFMGGetMaxIter( solver, value );
   }
   else if ( strcmp(name,"MaxLevels")==0 )
   {
      ierr += HYPRE_StructPFMGGetMaxLevels( solver, value );
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"relative change test")==0 )
   {
      ierr += HYPRE_StructPFMGGetRelChange( solver, value );
   }
   else if ( strcmp(name,"NonZeroGuess")==0 || strcmp(name,"nonzero guess")==0 )
   {
      ierr += HYPRE_StructPFMGGetZeroGuess( solver, value );
      if ( *value==0 )
         *value = 1;
      else if ( *value==1 )
         *value = 0;
      else
         ++ierr;
   }
   else if ( strcmp(name,"ZeroGuess")==0 || strcmp(name,"zero guess")==0 )
   {
      ierr += HYPRE_StructPFMGGetZeroGuess( solver, value );
   }
   else if ( strcmp(name,"RelaxType")==0 )
   {
      ierr += HYPRE_StructPFMGGetRelaxType( solver, value );
   }
   else if ( strcmp(name,"RapType")==0 || strcmp(name,"rap type")==0 )
   {
      ierr += HYPRE_StructPFMGGetRAPType( solver, value );
   }
   else if ( strcmp(name,"NumPreRelax")==0 || strcmp(name,"num prerelax")==0 )
   {
      ierr += HYPRE_StructPFMGGetNumPreRelax( solver, value );
   }
   else if ( strcmp(name,"NumPostRelax")==0 || strcmp(name,"num postrelax")==0 )
   {
      ierr += HYPRE_StructPFMGGetNumPostRelax( solver, value );
   }
   else if ( strcmp(name,"SkipRelax")==0 || strcmp(name,"skip relax")==0 )
   {
      ierr += HYPRE_StructPFMGGetSkipRelax( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_StructPFMGGetLogging( solver, value );
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      ierr += HYPRE_StructPFMGGetPrintLevel( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_GetDoubleValue(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 )
   {
      ierr = HYPRE_StructPFMGGetFinalRelativeResidualNorm( solver, value );
   }
   else if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 )
   {
      ierr += HYPRE_StructPFMGGetTol( solver, value );      
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_Setup(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;
   struct bHYPRE_StructMatrix__data * dataA;
   struct bHYPRE_StructVector__data * datab, * datax;
   bHYPRE_StructMatrix A;
   HYPRE_StructMatrix HA;
   bHYPRE_StructVector bHYPREP_b, bHYPREP_x;
   HYPRE_StructVector Hb, Hx;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;
   A = data->matrix;
   dataA = bHYPRE_StructMatrix__get_data( A );
   HA = dataA -> matrix;

   bHYPREP_b = (bHYPRE_StructVector) bHYPRE_Vector__cast2( b, "bHYPRE.StructVector", _ex ); SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)x );

   datab = bHYPRE_StructVector__get_data( bHYPREP_b );
   Hb = datab -> vec;

   bHYPREP_x = (bHYPRE_StructVector) bHYPRE_Vector__cast2( x, "bHYPRE.StructVector", _ex ); SIDL_CHECK(*_ex);
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)(x) );

   datax = bHYPRE_StructVector__get_data( bHYPREP_x );
   Hx = datax -> vec;

   ierr += HYPRE_StructPFMGSetup( solver, HA, Hb, Hx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_Apply(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructPFMG__data * data;
   struct bHYPRE_StructMatrix__data * dataA;
   struct bHYPRE_StructVector__data * datab, * datax;
   bHYPRE_StructMatrix A;
   HYPRE_StructMatrix HA;
   bHYPRE_StructVector bHYPREP_b, bHYPREP_x;
   HYPRE_StructVector Hb, Hx;

   data = bHYPRE_StructPFMG__get_data( self );
   solver = data->solver;
   A = data->matrix;
   dataA = bHYPRE_StructMatrix__get_data( A );
   HA = dataA -> matrix;

   bHYPREP_b = (bHYPRE_StructVector) bHYPRE_Vector__cast2( b, "bHYPRE.StructVector", _ex ); SIDL_CHECK(*_ex);
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

   ierr += HYPRE_StructPFMGSolve( solver, HA, Hb, Hx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructPFMG_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructPFMG_ApplyAdjoint(
  /* in */ bHYPRE_StructPFMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructPFMG.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.StructPFMG.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructPFMG.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructPFMG_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructPFMG_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Solver(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_StructPFMG_fcast_bHYPRE_Solver(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_StructMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructPFMG_fcast_bHYPRE_StructMatrix(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructMatrix__cast(bi, _ex);
}
struct bHYPRE_StructPFMG__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_StructPFMG(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructPFMG__connectI(url, ar, _ex);
}
struct bHYPRE_StructPFMG__object* 
  impl_bHYPRE_StructPFMG_fcast_bHYPRE_StructPFMG(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructPFMG__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructPFMG_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_StructPFMG_fcast_bHYPRE_Vector(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructPFMG_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructPFMG_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructPFMG_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructPFMG_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructPFMG_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
