/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/


/*
 * File:          bHYPRE_Schwarz_Impl.c
 * Symbol:        bHYPRE.Schwarz-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.Schwarz
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.Schwarz" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * Schwarz requires an IJParCSR matrix
 */

#include "bHYPRE_Schwarz_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz._includes) */
/* Insert-Code-Here {bHYPRE.Schwarz._includes} (includes and arbitrary code) */


#include <assert.h>
#include "hypre_babel_exception_handler.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Schwarz__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz._load) */
  /* Insert-Code-Here {bHYPRE.Schwarz._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Schwarz__ctor(
  /* in */ bHYPRE_Schwarz self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz._ctor) */
  /* Insert-Code-Here {bHYPRE.Schwarz._ctor} (constructor method) */

   /* Note: user calls of __create() are DEPRECATED, _Create also calls this function */

   struct bHYPRE_Schwarz__data * data;
   data = hypre_CTAlloc( struct bHYPRE_Schwarz__data, 1 );
   data -> solver = (HYPRE_Solver) NULL;
   data -> matrix = (bHYPRE_IJParCSRMatrix) NULL;
   /* set any other data components here */
   bHYPRE_Schwarz__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Schwarz__ctor2(
  /* in */ bHYPRE_Schwarz self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz._ctor2) */
    /* Insert-Code-Here {bHYPRE.Schwarz._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Schwarz__dtor(
  /* in */ bHYPRE_Schwarz self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz._dtor) */
  /* Insert-Code-Here {bHYPRE.Schwarz._dtor} (destructor method) */

   int ierr = 0;
   struct bHYPRE_Schwarz__data * data;

   data = bHYPRE_Schwarz__get_data( self );
   if ( data->matrix != (bHYPRE_IJParCSRMatrix) NULL )
   {
      bHYPRE_IJParCSRMatrix_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   }
   ierr += HYPRE_SchwarzDestroy( data->solver );
   hypre_assert( ierr== 0 );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_Schwarz
impl_bHYPRE_Schwarz_Create(
  /* in */ bHYPRE_IJParCSRMatrix A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.Create) */
  /* Insert-Code-Here {bHYPRE.Schwarz.Create} (Create method) */

   int ierr = 0;
   HYPRE_Solver dummy;
   HYPRE_Solver * Hsolver = &dummy;
   bHYPRE_Schwarz solver = bHYPRE_Schwarz__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_Schwarz__data * data = bHYPRE_Schwarz__get_data( solver );

   ierr += HYPRE_SchwarzCreate( Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   data->matrix = A;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetOperator(
  /* in */ bHYPRE_Schwarz self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetOperator) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetOperator} (SetOperator method) */

   int ierr = 0;
   struct bHYPRE_Schwarz__data * data;
   bHYPRE_IJParCSRMatrix Amat;

   Amat = (bHYPRE_IJParCSRMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.IJParCSRMatrix", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( Amat!=NULL );

   data = bHYPRE_Schwarz__get_data( self );
   data->matrix = Amat;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetTolerance(
  /* in */ bHYPRE_Schwarz self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetTolerance} (SetTolerance method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetMaxIterations(
  /* in */ bHYPRE_Schwarz self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetMaxIterations} (SetMaxIterations method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_Schwarz_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetLogging(
  /* in */ bHYPRE_Schwarz self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetLogging) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetLogging} (SetLogging method) */

   return 0; /* ignored */

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_Schwarz_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetPrintLevel(
  /* in */ bHYPRE_Schwarz self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetPrintLevel} (SetPrintLevel method) */

   return 0;  /* ignored */

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_GetNumIterations(
  /* in */ bHYPRE_Schwarz self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.Schwarz.GetNumIterations} (GetNumIterations method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_GetRelResidualNorm(
  /* in */ bHYPRE_Schwarz self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.Schwarz.GetRelResidualNorm} (GetRelResidualNorm method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetCommunicator(
  /* in */ bHYPRE_Schwarz self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetCommunicator} (SetCommunicator method) */
   return 1;  /* no MPI in this solver, and if there were I still wouldn't
                 implement this deprecated function */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Schwarz_Destroy(
  /* in */ bHYPRE_Schwarz self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.Destroy) */
    /* Insert-Code-Here {bHYPRE.Schwarz.Destroy} (Destroy method) */
     bHYPRE_Schwarz_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetIntParameter(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetIntParameter} (SetIntParameter method) */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_Schwarz__data * data;

   data = bHYPRE_Schwarz__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Variant")==0 )
   {
      ierr += HYPRE_SchwarzSetVariant( solver, value );
   }
   else if ( strcmp(name,"Overlap")==0 )
   {
      ierr += HYPRE_SchwarzSetOverlap( solver, value );
   }
   else if ( strcmp(name,"DomainType")==0 )
   {
      ierr += HYPRE_SchwarzSetDomainType( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetDoubleParameter(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetDoubleParameter} (SetDoubleParameter method) */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_Schwarz__data * data;

   data = bHYPRE_Schwarz__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"RelaxWeight")==0 )
   {
      ierr += HYPRE_SchwarzSetRelaxWeight( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetStringParameter(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetStringParameter} (SetStringParameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetIntArray1Parameter(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetIntArray2Parameter(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_GetIntValue(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.Schwarz.GetIntValue} (GetIntValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_GetDoubleValue(
  /* in */ bHYPRE_Schwarz self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.Schwarz.GetDoubleValue} (GetDoubleValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_Setup(
  /* in */ bHYPRE_Schwarz self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.Setup) */
  /* Insert-Code-Here {bHYPRE.Schwarz.Setup} (Setup method) */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_Schwarz__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix HA;
   bHYPRE_IJParCSRVector bHb, bHx;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_Schwarz__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   HA = (HYPRE_ParCSRMatrix) objectA;

   bHb = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bHb!=NULL );

   datab = bHYPRE_IJParCSRVector__get_data( bHb );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   bHx = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( x, "bHYPRE.IJParCSRVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bHx!=NULL );

   datax = bHYPRE_IJParCSRVector__get_data( bHx );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   ierr += HYPRE_SchwarzSetup( solver, HA, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHb, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHx, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_Apply(
  /* in */ bHYPRE_Schwarz self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.Apply) */
  /* Insert-Code-Here {bHYPRE.Schwarz.Apply} (Apply method) */


   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_Schwarz__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHA;
   bHYPRE_IJParCSRVector bHb, bHx;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_Schwarz__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHA = (HYPRE_ParCSRMatrix) objectA;

   bHb = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bHb!=NULL );

   datab = bHYPRE_IJParCSRVector__get_data( bHb );
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

   bHx = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( *x, "bHYPRE.IJParCSRVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bHx!=NULL );

   datax = bHYPRE_IJParCSRVector__get_data( bHx );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_SchwarzSolve( solver, bHA, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHb, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHx, _ex ); SIDL_CHECK(*_ex);
   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_ApplyAdjoint(
  /* in */ bHYPRE_Schwarz self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.Schwarz.ApplyAdjoint} (ApplyAdjoint method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_Schwarz_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_IJParCSRMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_Schwarz_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_IJParCSRMatrix__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Schwarz_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Schwarz_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_Schwarz_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_Schwarz_fcast_bHYPRE_Operator(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_Schwarz__object* 
  impl_bHYPRE_Schwarz_fconnect_bHYPRE_Schwarz(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Schwarz__connectI(url, ar, _ex);
}
struct bHYPRE_Schwarz__object* impl_bHYPRE_Schwarz_fcast_bHYPRE_Schwarz(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Schwarz__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_Schwarz_fconnect_bHYPRE_Solver(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_Schwarz_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Schwarz_fconnect_bHYPRE_Vector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Schwarz_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_Schwarz_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_Schwarz_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Schwarz_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Schwarz_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_Schwarz_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_Schwarz_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_Schwarz_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_Schwarz_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
