/*
 * File:          bHYPRE_Euclid_Impl.c
 * Symbol:        bHYPRE.Euclid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.Euclid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.Euclid" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * Although the usual Solver SetParameter functions are available,
 * a Euclid-stype parameter-setting function is also available, SetParameters.
 */

#include "bHYPRE_Euclid_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid._includes) */
/* Insert-Code-Here {bHYPRE.Euclid._includes} (includes and arbitrary code) */

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

#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
#include <assert.h>
#include "hypre_babel_exception_handler.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.Euclid._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Euclid__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid._load) */
  /* Insert-Code-Here {bHYPRE.Euclid._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Euclid__ctor(
  /* in */ bHYPRE_Euclid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid._ctor) */
  /* Insert-Code-Here {bHYPRE.Euclid._ctor} (constructor method) */

   /* Note: user calls of __create() are DEPRECATED, _Create also calls this function */

   struct bHYPRE_Euclid__data * data;
   data = hypre_CTAlloc( struct bHYPRE_Euclid__data, 1 );
   data -> comm = (MPI_Comm) NULL;
   data -> solver = (HYPRE_Solver) NULL;
   data -> matrix = (bHYPRE_IJParCSRMatrix) NULL;

   bHYPRE_Euclid__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Euclid__ctor2(
  /* in */ bHYPRE_Euclid self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid._ctor2) */
    /* Insert-Code-Here {bHYPRE.Euclid._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_Euclid__dtor(
  /* in */ bHYPRE_Euclid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid._dtor) */
  /* Insert-Code-Here {bHYPRE.Euclid._dtor} (destructor method) */

   int ierr = 0;
   struct bHYPRE_Euclid__data * data;

   data = bHYPRE_Euclid__get_data( self );
   if ( data->matrix != (bHYPRE_IJParCSRMatrix) NULL )
      bHYPRE_IJParCSRMatrix_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   ierr += HYPRE_EuclidDestroy( data->solver );
   hypre_assert( ierr== 0 );

   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_Euclid
impl_bHYPRE_Euclid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_IJParCSRMatrix A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.Create) */
  /* Insert-Code-Here {bHYPRE.Euclid.Create} (Create method) */

   int ierr = 0;
   HYPRE_Solver dummy;
   HYPRE_Solver * Hsolver = &dummy;
   bHYPRE_Euclid solver = bHYPRE_Euclid__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_Euclid__data * data = bHYPRE_Euclid__get_data( solver );

   data->matrix = A;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   ierr += HYPRE_EuclidCreate( (data->comm), Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.Create) */
  }
}

/*
 * Method:  SetParameters[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetParameters"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetParameters(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t argc,
  /* inout */ char** argv,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetParameters) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetParameters} (SetParameters method) */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_Euclid__data * data;
   data = bHYPRE_Euclid__get_data( self );
   solver = data->solver;

   ierr +=  HYPRE_EuclidSetParams( solver, argc, argv );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetParameters) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetOperator(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetOperator) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetOperator} (SetOperator method) */


   int ierr = 0;
   struct bHYPRE_Euclid__data * data;
   bHYPRE_IJParCSRMatrix Amat;

   Amat = (bHYPRE_IJParCSRMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.IJParCSRMatrix" , _ex );
   SIDL_CHECK(*_ex);
   if ( Amat==NULL ) hypre_assert( "Unrecognized operator type."==(char *)A );

   data = bHYPRE_Euclid__get_data( self );
   data->matrix = Amat;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetTolerance(
  /* in */ bHYPRE_Euclid self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetTolerance} (SetTolerance method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetMaxIterations(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetMaxIterations} (SetMaxIterations method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_Euclid_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetLogging(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetLogging) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetLogging} (SetLogging method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_Euclid_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetPrintLevel(
  /* in */ bHYPRE_Euclid self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetPrintLevel} (SetPrintLevel method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_GetNumIterations(
  /* in */ bHYPRE_Euclid self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.Euclid.GetNumIterations} (GetNumIterations method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_GetRelResidualNorm(
  /* in */ bHYPRE_Euclid self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.Euclid.GetRelResidualNorm} (GetRelResidualNorm method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetCommunicator(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetCommunicator} (SetCommunicator method) */
   return 1;  /* DEPRECATED and will never be implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetCommunicator) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetIntParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetIntParameter} (SetIntParameter method) */

   /* Euclid has its own parameter parsing system, based on a C-program-style
      argc and argv.  We'll declare and set argc and argv based on the standard
      hypre-Babel parameters. Unlike most other solver interfaces, this one has
      no need to check the parameter name, Euclid does that.
   */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_Euclid__data * data;
   int argc = 1;
   char * dummy;
   char ** argv = &dummy;

   *argv = hypre_CTAlloc( char, 256 );  /* automatically initialized to 0 */

   data = bHYPRE_Euclid__get_data( self );
   solver = data->solver;

   printf("Euclid SetIntParameter name=%s  value=%i\n", name, value ); /* for testing */

   sprintf( *argv, "%s %i", name, value );
   printf( *argv ); printf("\n"); /* for testing */
   ierr +=  HYPRE_EuclidSetParams( solver, argc, argv );

   hypre_TFree( *argv );
   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetDoubleParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetDoubleParameter} (SetDoubleParameter method) */


   /* Euclid has its own parameter parsing system, based on a C-program-style
      argc and argv.  We'll declare and set argc and argv based on the standard
      hypre-Babel parameters. Unlike most other solver interfaces, this one has
      no need to check the parameter name, Euclid does that.
   */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_Euclid__data * data;
   int argc = 1;
   char * dummy;
   char ** argv = &dummy;

   *argv = hypre_CTAlloc( char, 256 );  /* automatically initialized to 0 */

   data = bHYPRE_Euclid__get_data( self );
   solver = data->solver;

   printf("Euclid SetDoubleParameter name=%s  value=%e\n", name, value ); /* for testing */

   sprintf( *argv, "%s %e", name, value );
   printf( *argv ); printf("\n"); /* for testing */
   ierr +=  HYPRE_EuclidSetParams( solver, argc, argv );

   hypre_TFree( *argv );
   return ierr;


  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetStringParameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetStringParameter} (SetStringParameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetIntArray1Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetIntArray2Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetDoubleArray1Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_SetDoubleArray2Parameter(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.Euclid.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_GetIntValue(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.Euclid.GetIntValue} (GetIntValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_GetDoubleValue(
  /* in */ bHYPRE_Euclid self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.Euclid.GetDoubleValue} (GetDoubleValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_Setup(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.Setup) */
  /* Insert-Code-Here {bHYPRE.Euclid.Setup} (Setup method) */


   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_Euclid__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_Euclid__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;


   bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)x );

   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   bHYPREP_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( x, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)(x) );

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   ierr += HYPRE_EuclidSetup( solver, bHYPREP_A, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_Apply(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.Apply) */
  /* Insert-Code-Here {bHYPRE.Euclid.Apply} (Apply method) */


   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_Euclid__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_Euclid__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;

   bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)x );

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
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)(*x) );

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_EuclidSolve( solver, bHYPREP_A, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Euclid_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Euclid_ApplyAdjoint(
  /* in */ bHYPRE_Euclid self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Euclid.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.Euclid.ApplyAdjoint} (ApplyAdjoint method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Euclid.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Euclid__object* impl_bHYPRE_Euclid_fconnect_bHYPRE_Euclid(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Euclid__connectI(url, ar, _ex);
}
struct bHYPRE_Euclid__object* impl_bHYPRE_Euclid_fcast_bHYPRE_Euclid(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Euclid__cast(bi, _ex);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_IJParCSRMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_Euclid_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_IJParCSRMatrix__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Euclid_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_Euclid_fconnect_bHYPRE_Operator(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_Euclid_fcast_bHYPRE_Operator(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_Euclid_fconnect_bHYPRE_Solver(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_Euclid_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Euclid_fconnect_bHYPRE_Vector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Euclid_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_Euclid_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_Euclid_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Euclid_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Euclid_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_Euclid_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_Euclid_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_Euclid_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_Euclid_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
