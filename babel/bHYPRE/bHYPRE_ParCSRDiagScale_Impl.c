/*
 * File:          bHYPRE_ParCSRDiagScale_Impl.c
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.ParCSRDiagScale" (version 1.0.0)
 * 
 * Diagonal scaling preconditioner for ParCSR matrix class.
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 */

#include "bHYPRE_ParCSRDiagScale_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._includes) */
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
#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParCSRDiagScale__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._load) */
  /* Insert-Code-Here {bHYPRE.ParCSRDiagScale._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParCSRDiagScale__ctor(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* Note: user calls of __create() are DEPRECATED, _Create also calls this function */

   struct bHYPRE_ParCSRDiagScale__data * data;
   data = hypre_CTAlloc( struct bHYPRE_ParCSRDiagScale__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> matrix = NULL;
   bHYPRE_ParCSRDiagScale__set_data( self, data );
   /* hypre diagonal scaling requires no constructor or setup. cf
    * parcsr/HYPRE_parcsr_pcg.c, function HYPRE_ParCSRDiagScale */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParCSRDiagScale__ctor2(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._ctor2) */
    /* Insert-Code-Here {bHYPRE.ParCSRDiagScale._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParCSRDiagScale__dtor(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._dtor) */
  /* Insert the implementation of the destructor method here... */

   struct bHYPRE_ParCSRDiagScale__data * data;
   data = bHYPRE_ParCSRDiagScale__get_data( self );

   bHYPRE_IJParCSRMatrix_deleteRef( data->matrix, _ex ); SIDL_CHECK(*_ex);
   /* delete any nontrivial data components here */
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_ParCSRDiagScale
impl_bHYPRE_ParCSRDiagScale_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_IJParCSRMatrix A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.Create) */
  /* Insert-Code-Here {bHYPRE.ParCSRDiagScale.Create} (Create method) */

   bHYPRE_ParCSRDiagScale solver = bHYPRE_ParCSRDiagScale__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_ParCSRDiagScale__data * data = bHYPRE_ParCSRDiagScale__get_data( solver );

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   data->matrix = A;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.Create) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetOperator(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_ParCSRDiagScale__data * data;
   bHYPRE_IJParCSRMatrix bH_A;

   bH_A = (bHYPRE_IJParCSRMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.IJParCSRMatrix", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bH_A!=NULL );

   data = bHYPRE_ParCSRDiagScale__get_data( self );
   data->matrix = bH_A;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetTolerance(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetMaxIterations(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetLogging(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetPrintLevel(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_GetNumIterations(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   /* diagonal scaling is like 1 step of Jacobi */
   *num_iterations = 1;
   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetCommunicator(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED   Use Create */

   int ierr = 0;
   struct bHYPRE_ParCSRDiagScale__data * data;
   data = bHYPRE_ParCSRDiagScale__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   bHYPRE_ParCSRDiagScale__set_data( self, data );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParCSRDiagScale_Destroy(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.Destroy) */
    /* Insert-Code-Here {bHYPRE.ParCSRDiagScale.Destroy} (Destroy method) */
     bHYPRE_ParCSRDiagScale_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetStringParameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetIntArray1Parameter) 
    */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetIntArray2Parameter) 
    */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_GetIntValue(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_GetDoubleValue(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_Setup(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.Setup) */
  /* Insert the implementation of the Setup method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_Apply(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   MPI_Comm comm;
   HYPRE_Solver dummy;
   HYPRE_Solver * solver = &dummy;
   struct bHYPRE_ParCSRDiagScale__data * data;
   /* not used HYPRE_Matrix HYPRE_A;*/
   bHYPRE_IJParCSRMatrix bH_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   /* not used HYPRE_Vector HYPRE_x, HYPRE_b;*/
   bHYPRE_IJParCSRVector bH_b, bH_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = bHYPRE_ParCSRDiagScale__get_data( self );
   comm = data->comm;
   /* SetCommunicator should have been called earlier */
   hypre_assert( comm != MPI_COMM_NULL );
   bH_A = data->matrix;
   hypre_assert( bH_A != NULL );

   if ( *x==NULL )
   {  /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or
       * hypre_assert(x-has-the-right-size) */
      bHYPRE_Vector_Clone( b, x, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Vector_Clear( *x, _ex ); SIDL_CHECK(*_ex);
   }

   bH_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bH_b!=NULL );

   datab = bHYPRE_IJParCSRVector__get_data( bH_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;
   /* not used HYPRE_b = (HYPRE_Vector) bb;*/

   bH_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( *x, "bHYPRE.IJParCSRVector", _ex ); SIDL_CHECK(*_ex);
   hypre_assert( bH_x!=NULL );

   datax = bHYPRE_IJParCSRVector__get_data( bH_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   /* not used HYPRE_b = (HYPRE_Vector) xx;*/

   dataA = bHYPRE_IJParCSRMatrix__get_data( bH_A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   AA = (HYPRE_ParCSRMatrix) objectA;
   /* not used HYPRE_A = (HYPRE_Matrix) AA;*/

   /* does x = y/diagA as approximation to solving Ax=y for x ... */
   ierr += HYPRE_ParCSRDiagScale( *solver, AA, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bH_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bH_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParCSRDiagScale_ApplyAdjoint(
  /* in */ bHYPRE_ParCSRDiagScale self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.ParCSRDiagScale.ApplyAdjoint} (ApplyAdjoint method) */

   /* The adjoint of a (real) diagonal matrix is itself, so just call Apply: */
     int32_t ierr = impl_bHYPRE_ParCSRDiagScale_Apply( self, b, x, _ex ); SIDL_CHECK(*_ex);
     return ierr;

     hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_IJParCSRMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_IJParCSRMatrix__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_ParCSRDiagScale__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_ParCSRDiagScale(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_ParCSRDiagScale__connectI(url, ar, _ex);
}
struct bHYPRE_ParCSRDiagScale__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_ParCSRDiagScale(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_ParCSRDiagScale__cast(bi, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Solver(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Solver(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_ParCSRDiagScale_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_ParCSRDiagScale_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
