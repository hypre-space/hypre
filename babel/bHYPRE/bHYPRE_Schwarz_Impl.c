/*
 * File:          bHYPRE_Schwarz_Impl.c
 * Symbol:        bHYPRE.Schwarz-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.Schwarz
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
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
 * 
 * 
 */

#include "bHYPRE_Schwarz_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz._includes) */
/* Insert-Code-Here {bHYPRE.Schwarz._includes} (includes and arbitrary code) */

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
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz._includes) */

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
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz._load) */
  /* Insert-Code-Here {bHYPRE.Schwarz._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz._load) */
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
  /* in */ bHYPRE_Schwarz self)
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
  /* in */ bHYPRE_Schwarz self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz._dtor) */
  /* Insert-Code-Here {bHYPRE.Schwarz._dtor} (destructor method) */

   int ierr = 0;
   struct bHYPRE_Schwarz__data * data;

   data = bHYPRE_Schwarz__get_data( self );
   if ( data->matrix != (bHYPRE_IJParCSRMatrix) NULL )
      bHYPRE_IJParCSRMatrix_deleteRef( data->matrix );
   ierr += HYPRE_SchwarzDestroy( data->solver );
   hypre_assert( ierr== 0 );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz._dtor) */
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
  /* in */ bHYPRE_IJParCSRMatrix A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.Create) */
  /* Insert-Code-Here {bHYPRE.Schwarz.Create} (Create method) */

   int ierr = 0;
   HYPRE_Solver dummy;
   HYPRE_Solver * Hsolver = &dummy;
   bHYPRE_Schwarz solver = bHYPRE_Schwarz__create();
   struct bHYPRE_Schwarz__data * data = bHYPRE_Schwarz__get_data( solver );

   ierr += HYPRE_SchwarzCreate( Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   data->matrix = A;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix );

   return solver;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.Create) */
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetCommunicator(
  /* in */ bHYPRE_Schwarz self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetCommunicator} (SetCommunicator method) */
   return 1;  /* no MPI in this solver, and if there were I still wouldn't
                 implement this deprecated function */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
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
  /* in */ int32_t value)
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

/*
 * Set the double parameter associated with {\tt name}.
 * 
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
  /* in */ double value)
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

/*
 * Set the string parameter associated with {\tt name}.
 * 
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
  /* in */ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetStringParameter} (SetStringParameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
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
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
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
  /* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
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
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
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
  /* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
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
  /* out */ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.Schwarz.GetIntValue} (GetIntValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
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
  /* out */ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.Schwarz.GetDoubleValue} (GetDoubleValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
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
  /* in */ bHYPRE_Vector x)
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

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHb = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   datab = bHYPRE_IJParCSRVector__get_data( bHb );
   bHYPRE_IJParCSRVector_deleteRef( bHb );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   if ( bHYPRE_Vector_queryInt( x, "bHYPRE.IJParCSRVector" ) )
   {
      bHx = bHYPRE_IJParCSRVector__cast( x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)(x) );
   }

   datax = bHYPRE_IJParCSRVector__get_data( bHx );
   bHYPRE_IJParCSRVector_deleteRef( bHx );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   ierr += HYPRE_SchwarzSetup( solver, HA, bb, xx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
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
  /* inout */ bHYPRE_Vector* x)
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

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHb = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   datab = bHYPRE_IJParCSRVector__get_data( bHb );
   bHYPRE_IJParCSRVector_deleteRef( bHb );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   if ( *x==NULL )
   {
      /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or hypre_assert(x
       * has the right size) */
      bHYPRE_Vector_Clone( b, x );
      bHYPRE_Vector_Clear( *x );
   }
   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.IJParCSRVector" ) )
   {
      bHx = bHYPRE_IJParCSRVector__cast( *x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)(*x) );
   }

   datax = bHYPRE_IJParCSRVector__get_data( bHx );
   bHYPRE_IJParCSRVector_deleteRef( bHx );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_SchwarzSolve( solver, bHA, bb, xx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.Apply) */
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
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
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.Schwarz.ApplyAdjoint} (ApplyAdjoint method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.ApplyAdjoint) */
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetOperator(
  /* in */ bHYPRE_Schwarz self,
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetOperator) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetOperator} (SetOperator method) */

   int ierr = 0;
   struct bHYPRE_Schwarz__data * data;
   bHYPRE_IJParCSRMatrix Amat;

   if ( bHYPRE_Operator_queryInt( A, "bHYPRE.IJParCSRMatrix" ) )
   {
      Amat = bHYPRE_IJParCSRMatrix__cast( A );
      bHYPRE_IJParCSRMatrix_deleteRef( Amat ); /* extra ref from queryInt */
   }
   else
   {
      hypre_assert( "Unrecognized operator type."==(char *)A );
   }

   data = bHYPRE_Schwarz__get_data( self );
   data->matrix = Amat;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix );

   return ierr;


  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetTolerance(
  /* in */ bHYPRE_Schwarz self,
  /* in */ double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetTolerance} (SetTolerance method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetMaxIterations(
  /* in */ bHYPRE_Schwarz self,
  /* in */ int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetMaxIterations} (SetMaxIterations method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetMaxIterations) */
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetLogging(
  /* in */ bHYPRE_Schwarz self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetLogging) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetLogging} (SetLogging method) */

   return 0; /* ignored */

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetLogging) */
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_SetPrintLevel(
  /* in */ bHYPRE_Schwarz self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.Schwarz.SetPrintLevel} (SetPrintLevel method) */

   return 0;  /* ignored */

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_GetNumIterations(
  /* in */ bHYPRE_Schwarz self,
  /* out */ int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.Schwarz.GetNumIterations} (GetNumIterations method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Schwarz_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_Schwarz_GetRelResidualNorm(
  /* in */ bHYPRE_Schwarz self,
  /* out */ double* norm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Schwarz.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.Schwarz.GetRelResidualNorm} (GetRelResidualNorm method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.Schwarz.GetRelResidualNorm) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Solver__object* impl_bHYPRE_Schwarz_fconnect_bHYPRE_Solver(char* 
  url, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* 
  obj) {
  return bHYPRE_Solver__getURL(obj);
}
struct bHYPRE_Schwarz__object* 
  impl_bHYPRE_Schwarz_fconnect_bHYPRE_Schwarz(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Schwarz__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_bHYPRE_Schwarz(struct 
  bHYPRE_Schwarz__object* obj) {
  return bHYPRE_Schwarz__getURL(obj);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_Schwarz_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_Schwarz_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) {
  return bHYPRE_Operator__getURL(obj);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_Schwarz_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_IJParCSRMatrix__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj) {
  return bHYPRE_IJParCSRMatrix__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_Schwarz_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* impl_bHYPRE_Schwarz_fconnect_bHYPRE_Vector(char* 
  url, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* 
  obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_Schwarz_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_Schwarz_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_Schwarz_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
