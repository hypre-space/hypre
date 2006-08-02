/*
 * File:          bHYPRE_SStructSplit_Impl.c
 * Symbol:        bHYPRE.SStructSplit-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.SStructSplit
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
 * Symbol "bHYPRE.SStructSplit" (version 1.0.0)
 * 
 * 
 * Documentation goes here.
 * 
 * The SStructSplit solver requires a SStruct matrix.
 * 
 * 
 */

#include "bHYPRE_SStructSplit_Impl.h"

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
#include "bHYPRE_SStructMatrix.h"
#include "bHYPRE_SStructMatrix_Impl.h"
#include "bHYPRE_SStructVector.h"
#include "bHYPRE_SStructVector_Impl.h"
#include "HYPRE_sstruct_ls.h"
#include "sstruct_ls.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._includes) */

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
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._load) */
  /* Insert-Code-Here {bHYPRE.SStructSplit._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._load) */
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
  /* in */ bHYPRE_SStructSplit self)
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
  /* in */ bHYPRE_SStructSplit self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit._dtor) */
  /* Insert-Code-Here {bHYPRE.SStructSplit._dtor} (destructor method) */

   int ierr = 0;
   struct bHYPRE_SStructSplit__data * data;
   data = bHYPRE_SStructSplit__get_data( self );
   ierr += HYPRE_SStructSplitDestroy( data->solver );
   bHYPRE_SStructMatrix_deleteRef( data->matrix );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit._dtor) */
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
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.Create) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.Create} (Create method) */

   int ierr = 0;
   bHYPRE_SStructSplit solver = bHYPRE_SStructSplit__create();
   struct bHYPRE_SStructSplit__data * data = bHYPRE_SStructSplit__get_data( solver );
   HYPRE_SStructSolver dummy;
   HYPRE_SStructSolver * Hsolver = &dummy;
   bHYPRE_SStructMatrix Amat;

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   ierr += HYPRE_SStructSplitCreate( (data->comm), Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   if ( bHYPRE_Operator_queryInt( A, "bHYPRE.SStructMatrix" ) )
   {
      Amat = bHYPRE_SStructMatrix__cast( A );
      bHYPRE_SStructMatrix_deleteRef( Amat ); /* extra ref from queryInt */
   }
   else
   {
      hypre_assert( "Unrecognized operator type."==(char *)A );
   }
   data->matrix = Amat;
   bHYPRE_SStructMatrix_addRef( data->matrix );

   return solver;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.Create) */
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetCommunicator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetCommunicator} (SetCommunicator method) */
   return 1; /* deprecated and will never be implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
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
  /* in */ int32_t value)
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
      ierr += bHYPRE_SStructSplit_SetMaxIterations( self, value );
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

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
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
  /* in */ double value)
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
      ierr += bHYPRE_SStructSplit_SetTolerance( self, value );      
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
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
  /* in */ const char* value)
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

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
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
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
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
  /* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
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
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
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
  /* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
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
  /* out */ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.GetIntValue} (GetIntValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
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
  /* out */ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.GetDoubleValue} (GetDoubleValue method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
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
  /* in */ bHYPRE_Vector x)
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

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.SStructVector" ) )
   {
      bHb = bHYPRE_SStructVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }
   datab = bHYPRE_SStructVector__get_data( bHb );
   bHYPRE_SStructVector_deleteRef( bHb );
   Hb = datab -> vec;

   if ( bHYPRE_Vector_queryInt( x, "bHYPRE.SStructVector" ) )
   {
      bHx = bHYPRE_SStructVector__cast( x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)(x) );
   }
   datax = bHYPRE_SStructVector__get_data( bHx );
   bHYPRE_SStructVector_deleteRef( bHx );
   Hx = datax -> vec;

   ierr += HYPRE_SStructSplitSetup( solver, HA, Hb, Hx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
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
  /* inout */ bHYPRE_Vector* x)
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

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.SStructVector" ) )
   {
      bHb = bHYPRE_SStructVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }
   datab = bHYPRE_SStructVector__get_data( bHb );
   bHYPRE_SStructVector_deleteRef( bHb );
   Hb = datab -> vec;

   if ( *x==NULL )
   {
      /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or hypre_assert(x
       * has the right size) */
      bHYPRE_Vector_Clone( b, x );
      bHYPRE_Vector_Clear( *x );
   }
   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.SStructVector" ) )
   {
      bHx = bHYPRE_SStructVector__cast( *x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)(*x) );
   }
   datax = bHYPRE_SStructVector__get_data( bHx );
   bHYPRE_SStructVector_deleteRef( bHx );
   Hx = datax -> vec;

   ierr += HYPRE_SStructSplitSolve( solver, HA, Hb, Hx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.Apply) */
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
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
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.ApplyAdjoint} (ApplyAdjoint method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.ApplyAdjoint) */
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetOperator(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetOperator) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetOperator} (SetOperator method) */

   int ierr = 0;
   struct bHYPRE_SStructSplit__data * data;
   bHYPRE_SStructMatrix Amat;

   if ( bHYPRE_Operator_queryInt( A, "bHYPRE.SStructMatrix" ) )
   {
      Amat = bHYPRE_SStructMatrix__cast( A );
      bHYPRE_SStructMatrix_deleteRef( Amat ); /* extra ref from queryInt */
   }
   else
   {
      hypre_assert( "Unrecognized operator type."==(char *)A );
   }

   data = bHYPRE_SStructSplit__get_data( self );
   data->matrix = Amat;
   bHYPRE_SStructMatrix_addRef( data->matrix );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetTolerance(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ double tolerance)
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

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetMaxIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t max_iterations)
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
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetLogging(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetLogging) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetLogging} (SetLogging method) */

  /* ignored by HYPRE_SStructSplit, but it sets Logging to 0 for solvers it calls */
   if ( level==0 ) return 0;
   else return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_SStructSplit_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_SetPrintLevel(
  /* in */ bHYPRE_SStructSplit self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructSplit.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.SStructSplit.SetPrintLevel} (SetPrintLevel method) */

  /* ignored by HYPRE_SStructSplit, but it sets PrintLevel to 0 for solvers it calls */
   if ( level==0 ) return 0;
   else return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructSplit.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_GetNumIterations(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ int32_t* num_iterations)
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

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructSplit_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructSplit_GetRelResidualNorm(
  /* in */ bHYPRE_SStructSplit self,
  /* out */ double* norm)
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
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Solver__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connect(url, _ex);
}
char * impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj) {
  return bHYPRE_Solver__getURL(obj);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connect(url, _ex);
}
char * impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) {
  return bHYPRE_Operator__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_SStructSplit_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct bHYPRE_SStructSplit__object* 
  impl_bHYPRE_SStructSplit_fconnect_bHYPRE_SStructSplit(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_SStructSplit__connect(url, _ex);
}
char * impl_bHYPRE_SStructSplit_fgetURL_bHYPRE_SStructSplit(struct 
  bHYPRE_SStructSplit__object* obj) {
  return bHYPRE_SStructSplit__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructSplit_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_SStructSplit_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
