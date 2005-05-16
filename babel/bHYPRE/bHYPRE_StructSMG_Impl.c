/*
 * File:          bHYPRE_StructSMG_Impl.c
 * Symbol:        bHYPRE.StructSMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.StructSMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructSMG" (version 1.0.0)
 */

#include "bHYPRE_StructSMG_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "bHYPRE_StructMatrix.h"
#include "bHYPRE_StructMatrix_Impl.h"
#include "bHYPRE_StructVector.h"
#include "bHYPRE_StructVector_Impl.h"
#include "HYPRE_struct_ls.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG__ctor"

void
impl_bHYPRE_StructSMG__ctor(
  /*in*/ bHYPRE_StructSMG self)
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

   struct bHYPRE_StructSMG__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructSMG__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> solver = (HYPRE_StructSolver) NULL;
   data -> matrix = (bHYPRE_StructMatrix) NULL;
   bHYPRE_StructSMG__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG__dtor"

void
impl_bHYPRE_StructSMG__dtor(
  /*in*/ bHYPRE_StructSMG self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructSMG__data * data;
   data = bHYPRE_StructSMG__get_data( self );
   ierr += HYPRE_StructSMGDestroy( data->solver );
   bHYPRE_StructMatrix_deleteRef( data->matrix );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetCommunicator"

int32_t
impl_bHYPRE_StructSMG_SetCommunicator(
  /*in*/ bHYPRE_StructSMG self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   HYPRE_StructSolver dummy;
   HYPRE_StructSolver * solver = &dummy;
   struct bHYPRE_StructSMG__data * data = bHYPRE_StructSMG__get_data( self );

   data -> comm = (MPI_Comm) mpi_comm;

   ierr += HYPRE_StructSMGCreate( (data->comm), solver );
   data -> solver = *solver;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetIntParameter"

int32_t
impl_bHYPRE_StructSMG_SetIntParameter(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name, /*in*/ int32_t value)
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
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"max iter")==0 )
   {
      ierr += HYPRE_StructSMGSetMaxIter( solver, value );
   }
   else if ( strcmp(name,"RelChange")==0 || strcmp(name,"rel change")==0 )
   {
      ierr += HYPRE_StructSMGSetRelChange( solver, value );
   }
   else if ( strcmp(name,"NonZeroGuess")==0 || strcmp(name,"nonzero guess")==0 )
   {
      ierr += HYPRE_StructSMGSetNonZeroGuess( solver );
   }
   else if ( strcmp(name,"ZeroGuess")==0 || strcmp(name,"zero guess")==0 )
   {
      ierr += HYPRE_StructSMGSetZeroGuess( solver );
   }
   else if ( strcmp(name,"NumPreRelax")==0 || strcmp(name,"num prerelax")==0 )
   {
      ierr += HYPRE_StructSMGSetNumPreRelax( solver, value );
   }
   else if ( strcmp(name,"NumPostRelax")==0 || strcmp(name,"num postrelax")==0 )
   {
      ierr += HYPRE_StructSMGSetNumPostRelax( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 || strcmp(name,"logging")==0 )
   {
      ierr += HYPRE_StructSMGSetLogging( solver, value );
   }
   else if ( strcmp(name,"PrintLevel")==0 || strcmp(name,"print level")==0 )
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

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetDoubleParameter"

int32_t
impl_bHYPRE_StructSMG_SetDoubleParameter(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name, /*in*/ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructSMG__data * data;

   data = bHYPRE_StructSMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Tol")==0 || strcmp(name,"tol")==0 )
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

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetStringParameter"

int32_t
impl_bHYPRE_StructSMG_SetStringParameter(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name,
    /*in*/ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetIntArray1Parameter"

int32_t
impl_bHYPRE_StructSMG_SetIntArray1Parameter(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetIntArray2Parameter"

int32_t
impl_bHYPRE_StructSMG_SetIntArray2Parameter(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_StructSMG_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_StructSMG_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_GetIntValue"

int32_t
impl_bHYPRE_StructSMG_GetIntValue(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name, /*out*/ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_GetDoubleValue"

int32_t
impl_bHYPRE_StructSMG_GetDoubleValue(
  /*in*/ bHYPRE_StructSMG self, /*in*/ const char* name, /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1; /* not implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_Setup"

int32_t
impl_bHYPRE_StructSMG_Setup(
  /*in*/ bHYPRE_StructSMG self, /*in*/ bHYPRE_Vector b, /*in*/ bHYPRE_Vector x)
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

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.StructVector" ) )
   {
      bHYPREP_b = bHYPRE_StructVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }
   datab = bHYPRE_StructVector__get_data( bHYPREP_b );
   bHYPRE_StructVector_deleteRef( bHYPREP_b );
   Hb = datab -> vec;

   if ( bHYPRE_Vector_queryInt( x, "bHYPRE.StructVector" ) )
   {
      bHYPREP_x = bHYPRE_StructVector__cast( x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)(x) );
   }
   datax = bHYPRE_StructVector__get_data( bHYPREP_x );
   bHYPRE_StructVector_deleteRef( bHYPREP_x );
   Hx = datax -> vec;

   ierr += HYPRE_StructSMGSetup( solver, HA, Hb, Hx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_Apply"

int32_t
impl_bHYPRE_StructSMG_Apply(
  /*in*/ bHYPRE_StructSMG self, /*in*/ bHYPRE_Vector b,
    /*inout*/ bHYPRE_Vector* x)
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

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.StructVector" ) )
   {
      bHYPREP_b = bHYPRE_StructVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }
   datab = bHYPRE_StructVector__get_data( bHYPREP_b );
   bHYPRE_StructVector_deleteRef( bHYPREP_b );
   Hb = datab -> vec;

   if ( *x==NULL )
   {
      /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or assert(x
       * has the right size) */
      bHYPRE_Vector_Clone( b, x );
      bHYPRE_Vector_Clear( *x );
   }
   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.StructVector" ) )
   {
      bHYPREP_x = bHYPRE_StructVector__cast( *x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)(*x) );
   }
   datax = bHYPRE_StructVector__get_data( bHYPREP_x );
   bHYPRE_StructVector_deleteRef( bHYPREP_x );
   Hx = datax -> vec;

   ierr += HYPRE_StructSMGSolve( solver, HA, Hb, Hx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetOperator"

int32_t
impl_bHYPRE_StructSMG_SetOperator(
  /*in*/ bHYPRE_StructSMG self, /*in*/ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructSMG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_StructSMG__data * data;
   bHYPRE_StructMatrix Amat;

   if ( bHYPRE_Operator_queryInt( A, "bHYPRE.StructMatrix" ) )
   {
      Amat = bHYPRE_StructMatrix__cast( A );
      bHYPRE_StructMatrix_deleteRef( Amat ); /* extra ref from queryInt */
   }
   else
   {
      assert( "Unrecognized operator type."==(char *)A );
   }

   data = bHYPRE_StructSMG__get_data( self );
   data->matrix = Amat;
   bHYPRE_StructMatrix_addRef( data->matrix );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructSMG.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetTolerance"

int32_t
impl_bHYPRE_StructSMG_SetTolerance(
  /*in*/ bHYPRE_StructSMG self, /*in*/ double tolerance)
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

/*
 * (Optional) Set maximum number of iterations.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetMaxIterations"

int32_t
impl_bHYPRE_StructSMG_SetMaxIterations(
  /*in*/ bHYPRE_StructSMG self, /*in*/ int32_t max_iterations)
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

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetLogging"

int32_t
impl_bHYPRE_StructSMG_SetLogging(
  /*in*/ bHYPRE_StructSMG self, /*in*/ int32_t level)
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

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_SetPrintLevel"

int32_t
impl_bHYPRE_StructSMG_SetPrintLevel(
  /*in*/ bHYPRE_StructSMG self, /*in*/ int32_t level)
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

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_GetNumIterations"

int32_t
impl_bHYPRE_StructSMG_GetNumIterations(
  /*in*/ bHYPRE_StructSMG self, /*out*/ int32_t* num_iterations)
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

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructSMG_GetRelResidualNorm"

int32_t
impl_bHYPRE_StructSMG_GetRelResidualNorm(
  /*in*/ bHYPRE_StructSMG self, /*out*/ double* norm)
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
