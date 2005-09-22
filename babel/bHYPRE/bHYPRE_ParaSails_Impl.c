/*
 * File:          bHYPRE_ParaSails_Impl.c
 * Symbol:        bHYPRE.ParaSails-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for bHYPRE.ParaSails
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.ParaSails" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * ParaSails requires an IJParCSR matrix
 * 
 * 
 */

#include "bHYPRE_ParaSails_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParaSails__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._load) */
  /* Insert-Code-Here {bHYPRE.ParaSails._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParaSails__ctor(
  /* in */ bHYPRE_ParaSails self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* Note: user calls of __create() are DEPRECATED, _Create also calls this function */

   struct bHYPRE_ParaSails__data * data;
   data = hypre_CTAlloc( struct bHYPRE_ParaSails__data, 1 );
   data -> comm = (MPI_Comm) NULL;
   data -> solver = (HYPRE_Solver) NULL;
   /* set any other data components here */
   bHYPRE_ParaSails__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ParaSails__dtor(
  /* in */ bHYPRE_ParaSails self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   bHYPRE_IJParCSRMatrix_deleteRef( data->matrix );
   ierr += HYPRE_ParaSailsDestroy( data->solver );
   hypre_assert( ierr== 0 );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_ParaSails
impl_bHYPRE_ParaSails_Create(
  /* in */ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.Create) */
  /* Insert-Code-Here {bHYPRE.ParaSails.Create} (Create method) */

   int ierr = 0;
   HYPRE_Solver dummy;
   HYPRE_Solver * Hsolver = &dummy;
   bHYPRE_ParaSails solver = bHYPRE_ParaSails__create();
   struct bHYPRE_ParaSails__data * data = bHYPRE_ParaSails__get_data( solver );

   data -> comm = (MPI_Comm) mpi_comm;
   ierr += HYPRE_ParCSRParaSailsCreate( (data->comm), Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   return solver;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.Create) */
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetCommunicator(
  /* in */ bHYPRE_ParaSails self,
  /* in */ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED  Use _Create */

   int ierr = 0;
   HYPRE_Solver dummy;
   HYPRE_Solver * solver = &dummy;

   struct bHYPRE_ParaSails__data * data = bHYPRE_ParaSails__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   ierr += HYPRE_ParCSRParaSailsCreate( (data->comm), solver );
   data -> solver = *solver;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetIntParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Nlevels")==0 )
   {
      ierr += HYPRE_ParaSailsSetNlevels( solver, value );
   }
   else if ( strcmp(name,"Sym")==0 )
   {
      ierr += HYPRE_ParaSailsSetSym( solver, value );
   }
   else if ( strcmp(name,"Reuse")==0 )
   {
      ierr += HYPRE_ParaSailsSetReuse( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_ParaSailsSetLogging( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetDoubleParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Thresh")==0 )
   {
      ierr += HYPRE_ParaSailsSetThresh( solver, value );
   }
   else if (strcmp(name,"Loadbal")==0 )
   {
      ierr += HYPRE_ParaSailsSetLoadbal( solver, value );
   }
   else if (strcmp(name,"Filter")==0 )
   {
      ierr += HYPRE_ParaSailsSetFilter( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetStringParameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in */ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetIntArray1Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetIntArray2Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetDoubleArray1Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetDoubleArray2Parameter(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_GetIntValue(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Nlevels")==0 )
   {
      ierr += HYPRE_ParaSailsGetNlevels( solver, value );
   }
   else if ( strcmp(name,"Sym")==0 )
   {
      ierr += HYPRE_ParaSailsGetSym( solver, value );
   }
   else if ( strcmp(name,"Reuse")==0 )
   {
      ierr += HYPRE_ParaSailsGetReuse( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_ParaSailsGetLogging( solver, value );
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_GetDoubleValue(
  /* in */ bHYPRE_ParaSails self,
  /* in */ const char* name,
  /* out */ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Thresh")==0 )
   {
      ierr += HYPRE_ParaSailsGetThresh( solver, value );
   }
   else if (strcmp(name,"Loadbal")==0 )
   {
      ierr += HYPRE_ParaSailsGetLoadbal( solver, value );
   }
   else if (strcmp(name,"Filter")==0 )
   {
      ierr += HYPRE_ParaSailsGetFilter( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_Setup(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.Setup) */
  /* Insert the implementation of the Setup method here... */

   /* SetCommunicator or Create should have been called by now, to trigger the
      solver's Create call. */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   if ( bHYPRE_Vector_queryInt( x, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_x = bHYPRE_IJParCSRVector__cast( x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)(x) );
   }

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   ierr += HYPRE_ParaSailsSetup( solver, bHYPREP_A, bb, xx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_Apply(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b );
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
      bHYPREP_x = bHYPRE_IJParCSRVector__cast( *x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)(*x) );
   }

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_ParaSailsSolve( solver, bHYPREP_A, bb, xx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetOperator(
  /* in */ bHYPRE_ParaSails self,
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_ParaSails__data * data;
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

   data = bHYPRE_ParaSails__get_data( self );
   data->matrix = Amat;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetTolerance(
  /* in */ bHYPRE_ParaSails self,
  /* in */ double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetMaxIterations(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_ParaSails_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetLogging(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   /* This function should be called before Setup.  Log level changes
    * may require allocation or freeing of arrays, which is presently
    * only done there.  It may be possible to support log_level
    * changes at other times, but there is little need.  */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_ParaSails__data * data;

   data = bHYPRE_ParaSails__get_data( self );
   solver = data->solver;

   ierr += HYPRE_ParaSailsSetLogging( solver, level );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_ParaSails_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_SetPrintLevel(
  /* in */ bHYPRE_ParaSails self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_GetNumIterations(
  /* in */ bHYPRE_ParaSails self,
  /* out */ int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParaSails_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ParaSails_GetRelResidualNorm(
  /* in */ bHYPRE_ParaSails self,
  /* out */ double* norm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParaSails.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   return 1;  /* no such function, this shouldn't be called */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParaSails.GetRelResidualNorm) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Solver__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connect(url, _ex);
}
char * impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj) {
  return bHYPRE_Solver__getURL(obj);
}
struct bHYPRE_ParaSails__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_ParaSails(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_ParaSails__connect(url, _ex);
}
char * impl_bHYPRE_ParaSails_fgetURL_bHYPRE_ParaSails(struct 
  bHYPRE_ParaSails__object* obj) {
  return bHYPRE_ParaSails__getURL(obj);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connect(url, _ex);
}
char * impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) {
  return bHYPRE_Operator__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_ParaSails_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_ParaSails_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_ParaSails_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_ParaSails_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_ParaSails_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_ParaSails_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
