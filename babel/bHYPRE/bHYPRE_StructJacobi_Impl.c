/*
 * File:          bHYPRE_StructJacobi_Impl.c
 * Symbol:        bHYPRE.StructJacobi-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.StructJacobi
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
 * Symbol "bHYPRE.StructJacobi" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects
 * using the {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 * The StructJacobi solver requires a Struct matrix.
 * 
 * 
 */

#include "bHYPRE_StructJacobi_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi._includes) */
/* Insert-Code-Here {bHYPRE.StructJacobi._includes} (includes and arbitrary code) */
#include <assert.h>
#include "bHYPRE_StructMatrix.h"
#include "bHYPRE_StructMatrix_Impl.h"
#include "bHYPRE_StructVector.h"
#include "bHYPRE_StructVector_Impl.h"
#include "HYPRE_struct_ls.h"
#include "struct_ls.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructJacobi__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi._load) */
  /* Insert-Code-Here {bHYPRE.StructJacobi._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructJacobi__ctor(
  /* in */ bHYPRE_StructJacobi self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi._ctor) */
  /* Insert-Code-Here {bHYPRE.StructJacobi._ctor} (constructor method) */

   struct bHYPRE_StructJacobi__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructJacobi__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> solver = (HYPRE_StructSolver) NULL;
   data -> matrix = (bHYPRE_StructMatrix) NULL;
   bHYPRE_StructJacobi__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructJacobi__dtor(
  /* in */ bHYPRE_StructJacobi self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi._dtor) */
  /* Insert-Code-Here {bHYPRE.StructJacobi._dtor} (destructor method) */

   int ierr = 0;
   struct bHYPRE_StructJacobi__data * data;
   data = bHYPRE_StructJacobi__get_data( self );
   ierr += HYPRE_StructJacobiDestroy( data->solver );
   bHYPRE_StructMatrix_deleteRef( data->matrix );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructJacobi
impl_bHYPRE_StructJacobi_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructMatrix A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.Create) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.Create} (Create method) */

   int ierr = 0;
   bHYPRE_StructJacobi solver = bHYPRE_StructJacobi__create();
   struct bHYPRE_StructJacobi__data * data = bHYPRE_StructJacobi__get_data( solver );
   HYPRE_StructSolver dummy;
   HYPRE_StructSolver * Hsolver = &dummy;

   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   ierr += HYPRE_StructJacobiCreate( (data->comm), Hsolver );
   hypre_assert( ierr==0 );
   data -> solver = *Hsolver;

   data->matrix = A;
   bHYPRE_StructMatrix_addRef( data->matrix );

   return solver;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.Create) */
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetCommunicator(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetCommunicator) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetCommunicator} (SetCommunicator method) */
   return 1; /* deprecated, will not be implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetIntParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetIntParameter) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetIntParameter} (SetIntParameter method) */


   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;

   data = bHYPRE_StructJacobi__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      ierr += bHYPRE_StructJacobi_SetMaxIterations( self, value );
   }
   else if ( strcmp(name,"NonZeroGuess")==0 || strcmp(name,"nonzero guess")==0 )
   {
      if ( value==0 )
      {
         ierr += HYPRE_StructJacobiSetZeroGuess( solver );
      }
      else if ( value==1 )
      {
         ierr += HYPRE_StructJacobiSetNonZeroGuess( solver );
      }
      else
      {
         ierr += HYPRE_StructJacobiSetNonZeroGuess( solver );
         ++ierr;
      }
   }
   else if ( strcmp(name,"ZeroGuess")==0 || strcmp(name,"zero guess")==0 )
   {
      if ( value==0 )
      {
         ierr += HYPRE_StructJacobiSetNonZeroGuess( solver );
      }
      else if ( value==1 )
      {
         ierr += HYPRE_StructJacobiSetZeroGuess( solver );
      }
      else
      {
         ierr += HYPRE_StructJacobiSetZeroGuess( solver );
         ++ierr;
      }
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      /* legitimate parameter name, but Jacobi solver ignores it. */
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      /* legitimate parameter name, but Jacobi solver ignores it. */
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetDoubleParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetDoubleParameter) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetDoubleParameter} (SetDoubleParameter method) */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;

   data = bHYPRE_StructJacobi__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 )
   {
      ierr += bHYPRE_StructJacobi_SetTolerance( self, value );      
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetStringParameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in */ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetStringParameter) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetStringParameter} (SetStringParameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetIntArray1Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetIntArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetIntArray1Parameter} (SetIntArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetIntArray2Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetIntArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetIntArray2Parameter} (SetIntArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetDoubleArray1Parameter) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetDoubleArray1Parameter} (SetDoubleArray1Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetDoubleArray2Parameter) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetDoubleArray2Parameter} (SetDoubleArray2Parameter method) */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_GetIntValue(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.GetIntValue) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.GetIntValue} (GetIntValue method) */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;

   data = bHYPRE_StructJacobi__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"NumIterations")==0 )
   {
      ierr = HYPRE_StructJacobiGetNumIterations( solver, value );
   }
   else if ( strcmp(name,"MaxIter")==0 || strcmp(name,"MaxIterations")==0 )
   {
      ierr += HYPRE_StructJacobiGetMaxIter( solver, value );
   }
   else if ( strcmp(name,"NonZeroGuess")==0 || strcmp(name,"nonzero guess")==0 )
   {
      ierr += HYPRE_StructJacobiGetZeroGuess( solver, value );
      if ( *value==0 )
         *value = 1;
      else if ( *value==1 )
         *value = 0;
      else
         ++ierr;
   }
   else if ( strcmp(name,"ZeroGuess")==0 || strcmp(name,"zero guess")==0 )
   {
      ierr += HYPRE_StructJacobiGetZeroGuess( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      /* legitimate parameter name, but Jacobi solver ignores it. */
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      /* legitimate parameter name, but Jacobi solver ignores it. */
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_GetDoubleValue(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ const char* name,
  /* out */ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.GetDoubleValue) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.GetDoubleValue} (GetDoubleValue method) */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;

   data = bHYPRE_StructJacobi__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 )
   {
      ierr += bHYPRE_StructJacobi_GetRelResidualNorm( self, value );
   }
   else if ( strcmp(name,"Tol")==0 || strcmp(name,"Tolerance")==0 )
   {
      ierr += HYPRE_StructJacobiGetTol( solver, value );      
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_Setup(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.Setup) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.Setup} (Setup method) */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;
   struct bHYPRE_StructMatrix__data * dataA;
   struct bHYPRE_StructVector__data * datab, * datax;
   bHYPRE_StructMatrix A;
   HYPRE_StructMatrix HA;
   bHYPRE_StructVector bHYPREP_b, bHYPREP_x;
   HYPRE_StructVector Hb, Hx;

   data = bHYPRE_StructJacobi__get_data( self );
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
      hypre_assert( "Unrecognized vector type."==(char *)x );
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
      hypre_assert( "Unrecognized vector type."==(char *)(x) );
   }
   datax = bHYPRE_StructVector__get_data( bHYPREP_x );
   bHYPRE_StructVector_deleteRef( bHYPREP_x );
   Hx = datax -> vec;

   ierr += HYPRE_StructJacobiSetup( solver, HA, Hb, Hx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_Apply(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.Apply) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.Apply} (Apply method) */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;
   struct bHYPRE_StructMatrix__data * dataA;
   struct bHYPRE_StructVector__data * datab, * datax;
   bHYPRE_StructMatrix A;
   HYPRE_StructMatrix HA;
   bHYPRE_StructVector bHb, bHx, bHr;
   bHYPRE_Vector r;
   HYPRE_StructVector Hb, Hx;
   double rnorm2, bnorm2;

   data = bHYPRE_StructJacobi__get_data( self );
   solver = data->solver;
   A = data->matrix;
   dataA = bHYPRE_StructMatrix__get_data( A );
   HA = dataA -> matrix;

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.StructVector" ) )
   {
      bHb = bHYPRE_StructVector__cast( b );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }
   datab = bHYPRE_StructVector__get_data( bHb );
   bHYPRE_StructVector_deleteRef( bHb );
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
   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.StructVector" ) )
   {
      bHx = bHYPRE_StructVector__cast( *x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)(*x) );
   }
   datax = bHYPRE_StructVector__get_data( bHx );
   bHYPRE_StructVector_deleteRef( bHx );
   Hx = datax -> vec;

   ierr += HYPRE_StructJacobiSolve( solver, HA, Hb, Hx ); /* solve Ax=b for x */

   /* Compute the relative residual norm, as the HYPRE solver doesn't do it but
      our interface requires us to make one available... */
   ierr += bHYPRE_StructVector_Clone( bHb, &r );
   if ( bHYPRE_Vector_queryInt( r, "bHYPRE.StructVector" ) )
   {
      bHr = bHYPRE_StructVector__cast( r );
      bHYPRE_Vector_deleteRef( r ); /* extra ref from queryInt */
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }
   ierr += bHYPRE_StructMatrix_Apply( A, *x, &r );     /* r = Ax */
   ierr += bHYPRE_StructVector_Axpy( bHr, -1.0, b );   /* r = r - b = Ax - b */
   ierr += bHYPRE_StructVector_Dot( bHr, r, &rnorm2 ); /* rnorm2 = <r,r> */
   ierr += bHYPRE_StructVector_Dot( bHb, b, &bnorm2 ); /* bnorm2 = <b,b> */
   if ( bnorm2 == 0 ) bnorm2 = 1.0; /* there are plenty of other possible overflow
                                       conditions, which I'll deal with if needed. */
   data -> rel_resid_norm = sqrt( rnorm2 / bnorm2 );
   bHYPRE_StructVector_deleteRef( bHr );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.Apply) */
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_ApplyAdjoint(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.ApplyAdjoint) */
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetOperator(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetOperator) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetOperator} (SetOperator method) */

   int ierr = 0;
   struct bHYPRE_StructJacobi__data * data;
   bHYPRE_StructMatrix Amat;

   if ( bHYPRE_Operator_queryInt( A, "bHYPRE.StructMatrix" ) )
   {
      Amat = bHYPRE_StructMatrix__cast( A );
      bHYPRE_StructMatrix_deleteRef( Amat ); /* extra ref from queryInt */
   }
   else
   {
      hypre_assert( "Unrecognized operator type."==(char *)A );
   }

   data = bHYPRE_StructJacobi__get_data( self );
   data->matrix = Amat;
   bHYPRE_StructMatrix_addRef( data->matrix );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetTolerance(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetTolerance) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetTolerance} (SetTolerance method) */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;

   data = bHYPRE_StructJacobi__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructJacobiSetTol( solver, tolerance );
   /* ... I believe this is ignored in the solver, it just does max_iter iterations */

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetMaxIterations(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetMaxIterations) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetMaxIterations} (SetMaxIterations method) */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;

   data = bHYPRE_StructJacobi__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructJacobiSetMaxIter( solver, max_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetLogging(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetLogging) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetLogging} (SetLogging method) */

   return 0;  /* The Jacobi solver has no logging parameter */

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_StructJacobi_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_SetPrintLevel(
  /* in */ bHYPRE_StructJacobi self,
  /* in */ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.SetPrintLevel) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.SetPrintLevel} (SetPrintLevel method) */

   return 0;  /* The Jacobi solver has no print level parameter */

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_GetNumIterations(
  /* in */ bHYPRE_StructJacobi self,
  /* out */ int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.GetNumIterations) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.GetNumIterations} (GetNumIterations method) */

   int ierr = 0;
   HYPRE_StructSolver solver;
   struct bHYPRE_StructJacobi__data * data;

   data = bHYPRE_StructJacobi__get_data( self );
   solver = data->solver;

   ierr = HYPRE_StructJacobiGetNumIterations( solver, num_iterations );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructJacobi_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructJacobi_GetRelResidualNorm(
  /* in */ bHYPRE_StructJacobi self,
  /* out */ double* norm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructJacobi.GetRelResidualNorm) */
  /* Insert-Code-Here {bHYPRE.StructJacobi.GetRelResidualNorm} (GetRelResidualNorm method) */

   int ierr = 0;
   /* HYPRE_StructSolver solver;*/
   struct bHYPRE_StructJacobi__data * data;

   data = bHYPRE_StructJacobi__get_data( self );
   /* solver = data->solver;*/

   /* ierr = HYPRE_StructJacobiGetFinalRelativeResidualNorm( solver, norm ); */
   /* ... This function exists but hasn't been implemented. In fact the solver
      doesn't have any kind of error test implemented. */
   *norm = data -> rel_resid_norm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructJacobi.GetRelResidualNorm) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Solver__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj) {
  return bHYPRE_Solver__getURL(obj);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_StructMatrix(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_StructMatrix__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_StructMatrix(struct 
  bHYPRE_StructMatrix__object* obj) {
  return bHYPRE_StructMatrix__getURL(obj);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) {
  return bHYPRE_Operator__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct bHYPRE_StructJacobi__object* 
  impl_bHYPRE_StructJacobi_fconnect_bHYPRE_StructJacobi(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_StructJacobi__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_bHYPRE_StructJacobi(struct 
  bHYPRE_StructJacobi__object* obj) {
  return bHYPRE_StructJacobi__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructJacobi_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_StructJacobi_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
