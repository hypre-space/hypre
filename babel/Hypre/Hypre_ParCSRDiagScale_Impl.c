/*
 * File:          Hypre_ParCSRDiagScale_Impl.c
 * Symbol:        Hypre.ParCSRDiagScale-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1152
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.ParCSRDiagScale" (version 0.1.7)
 * 
 * Diagonal scaling preconditioner for ParCSR matrix class.
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Changed name from 'ParDiagScale' (x)
 * 
 */

#include "Hypre_ParCSRDiagScale_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "Hypre_IJParCSRMatrix.h"
#include "Hypre_IJParCSRMatrix_Impl.h"
#include "Hypre_IJParCSRVector.h"
#include "Hypre_IJParCSRVector_Impl.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
/* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale__ctor"

void
impl_Hypre_ParCSRDiagScale__ctor(
  Hypre_ParCSRDiagScale self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct Hypre_ParCSRDiagScale__data * data;
   data = hypre_CTAlloc( struct Hypre_ParCSRDiagScale__data, 1 );
   data -> comm = NULL;
   data -> matrix = NULL;
   Hypre_ParCSRDiagScale__set_data( self, data );
   /* hypre diagonal scaling requires no constructor or setup. cf
    * parcsr/HYPRE_parcsr_pcg.c, function HYPRE_ParCSRDiagScale */

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale__dtor"

void
impl_Hypre_ParCSRDiagScale__dtor(
  Hypre_ParCSRDiagScale self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale._dtor) */
  /* Insert the implementation of the destructor method here... */

   struct Hypre_ParCSRDiagScale__data * data;
   data = Hypre_ParCSRDiagScale__get_data( self );

   Hypre_Operator_deleteRef( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetCommunicator"

int32_t
impl_Hypre_ParCSRDiagScale_SetCommunicator(
  Hypre_ParCSRDiagScale self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct Hypre_ParCSRDiagScale__data * data;
   data = Hypre_ParCSRDiagScale__get_data( self );
   data -> comm = (MPI_Comm *) mpi_comm;
   Hypre_ParCSRDiagScale__set_data( self, data );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetIntParameter"

int32_t
impl_Hypre_ParCSRDiagScale_SetIntParameter(
  Hypre_ParCSRDiagScale self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetDoubleParameter"

int32_t
impl_Hypre_ParCSRDiagScale_SetDoubleParameter(
  Hypre_ParCSRDiagScale self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetStringParameter"

int32_t
impl_Hypre_ParCSRDiagScale_SetStringParameter(
  Hypre_ParCSRDiagScale self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetStringParameter) */
}

/*
 * Set the int array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetIntArrayParameter"

int32_t
impl_Hypre_ParCSRDiagScale_SetIntArrayParameter(
  Hypre_ParCSRDiagScale self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetIntArrayParameter) */
}

/*
 * Set the double array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetDoubleArrayParameter"

int32_t
impl_Hypre_ParCSRDiagScale_SetDoubleArrayParameter(
  Hypre_ParCSRDiagScale self, const char* name,
    struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetDoubleArrayParameter) 
    */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetDoubleArrayParameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_GetIntValue"

int32_t
impl_Hypre_ParCSRDiagScale_GetIntValue(
  Hypre_ParCSRDiagScale self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_GetDoubleValue"

int32_t
impl_Hypre_ParCSRDiagScale_GetDoubleValue(
  Hypre_ParCSRDiagScale self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_Setup"

int32_t
impl_Hypre_ParCSRDiagScale_Setup(
  Hypre_ParCSRDiagScale self, Hypre_Vector b, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.Setup) */
  /* Insert the implementation of the Setup method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_Apply"

int32_t
impl_Hypre_ParCSRDiagScale_Apply(
  Hypre_ParCSRDiagScale self, Hypre_Vector b, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   MPI_Comm * comm;
   HYPRE_Solver dummy;
   HYPRE_Solver * solver = &dummy;
   struct Hypre_ParCSRDiagScale__data * data;
   Hypre_Operator mat;
   /* not used HYPRE_Matrix HYPRE_A;*/
   Hypre_IJParCSRMatrix HypreP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   /* not used HYPRE_Vector HYPRE_x, HYPRE_b;*/
   Hypre_IJParCSRVector HypreP_b, HypreP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct Hypre_IJParCSRMatrix__data * dataA;
   struct Hypre_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = Hypre_ParCSRDiagScale__get_data( self );
   comm = data->comm;
   /* SetCommunicator should have been called earlier */
   assert( comm != NULL );
   mat = data->matrix;
   /* SetOperator should have been called earlier */
   assert( mat != NULL );

   if ( *x==NULL )
   {  /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or
       * assert(x-has-the-right-size) */
      Hypre_Vector_Clone( b, x );
      Hypre_Vector_Clear( *x );
   }

   HypreP_b = Hypre_IJParCSRVector__cast
      ( Hypre_Vector_queryInt( b, "Hypre.IJParCSRVector") );
   datab = Hypre_IJParCSRVector__get_data( HypreP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;
   /* not used HYPRE_b = (HYPRE_Vector) bb;*/

   HypreP_x = Hypre_IJParCSRVector__cast
      ( Hypre_Vector_queryInt( *x, "Hypre.IJParCSRVector") );
   datax = Hypre_IJParCSRVector__get_data( HypreP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   /* not used HYPRE_b = (HYPRE_Vector) xx;*/

   HypreP_A = Hypre_IJParCSRMatrix__cast
      ( Hypre_Operator_queryInt( mat, "Hypre.IJParCSRMatrix") );
   dataA = Hypre_IJParCSRMatrix__get_data( HypreP_A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   AA = (HYPRE_ParCSRMatrix) objectA;
   /* not used HYPRE_A = (HYPRE_Matrix) AA;*/

   /* does x = y/diagA as approximation to solving Ax=y for x ... */
   ierr += HYPRE_ParCSRDiagScale( *solver, AA, xx, bb );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetOperator"

int32_t
impl_Hypre_ParCSRDiagScale_SetOperator(
  Hypre_ParCSRDiagScale self, Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct Hypre_ParCSRDiagScale__data * data;

   data = Hypre_ParCSRDiagScale__get_data( self );
   data->matrix = A;

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetTolerance"

int32_t
impl_Hypre_ParCSRDiagScale_SetTolerance(
  Hypre_ParCSRDiagScale self, double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetMaxIterations"

int32_t
impl_Hypre_ParCSRDiagScale_SetMaxIterations(
  Hypre_ParCSRDiagScale self, int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetMaxIterations) */
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
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetLogging"

int32_t
impl_Hypre_ParCSRDiagScale_SetLogging(
  Hypre_ParCSRDiagScale self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetLogging) */
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
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_SetPrintLevel"

int32_t
impl_Hypre_ParCSRDiagScale_SetPrintLevel(
  Hypre_ParCSRDiagScale self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_GetNumIterations"

int32_t
impl_Hypre_ParCSRDiagScale_GetNumIterations(
  Hypre_ParCSRDiagScale self, int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   /* diagonal scaling is like 1 step of Jacobi */
   *num_iterations = 1;
   return 0;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRDiagScale_GetRelResidualNorm"

int32_t
impl_Hypre_ParCSRDiagScale_GetRelResidualNorm(
  Hypre_ParCSRDiagScale self, double* norm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRDiagScale.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRDiagScale.GetRelResidualNorm) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
