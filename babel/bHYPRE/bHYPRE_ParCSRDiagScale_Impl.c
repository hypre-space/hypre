/*
 * File:          bHYPRE_ParCSRDiagScale_Impl.c
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:30 PST
 * Description:   Server-side implementation for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1140
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
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
 * 
 */

#include "bHYPRE_ParCSRDiagScale_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "bHYPRE_IJParCSRMatrix.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale__ctor"

void
impl_bHYPRE_ParCSRDiagScale__ctor(
  bHYPRE_ParCSRDiagScale self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_ParCSRDiagScale__data * data;
   data = hypre_CTAlloc( struct bHYPRE_ParCSRDiagScale__data, 1 );
   data -> comm = NULL;
   data -> matrix = NULL;
   bHYPRE_ParCSRDiagScale__set_data( self, data );
   /* hypre diagonal scaling requires no constructor or setup. cf
    * parcsr/HYPRE_parcsr_pcg.c, function HYPRE_ParCSRDiagScale */

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale__dtor"

void
impl_bHYPRE_ParCSRDiagScale__dtor(
  bHYPRE_ParCSRDiagScale self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale._dtor) */
  /* Insert the implementation of the destructor method here... */

   struct bHYPRE_ParCSRDiagScale__data * data;
   data = bHYPRE_ParCSRDiagScale__get_data( self );

   bHYPRE_Operator_deleteRef( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetCommunicator"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetCommunicator(
  bHYPRE_ParCSRDiagScale self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct bHYPRE_ParCSRDiagScale__data * data;
   data = bHYPRE_ParCSRDiagScale__get_data( self );
   data -> comm = (MPI_Comm *) mpi_comm;
   bHYPRE_ParCSRDiagScale__set_data( self, data );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetIntParameter"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntParameter(
  bHYPRE_ParCSRDiagScale self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleParameter(
  bHYPRE_ParCSRDiagScale self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetStringParameter"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetStringParameter(
  bHYPRE_ParCSRDiagScale self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray1Parameter(
  bHYPRE_ParCSRDiagScale self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetIntArray1Parameter) 
    */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetIntArray2Parameter(
  bHYPRE_ParCSRDiagScale self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetIntArray2Parameter) 
    */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter(
  bHYPRE_ParCSRDiagScale self, const char* name,
    struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE 
    splicer.begin(bHYPRE.ParCSRDiagScale.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetDoubleArray1Parameter) 
    */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter(
  bHYPRE_ParCSRDiagScale self, const char* name,
    struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE 
    splicer.begin(bHYPRE.ParCSRDiagScale.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetDoubleArray2Parameter) 
    */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_GetIntValue"

int32_t
impl_bHYPRE_ParCSRDiagScale_GetIntValue(
  bHYPRE_ParCSRDiagScale self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_GetDoubleValue"

int32_t
impl_bHYPRE_ParCSRDiagScale_GetDoubleValue(
  bHYPRE_ParCSRDiagScale self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_Setup"

int32_t
impl_bHYPRE_ParCSRDiagScale_Setup(
  bHYPRE_ParCSRDiagScale self, bHYPRE_Vector b, bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.Setup) */
  /* Insert the implementation of the Setup method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_Apply"

int32_t
impl_bHYPRE_ParCSRDiagScale_Apply(
  bHYPRE_ParCSRDiagScale self, bHYPRE_Vector b, bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   MPI_Comm * comm;
   HYPRE_Solver dummy;
   HYPRE_Solver * solver = &dummy;
   struct bHYPRE_ParCSRDiagScale__data * data;
   bHYPRE_Operator mat;
   /* not used HYPRE_Matrix HYPRE_A;*/
   bHYPRE_IJParCSRMatrix bHYPREP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   /* not used HYPRE_Vector HYPRE_x, HYPRE_b;*/
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   void * objectA, * objectb, * objectx;

   data = bHYPRE_ParCSRDiagScale__get_data( self );
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
      bHYPRE_Vector_Clone( b, x );
      bHYPRE_Vector_Clear( *x );
   }

   bHYPREP_b = bHYPRE_IJParCSRVector__cast
      ( bHYPRE_Vector_queryInt( b, "bHYPRE.IJParCSRVector") );
   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b ); /* extra reference from queryInt */
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;
   /* not used HYPRE_b = (HYPRE_Vector) bb;*/

   bHYPREP_x = bHYPRE_IJParCSRVector__cast
      ( bHYPRE_Vector_queryInt( *x, "bHYPRE.IJParCSRVector") );
   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = datax -> ij_b;
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x ); /* extra reference from queryInt */
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   /* not used HYPRE_b = (HYPRE_Vector) xx;*/

   bHYPREP_A = bHYPRE_IJParCSRMatrix__cast
      ( bHYPRE_Operator_queryInt( mat, "bHYPRE.IJParCSRMatrix") );
   dataA = bHYPRE_IJParCSRMatrix__get_data( bHYPREP_A );
   bHYPRE_IJParCSRMatrix_deleteRef( bHYPREP_A ); /* extra reference from queryInt */
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   AA = (HYPRE_ParCSRMatrix) objectA;
   /* not used HYPRE_A = (HYPRE_Matrix) AA;*/

   /* does x = y/diagA as approximation to solving Ax=y for x ... */
   ierr += HYPRE_ParCSRDiagScale( *solver, AA, xx, bb );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetOperator"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetOperator(
  bHYPRE_ParCSRDiagScale self, bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_ParCSRDiagScale__data * data;

   data = bHYPRE_ParCSRDiagScale__get_data( self );
   data->matrix = A;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetTolerance"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetTolerance(
  bHYPRE_ParCSRDiagScale self, double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetMaxIterations"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetMaxIterations(
  bHYPRE_ParCSRDiagScale self, int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetLogging"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetLogging(
  bHYPRE_ParCSRDiagScale self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_SetPrintLevel"

int32_t
impl_bHYPRE_ParCSRDiagScale_SetPrintLevel(
  bHYPRE_ParCSRDiagScale self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_GetNumIterations"

int32_t
impl_bHYPRE_ParCSRDiagScale_GetNumIterations(
  bHYPRE_ParCSRDiagScale self, int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   /* diagonal scaling is like 1 step of Jacobi */
   *num_iterations = 1;
   return 0;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm"

int32_t
impl_bHYPRE_ParCSRDiagScale_GetRelResidualNorm(
  bHYPRE_ParCSRDiagScale self, double* norm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ParCSRDiagScale.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.ParCSRDiagScale.GetRelResidualNorm) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
