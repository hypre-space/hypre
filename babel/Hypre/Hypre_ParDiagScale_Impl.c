/*
 * File:          Hypre_ParDiagScale_Impl.c
 * Symbol:        Hypre.ParDiagScale-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020904 10:05:22 PDT
 * Generated:     20020904 10:05:30 PDT
 * Description:   Server-side implementation for Hypre.ParDiagScale
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.ParDiagScale" (version 0.1.5)
 */

#include "Hypre_ParDiagScale_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale._includes) */
/* Put additional includes or other arbitrary code here... */
#include "Hypre_ParCSRMatrix.h"
#include "Hypre_ParCSRMatrix_Impl.h"
#include "Hypre_ParCSRVector.h"
#include "Hypre_ParCSRVector_Impl.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include <assert.h>
/* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale__ctor"

void
impl_Hypre_ParDiagScale__ctor(
  Hypre_ParDiagScale self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale._ctor) */
  /* Insert the implementation of the constructor method here... */
   struct Hypre_ParDiagScale__data * data;
   data = hypre_CTAlloc( struct Hypre_ParDiagScale__data, 1 );
   data -> comm = NULL;
   data -> matrix = NULL;
   Hypre_ParDiagScale__set_data( self, data );
   /* hypre diagonal scaling requires no constructor or setup. cf parcsr/HYPRE_parcsr_pcg.c,
      function HYPRE_ParCSRDiagScale */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale__dtor"

void
impl_Hypre_ParDiagScale__dtor(
  Hypre_ParDiagScale self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale._dtor) */
  /* Insert the implementation of the destructor method here... */
   int ierr = 0;
   struct Hypre_ParDiagScale__data * data;
   data = Hypre_ParDiagScale__get_data( self );

   Hypre_Operator_deleteReference( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale._dtor) */
}

/*
 * Method:  Apply
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_Apply"

int32_t
impl_Hypre_ParDiagScale_Apply(
  Hypre_ParDiagScale self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.Apply) */
  /* Insert the implementation of the Apply method here... */
   int ierr = 0;
   MPI_Comm * comm;
   HYPRE_Solver * solver;
   struct Hypre_ParDiagScale__data * data;
   Hypre_Operator mat;
   HYPRE_Matrix HYPRE_A;
   Hypre_ParCSRMatrix HypreP_A;
   HYPRE_ParCSRMatrix AA;
   HYPRE_IJMatrix ij_A;
   HYPRE_Vector HYPRE_y, HYPRE_x;
   Hypre_ParCSRVector HypreP_x, HypreP_y;
   HYPRE_ParVector xx, yy;
   HYPRE_IJVector ij_x, ij_y;
   struct Hypre_ParCSRMatrix__data * dataA;
   struct Hypre_ParCSRVector__data * datax, * datay;
   void * objectA, * objectx, * objecty;

   data = Hypre_ParDiagScale__get_data( self );
   comm = data->comm;
   assert( comm != NULL ); /* SetCommunicator should have been called earlier */
   mat = data->matrix;
   assert( mat != NULL ); /* SetOperator should have been called earlier */

   HypreP_x = Hypre_Vector__cast2
      ( Hypre_Vector_queryInterface( x, "Hypre.ParCSRVector"),
        "Hypre.ParCSRVector" );
   datax = Hypre_ParCSRVector__get_data( HypreP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   HYPRE_x = (HYPRE_Vector) xx;

   HypreP_y = Hypre_Vector__cast2
      ( Hypre_Vector_queryInterface( *y, "Hypre.ParCSRVector"),
        "Hypre.ParCSRVector" );
   datay = Hypre_ParCSRVector__get_data( HypreP_y );
   ij_y = datay -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_y, &objecty );
   yy = (HYPRE_ParVector) objecty;
   HYPRE_x = (HYPRE_Vector) yy;

   HypreP_A = Hypre_Operator__cast2
      ( Hypre_Operator_queryInterface( mat, "Hypre.ParCSRVector"),
        "Hypre.ParCSRVector" );
   dataA = Hypre_ParCSRMatrix__get_data( HypreP_A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   AA = (HYPRE_ParCSRMatrix) objectA;
   HYPRE_A = (HYPRE_Matrix) AA;

   /* does x = y/diagA as approximation to solving Ax=y for x ... */
   ierr += HYPRE_ParCSRDiagScale( *solver, AA, yy, xx );

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.Apply) */
}

/*
 * Method:  GetDoubleValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_GetDoubleValue"

int32_t
impl_Hypre_ParDiagScale_GetDoubleValue(
  Hypre_ParDiagScale self,
  const char* name,
  double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.GetDoubleValue) */
}

/*
 * Method:  GetIntValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_GetIntValue"

int32_t
impl_Hypre_ParDiagScale_GetIntValue(
  Hypre_ParDiagScale self,
  const char* name,
  int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.GetIntValue) */
}

/*
 * Method:  GetResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_GetResidual"

int32_t
impl_Hypre_ParDiagScale_GetResidual(
  Hypre_ParDiagScale self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.GetResidual) */
  /* Insert the implementation of the GetResidual method here... */
   /* >>>>>>>>>>>> TO DO <<<<<<<<<<<<<<<<< 
      but shouldn't be needed.  And it requires setting a log level flag,
      allocating an array, etc. Much better for the user to do his own
      computation. */
   printf("***** Diagonal Scaling does not support GetResidual *****\n" );
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.GetResidual) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetCommunicator"

int32_t
impl_Hypre_ParDiagScale_SetCommunicator(
  Hypre_ParDiagScale self,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   int ierr = 0;
   struct Hypre_ParDiagScale__data * data;
   data = Hypre_ParDiagScale__get_data( self );
   data -> comm = (MPI_Comm *) comm;
   Hypre_ParDiagScale__set_data( self, data );

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetCommunicator) */
}

/*
 * Method:  SetDoubleArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetDoubleArrayParameter"

int32_t
impl_Hypre_ParDiagScale_SetDoubleArrayParameter(
  Hypre_ParDiagScale self,
  const char* name,
  struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
   /* hypre diagonal scaling has no parameters */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetDoubleArrayParameter) */
}

/*
 * Method:  SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetDoubleParameter"

int32_t
impl_Hypre_ParDiagScale_SetDoubleParameter(
  Hypre_ParDiagScale self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   /* hypre diagonal scaling has no parameters */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetDoubleParameter) */
}

/*
 * Method:  SetIntArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetIntArrayParameter"

int32_t
impl_Hypre_ParDiagScale_SetIntArrayParameter(
  Hypre_ParDiagScale self,
  const char* name,
  struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
   /* hypre diagonal scaling has no parameters */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetIntArrayParameter) */
}

/*
 * Method:  SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetIntParameter"

int32_t
impl_Hypre_ParDiagScale_SetIntParameter(
  Hypre_ParDiagScale self,
  const char* name,
  int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   /* hypre diagonal scaling has no parameters */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetIntParameter) */
}

/*
 * Method:  SetLogging
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetLogging"

int32_t
impl_Hypre_ParDiagScale_SetLogging(
  Hypre_ParDiagScale self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
   /* hypre diagonal scaling does no logging */
   return 0;
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetLogging) */
}

/*
 * Method:  SetOperator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetOperator"

int32_t
impl_Hypre_ParDiagScale_SetOperator(
  Hypre_ParDiagScale self,
  Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   int ierr = 0;
   struct Hypre_ParDiagScale__data * data;

   data = Hypre_ParDiagScale__get_data( self );
   data->matrix = A;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetOperator) */
}

/*
 * Method:  SetPrintLevel
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetPrintLevel"

int32_t
impl_Hypre_ParDiagScale_SetPrintLevel(
  Hypre_ParDiagScale self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
   /* hypre diagonal scaling does no printing */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetPrintLevel) */
}

/*
 * Method:  SetStringParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_SetStringParameter"

int32_t
impl_Hypre_ParDiagScale_SetStringParameter(
  Hypre_ParDiagScale self,
  const char* name,
  const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   /* hypre diagonal scaling has no parameters */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.SetStringParameter) */
}

/*
 * Method:  Setup
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParDiagScale_Setup"

int32_t
impl_Hypre_ParDiagScale_Setup(
  Hypre_ParDiagScale self,
  Hypre_Vector x,
  Hypre_Vector y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParDiagScale.Setup) */
  /* Insert the implementation of the Setup method here... */
   /* hypre diagonal scaling has no setup */
  /* DO-NOT-DELETE splicer.end(Hypre.ParDiagScale.Setup) */
}
