/*
 * File:          Hypre_ParAMG_Impl.c
 * Symbol:        Hypre.ParAMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:44 PDT
 * Description:   Server-side implementation for Hypre.ParAMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.ParAMG" (version 0.1.5)
 */

#include "Hypre_ParAMG_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.ParAMG._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "Hypre_ParCSRMatrix_Impl.h"
#include "Hypre_ParCSRVector_Impl.h"
/* DO-NOT-DELETE splicer.end(Hypre.ParAMG._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG__ctor"

void
impl_Hypre_ParAMG__ctor(
  Hypre_ParAMG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG._ctor) */
  /* Insert the implementation of the constructor method here... */
   int ierr=0;
   HYPRE_Solver * solver;
   struct Hypre_ParAMG__data * data;
   data = hypre_CTAlloc( struct Hypre_ParAMG__data, 1 );
   data -> comm = NULL;
   ierr += HYPRE_BoomerAMGCreate( solver );
   data -> solver = *solver;
   /* set any other data components here */
   Hypre_ParAMG__set_data( self, data );
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG__dtor"

void
impl_Hypre_ParAMG__dtor(
  Hypre_ParAMG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG._dtor) */
  /* Insert the implementation of the destructor method here... */
   int ierr = 0;
   struct Hypre_ParAMG__data * data;
   data = Hypre_ParAMG__get_data( self );
   ierr += HYPRE_BoomerAMGDestroy( data->solver );
   Hypre_ParCSRMatrix_deleteReference( data->matrix );
   /* delete any nontrivial data components here */
   hypre_TFree( data );
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG._dtor) */
}

/*
 * Method:  Apply
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_Apply"

int32_t
impl_Hypre_ParAMG_Apply(
  Hypre_ParAMG self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.Apply) */
  /* Insert the implementation of the Apply method here... */
   int ierr = 0;
   void * objectA, * objectx, * objecty;
   HYPRE_Solver solver;
   struct Hypre_ParAMG__data * data;
   struct Hypre_ParCSRMatrix__data * dataA;
   struct Hypre_ParCSRVector__data * datax, * datay;
   Hypre_ParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix HypreP_A;
   Hypre_ParCSRVector HypreP_x, HypreP_y;
   HYPRE_ParVector xx, yy;
   HYPRE_IJVector ij_x, ij_y;

   data = Hypre_ParAMG__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = Hypre_ParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   HypreP_A = (HYPRE_ParCSRMatrix) objectA;

   HypreP_x = Hypre_Vector__cast2
      ( Hypre_Vector_queryInterface( x, "Hypre.ParCSRVector"), "Hypre.ParCSRVector" );
   assert( HypreP_x!=NULL );
   datax = Hypre_ParCSRVector__get_data( HypreP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   HypreP_y = Hypre_Vector__cast2
      ( Hypre_Vector_queryInterface( *y, "Hypre.ParCSRVector"), "Hypre.ParCSRVector" );
   assert( HypreP_y!=NULL );
   datay = Hypre_ParCSRVector__get_data( HypreP_y );
   ij_y = datay -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_y, &objecty );
   yy = (HYPRE_ParVector) objecty;

   ierr += HYPRE_BoomerAMGSetup( solver, HypreP_A, xx, yy );
   ierr += HYPRE_BoomerAMGSolve( solver, HypreP_A, xx, yy );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.Apply) */
}

/*
 * Method:  GetResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_GetResidual"

int32_t
impl_Hypre_ParAMG_GetResidual(
  Hypre_ParAMG self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.GetResidual) */
  /* Insert the implementation of the GetResidual method here... */
   /* The existence of this function assumes that a residual vector is available
      from many solvers.  BoomerAMG computes the residual in a temporary vector,
      with no provision for saving it.  See parcsr_ls/par_amg_solve.c, line 194.
      For this function we could recompute the residual.  That requires the
      solution and rhs.  Or we can change par_amg_solve.c  Both options require
      saving data which usually isn't needed, on the chance that GetResidual
      may be called. */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.GetResidual) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetCommunicator"

int32_t
impl_Hypre_ParAMG_SetCommunicator(
  Hypre_ParAMG self,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   int ierr = 0;
   struct Hypre_ParAMG__data * data = Hypre_ParAMG__get_data( self );
   data -> comm = (MPI_Comm *) comm;
   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetCommunicator) */
}

/*
 * Method:  SetDoubleArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetDoubleArrayParameter"

int32_t
impl_Hypre_ParAMG_SetDoubleArrayParameter(
  Hypre_ParAMG self,
  const char* name,
  struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
   int ierr = 0, stride, dim;
   HYPRE_Solver solver;
   struct Hypre_ParAMG__data * data;

   data = Hypre_ParAMG__get_data( self );
   solver = data->solver;

   dim = SIDL_double__array_dimen( value );
/* strides don't seem to be supported in babel 0.6:   stride = SIDL_double__array_stride( value, 0 );
   assert( stride==1 );*/

   if ( name=="RelaxWeight" || name=="Relax Weight" ) {
      assert( dim==1 );
      HYPRE_BoomerAMGSetRelaxWeight( solver, value->d_firstElement );
   }
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetDoubleArrayParameter) */
}

/*
 * Method:  SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetDoubleParameter"

int32_t
impl_Hypre_ParAMG_SetDoubleParameter(
  Hypre_ParAMG self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_ParAMG__data * data;

   data = Hypre_ParAMG__get_data( self );
   solver = data->solver;

   if ( name=="Tolerance" || name=="Tol" ) {
      HYPRE_BoomerAMGSetTol( solver, value );
   }
   else if ( name=="StrongThreshold" || name=="Strong Threshold" ) {
      HYPRE_BoomerAMGSetStrongThreshold( solver, value );
   }
   else if ( name=="TruncFactor" || name=="Trunc Factor" || name=="Truncation Factor" ) {
      HYPRE_BoomerAMGSetTruncFactor( solver, value );
   }
   else if ( name=="SchwarzRlxWeight" || name=="Schwarz Rlx Weight" ||
             name=="SchwarzRelaxWeight" || name=="Schwarz Relax Weight" ||
             name=="SchwarzRelaxationWeight" || name=="Schwarz Relaxation Weight" ){
      HYPRE_BoomerAMGSetSchwarzRlxWeight( solver, value );
   }
   else if ( name=="MaxRowSum" || name=="Max Row Sum" ) {
      HYPRE_BoomerAMGSetMaxRowSum( solver, value );
   }
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetDoubleParameter) */
}

/*
 * Method:  SetIntArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetIntArrayParameter"

int32_t
impl_Hypre_ParAMG_SetIntArrayParameter(
  Hypre_ParAMG self,
  const char* name,
  struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
   int ierr = 0;
   int stride, dim, lb0, ub0, lb1, ub1, i, j;
   int * data1_c;  /* the data in a C 1d array */
   int ** data2_c; /* the data in a C 2d array */
   HYPRE_Solver solver;
   struct Hypre_ParAMG__data * data;

   data = Hypre_ParAMG__get_data( self );
   solver = data->solver;
   data1_c = value->d_firstElement;

   dim = SIDL_int__array_dimen( value );
/* strides don't seem to be supported in babel 0.6:   stride = SIDL_int__array_stride( value, 0 );
   assert( stride==1 );*/

   if ( name=="NumGridSweeps" || name=="Num Grid Sweeps" ||
             name=="NumberGridSweeps" || name=="Number Grid Sweeps" ||
             name=="Number of Grid Sweeps" ) {
      assert( dim==1 );
      HYPRE_BoomerAMGSetNumGridSweeps( solver, data1_c );
   }
   else if ( name=="GridRelaxType" || name=="Grid Relax Type" ) {
      assert( dim==1 );
      HYPRE_BoomerAMGSetGridRelaxType( solver, data1_c );
   }
   else if ( name=="SmoothOption" || name=="Smooth Option" ) {
      assert( dim==1 );
      HYPRE_BoomerAMGSetSmoothOption( solver, data1_c );
   }
   else if ( name=="GridRelaxPoints" || name=="Grid Relax Points" ) {
      assert( dim==2 );
/* strides don't seem to be supported in babel 0.6:      stride = SIDL_int__array_stride( value, 1 );
   assert( stride==1 );*/
      lb0 = SIDL_int__array_lower( value, 0 );
      ub0 = SIDL_int__array_upper( value, 0 );
      lb1 = SIDL_int__array_lower( value, 1 );
      ub1 = SIDL_int__array_upper( value, 1 );
      assert( lb0==0 );
      assert( lb1==0 );
      data2_c = hypre_CTAlloc(int *,ub0+1);
      for ( i=0; i<4; ++i ) {
         data2_c[i] = hypre_CTAlloc(int,ub1+1);
         for ( j=0; j<ub1+1; ++j ) data2_c[i][j] == SIDL_int__array_get2( value, i, j );
      };
      HYPRE_BoomerAMGSetGridRelaxPoints( solver, data2_c );
   }
   else if ( name=="DofFunc" || name=="Dof Func" || name=="DOF Func" ) {
      assert( dim==1 );
      HYPRE_BoomerAMGSetDofFunc( solver, data1_c );
   }
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetIntArrayParameter) */
}

/*
 * Method:  SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetIntParameter"

int32_t
impl_Hypre_ParAMG_SetIntParameter(
  Hypre_ParAMG self,
  const char* name,
  int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_ParAMG__data * data;

   data = Hypre_ParAMG__get_data( self );
   solver = data->solver;

   if ( name=="CoarsenType" || name=="Coarsen Type" ) {
      HYPRE_BoomerAMGSetCoarsenType( solver, value );      
   }
   else if ( name=="MeasureType" || name=="Measure Type" ) {
      HYPRE_BoomerAMGSetMeasureType( solver, value );
   }
   else if ( name=="Logging" ) {  /* >>> need a way to set the filename */
      HYPRE_BoomerAMGSetLogging( solver, value, "driver.out.log");
   }
   else if ( name=="CycleType" || name=="Cycle Type" ) {
      HYPRE_BoomerAMGSetCycleType( solver, value );
   }
   else if ( name=="SmothNumSweep" || name=="Smooth Num Sweep" ) {
      HYPRE_BoomerAMGSetSmoothNumSweep( solver, value );
   }
   else if ( name=="MaxLevels" || name=="Max Levels" ) {
      HYPRE_BoomerAMGSetMaxLevels( solver, value );
   }
   else if ( name=="DebugFlag" || name=="Debug Flag" ) {
      HYPRE_BoomerAMGSetDebugFlag( solver, value );
   }
   else if ( name=="Variant" ) {
      HYPRE_BoomerAMGSetVariant( solver, value );
   }
   else if ( name=="Overlap" ) {
      HYPRE_BoomerAMGSetOverlap( solver, value );
   }
   else if ( name=="DomainType" || name=="Domain Type" ) {
      HYPRE_BoomerAMGSetDomainType( solver, value );
   }
   else if ( name=="NumFunctions" || name=="Num Functions" ||
             name=="NumberFunctions" || name=="Number Functions" ) {
      HYPRE_BoomerAMGSetNumFunctions( solver, value );
   }
   else ierr=1;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetIntParameter) */
}

/*
 * Method:  SetLogging
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetLogging"

int32_t
impl_Hypre_ParAMG_SetLogging(
  Hypre_ParAMG self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetLogging) */
}

/*
 * Method:  SetOperator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetOperator"

int32_t
impl_Hypre_ParAMG_SetOperator(
  Hypre_ParAMG self,
  Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   int ierr = 0;
   void * objectA;
   HYPRE_Solver solver;
   struct Hypre_ParAMG__data * data;
   Hypre_ParCSRMatrix Amat;

   Amat = Hypre_Operator__cast2
      ( Hypre_Operator_queryInterface( A, "Hypre.ParCSRMatrix"), "Hypre.ParCSRMatrix" );
   assert( Amat!=NULL );

   data = Hypre_ParAMG__get_data( self );
   data->matrix = Amat;

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetOperator) */
}

/*
 * Method:  SetPrintLevel
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetPrintLevel"

int32_t
impl_Hypre_ParAMG_SetPrintLevel(
  Hypre_ParAMG self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetPrintLevel) */
}

/*
 * Method:  SetStringParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetStringParameter"

int32_t
impl_Hypre_ParAMG_SetStringParameter(
  Hypre_ParAMG self,
  const char* name,
  const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   int ierr = 0;
   HYPRE_Solver solver;
   struct Hypre_ParAMG__data * data;

   data = Hypre_ParAMG__get_data( self );
   solver = data->solver;

   ierr=1; /* There is no string parameter */

   return ierr;
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetStringParameter) */
}

/*
 * Method:  Setup
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_Setup"

int32_t
impl_Hypre_ParAMG_Setup(
  Hypre_ParAMG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.Setup) */
  /* Insert the implementation of the Setup method here... */

   /* nothing to implement - HYPRE_BoomerAMGSetup requires
      the vectors x,y in Ay=x.  So we do Setup in Apply. */

  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.Setup) */
}
