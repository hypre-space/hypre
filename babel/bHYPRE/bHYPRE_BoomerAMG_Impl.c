/*
 * File:          bHYPRE_BoomerAMG_Impl.c
 * Symbol:        bHYPRE.BoomerAMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.BoomerAMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.BoomerAMG" (version 1.0.0)
 * 
 * Algebraic multigrid solver, based on classical Ruge-Stueben.
 * 
 * BoomerAMG requires an IJParCSR matrix
 * 
 * The following optional parameters are available and may be set
 * using the appropriate {\tt Parameter} function (as indicated in
 * parentheses):
 * 
 * \begin{description}
 * 
 * \item[MaxLevels] ({\tt Int}) - maximum number of multigrid
 * levels.
 * 
 * \item[StrongThreshold] ({\tt Double}) - AMG strength threshold.
 * 
 * \item[MaxRowSum] ({\tt Double}) -
 * 
 * \item[CoarsenType] ({\tt Int}) - type of parallel coarsening
 * algorithm used.
 * 
 * \item[MeasureType] ({\tt Int}) - type of measure used; local or
 * global.
 * 
 * \item[CycleType] ({\tt Int}) - type of cycle used; a V-cycle
 * (default) or a W-cycle.
 * 
 * \item[NumGridSweeps] ({\tt IntArray 1D}) - number of sweeps for
 * fine and coarse grid, up and down cycle. DEPRECATED:
 * Use NumSweeps or Cycle?NumSweeps instead.
 * 
 * \item[NumSweeps] ({\tt Int}) - number of sweeps for fine grid, up and
 * down cycle.
 * 
 * \item[Cycle1NumSweeps] ({\tt Int}) - number of sweeps for down cycle
 * 
 * \item[Cycle2NumSweeps] ({\tt Int}) - number of sweeps for up cycle
 * 
 * \item[Cycle3NumSweeps] ({\tt Int}) - number of sweeps for coarse grid
 * 
 * \item[GridRelaxType] ({\tt IntArray 1D}) - type of smoother used on
 * fine and coarse grid, up and down cycle. DEPRECATED:
 * Use RelaxType or Cycle?RelaxType instead.
 * 
 * \item[RelaxType] ({\tt Int}) - type of smoother for fine grid, up and
 * down cycle.
 * 
 * \item[Cycle1RelaxType] ({\tt Int}) - type of smoother for down cycle
 * 
 * \item[Cycle2RelaxType] ({\tt Int}) - type of smoother for up cycle
 * 
 * \item[Cycle3RelaxType] ({\tt Int}) - type of smoother for coarse grid
 * 
 * \item[GridRelaxPoints] ({\tt IntArray 2D}) - point ordering used in
 * relaxation.  DEPRECATED.
 * 
 * \item[RelaxWeight] ({\tt DoubleArray 1D}) - relaxation weight for
 * smoothed Jacobi and hybrid SOR.  DEPRECATED:
 * Instead, use the RelaxWt parameter and the SetLevelRelaxWt function.
 * 
 * \item[RelaxWt] ({\tt Int}) - relaxation weight for all levels for
 * smoothed Jacobi and hybrid SOR.
 * 
 * \item[TruncFactor] ({\tt Double}) - truncation factor for
 * interpolation.
 * 
 * \item[JacobiTruncThreshold] ({\tt Double}) - threshold for truncation
 * of Jacobi interpolation.
 * 
 * \item[SmoothType] ({\tt Int}) - more complex smoothers.
 * 
 * \item[SmoothNumLevels] ({\tt Int}) - number of levels for more
 * complex smoothers.
 * 
 * \item[SmoothNumSweeps] ({\tt Int}) - number of sweeps for more
 * complex smoothers.
 * 
 * \item[PrintFileName] ({\tt String}) - name of file printed to in
 * association with {\tt SetPrintLevel}.
 * 
 * \item[NumFunctions] ({\tt Int}) - size of the system of PDEs
 * (when using the systems version).
 * 
 * \item[DOFFunc] ({\tt IntArray 1D}) - mapping that assigns the
 * function to each variable (when using the systems version).
 * 
 * \item[Variant] ({\tt Int}) - variant of Schwarz used.
 * 
 * \item[Overlap] ({\tt Int}) - overlap for Schwarz.
 * 
 * \item[DomainType] ({\tt Int}) - type of domain used for Schwarz.
 * 
 * \item[SchwarzRlxWeight] ({\tt Double}) - the smoothing parameter
 * for additive Schwarz.
 * 
 * \item[Tolerance] ({\tt Double}) - convergence tolerance, if this
 * is used as a solver; ignored if this is used as a preconditioner
 * 
 * \item[DebugFlag] ({\tt Int}) -
 * 
 * \item[InterpType] ({\tt Int}) - Defines which parallel interpolation
 * operator is used. There are the following options for interp\_type: 
 * 
 * \begin{tabular}{|c|l|} \hline
 * 0 &	classical modified interpolation \\
 * 1 &	LS interpolation (for use with GSMG) \\
 * 2 &	classical modified interpolation for hyperbolic PDEs \\
 * 3 &	direct interpolation (with separation of weights) \\
 * 4 &	multipass interpolation \\
 * 5 &	multipass interpolation (with separation of weights) \\
 * 6 &  extended classical modified interpolation \\
 * 7 &  extended (if no common C neighbor) classical modified interpolation \\
 * 8 &	standard interpolation \\
 * 9 &	standard interpolation (with separation of weights) \\
 * 10 &	classical block interpolation (for use with nodal systems version only) \\
 * 11 &	classical block interpolation (for use with nodal systems version only) \\
 * &	with diagonalized diagonal blocks \\
 * 12 &	FF interpolation \\
 * 13 &	FF1 interpolation \\
 * \hline
 * \end{tabular}
 * 
 * The default is 0. 
 * 
 * \item[NumSamples] ({\tt Int}) - Defines the number of sample vectors used
 * in GSMG or LS interpolation.
 * 
 * \item[MaxIterations] ({\tt Int}) - maximum number of iterations
 * 
 * \item[Logging] ({\tt Int}) - Set the {\it logging level}, specifying the
 * degree of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 * \item[PrintLevel] ({\tt Int}) - Set the {\it print level}, specifying the
 * degree of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 * \end{description}
 * 
 * The following function is specific to this class:
 * 
 * \begin{description}
 * 
 * \item[SetLevelRelxWeight] ({\tt Double , \tt Int}) -
 * relaxation weight for one specified level of smoothed Jacobi and hybrid SOR.
 * 
 * \end{description}
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 */

#include "bHYPRE_BoomerAMG_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._includes) */
/* Put additional includes or other arbitrary code here... */



#include "hypre_babel_exception_handler.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BoomerAMG__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._load) */
  /* Insert-Code-Here {bHYPRE.BoomerAMG._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BoomerAMG__ctor(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* Note: user calls of __create() are DEPRECATED, _Create also calls this function */

   int ierr=0;
   HYPRE_Solver dummy;
   /* will really be initialized by Create call */
   HYPRE_Solver * solver = &dummy;
   struct bHYPRE_BoomerAMG__data * data;
   data = hypre_CTAlloc( struct bHYPRE_BoomerAMG__data, 1 );
   data->comm = MPI_COMM_NULL;
   ierr += HYPRE_BoomerAMGCreate( solver );
   data -> solver = *solver;
   /* set any other data components here */
   bHYPRE_BoomerAMG__set_data( self, data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BoomerAMG__ctor2(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._ctor2) */
    /* Insert-Code-Here {bHYPRE.BoomerAMG._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BoomerAMG__dtor(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   bHYPRE_IJParCSRMatrix_deleteRef( data->matrix, _ex );  SIDL_CHECK(*_ex);
   ierr += HYPRE_BoomerAMGDestroy( data->solver );
   hypre_assert( ierr== 0 );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

   return; hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._dtor) */
  }
}

/*
 *  This function is the preferred way to create a BoomerAMG solver. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_BoomerAMG
impl_bHYPRE_BoomerAMG_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_IJParCSRMatrix A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.Create) */
  /* Insert-Code-Here {bHYPRE.BoomerAMG.Create} (Create method) */

   bHYPRE_BoomerAMG solver = bHYPRE_BoomerAMG__create(_ex);  SIDL_CHECK(*_ex);
   struct bHYPRE_BoomerAMG__data * data = bHYPRE_BoomerAMG__get_data( solver );
   struct bHYPRE_MPICommunicator__data * mpi_data =
      bHYPRE_MPICommunicator__get_data(mpi_comm);

   data->comm = mpi_data->mpi_comm;

   data->matrix = A;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex );  SIDL_CHECK(*_ex);

   return solver;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.Create) */
  }
}

/*
 * Method:  SetLevelRelaxWt[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetLevelRelaxWt"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetLevelRelaxWt(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ double relax_wt,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetLevelRelaxWt) */
  /* Insert the implementation of the SetLevelRelaxWt method here... */

   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   return HYPRE_BoomerAMGSetLevelRelaxWt( solver, relax_wt, level );

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetLevelRelaxWt) */
  }
}

/*
 * Method:  InitGridRelaxation[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_InitGridRelaxation"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_InitGridRelaxation(
  /* in */ bHYPRE_BoomerAMG self,
  /* out array<int,column-major> */ struct sidl_int__array** num_grid_sweeps,
  /* out array<int,column-major> */ struct sidl_int__array** grid_relax_type,
  /* out array<int,2,column-major> */ struct sidl_int__array** 
    grid_relax_points,
  /* in */ int32_t coarsen_type,
  /* out array<double,column-major> */ struct sidl_double__array** 
    relax_weights,
  /* in */ int32_t max_levels,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.InitGridRelaxation) */
   /* Insert-Code-Here {bHYPRE.BoomerAMG.InitGridRelaxation} (InitGridRelaxation method) */

   /* This function is for convenience in writing test drivers. */

   int ierr = 0;
   int lower[2], upper[2], stride[2], upper_rw[1];
   int * dummy_ngsp;
   int * dummy_grtp;
   int * dummy_grp1p;
   int ** dummy_grpp = &dummy_grp1p;
   double * dummy_rwp;
   int ** num_grid_sweeps_ptr = &dummy_ngsp;
   int ** grid_relax_type_ptr = &dummy_grtp;
   int *** grid_relax_points_ptr = &dummy_grpp;
   double ** relax_weights_ptr = &dummy_rwp;
   int i, j;

   lower[0] = 0; lower[1] = 0; upper[0] = 4; upper[1] = 4;
   stride[0]=1; stride[1] = 1;
   upper_rw[0] = max_levels;

   ierr += HYPRE_BoomerAMGInitGridRelaxation(
      num_grid_sweeps_ptr, grid_relax_type_ptr, grid_relax_points_ptr,
      coarsen_type, relax_weights_ptr, max_levels  );

   *num_grid_sweeps = sidl_int__array_borrow( *num_grid_sweeps_ptr, 1,
                                              lower, upper, stride );
   *grid_relax_type = sidl_int__array_borrow( *grid_relax_type_ptr, 1
                                              , lower, upper, stride );
   *relax_weights = sidl_double__array_borrow( *relax_weights_ptr, 1,
                                               lower, upper_rw, stride );
   *grid_relax_points = sidl_int__array_createCol( 2, lower, upper );
   for ( i=0; i<4; ++i )
   {
      for ( j=0; j<(*num_grid_sweeps_ptr)[i]; ++j )
      {
         sidl_int__array_set2( *grid_relax_points, i, j,
                               (*grid_relax_points_ptr)[i][j] );
      }
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.InitGridRelaxation) */
  }
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetOperator(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Operator A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_BoomerAMG__data * data;
   bHYPRE_IJParCSRMatrix Amat;

   Amat = (bHYPRE_IJParCSRMatrix) bHYPRE_Operator__cast2( A, "bHYPRE.IJParCSRMatrix", _ex ); SIDL_CHECK(*_ex);
   if ( Amat==NULL ) hypre_assert( "Unrecognized operator type."==(char *)A );

   data = bHYPRE_BoomerAMG__get_data( self );
   data->matrix = Amat;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetOperator) */
  }
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetTolerance(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ double tolerance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_BoomerAMGSetTol( solver, tolerance );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetTolerance) */
  }
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetMaxIterations(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t max_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr = HYPRE_BoomerAMGSetMaxIter( solver, max_iterations );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetMaxIterations) */
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
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetLogging(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */

   /* This function should be called before Setup.  Log level changes
    * may require allocation or freeing of arrays, which is presently
    * only done there.  It may be possible to support log_level
    * changes at other times, but there is little need.  */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr += HYPRE_BoomerAMGSetLogging( solver, level );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetLogging) */
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
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetPrintLevel(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t level,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr += HYPRE_BoomerAMGSetPrintLevel( solver, level );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetPrintLevel) */
  }
}

/*
 * (Optional) Return the number of iterations taken.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_GetNumIterations(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ int32_t* num_iterations,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr += HYPRE_BoomerAMGGetNumIterations( solver, num_iterations );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.GetNumIterations) */
  }
}

/*
 * (Optional) Return the norm of the relative residual.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_GetRelResidualNorm(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ double* norm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   ierr += HYPRE_BoomerAMGGetFinalRelativeResidualNorm( solver, norm );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.GetRelResidualNorm) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetCommunicator(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED  Use Create */

   int ierr = 0;
   struct bHYPRE_BoomerAMG__data * data = bHYPRE_BoomerAMG__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_BoomerAMG_Destroy(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.Destroy) */
    /* Insert-Code-Here {bHYPRE.BoomerAMG.Destroy} (Destroy method) */
     bHYPRE_BoomerAMG_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.Destroy) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetIntParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"CoarsenType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCoarsenType( solver, value );      
   }
   else if ( strcmp(name,"MeasureType")==0 ) 
   {
      ierr += HYPRE_BoomerAMGSetMeasureType( solver, value );
   }
   else if ( strcmp(name,"CycleType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleType( solver, value );
   }
   else if ( strcmp(name,"NumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGSetNumSweeps( solver, value );
   }
   else if ( strcmp(name,"Cycle1NumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleNumSweeps( solver, value, 1 );
   }
   else if ( strcmp(name,"Cycle2NumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleNumSweeps( solver, value, 2 );
   }
   else if ( strcmp(name,"Cycle3NumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleNumSweeps( solver, value, 3 );
   }
   else if ( strcmp(name,"RelaxType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetRelaxType( solver, value );
   }
   else if ( strcmp(name,"Cycle1RelaxType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleRelaxType( solver, value, 1 );
   }
   else if ( strcmp(name,"Cycle2RelaxType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleRelaxType( solver, value, 2 );
   }
   else if ( strcmp(name,"Cycle3RelaxType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleRelaxType( solver, value, 3 );
   }
   else if ( strcmp(name,"RelaxWt")==0 )
   {
      ierr += HYPRE_BoomerAMGSetRelaxWt( solver, value );
   }
   else if ( strcmp(name,"SmoothType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSmoothType( solver, value );
   }
   else if ( strcmp(name,"SmoothNumLevels")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSmoothNumLevels( solver, value );
   }
   else if ( strcmp(name,"SmoothNumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSmoothNumSweeps( solver, value );
   }
   else if ( strcmp(name,"MaxLevels")==0 )
   {
      ierr += HYPRE_BoomerAMGSetMaxLevels( solver, value );
   }
   else if ( strcmp(name,"DebugFlag")==0 )
   {
      ierr += HYPRE_BoomerAMGSetDebugFlag( solver, value );
   }
   else if ( strcmp(name,"InterpType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetInterpType( solver, value );
   }
   else if ( strcmp(name,"NumSamples")==0 )
   {
      ierr += HYPRE_BoomerAMGSetNumSamples( solver, value );
   }
   else if ( strcmp(name,"Variant")==0 )
   {
      ierr += HYPRE_BoomerAMGSetVariant( solver, value );
   }
   else if ( strcmp(name,"Overlap")==0 )
   {
      ierr += HYPRE_BoomerAMGSetOverlap( solver, value );
   }
   else if ( strcmp(name,"DomainType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetDomainType( solver, value );
   }
   else if ( strcmp(name,"NumFunctions")==0 )
   {
      ierr += HYPRE_BoomerAMGSetNumFunctions( solver, value );
   }
   else if ( strcmp(name,"MaxIterations")==0 || strcmp(name,"MaxIter")==0 )
   {
      ierr += HYPRE_BoomerAMGSetMaxIter( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_BoomerAMGSetLogging( solver, value );
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      ierr += HYPRE_BoomerAMGSetPrintLevel( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"StrongThreshold")==0 )
   {
      ierr += HYPRE_BoomerAMGSetStrongThreshold( solver, value );
   }
   else if ( strcmp(name,"TruncFactor")==0 )
   {
      ierr += HYPRE_BoomerAMGSetTruncFactor( solver, value );
   }
   else if ( strcmp(name,"JacobiTruncThreshold")==0 )
   {
      ierr += HYPRE_BoomerAMGSetJacobiTruncThreshold( solver, value );
   }
   else if ( strcmp(name,"SchwarzRlxWeight")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSchwarzRlxWeight( solver, value );
   }
   else if ( strcmp(name,"MaxRowSum")==0 )
   {
      ierr += HYPRE_BoomerAMGSetMaxRowSum( solver, value );
   }
   else if ( strcmp(name,"Tolerance")==0 || strcmp(name,"Tol")==0 )
   {
      ierr += HYPRE_BoomerAMGSetTol( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetStringParameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"PrintFileName")==0 )
   {
      ierr += hypre_BoomerAMGSetPrintFileName( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetIntArray1Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"NumGridSweeps")==0 )
   {
      /* *** DEPRECATED *** Use IntParameter NumSweeps and Cycle?NumSweeps instead. */
      ierr += HYPRE_BoomerAMGSetNumGridSweeps( solver, value );
   }
   else if ( strcmp(name,"GridRelaxType")==0 )
   {
      /* *** DEPRECATED *** Use RelaxType and Cycle?RelaxType instead. */
      ierr += HYPRE_BoomerAMGSetGridRelaxType( solver, value );
   }
   else if ( strcmp(name,"DOFFunc")==0 )
   {
      ierr += HYPRE_BoomerAMGSetDofFunc( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetIntArray2Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   int ierr = 0;
   int dim, lb0, ub0, lb1, ub1, i, j;
   int ** data2_c;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   dim = sidl_int__array_dimen( value );

   if ( strcmp(name,"GridRelaxPoints")==0 )
   {
      /* *** DEPRECATED ***  There is no substitute because Ulrike Yang
         thinks nobody uses this anyway. */
      hypre_assert( dim==2 );
      lb0 = sidl_int__array_lower( value, 0 );
      ub0 = sidl_int__array_upper( value, 0 );
      lb1 = sidl_int__array_lower( value, 1 );
      ub1 = sidl_int__array_upper( value, 1 );
      hypre_assert( lb0==0 );
      hypre_assert( lb1==0 );
      data2_c = hypre_CTAlloc(int *,ub0);
      for ( i=0; i<ub0; ++i )
      {
         data2_c[i] = hypre_CTAlloc(int,ub1);
         for ( j=0; j<ub1; ++j )
         {
            data2_c[i][j] = sidl_int__array_get2( value, i, j );
         }
      }
      ierr += HYPRE_BoomerAMGSetGridRelaxPoints( solver, data2_c );
   }
   else
   {
      ierr=1;
   }

   return ierr;
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleArray1Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"RelaxWeight")==0 )
   {
      /* *** DEPRECATED *** Use the RelaxWt parameter and SetLevelRelaxWt function
         instead. */
      ierr += HYPRE_BoomerAMGSetRelaxWeight( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetDoubleArray2Parameter(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_GetIntValue(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;
   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"NumIterations")==0 )
   {
      ierr += HYPRE_BoomerAMGGetNumIterations( solver, value );
   }
   /* everything following appears in SetIntParameter */
   else if ( strcmp(name,"CoarsenType")==0 )
   {
      ierr += HYPRE_BoomerAMGGetCoarsenType( solver, value );      
   }
   else if ( strcmp(name,"MeasureType")==0 ) 
   {
      ierr += HYPRE_BoomerAMGGetMeasureType( solver, value );
   }
   else if ( strcmp(name,"CycleType")==0 )
   {
      ierr += HYPRE_BoomerAMGGetCycleType( solver, value );
   }
   else if ( strcmp(name,"NumSweeps")==0 )
   {
      /* no Get function.  Here's why: Set...NumSweeps uses one input parameter
         to set an array of parameters.  We can't always return just a single
         parameter because they may not still be all the same. */
      ++ierr;
      return ierr;
   }
   else if ( strcmp(name,"Cycle1NumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGGetCycleNumSweeps( solver, value, 1 );
   }
   else if ( strcmp(name,"Cycle2NumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGGetCycleNumSweeps( solver, value, 2 );
   }
   else if ( strcmp(name,"Cycle3NumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGGetCycleNumSweeps( solver, value, 3 );
   }
   else if ( strcmp(name,"RelaxType")==0 )
   {
      /* no Get function.  Here's why: Set...RelaxType uses one input parameter
         to set an array of parameters.  We can't always return just a single
         parameter because they may not still be all the same. */
      ++ierr;
   }
   else if ( strcmp(name,"Cycle1RelaxType")==0 )
   {
      ierr += HYPRE_BoomerAMGGetCycleRelaxType( solver, value, 1 );
   }
   else if ( strcmp(name,"Cycle2RelaxType")==0 )
   {
      ierr += HYPRE_BoomerAMGGetCycleRelaxType( solver, value, 2 );
   }
   else if ( strcmp(name,"Cycle3RelaxType")==0 )
   {
      ierr += HYPRE_BoomerAMGGetCycleRelaxType( solver, value, 3 );
   }
   else if ( strcmp(name,"RelaxWt")==0 )
   {
      /* no Get function.  Here's why: Set...RelaxWt uses one input parameter
         to set an array of parameters.  We can't always return just a single
         parameter because they may not still be all the same. */
      ++ierr;
   }
   else if ( strcmp(name,"SmoothType")==0 )
   {
      ierr += HYPRE_BoomerAMGGetSmoothType( solver, value );
   }
   else if ( strcmp(name,"SmoothNumLevels")==0 )
   {
      ierr += HYPRE_BoomerAMGGetSmoothNumLevels( solver, value );
   }
   else if ( strcmp(name,"SmoothNumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGGetSmoothNumSweeps( solver, value );
   }
   else if ( strcmp(name,"MaxLevels")==0 )
   {
      ierr += HYPRE_BoomerAMGGetMaxLevels( solver, value );
   }
   else if ( strcmp(name,"DebugFlag")==0 )
   {
      ierr += HYPRE_BoomerAMGGetDebugFlag( solver, value );
   }
   else if ( strcmp(name,"Variant")==0 )
   {
      ierr += HYPRE_BoomerAMGGetVariant( solver, value );
   }
   else if ( strcmp(name,"Overlap")==0 )
   {
      ierr += HYPRE_BoomerAMGGetOverlap( solver, value );
   }
   else if ( strcmp(name,"DomainType")==0 )
   {
      ierr += HYPRE_BoomerAMGGetDomainType( solver, value );
   }
   else if ( strcmp(name,"NumFunctions")==0 )
   {
      ierr += HYPRE_BoomerAMGGetNumFunctions( solver, value );
   }
   else if ( strcmp(name,"MaxIterations")==0 || strcmp(name,"MaxIter")==0 )
   {
      ierr += HYPRE_BoomerAMGGetMaxIter( solver, value );
   }
   else if ( strcmp(name,"Logging")==0 )
   {
      ierr += HYPRE_BoomerAMGGetLogging( solver, value );
   }
   else if ( strcmp(name,"PrintLevel")==0 )
   {
      ierr += HYPRE_BoomerAMGGetPrintLevel( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_GetDoubleValue(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   int ierr = 0;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;

   if ( strcmp(name,"FinalRelativeResidualNorm")==0 ||
        strcmp(name,"Final Relative Residual Norm")==0 ||
        strcmp(name,"RelativeResidualNorm")==0 ||
        strcmp(name,"RelResidualNorm")==0 )
   {
      ierr += HYPRE_BoomerAMGGetFinalRelativeResidualNorm( solver, value );
   }
   else if ( strcmp(name,"StrongThreshold")==0 )
   {
      ierr += HYPRE_BoomerAMGGetStrongThreshold( solver, value );
   }
   else if ( strcmp(name,"JacobiTruncThreshold")==0 )
   {
      ierr += HYPRE_BoomerAMGGetJacobiTruncThreshold( solver, value );
   }
   else if ( strcmp(name,"SchwarzRlxWeight")==0 )
   {
      ierr += HYPRE_BoomerAMGGetSchwarzRlxWeight( solver, value );
   }
   else if ( strcmp(name,"MaxRowSum")==0 )
   {
      ierr += HYPRE_BoomerAMGGetMaxRowSum( solver, value );
   }
   else if ( strcmp(name,"Tolerance")==0 || strcmp(name,"Tol")==0 )
   {
      ierr += HYPRE_BoomerAMGGetTol( solver, value );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_Setup(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;

   bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)x );

   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   bHYPREP_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( x, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)(x) );

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   ierr += HYPRE_BoomerAMGSetup( solver, bHYPREP_A, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_Apply(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.Apply) */
  /* Insert the implementation of the Apply method here... */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;

   bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2(b, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)b );

   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   if ( *x==NULL )
   {
      /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or hypre_assert(x
       * has the right size) */
      bHYPRE_Vector_Clone( b, x, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Vector_Clear( *x, _ex ); SIDL_CHECK(*_ex);
   }
   bHYPREP_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( *x, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)(*x) );

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_BoomerAMGSolve( solver, bHYPREP_A, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_ApplyAdjoint(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.BoomerAMG.ApplyAdjoint} (ApplyAdjoint method) */

   int ierr = 0;
   void * objectA, * objectb, * objectx;
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;
   struct bHYPRE_IJParCSRMatrix__data * dataA;
   struct bHYPRE_IJParCSRVector__data * datab, * datax;
   bHYPRE_IJParCSRMatrix A;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_ParVector bb, xx;
   HYPRE_IJVector ij_b, ij_x;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;
   A = data->matrix;

   dataA = bHYPRE_IJParCSRMatrix__get_data( A );
   ij_A = dataA -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &objectA );
   bHYPREP_A = (HYPRE_ParCSRMatrix) objectA;

   bHYPREP_b = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( b, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)x );

   datab = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = datab -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &objectb );
   bb = (HYPRE_ParVector) objectb;

   if ( *x==NULL )
   {
      /* If vector not supplied, make one...*/
      /* There's no good way to check the size of x.  It would be good
       * to do something similar if x had zero length.  Or hypre_assert(x
       * has the right size) */
      bHYPRE_Vector_Clone( b, x, _ex ); SIDL_CHECK(*_ex);
      bHYPRE_Vector_Clear( *x, _ex ); SIDL_CHECK(*_ex);
   }
   bHYPREP_x = (bHYPRE_IJParCSRVector) bHYPRE_Vector__cast2( *x, "bHYPRE.IJParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)(*x) );

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_BoomerAMGSolveT( solver, bHYPREP_A, bb, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);
   return ierr;

   hypre_babel_exception_return_error(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_BoomerAMG__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_BoomerAMG__connectI(url, ar, _ex);
}
struct bHYPRE_BoomerAMG__object* impl_bHYPRE_BoomerAMG_fcast_bHYPRE_BoomerAMG(
  void* bi, sidl_BaseInterface* _ex) {
  return bHYPRE_BoomerAMG__cast(bi, _ex);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_IJParCSRMatrix(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_IJParCSRMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_IJParCSRMatrix(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_IJParCSRMatrix__cast(bi, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_BoomerAMG_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Operator(
  void* bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connectI(url, ar, _ex);
}
struct bHYPRE_Solver__object* impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Solver(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Solver__cast(bi, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* impl_bHYPRE_BoomerAMG_fcast_bHYPRE_Vector(void* 
  bi, sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_BoomerAMG_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_BoomerAMG_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_BoomerAMG_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
