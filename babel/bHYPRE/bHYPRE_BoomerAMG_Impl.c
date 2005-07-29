/*
 * File:          bHYPRE_BoomerAMG_Impl.c
 * Symbol:        bHYPRE.BoomerAMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.BoomerAMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
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
 * \item[Cycle0NumSweeps] ({\tt Int}) - number of sweeps for fine grid
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
 * \item[Cycle0RelaxType] ({\tt Int}) - type of smoother for fine grid
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
 * \item[SmoothType] ({\tt Int}) - more complex smoothers.
 * 
 * \item[SmoothNumLevels] ({\tt Int}) - number of levels for more
 * complex smoothers.
 * 
 * \item[SmoothNumSweeps] ({\tt Int}) - number of sweeps for more
 * complex smoothers.
 * 
 * \item[PrintFileName] ({\tt String}) - name of file printed to in
 * association with {\tt SetPrintLevel}.  (not yet implemented).
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
 * \item[DebugFlag] ({\tt Int}) -
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
 * 
 */

#include "bHYPRE_BoomerAMG_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._includes) */

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
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._load) */
  /* Insert-Code-Here {bHYPRE.BoomerAMG._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._load) */
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
  /* in */ bHYPRE_BoomerAMG self)
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
   data -> comm = NULL;
   ierr += HYPRE_BoomerAMGCreate( solver );
   data -> solver = *solver;
   /* set any other data components here */
   bHYPRE_BoomerAMG__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._ctor) */
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
  /* in */ bHYPRE_BoomerAMG self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   bHYPRE_IJParCSRMatrix_deleteRef( data->matrix );
   ierr += HYPRE_BoomerAMGDestroy( data->solver );
   assert( ierr== 0 );
   /* delete any nontrivial data components here */
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_BoomerAMG
impl_bHYPRE_BoomerAMG_Create(
  /* in */ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.Create) */
  /* Insert-Code-Here {bHYPRE.BoomerAMG.Create} (Create method) */

   bHYPRE_BoomerAMG solver = bHYPRE_BoomerAMG__create();
   struct bHYPRE_BoomerAMG__data * data = bHYPRE_BoomerAMG__get_data( solver );

   data -> comm = (MPI_Comm *) mpi_comm;

   return solver;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.Create) */
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
  /* in */ int32_t level)
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

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetCommunicator(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED  Use Create */

   int ierr = 0;
   struct bHYPRE_BoomerAMG__data * data = bHYPRE_BoomerAMG__get_data( self );
   data -> comm = (MPI_Comm *) mpi_comm;
   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
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
  /* in */ int32_t value)
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
   else if ( strcmp(name,"Cycle0NumSweeps")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleNumSweeps( solver, value, 0 );
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
   else if ( strcmp(name,"Cycle0RelaxType")==0 )
   {
      ierr += HYPRE_BoomerAMGSetCycleRelaxType( solver, value, 0 );
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
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
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
  /* in */ double value)
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
   else if ( strcmp(name,"SchwarzRlxWeight")==0 )
   {
      ierr += HYPRE_BoomerAMGSetSchwarzRlxWeight( solver, value );
   }
   else if ( strcmp(name,"MaxRowSum")==0 )
   {
      ierr += HYPRE_BoomerAMGSetMaxRowSum( solver, value );
   }
   else
   {
      ierr=1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
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
  /* in */ const char* value)
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

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
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
  /* in */ int32_t* value,
  /* in */ int32_t nvalues)
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

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
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
  /* in */ struct sidl_int__array* value)
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
      assert( dim==2 );
      lb0 = sidl_int__array_lower( value, 0 );
      ub0 = sidl_int__array_upper( value, 0 );
      lb1 = sidl_int__array_lower( value, 1 );
      ub1 = sidl_int__array_upper( value, 1 );
      assert( lb0==0 );
      assert( lb1==0 );
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

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
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
  /* in */ double* value,
  /* in */ int32_t nvalues)
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

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
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
  /* in */ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
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
  /* out */ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
/* not needed until we have something to return...
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;
*/
   ierr = 1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
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
  /* out */ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   int ierr = 0;
/* not needed until we have something to return...
   HYPRE_Solver solver;
   struct bHYPRE_BoomerAMG__data * data;

   data = bHYPRE_BoomerAMG__get_data( self );
   solver = data->solver;
*/
   ierr = 1;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
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
  /* in */ bHYPRE_Vector x)
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

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
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
      assert( "Unrecognized vector type."==(char *)(x) );
   }

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;
   ierr += HYPRE_BoomerAMGSetup( solver, bHYPREP_A, bb, xx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
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
  /* inout */ bHYPRE_Vector* x)
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

   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
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
       * to do something similar if x had zero length.  Or assert(x
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
      assert( "Unrecognized vector type."==(char *)(*x) );
   }

   datax = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x );
   ij_x = datax -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_BoomerAMGSolve( solver, bHYPREP_A, bb, xx );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetOperator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetOperator(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.BoomerAMG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */

   int ierr = 0;
   struct bHYPRE_BoomerAMG__data * data;
   bHYPRE_IJParCSRMatrix Amat;

   if ( bHYPRE_Operator_queryInt( A, "bHYPRE.IJParCSRMatrix" ) )
   {
      Amat = bHYPRE_IJParCSRMatrix__cast( A );
      bHYPRE_IJParCSRMatrix_deleteRef( Amat ); /* extra ref from queryInt */
   }
   else
   {
      assert( "Unrecognized operator type."==(char *)A );
   }

   data = bHYPRE_BoomerAMG__get_data( self );
   data->matrix = Amat;
   bHYPRE_IJParCSRMatrix_addRef( data->matrix );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.BoomerAMG.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetTolerance"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetTolerance(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ double tolerance)
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

/*
 * (Optional) Set maximum number of iterations.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetMaxIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetMaxIterations(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t max_iterations)
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

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetLogging"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetLogging(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t level)
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

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_SetPrintLevel"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_SetPrintLevel(
  /* in */ bHYPRE_BoomerAMG self,
  /* in */ int32_t level)
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

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_GetNumIterations"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_GetNumIterations(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ int32_t* num_iterations)
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

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_BoomerAMG_GetRelResidualNorm"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_BoomerAMG_GetRelResidualNorm(
  /* in */ bHYPRE_BoomerAMG self,
  /* out */ double* norm)
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
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_Solver__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Solver__connect(url, _ex);
}
char * impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Solver(struct 
  bHYPRE_Solver__object* obj) {
  return bHYPRE_Solver__getURL(obj);
}
struct bHYPRE_BoomerAMG__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_BoomerAMG__connect(url, _ex);
}
char * impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_BoomerAMG(struct 
  bHYPRE_BoomerAMG__object* obj) {
  return bHYPRE_BoomerAMG__getURL(obj);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connect(url, _ex);
}
char * impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) {
  return bHYPRE_Operator__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_BoomerAMG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_BoomerAMG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_BoomerAMG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
