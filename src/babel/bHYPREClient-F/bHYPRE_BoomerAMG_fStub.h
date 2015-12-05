/*
 * File:          bHYPRE_BoomerAMG_fStub.h
 * Symbol:        bHYPRE.BoomerAMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side documentation text for bHYPRE.BoomerAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_BoomerAMG_fStub_h
#define included_bHYPRE_BoomerAMG_fStub_h

/**
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
 */

#ifndef included_bHYPRE_BoomerAMG_IOR_h
#include "bHYPRE_BoomerAMG_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak bHYPRE_BoomerAMG__connectI

#pragma weak bHYPRE_BoomerAMG__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_BoomerAMG__object*
bHYPRE_BoomerAMG__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_BoomerAMG__object*
bHYPRE_BoomerAMG__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
