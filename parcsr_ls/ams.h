/*BHEADER**********************************************************************
 * (c) 2006   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef hypre_AMS_DATA_HEADER
#define hypre_AMS_DATA_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary space Maxwell Solver data
 *--------------------------------------------------------------------------*/
typedef struct
{
   /* Space dimension (2 or 3) */
   int dim;

   /* Edge element (ND1) stiffness matrix */
   hypre_ParCSRMatrix *A;

   /* Discrete gradient matrix (vertex-to-edge) */
   hypre_ParCSRMatrix *G;
   /* Coarse grid matrix on the range of G^T */
   hypre_ParCSRMatrix *A_G;
   /* AMG solver for A_G */
   HYPRE_Solver B_G;
   /* Is the mass term coefficient zero? */
   int beta_is_zero;

   /* Nedelec interpolation matrix (vertex^dim-to-edge) */
   hypre_ParCSRMatrix *Pi;
   /* Coarse grid matrix on the range of Pi^T */
   hypre_ParCSRMatrix *A_Pi;
   /* AMG solver for A_Pi */
   HYPRE_Solver B_Pi;

   /* Does the solver own the coarse grid matrices? */
   int owns_A_G, owns_A_Pi;

   /* Coordinates of the vertices (z = 0 if dim == 2) */
   hypre_ParVector *x, *y, *z;

   /* Representations of the constant vectors in the Nedelec basis */
   hypre_ParVector *Gx, *Gy, *Gz;

   /* Solver options */
   int maxit;
   double tol;
   int cycle_type;
   int print_level;

   /* Smoothing options for A */
   int A_relax_type;
   int A_relax_times;
   double *A_l1_norms;
   double A_relax_weight;
   double A_omega;

   /* AMG options for B_G */
   int B_G_coarsen_type;
   int B_G_agg_levels;
   int B_G_relax_type;
   double B_G_theta;

   /* AMG options for B_Pi */
   int B_Pi_coarsen_type;
   int B_Pi_agg_levels;
   int B_Pi_relax_type;
   double B_Pi_theta;

   /* Temporary vectors */
   hypre_ParVector *r0, *g0, *r1, *g1, *r2, *g2;

} hypre_AMSData;

/* Space dimension */
#define hypre_AMSDataDimension(ams_data) ((ams_data)->dim)

/* Edge stiffness matrix */
#define hypre_AMSDataA(ams_data) ((ams_data)->A)

/* Vertex space data */
#define hypre_AMSDataDiscreteGradient(ams_data) ((ams_data)->G)
#define hypre_AMSDataPoissonBeta(ams_data) ((ams_data)->A_G)
#define hypre_AMSDataPoissonBetaAMG(ams_data) ((ams_data)->B_G)
#define hypre_AMSDataOwnsPoissonBeta(ams_data) ((ams_data)->owns_A_G)
#define hypre_AMSDataBetaIsZero(ams_data) ((ams_data)->beta_is_zero)

/* Vector vertex space data */
#define hypre_AMSDataPiInterpolation(ams_data) ((ams_data)->Pi)
#define hypre_AMSDataPoissonAlpha(ams_data) ((ams_data)->A_Pi)
#define hypre_AMSDataPoissonAlphaAMG(ams_data) ((ams_data)->B_Pi)
#define hypre_AMSDataOwnsPoissonAlpha(ams_data) ((ams_data)->owns_A_Pi)

/* Coordinates of the vertices */
#define hypre_AMSDataVertexCoordinateX(ams_data) ((ams_data)->x)
#define hypre_AMSDataVertexCoordinateY(ams_data) ((ams_data)->y)
#define hypre_AMSDataVertexCoordinateZ(ams_data) ((ams_data)->z)

/* Representations of the constant vectors in the Nedelec basis */
#define hypre_AMSDataEdgeConstantX(ams_data) ((ams_data)->Gx)
#define hypre_AMSDataEdgeConstantY(ams_data) ((ams_data)->Gy)
#define hypre_AMSDataEdgeConstantZ(ams_data) ((ams_data)->Gz)

/* Solver options */
#define hypre_AMSDataMaxIter(ams_data) ((ams_data)->maxit)
#define hypre_AMSDataTol(ams_data) ((ams_data)->tol)
#define hypre_AMSDataCycleType(ams_data) ((ams_data)->cycle_type)
#define hypre_AMSDataPrintLevel(ams_data) ((ams_data)->print_level)

/* Smoothing and AMG options */
#define hypre_AMSDataARelaxType(ams_data) ((ams_data)->A_relax_type)
#define hypre_AMSDataARelaxTimes(ams_data) ((ams_data)->A_relax_times)
#define hypre_AMSDataAL1Norms(ams_data) ((ams_data)->A_l1_norms)
#define hypre_AMSDataARelaxWeight(ams_data) ((ams_data)->A_relax_weight)
#define hypre_AMSDataAOmega(ams_data) ((ams_data)->A_omega)
#define hypre_AMSDataPoissonBetaAMGCoarsenType(ams_data) ((ams_data)->B_G_coarsen_type)
#define hypre_AMSDataPoissonBetaAMGAggLevels(ams_data) ((ams_data)->B_G_agg_levels)
#define hypre_AMSDataPoissonBetaAMGRelaxType(ams_data) ((ams_data)->B_G_relax_type)
#define hypre_AMSDataPoissonBetaAMGStrengthThreshold(ams_data) ((ams_data)->B_G_theta)
#define hypre_AMSDataPoissonAlphaAMGCoarsenType(ams_data) ((ams_data)->B_Pi_coarsen_type)
#define hypre_AMSDataPoissonAlphaAMGAggLevels(ams_data) ((ams_data)->B_Pi_agg_levels)
#define hypre_AMSDataPoissonAlphaAMGRelaxType(ams_data) ((ams_data)->B_Pi_relax_type)
#define hypre_AMSDataPoissonAlphaAMGStrengthThreshold(ams_data) ((ams_data)->B_Pi_theta)

/* Temporary vectors */
#define hypre_AMSDataTempEdgeVectorR(ams_data) ((ams_data)->r0)
#define hypre_AMSDataTempEdgeVectorG(ams_data) ((ams_data)->g0)
#define hypre_AMSDataTempVertexVectorR(ams_data) ((ams_data)->r1)
#define hypre_AMSDataTempVertexVectorG(ams_data) ((ams_data)->g1)
#define hypre_AMSDataTempVecVertexVectorR(ams_data) ((ams_data)->r2)
#define hypre_AMSDataTempVecVertexVectorG(ams_data) ((ams_data)->g2)

#endif
