/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ADS_DATA_HEADER
#define hypre_ADS_DATA_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary space Divergence Solver data
 *--------------------------------------------------------------------------*/
typedef struct
{
   /* Face element (RT0) stiffness matrix */
   hypre_ParCSRMatrix *A;

   /* Discrete curl matrix (edge-to-face) */
   hypre_ParCSRMatrix *C;
   /* Coarse grid matrix on the range of C^T */
   hypre_ParCSRMatrix *A_C;
   /* AMS solver for A_C */
   HYPRE_Solver B_C;

   /* Raviart-Thomas nodal interpolation matrix (vertex^3-to-face) */
   hypre_ParCSRMatrix *Pi;
   /* Coarse grid matrix on the range of Pi^T */
   hypre_ParCSRMatrix *A_Pi;
   /* AMG solver for A_Pi */
   HYPRE_Solver B_Pi;

   /* Components of the face interpolation matrix (vertex-to-face each) */
   hypre_ParCSRMatrix *Pix, *Piy, *Piz;
   /* Coarse grid matrices on the ranges of Pi{x,y,z}^T */
   hypre_ParCSRMatrix *A_Pix, *A_Piy, *A_Piz;
   /* AMG solvers for A_Pi{x,y,z} */
   HYPRE_Solver B_Pix, B_Piy, B_Piz;

   /* Does the solver own the RT/ND interpolations matrices? */
   HYPRE_Int owns_Pi;
   /* The (high-order) edge interpolation matrix and its components */
   hypre_ParCSRMatrix *ND_Pi, *ND_Pix, *ND_Piy, *ND_Piz;

   /* Discrete gradient matrix (vertex-to-edge) */
   hypre_ParCSRMatrix *G;
   /* Coordinates of the vertices */
   hypre_ParVector *x, *y, *z;

   /* Solver options */
   HYPRE_Int maxit;
   HYPRE_Real tol;
   HYPRE_Int cycle_type;
   HYPRE_Int print_level;

   /* Smoothing options for A */
   HYPRE_Int A_relax_type;
   HYPRE_Int A_relax_times;
   hypre_Vector *A_l1_norms;
   HYPRE_Real A_relax_weight;
   HYPRE_Real A_omega;
   HYPRE_Real A_max_eig_est;
   HYPRE_Real A_min_eig_est;
   HYPRE_Int A_cheby_order;
   HYPRE_Real  A_cheby_fraction;

   /* AMS options for B_C */
   HYPRE_Int B_C_cycle_type;
   HYPRE_Int B_C_coarsen_type;
   HYPRE_Int B_C_agg_levels;
   HYPRE_Int B_C_relax_type;
   HYPRE_Real B_C_theta;
   HYPRE_Int B_C_interp_type;
   HYPRE_Int B_C_Pmax;

   /* AMG options for B_Pi */
   HYPRE_Int B_Pi_coarsen_type;
   HYPRE_Int B_Pi_agg_levels;
   HYPRE_Int B_Pi_relax_type;
   HYPRE_Real B_Pi_theta;
   HYPRE_Int B_Pi_interp_type;
   HYPRE_Int B_Pi_Pmax;

   /* Temporary vectors */
   hypre_ParVector *r0, *g0, *r1, *g1, *r2, *g2, *zz;

   /* Output log info */
   HYPRE_Int num_iterations;
   HYPRE_Real rel_resid_norm;

} hypre_ADSData;

/* Face stiffness matrix */
#define hypre_ADSDataA(ads_data) ((ads_data)->A)

/* Face space data */
#define hypre_ADSDataDiscreteCurl(ads_data) ((ads_data)->C)
#define hypre_ADSDataCurlCurlA(ads_data) ((ads_data)->A_C)
#define hypre_ADSDataCurlCurlAMS(ads_data) ((ads_data)->B_C)

/* Vector vertex space data */
#define hypre_ADSDataPiInterpolation(ads_data) ((ads_data)->Pi)
#define hypre_ADSDataOwnsPiInterpolation(ads_data) ((ads_data)->owns_Pi)
#define hypre_ADSDataPoissonA(ads_data) ((ads_data)->A_Pi)
#define hypre_ADSDataPoissonAMG(ads_data) ((ads_data)->B_Pi)

/* Discrete gradient and coordinates of the vertices */
#define hypre_ADSDataDiscreteGradient(ads_data) ((ads_data)->G)
#define hypre_ADSDataVertexCoordinateX(ads_data) ((ads_data)->x)
#define hypre_ADSDataVertexCoordinateY(ads_data) ((ads_data)->y)
#define hypre_ADSDataVertexCoordinateZ(ads_data) ((ads_data)->z)

/* Solver options */
#define hypre_ADSDataMaxIter(ads_data) ((ads_data)->maxit)
#define hypre_ADSDataTol(ads_data) ((ads_data)->tol)
#define hypre_ADSDataCycleType(ads_data) ((ads_data)->cycle_type)
#define hypre_ADSDataPrintLevel(ads_data) ((ads_data)->print_level)

/* Smoothing options */
#define hypre_ADSDataARelaxType(ads_data) ((ads_data)->A_relax_type)
#define hypre_ADSDataARelaxTimes(ads_data) ((ads_data)->A_relax_times)
#define hypre_ADSDataAL1Norms(ads_data) ((ads_data)->A_l1_norms)
#define hypre_ADSDataARelaxWeight(ads_data) ((ads_data)->A_relax_weight)
#define hypre_ADSDataAOmega(ads_data) ((ads_data)->A_omega)
#define hypre_ADSDataAMaxEigEst(ads_data) ((ads_data)->A_max_eig_est)
#define hypre_ADSDataAMinEigEst(ads_data) ((ads_data)->A_min_eig_est)
#define hypre_ADSDataAChebyOrder(ads_data) ((ads_data)->A_cheby_order)
#define hypre_ADSDataAChebyFraction(ads_data) ((ads_data)->A_cheby_fraction)

/* AMS options */
#define hypre_ADSDataAMSCycleType(ads_data) ((ads_data)->B_C_cycle_type)
#define hypre_ADSDataAMSCoarsenType(ads_data) ((ads_data)->B_C_coarsen_type)
#define hypre_ADSDataAMSAggLevels(ads_data) ((ads_data)->B_C_agg_levels)
#define hypre_ADSDataAMSRelaxType(ads_data) ((ads_data)->B_C_relax_type)
#define hypre_ADSDataAMSStrengthThreshold(ads_data) ((ads_data)->B_C_theta)
#define hypre_ADSDataAMSInterpType(ads_data) ((ads_data)->B_C_interp_type)
#define hypre_ADSDataAMSPmax(ads_data) ((ads_data)->B_C_Pmax)

/* AMG options */
#define hypre_ADSDataAMGCoarsenType(ads_data) ((ads_data)->B_Pi_coarsen_type)
#define hypre_ADSDataAMGAggLevels(ads_data) ((ads_data)->B_Pi_agg_levels)
#define hypre_ADSDataAMGRelaxType(ads_data) ((ads_data)->B_Pi_relax_type)
#define hypre_ADSDataAMGStrengthThreshold(ads_data) ((ads_data)->B_Pi_theta)
#define hypre_ADSDataAMGInterpType(ads_data) ((ads_data)->B_Pi_interp_type)
#define hypre_ADSDataAMGPmax(ads_data) ((ads_data)->B_Pi_Pmax)

/* Temporary vectors */
#define hypre_ADSDataTempFaceVectorR(ads_data) ((ads_data)->r0)
#define hypre_ADSDataTempFaceVectorG(ads_data) ((ads_data)->g0)
#define hypre_ADSDataTempEdgeVectorR(ads_data) ((ads_data)->r1)
#define hypre_ADSDataTempEdgeVectorG(ads_data) ((ads_data)->g1)
#define hypre_ADSDataTempVertexVectorR(ads_data) ((ads_data)->r2)
#define hypre_ADSDataTempVertexVectorG(ads_data) ((ads_data)->g2)

#endif
