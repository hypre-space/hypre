/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_AMS_DATA_HEADER
#define hypre_AMS_DATA_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary space Maxwell Solver data
 *--------------------------------------------------------------------------*/
typedef struct
{
   /* Space dimension (2 or 3) */
   HYPRE_Int dim;

   /* Edge element (ND1) stiffness matrix */
   hypre_ParCSRMatrix *A;

   /* Discrete gradient matrix (vertex-to-edge) */
   hypre_ParCSRMatrix *G;
   /* Coarse grid matrix on the range of G^T */
   hypre_ParCSRMatrix *A_G;
   /* AMG solver for A_G */
   HYPRE_Solver B_G;
   /* Is the mass term coefficient zero? */
   HYPRE_Int beta_is_zero;

   /* Nedelec nodal interpolation matrix (vertex^dim-to-edge) */
   hypre_ParCSRMatrix *Pi;
   /* Coarse grid matrix on the range of Pi^T */
   hypre_ParCSRMatrix *A_Pi;
   /* AMG solver for A_Pi */
   HYPRE_Solver B_Pi;

   /* Components of the Nedelec interpolation matrix (vertex-to-edge each) */
   hypre_ParCSRMatrix *Pix, *Piy, *Piz;
   /* Coarse grid matrices on the ranges of Pi{x,y,z}^T */
   hypre_ParCSRMatrix *A_Pix, *A_Piy, *A_Piz;
   /* AMG solvers for A_Pi{x,y,z} */
   HYPRE_Solver B_Pix, B_Piy, B_Piz;

   /* Does the solver own the Nedelec interpolations? */
   HYPRE_Int owns_Pi;
   /* Does the solver own the coarse grid matrices? */
   HYPRE_Int owns_A_G, owns_A_Pi;

   /* Coordinates of the vertices (z = 0 if dim == 2) */
   hypre_ParVector *x, *y, *z;

   /* Representations of the constant vectors in the Nedelec basis */
   hypre_ParVector *Gx, *Gy, *Gz;

   /* Nodes in the interior of the zero-conductivity region */
   hypre_ParVector *interior_nodes;
   /* Discrete gradient matrix for the interior nodes only */
   hypre_ParCSRMatrix *G0;
   /* Coarse grid matrix on the interior nodes */
   hypre_ParCSRMatrix *A_G0;
   /* AMG solver for A_G0 */
   HYPRE_Solver B_G0;
   /* How frequently to project the r.h.s. onto Ker(G0^T)? */
   HYPRE_Int projection_frequency;
   /* Internal counter to use with projection_frequency in PCG */
   HYPRE_Int solve_counter;

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

   /* AMG options for B_G */
   HYPRE_Int B_G_coarsen_type;
   HYPRE_Int B_G_agg_levels;
   HYPRE_Int B_G_relax_type;
   HYPRE_Int B_G_coarse_relax_type;
   HYPRE_Real B_G_theta;
   HYPRE_Int B_G_interp_type;
   HYPRE_Int B_G_Pmax;

   /* AMG options for B_Pi */
   HYPRE_Int B_Pi_coarsen_type;
   HYPRE_Int B_Pi_agg_levels;
   HYPRE_Int B_Pi_relax_type;
   HYPRE_Int B_Pi_coarse_relax_type;
   HYPRE_Real B_Pi_theta;
   HYPRE_Int B_Pi_interp_type;
   HYPRE_Int B_Pi_Pmax;

   /* Temporary vectors */
   hypre_ParVector *r0, *g0, *r1, *g1, *r2, *g2, *zz;

   /* Output log info */
   HYPRE_Int num_iterations;
   HYPRE_Real rel_resid_norm;

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
#define hypre_AMSDataOwnsPiInterpolation(ams_data) ((ams_data)->owns_Pi)
#define hypre_AMSDataPoissonAlpha(ams_data) ((ams_data)->A_Pi)
#define hypre_AMSDataPoissonAlphaAMG(ams_data) ((ams_data)->B_Pi)
#define hypre_AMSDataOwnsPoissonAlpha(ams_data) ((ams_data)->owns_A_Pi)

/* Vector vertex components data */
#define hypre_AMSDataPiXInterpolation(ams_data) ((ams_data)->Pix)
#define hypre_AMSDataPiYInterpolation(ams_data) ((ams_data)->Piy)
#define hypre_AMSDataPiZInterpolation(ams_data) ((ams_data)->Piz)
#define hypre_AMSDataPoissonAlphaX(ams_data) ((ams_data)->A_Pix)
#define hypre_AMSDataPoissonAlphaY(ams_data) ((ams_data)->A_Piy)
#define hypre_AMSDataPoissonAlphaZ(ams_data) ((ams_data)->A_Piz)
#define hypre_AMSDataPoissonAlphaXAMG(ams_data) ((ams_data)->B_Pix)
#define hypre_AMSDataPoissonAlphaYAMG(ams_data) ((ams_data)->B_Piy)
#define hypre_AMSDataPoissonAlphaZAMG(ams_data) ((ams_data)->B_Piz)

/* Coordinates of the vertices */
#define hypre_AMSDataVertexCoordinateX(ams_data) ((ams_data)->x)
#define hypre_AMSDataVertexCoordinateY(ams_data) ((ams_data)->y)
#define hypre_AMSDataVertexCoordinateZ(ams_data) ((ams_data)->z)

/* Representations of the constant vectors in the Nedelec basis */
#define hypre_AMSDataEdgeConstantX(ams_data) ((ams_data)->Gx)
#define hypre_AMSDataEdgeConstantY(ams_data) ((ams_data)->Gy)
#define hypre_AMSDataEdgeConstantZ(ams_data) ((ams_data)->Gz)

/* Interior zero conductivity region */
#define hypre_AMSDataInteriorNodes(ams_data) ((ams_data)->interior_nodes)
#define hypre_AMSDataInteriorDiscreteGradient(ams_data) ((ams_data)->G0)
#define hypre_AMSDataInteriorPoissonBeta(ams_data) ((ams_data)->A_G0)
#define hypre_AMSDataInteriorPoissonBetaAMG(ams_data) ((ams_data)->B_G0)
#define hypre_AMSDataInteriorProjectionFrequency(ams_data) ((ams_data)->projection_frequency)
#define hypre_AMSDataInteriorSolveCounter(ams_data) ((ams_data)->solve_counter)

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
#define hypre_AMSDataAMaxEigEst(ams_data) ((ams_data)->A_max_eig_est)
#define hypre_AMSDataAMinEigEst(ams_data) ((ams_data)->A_min_eig_est)
#define hypre_AMSDataAChebyOrder(ams_data) ((ams_data)->A_cheby_order)
#define hypre_AMSDataAChebyFraction(ams_data) ((ams_data)->A_cheby_fraction)

#define hypre_AMSDataPoissonBetaAMGCoarsenType(ams_data) ((ams_data)->B_G_coarsen_type)
#define hypre_AMSDataPoissonBetaAMGAggLevels(ams_data) ((ams_data)->B_G_agg_levels)
#define hypre_AMSDataPoissonBetaAMGRelaxType(ams_data) ((ams_data)->B_G_relax_type)
#define hypre_AMSDataPoissonBetaAMGCoarseRelaxType(ams_data) ((ams_data)->B_G_coarse_relax_type)
#define hypre_AMSDataPoissonBetaAMGStrengthThreshold(ams_data) ((ams_data)->B_G_theta)
#define hypre_AMSDataPoissonBetaAMGInterpType(ams_data) ((ams_data)->B_G_interp_type)
#define hypre_AMSDataPoissonBetaAMGPMax(ams_data) ((ams_data)->B_G_Pmax)

#define hypre_AMSDataPoissonAlphaAMGCoarsenType(ams_data) ((ams_data)->B_Pi_coarsen_type)
#define hypre_AMSDataPoissonAlphaAMGAggLevels(ams_data) ((ams_data)->B_Pi_agg_levels)
#define hypre_AMSDataPoissonAlphaAMGRelaxType(ams_data) ((ams_data)->B_Pi_relax_type)
#define hypre_AMSDataPoissonAlphaAMGCoarseRelaxType(ams_data) ((ams_data)->B_Pi_coarse_relax_type)
#define hypre_AMSDataPoissonAlphaAMGStrengthThreshold(ams_data) ((ams_data)->B_Pi_theta)
#define hypre_AMSDataPoissonAlphaAMGInterpType(ams_data) ((ams_data)->B_Pi_interp_type)
#define hypre_AMSDataPoissonAlphaAMGPMax(ams_data) ((ams_data)->B_Pi_Pmax)

/* Temporary vectors */
#define hypre_AMSDataTempEdgeVectorR(ams_data) ((ams_data)->r0)
#define hypre_AMSDataTempEdgeVectorG(ams_data) ((ams_data)->g0)
#define hypre_AMSDataTempVertexVectorR(ams_data) ((ams_data)->r1)
#define hypre_AMSDataTempVertexVectorG(ams_data) ((ams_data)->g1)
#define hypre_AMSDataTempVecVertexVectorR(ams_data) ((ams_data)->r2)
#define hypre_AMSDataTempVecVertexVectorG(ams_data) ((ams_data)->g2)
#define hypre_AMSDataTempVecVertexVectorZZ(ams_data) ((ams_data)->zz)

/* Output log info */
#define hypre_AMSDataNumIterations(ams_data) ((ams_data)->num_iterations)
#define hypre_AMSDataResidualNorm(ams_data) ((ams_data)->rel_resid_norm)

#endif
