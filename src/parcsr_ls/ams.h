/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.17 $
 ***********************************************************************EHEADER*/





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
   double tol;
   HYPRE_Int cycle_type;
   HYPRE_Int print_level;

   /* Smoothing options for A */
   HYPRE_Int A_relax_type;
   HYPRE_Int A_relax_times;
   double *A_l1_norms;
   double A_relax_weight;
   double A_omega;
   double A_max_eig_est;
   double A_min_eig_est;
   HYPRE_Int A_cheby_order;
   double  A_cheby_fraction;

   /* AMG options for B_G */
   HYPRE_Int B_G_coarsen_type;
   HYPRE_Int B_G_agg_levels;
   HYPRE_Int B_G_relax_type;
   double B_G_theta;
   HYPRE_Int B_G_interp_type;
   HYPRE_Int B_G_Pmax;

   /* AMG options for B_Pi */
   HYPRE_Int B_Pi_coarsen_type;
   HYPRE_Int B_Pi_agg_levels;
   HYPRE_Int B_Pi_relax_type;
   double B_Pi_theta;
   HYPRE_Int B_Pi_interp_type;
   HYPRE_Int B_Pi_Pmax;

   /* Temporary vectors */
   hypre_ParVector *r0, *g0, *r1, *g1, *r2, *g2;

   /* Output log info */
   HYPRE_Int num_iterations;
   double rel_resid_norm;

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
#define hypre_AMSDataAMaxEigEst(ams_data) ((ams_data)->A_max_eig_est)
#define hypre_AMSDataAMinEigEst(ams_data) ((ams_data)->A_min_eig_est)
#define hypre_AMSDataAChebyOrder(ams_data) ((ams_data)->A_cheby_order)
#define hypre_AMSDataAChebyFraction(ams_data) ((ams_data)->A_cheby_fraction)

#define hypre_AMSDataPoissonAlphaAMGCoarsenType(ams_data) ((ams_data)->B_Pi_coarsen_type)
#define hypre_AMSDataPoissonAlphaAMGAggLevels(ams_data) ((ams_data)->B_Pi_agg_levels)
#define hypre_AMSDataPoissonAlphaAMGRelaxType(ams_data) ((ams_data)->B_Pi_relax_type)
#define hypre_AMSDataPoissonAlphaAMGStrengthThreshold(ams_data) ((ams_data)->B_Pi_theta)

#define hypre_AMSDataPoissonBetaAMGCoarsenType(ams_data) ((ams_data)->B_G_coarsen_type)
#define hypre_AMSDataPoissonBetaAMGAggLevels(ams_data) ((ams_data)->B_G_agg_levels)
#define hypre_AMSDataPoissonBetaAMGRelaxType(ams_data) ((ams_data)->B_G_relax_type)
#define hypre_AMSDataPoissonBetaAMGStrengthThreshold(ams_data) ((ams_data)->B_G_theta)

/* Temporary vectors */
#define hypre_AMSDataTempEdgeVectorR(ams_data) ((ams_data)->r0)
#define hypre_AMSDataTempEdgeVectorG(ams_data) ((ams_data)->g0)
#define hypre_AMSDataTempVertexVectorR(ams_data) ((ams_data)->r1)
#define hypre_AMSDataTempVertexVectorG(ams_data) ((ams_data)->g1)
#define hypre_AMSDataTempVecVertexVectorR(ams_data) ((ams_data)->r2)
#define hypre_AMSDataTempVecVertexVectorG(ams_data) ((ams_data)->g2)

#endif
