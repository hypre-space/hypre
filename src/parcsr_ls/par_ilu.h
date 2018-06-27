/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef hypre_ParILU_DATA_HEADER
#define hypre_ParILU_DATA_HEADER
/*--------------------------------------------------------------------------
 * hypre_ParILUData
 *--------------------------------------------------------------------------*/
typedef struct hypre_ParILUData_struct
{
   //general data
   HYPRE_Int global_solver;
   hypre_ParCSRMatrix *matA;
   hypre_ParCSRMatrix *matL;
   HYPRE_Real        *matD;
   hypre_ParCSRMatrix *matU;
   hypre_ParCSRMatrix *matS;
   HYPRE_Real        *droptol;/* should be an array of 3 element, for B, (E and F), S respectively */
   HYPRE_Int own_droptol_data;/* should I free droptols */
   HYPRE_Int        lfil;
   HYPRE_Int        maxRowNnz;
   HYPRE_Int *CF_marker_array;
   HYPRE_Int    *perm;
   hypre_ParVector    *F;
   hypre_ParVector    *U;
   hypre_ParVector    *residual;
   HYPRE_Real    *rel_res_norms;  
   HYPRE_Int     num_iterations;
   HYPRE_Real   *l1_norms;
   HYPRE_Real   final_rel_residual_norm;
   HYPRE_Real   tol;
   HYPRE_Real    operator_complexity;
   
   HYPRE_Int   logging;
   HYPRE_Int   print_level;
   HYPRE_Int   max_iter;
   
   HYPRE_Int   ilu_type;
   HYPRE_Int     nLU;
 
   /* temp vectors for solve phase */
   hypre_ParVector   *Utemp;
   hypre_ParVector   *Ftemp;
   
   /* data structure sor solving Schur System */
   HYPRE_Solver schur_solver;
   HYPRE_Solver schur_precond;
   hypre_ParVector *rhs;
   hypre_ParVector *x;
   
   /* schur solver data */
   HYPRE_Int ss_kDim;
   HYPRE_Int ss_max_iter;
   HYPRE_Real ss_tol;
   HYPRE_Real ss_absolute_tol;
   HYPRE_Int ss_logging;
   HYPRE_Int ss_print_level;
   HYPRE_Int ss_rel_change;
   
   /* schur precond data */
   HYPRE_Int sp_ilu_type;
   HYPRE_Int sp_ilu_lfil;
   HYPRE_Int sp_ilu_max_row_nnz;
   HYPRE_Real *sp_ilu_droptol;
   HYPRE_Int sp_own_droptol_data;
   HYPRE_Int sp_print_level;
   HYPRE_Int sp_max_iter;
   HYPRE_Real sp_tol;
   
} hypre_ParILUData;


#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define MAT_TOL 1e-14
#define EXPAND_FACT 1.3

#endif
