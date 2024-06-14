/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParILU_DATA_HEADER
#define hypre_ParILU_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParILUData
 *--------------------------------------------------------------------------*/

typedef struct hypre_ParILUData_struct
{
   /* Base solver data structure */
   hypre_Solver          base;

   /* General data */
   HYPRE_Int             global_solver;
   hypre_ParCSRMatrix   *matA;
   hypre_ParCSRMatrix   *matL;
   HYPRE_Real           *matD;
   hypre_ParCSRMatrix   *matU;
   hypre_ParCSRMatrix   *matmL;
   HYPRE_Real           *matmD;
   hypre_ParCSRMatrix   *matmU;
   hypre_ParCSRMatrix   *matS;
   HYPRE_Real           *droptol; /* Array of 3 elements, for B, (E and F), S respectively */
   HYPRE_Int             lfil;
   HYPRE_Int             maxRowNnz;
   HYPRE_Int            *CF_marker_array;
   HYPRE_Int            *perm;
   HYPRE_Int            *qperm;
   HYPRE_Real            tol_ddPQ;
   hypre_ParVector      *F;
   hypre_ParVector      *U;
   hypre_ParVector      *residual;
   HYPRE_Real           *rel_res_norms;
   HYPRE_Int             num_iterations;
   HYPRE_Real           *l1_norms;
   HYPRE_Real            final_rel_residual_norm;
   HYPRE_Real            tol;
   HYPRE_Real            operator_complexity;
   HYPRE_Int             logging;
   HYPRE_Int             print_level;
   HYPRE_Int             max_iter;
   HYPRE_Int             tri_solve;
   HYPRE_Int             lower_jacobi_iters;
   HYPRE_Int             upper_jacobi_iters;
   HYPRE_Int             ilu_type;
   HYPRE_Int             nLU;
   HYPRE_Int             nI;
   HYPRE_Int            *u_end; /* used when schur block is formed */

   /* Iterative ILU parameters */
   HYPRE_Int             iter_setup_type;
   HYPRE_Int             iter_setup_option;
   HYPRE_Int             setup_max_iter;
   HYPRE_Int             setup_num_iter;
   HYPRE_Real            setup_tolerance;
   HYPRE_Complex        *setup_history;

   /* temp vectors for solve phase */
   hypre_ParVector      *Utemp;
   hypre_ParVector      *Ftemp;
   hypre_ParVector      *Xtemp;
   hypre_ParVector      *Ytemp;
   hypre_ParVector      *Ztemp;
   HYPRE_Real           *uext;
   HYPRE_Real           *fext;

   /* On GPU, we have to form E and F explicitly, since we don't have much control to it */
#if defined(HYPRE_USING_GPU)
   hypre_CSRMatrix      *matALU_d; /* Matrix holding ILU of A (for A-smoothing) */
   hypre_CSRMatrix      *matBLU_d; /* Matrix holding ILU of B */
   hypre_CSRMatrix      *matSLU_d; /* Matrix holding ILU of S */
   hypre_CSRMatrix      *matE_d;
   hypre_CSRMatrix      *matF_d;
   hypre_ParCSRMatrix   *Aperm;
   hypre_ParCSRMatrix   *R;
   hypre_ParCSRMatrix   *P;
   hypre_Vector         *Ftemp_upper;
   hypre_Vector         *Utemp_lower;
   hypre_Vector         *Adiag_diag;
   hypre_Vector         *Sdiag_diag;
#endif

   /* data structure sor solving Schur System */
   HYPRE_Solver          schur_solver;
   HYPRE_Solver          schur_precond;
   hypre_ParVector      *rhs;
   hypre_ParVector      *x;

   /* Schur solver data */
   HYPRE_Int             ss_logging;
   HYPRE_Int             ss_print_level;

   /* Schur-GMRES */
   HYPRE_Int             ss_kDim;               /* max number of iterations for GMRES */
   HYPRE_Int             ss_max_iter;           /* max number of iterations for GMRES solve */
   HYPRE_Real            ss_tol;                /* stop iteration tol for GMRES */
   HYPRE_Real            ss_absolute_tol;       /* absolute tol for GMRES or tol for NSH solve */
   HYPRE_Int             ss_rel_change;

   /* Schur-NSH */
   HYPRE_Int             ss_nsh_setup_max_iter; /* number of iterations for NSH inverse */
   HYPRE_Int             ss_nsh_solve_max_iter; /* max number of iterations for NSH solve */
   HYPRE_Real            ss_nsh_setup_tol;      /* stop iteration tol for NSH inverse */
   HYPRE_Real            ss_nsh_solve_tol;      /* absolute tol for NSH solve */
   HYPRE_Int             ss_nsh_max_row_nnz;    /* max rows of nonzeros for NSH */
   HYPRE_Int             ss_nsh_mr_col_version; /* MR column version setting in NSH */
   HYPRE_Int             ss_nsh_mr_max_row_nnz; /* max rows for MR  */
   HYPRE_Real           *ss_nsh_droptol;        /* droptol array for NSH */
   HYPRE_Int             ss_nsh_mr_max_iter;    /* max MR iteration */
   HYPRE_Real            ss_nsh_mr_tol;

   /* Schur precond data */
   HYPRE_Int             sp_ilu_type;           /* ilu type is use ILU */
   HYPRE_Int             sp_ilu_lfil;           /* level of fill in for ILUK */
   HYPRE_Int             sp_ilu_max_row_nnz;    /* max rows for ILUT  */
   /* droptol for ILUT or MR
    * ILUT: [0], [1], [2] B, E&F, S respectively
    * NSH: [0] for MR, [1] for NSH
    */
   HYPRE_Real           *sp_ilu_droptol;        /* droptol array for ILUT */
   HYPRE_Int             sp_print_level;
   HYPRE_Int             sp_max_iter;           /* max precond iter or max MR iteration */
   HYPRE_Int             sp_tri_solve;
   HYPRE_Int             sp_lower_jacobi_iters;
   HYPRE_Int             sp_upper_jacobi_iters;
   HYPRE_Real            sp_tol;
   HYPRE_Int             test_opt; /* TODO (VPM): change this to something more descriptive*/

   /* local reordering */
   HYPRE_Int             reordering_type;
} hypre_ParILUData;

#define hypre_ParILUDataTestOption(ilu_data)                   ((ilu_data) -> test_opt)

#if defined(HYPRE_USING_GPU)
#define hypre_ParILUDataMatAILUDevice(ilu_data)                ((ilu_data) -> matALU_d)
#define hypre_ParILUDataMatBILUDevice(ilu_data)                ((ilu_data) -> matBLU_d)
#define hypre_ParILUDataMatSILUDevice(ilu_data)                ((ilu_data) -> matSLU_d)
#define hypre_ParILUDataMatEDevice(ilu_data)                   ((ilu_data) -> matE_d)
#define hypre_ParILUDataMatFDevice(ilu_data)                   ((ilu_data) -> matF_d)
#define hypre_ParILUDataAperm(ilu_data)                        ((ilu_data) -> Aperm)
#define hypre_ParILUDataR(ilu_data)                            ((ilu_data) -> R)
#define hypre_ParILUDataP(ilu_data)                            ((ilu_data) -> P)
#define hypre_ParILUDataFTempUpper(ilu_data)                   ((ilu_data) -> Ftemp_upper)
#define hypre_ParILUDataUTempLower(ilu_data)                   ((ilu_data) -> Utemp_lower)
#define hypre_ParILUDataADiagDiag(ilu_data)                    ((ilu_data) -> Adiag_diag)
#define hypre_ParILUDataSDiagDiag(ilu_data)                    ((ilu_data) -> Sdiag_diag)
#endif

#define hypre_ParILUDataGlobalSolver(ilu_data)                 ((ilu_data) -> global_solver)
#define hypre_ParILUDataMatA(ilu_data)                         ((ilu_data) -> matA)
#define hypre_ParILUDataMatL(ilu_data)                         ((ilu_data) -> matL)
#define hypre_ParILUDataMatD(ilu_data)                         ((ilu_data) -> matD)
#define hypre_ParILUDataMatU(ilu_data)                         ((ilu_data) -> matU)
#define hypre_ParILUDataMatLModified(ilu_data)                 ((ilu_data) -> matmL)
#define hypre_ParILUDataMatDModified(ilu_data)                 ((ilu_data) -> matmD)
#define hypre_ParILUDataMatUModified(ilu_data)                 ((ilu_data) -> matmU)
#define hypre_ParILUDataMatS(ilu_data)                         ((ilu_data) -> matS)
#define hypre_ParILUDataDroptol(ilu_data)                      ((ilu_data) -> droptol)
#define hypre_ParILUDataLfil(ilu_data)                         ((ilu_data) -> lfil)
#define hypre_ParILUDataMaxRowNnz(ilu_data)                    ((ilu_data) -> maxRowNnz)
#define hypre_ParILUDataCFMarkerArray(ilu_data)                ((ilu_data) -> CF_marker_array)
#define hypre_ParILUDataPerm(ilu_data)                         ((ilu_data) -> perm)
#define hypre_ParILUDataPPerm(ilu_data)                        ((ilu_data) -> perm)
#define hypre_ParILUDataQPerm(ilu_data)                        ((ilu_data) -> qperm)
#define hypre_ParILUDataTolDDPQ(ilu_data)                      ((ilu_data) -> tol_ddPQ)
#define hypre_ParILUDataF(ilu_data)                            ((ilu_data) -> F)
#define hypre_ParILUDataU(ilu_data)                            ((ilu_data) -> U)
#define hypre_ParILUDataResidual(ilu_data)                     ((ilu_data) -> residual)
#define hypre_ParILUDataRelResNorms(ilu_data)                  ((ilu_data) -> rel_res_norms)
#define hypre_ParILUDataNumIterations(ilu_data)                ((ilu_data) -> num_iterations)
#define hypre_ParILUDataL1Norms(ilu_data)                      ((ilu_data) -> l1_norms)
#define hypre_ParILUDataFinalRelResidualNorm(ilu_data)         ((ilu_data) -> final_rel_residual_norm)
#define hypre_ParILUDataTol(ilu_data)                          ((ilu_data) -> tol)
#define hypre_ParILUDataOperatorComplexity(ilu_data)           ((ilu_data) -> operator_complexity)
#define hypre_ParILUDataLogging(ilu_data)                      ((ilu_data) -> logging)
#define hypre_ParILUDataPrintLevel(ilu_data)                   ((ilu_data) -> print_level)
#define hypre_ParILUDataMaxIter(ilu_data)                      ((ilu_data) -> max_iter)
#define hypre_ParILUDataTriSolve(ilu_data)                     ((ilu_data) -> tri_solve)
#define hypre_ParILUDataLowerJacobiIters(ilu_data)             ((ilu_data) -> lower_jacobi_iters)
#define hypre_ParILUDataUpperJacobiIters(ilu_data)             ((ilu_data) -> upper_jacobi_iters)
#define hypre_ParILUDataIluType(ilu_data)                      ((ilu_data) -> ilu_type)
#define hypre_ParILUDataNLU(ilu_data)                          ((ilu_data) -> nLU)
#define hypre_ParILUDataNI(ilu_data)                           ((ilu_data) -> nI)
#define hypre_ParILUDataUEnd(ilu_data)                         ((ilu_data) -> u_end)
#define hypre_ParILUDataUTemp(ilu_data)                        ((ilu_data) -> Utemp)
#define hypre_ParILUDataFTemp(ilu_data)                        ((ilu_data) -> Ftemp)
#define hypre_ParILUDataXTemp(ilu_data)                        ((ilu_data) -> Xtemp)
#define hypre_ParILUDataYTemp(ilu_data)                        ((ilu_data) -> Ytemp)
#define hypre_ParILUDataZTemp(ilu_data)                        ((ilu_data) -> Ztemp)
#define hypre_ParILUDataUExt(ilu_data)                         ((ilu_data) -> uext)
#define hypre_ParILUDataFExt(ilu_data)                         ((ilu_data) -> fext)
#define hypre_ParILUDataSchurSolver(ilu_data)                  ((ilu_data) -> schur_solver)
#define hypre_ParILUDataSchurPrecond(ilu_data)                 ((ilu_data) -> schur_precond)
#define hypre_ParILUDataRhs(ilu_data)                          ((ilu_data) -> rhs)
#define hypre_ParILUDataX(ilu_data)                            ((ilu_data) -> x)
#define hypre_ParILUDataReorderingType(ilu_data)               ((ilu_data) -> reordering_type)

/* Iterative ILU setup */
#define hypre_ParILUDataIterativeSetupType(ilu_data)           ((ilu_data) -> iter_setup_type)
#define hypre_ParILUDataIterativeSetupOption(ilu_data)         ((ilu_data) -> iter_setup_option)
#define hypre_ParILUDataIterativeSetupMaxIter(ilu_data)        ((ilu_data) -> setup_max_iter)
#define hypre_ParILUDataIterativeSetupNumIter(ilu_data)        ((ilu_data) -> setup_num_iter)
#define hypre_ParILUDataIterativeSetupTolerance(ilu_data)      ((ilu_data) -> setup_tolerance)
#define hypre_ParILUDataIterativeSetupHistory(ilu_data)        ((ilu_data) -> setup_history)
#define hypre_ParILUDataIterSetupCorrectionNorm(ilu_data,i)    ((ilu_data) -> setup_history[i])
#define hypre_ParILUDataIterSetupResidualNorm(ilu_data,i)      (((ilu_data) -> setup_history + \
                                                                 (ilu_data) -> setup_num_iter)[i])

/* Schur System */
#define hypre_ParILUDataSchurGMRESKDim(ilu_data)               ((ilu_data) -> ss_kDim)
#define hypre_ParILUDataSchurGMRESMaxIter(ilu_data)            ((ilu_data) -> ss_max_iter)
#define hypre_ParILUDataSchurGMRESTol(ilu_data)                ((ilu_data) -> ss_tol)
#define hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data)        ((ilu_data) -> ss_absolute_tol)
#define hypre_ParILUDataSchurGMRESRelChange(ilu_data)          ((ilu_data) -> ss_rel_change)
#define hypre_ParILUDataSchurPrecondIluType(ilu_data)          ((ilu_data) -> sp_ilu_type)
#define hypre_ParILUDataSchurPrecondIluLfil(ilu_data)          ((ilu_data) -> sp_ilu_lfil)
#define hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data)     ((ilu_data) -> sp_ilu_max_row_nnz)
#define hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)       ((ilu_data) -> sp_ilu_droptol)
#define hypre_ParILUDataSchurPrecondPrintLevel(ilu_data)       ((ilu_data) -> sp_print_level)
#define hypre_ParILUDataSchurPrecondMaxIter(ilu_data)          ((ilu_data) -> sp_max_iter)
#define hypre_ParILUDataSchurPrecondTriSolve(ilu_data)         ((ilu_data) -> sp_tri_solve)
#define hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) ((ilu_data) -> sp_lower_jacobi_iters)
#define hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) ((ilu_data) -> sp_upper_jacobi_iters)
#define hypre_ParILUDataSchurPrecondTol(ilu_data)              ((ilu_data) -> sp_tol)

#define hypre_ParILUDataSchurNSHMaxNumIter(ilu_data)           ((ilu_data) -> ss_nsh_setup_max_iter)
#define hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data)         ((ilu_data) -> ss_nsh_solve_max_iter)
#define hypre_ParILUDataSchurNSHTol(ilu_data)                  ((ilu_data) -> ss_nsh_setup_tol)
#define hypre_ParILUDataSchurNSHSolveTol(ilu_data)             ((ilu_data) -> ss_nsh_solve_tol)
#define hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data)            ((ilu_data) -> ss_nsh_max_row_nnz)
#define hypre_ParILUDataSchurMRColVersion(ilu_data)            ((ilu_data) -> ss_nsh_mr_col_version)
#define hypre_ParILUDataSchurMRMaxRowNnz(ilu_data)             ((ilu_data) -> ss_nsh_mr_max_row_nnz)
#define hypre_ParILUDataSchurNSHDroptol(ilu_data)              ((ilu_data) -> ss_nsh_droptol)
#define hypre_ParILUDataSchurMRMaxIter(ilu_data)               ((ilu_data) -> ss_nsh_mr_max_iter)
#define hypre_ParILUDataSchurMRTol(ilu_data)                   ((ilu_data) -> ss_nsh_mr_tol)

#define hypre_ParILUDataSchurSolverLogging(ilu_data)           ((ilu_data) -> ss_logging)
#define hypre_ParILUDataSchurSolverPrintLevel(ilu_data)        ((ilu_data) -> ss_print_level)

#define FMRK   -1
#define CMRK    1
#define UMRK    0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define MAT_TOL     1e-14
#define EXPAND_FACT 1.3

/* NSH data structure */

typedef struct hypre_ParNSHData_struct
{
   /* solver information */
   HYPRE_Int              global_solver;
   hypre_ParCSRMatrix    *matA;
   hypre_ParCSRMatrix    *matM;
   hypre_ParVector       *F;
   hypre_ParVector       *U;
   hypre_ParVector       *residual;
   HYPRE_Real            *rel_res_norms;
   HYPRE_Int              num_iterations;
   HYPRE_Real            *l1_norms;
   HYPRE_Real             final_rel_residual_norm;
   HYPRE_Real             tol;
   HYPRE_Real             operator_complexity;
   HYPRE_Int              logging;
   HYPRE_Int              print_level;
   HYPRE_Int              max_iter;

   /* common data slots */
   HYPRE_Real            *droptol; /* 2 drop torelances for {MR, NSH}*/
   HYPRE_Int              own_droptol_data;

   /* temp vectors for solve phase */
   hypre_ParVector       *Utemp;
   hypre_ParVector       *Ftemp;

   /* data slots for local MR */
   HYPRE_Int              mr_max_iter;
   HYPRE_Real             mr_tol;
   HYPRE_Int              mr_max_row_nnz;
   HYPRE_Int              mr_col_version; /* global version or column version MR */

   /* data slots for global NSH */
   HYPRE_Int              nsh_max_iter;
   HYPRE_Real             nsh_tol;
   HYPRE_Int              nsh_max_row_nnz;
} hypre_ParNSHData;

#define hypre_ParNSHDataGlobalSolver(nsh_data)           ((nsh_data) -> global_solver)
#define hypre_ParNSHDataMatA(nsh_data)                   ((nsh_data) -> matA)
#define hypre_ParNSHDataMatM(nsh_data)                   ((nsh_data) -> matM)
#define hypre_ParNSHDataF(nsh_data)                      ((nsh_data) -> F)
#define hypre_ParNSHDataU(nsh_data)                      ((nsh_data) -> U)
#define hypre_ParNSHDataResidual(nsh_data)               ((nsh_data) -> residual)
#define hypre_ParNSHDataRelResNorms(nsh_data)            ((nsh_data) -> rel_res_norms)
#define hypre_ParNSHDataNumIterations(nsh_data)          ((nsh_data) -> num_iterations)
#define hypre_ParNSHDataL1Norms(nsh_data)                ((nsh_data) -> l1_norms)
#define hypre_ParNSHDataFinalRelResidualNorm(nsh_data)   ((nsh_data) -> final_rel_residual_norm)
#define hypre_ParNSHDataTol(nsh_data)                    ((nsh_data) -> tol)
#define hypre_ParNSHDataOperatorComplexity(nsh_data)     ((nsh_data) -> operator_complexity)
#define hypre_ParNSHDataLogging(nsh_data)                ((nsh_data) -> logging)
#define hypre_ParNSHDataPrintLevel(nsh_data)             ((nsh_data) -> print_level)
#define hypre_ParNSHDataMaxIter(nsh_data)                ((nsh_data) -> max_iter)
#define hypre_ParNSHDataDroptol(nsh_data)                ((nsh_data) -> droptol)
#define hypre_ParNSHDataOwnDroptolData(nsh_data)         ((nsh_data) -> own_droptol_data)
#define hypre_ParNSHDataUTemp(nsh_data)                  ((nsh_data) -> Utemp)
#define hypre_ParNSHDataFTemp(nsh_data)                  ((nsh_data) -> Ftemp)
#define hypre_ParNSHDataMRMaxIter(nsh_data)              ((nsh_data) -> mr_max_iter)
#define hypre_ParNSHDataMRTol(nsh_data)                  ((nsh_data) -> mr_tol)
#define hypre_ParNSHDataMRMaxRowNnz(nsh_data)            ((nsh_data) -> mr_max_row_nnz)
#define hypre_ParNSHDataMRColVersion(nsh_data)           ((nsh_data) -> mr_col_version)
#define hypre_ParNSHDataNSHMaxIter(nsh_data)             ((nsh_data) -> nsh_max_iter)
#define hypre_ParNSHDataNSHTol(nsh_data)                 ((nsh_data) -> nsh_tol)
#define hypre_ParNSHDataNSHMaxRowNnz(nsh_data)           ((nsh_data) -> nsh_max_row_nnz)

#endif /* #ifndef hypre_ParILU_DATA_HEADER */
