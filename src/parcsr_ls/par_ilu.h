/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
   //general data
   HYPRE_Int            global_solver;
   hypre_ParCSRMatrix   *matA;
   hypre_ParCSRMatrix   *matL;
   HYPRE_Real           *matD;
   hypre_ParCSRMatrix   *matU;
   hypre_ParCSRMatrix   *matS;
   HYPRE_Real           *droptol;/* should be an array of 3 element, for B, (E and F), S respectively */
   HYPRE_Int            own_droptol_data;/* should I free droptols */
   HYPRE_Int            lfil;
   HYPRE_Int            maxRowNnz;
   HYPRE_Int            *CF_marker_array;
   HYPRE_Int            *perm;
   HYPRE_Int            *qperm;
   HYPRE_Real           tol_ddPQ;
   hypre_ParVector      *F;
   hypre_ParVector      *U;
   hypre_ParVector      *residual;
   HYPRE_Real           *rel_res_norms;  
   HYPRE_Int            num_iterations;
   HYPRE_Real           *l1_norms;
   HYPRE_Real           final_rel_residual_norm;
   HYPRE_Real           tol;
   HYPRE_Real           operator_complexity;
   
   HYPRE_Int            logging;
   HYPRE_Int            print_level;
   HYPRE_Int            max_iter;
   
   HYPRE_Int            ilu_type;
   HYPRE_Int            nLU;
   HYPRE_Int            nI;
   
   /* used when schur block is formed */
   HYPRE_Int            *u_end;
 
   /* temp vectors for solve phase */
   hypre_ParVector      *Utemp;
   hypre_ParVector      *Ftemp;
   HYPRE_Real           *uext;
   HYPRE_Real           *fext;
   
   /* data structure sor solving Schur System */
   HYPRE_Solver         schur_solver;
   HYPRE_Solver         schur_precond;
   hypre_ParVector      *rhs;
   hypre_ParVector      *x;
   
   /* schur solver data */
   HYPRE_Int            ss_kDim;/* dim and max number of iterations for GMRES or max number of iterations for NSH inverse */
   HYPRE_Int            ss_max_iter;/* max number of iterations for NSH solve */
   HYPRE_Real           ss_tol;/* stop iteration tol for GMRES or NSH inverse */
   HYPRE_Real           ss_absolute_tol;/* absolute tol for GMRES or tol for NSH solve */
   HYPRE_Int            ss_logging;
   HYPRE_Int            ss_print_level;
   HYPRE_Int            ss_rel_change;
   
   /* schur precond data */
   HYPRE_Int            sp_ilu_type;/* ilu type is use ILU, or max rows of nonzeros for NSH */
   HYPRE_Int            sp_ilu_lfil;/* level of fill in for ILUK or MR column version setting*/
   HYPRE_Int            sp_ilu_max_row_nnz;/* max rows for ILUT or MR  */
   /* droptol for ILUT or MR 
    * ILUT: [0], [1], [2] B, E&F, S respectively
    * NSH: [0] for MR, [1] for NSH
    */
   HYPRE_Real           *sp_ilu_droptol;/* droptol array for ILUT or NSH */
   HYPRE_Int            sp_own_droptol_data;
   HYPRE_Int            sp_print_level;
   HYPRE_Int            sp_max_iter;/* max precond iter or max MR iteration */
   HYPRE_Real           sp_tol;
   
   /* local reordering */
   HYPRE_Int 	reordering_type;
   
} hypre_ParILUData;

#define hypre_ParILUDataGlobalSolver(ilu_data)                 ((ilu_data) -> global_solver)
#define hypre_ParILUDataMatA(ilu_data)                         ((ilu_data) -> matA)
#define hypre_ParILUDataMatL(ilu_data)                         ((ilu_data) -> matL)
#define hypre_ParILUDataMatD(ilu_data)                         ((ilu_data) -> matD)
#define hypre_ParILUDataMatU(ilu_data)                         ((ilu_data) -> matU)
#define hypre_ParILUDataMatS(ilu_data)                         ((ilu_data) -> matS)
#define hypre_ParILUDataDroptol(ilu_data)                      ((ilu_data) -> droptol)
#define hypre_ParILUDataOwnDroptolData(ilu_data)               ((ilu_data) -> own_droptol_data)
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
#define hypre_ParILUDataIluType(ilu_data)                      ((ilu_data) -> ilu_type)
#define hypre_ParILUDataNLU(ilu_data)                          ((ilu_data) -> nLU)
#define hypre_ParILUDataNI(ilu_data)                           ((ilu_data) -> nI)
#define hypre_ParILUDataUEnd(ilu_data)                         ((ilu_data) -> u_end)
#define hypre_ParILUDataUTemp(ilu_data)                        ((ilu_data) -> Utemp)
#define hypre_ParILUDataFTemp(ilu_data)                        ((ilu_data) -> Ftemp)
#define hypre_ParILUDataUExt(ilu_data)                         ((ilu_data) -> uext)
#define hypre_ParILUDataFExt(ilu_data)                         ((ilu_data) -> fext)
#define hypre_ParILUDataSchurSolver(ilu_data)                  ((ilu_data) -> schur_solver)
#define hypre_ParILUDataSchurPrecond(ilu_data)                 ((ilu_data) -> schur_precond)
#define hypre_ParILUDataRhs(ilu_data)                          ((ilu_data) -> rhs)
#define hypre_ParILUDataX(ilu_data)                            ((ilu_data) -> x)
#define hypre_ParILUDataReorderingType(ilu_data)                            ((ilu_data) -> reordering_type)
/* Schur System */
#define hypre_ParILUDataSchurGMRESKDim(ilu_data)               ((ilu_data) -> ss_kDim)
#define hypre_ParILUDataSchurNSHMaxNumIter(ilu_data)           ((ilu_data) -> ss_kDim)
#define hypre_ParILUDataSchurGMRESMaxIter(ilu_data)            ((ilu_data) -> ss_kDim)
#define hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data)         ((ilu_data) -> ss_max_iter)
#define hypre_ParILUDataSchurGMRESTol(ilu_data)                ((ilu_data) -> ss_tol)
#define hypre_ParILUDataSchurNSHTol(ilu_data)                  ((ilu_data) -> ss_tol)
#define hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data)        ((ilu_data) -> ss_absolute_tol)
#define hypre_ParILUDataSchurNSHSolveTol(ilu_data)             ((ilu_data) -> ss_absolute_tol)
#define hypre_ParILUDataSchurSolverLogging(ilu_data)           ((ilu_data) -> ss_logging)
#define hypre_ParILUDataSchurSolverPrintLevel(ilu_data)        ((ilu_data) -> ss_print_level)
#define hypre_ParILUDataSchurGMRESRelChange(ilu_data)          ((ilu_data) -> ss_rel_change)
#define hypre_ParILUDataSchurPrecondIluType(ilu_data)          ((ilu_data) -> sp_ilu_type)
#define hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data)            ((ilu_data) -> sp_ilu_type)
#define hypre_ParILUDataSchurPrecondIluLfil(ilu_data)          ((ilu_data) -> sp_ilu_lfil)
#define hypre_ParILUDataSchurMRColVersion(ilu_data)            ((ilu_data) -> sp_ilu_lfil)
#define hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data)     ((ilu_data) -> sp_ilu_max_row_nnz)
#define hypre_ParILUDataSchurMRMaxRowNnz(ilu_data)             ((ilu_data) -> sp_ilu_max_row_nnz)
#define hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)       ((ilu_data) -> sp_ilu_droptol)
#define hypre_ParILUDataSchurNSHDroptol(ilu_data)              ((ilu_data) -> sp_ilu_droptol)
#define hypre_ParILUDataSchurPrecondOwnDroptolData(ilu_data)   ((ilu_data) -> sp_own_droptol_data)
#define hypre_ParILUDataSchurNSHOwnDroptolData(ilu_data)       ((ilu_data) -> sp_own_droptol_data)
#define hypre_ParILUDataSchurPrecondPrintLevel(ilu_data)       ((ilu_data) -> sp_print_level)
#define hypre_ParILUDataSchurPrecondMaxIter(ilu_data)          ((ilu_data) -> sp_max_iter)
#define hypre_ParILUDataSchurMRMaxIter(ilu_data)               ((ilu_data) -> sp_max_iter)
#define hypre_ParILUDataSchurPrecondTol(ilu_data)              ((ilu_data) -> sp_tol)
#define hypre_ParILUDataSchurMRTol(ilu_data)                   ((ilu_data) -> sp_tol)


#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define MAT_TOL 1e-14
#define EXPAND_FACT 1.3





/* NSH data structure */

typedef struct hypre_ParNSHData_struct
{
   /* solver information */
   HYPRE_Int             global_solver;
   hypre_ParCSRMatrix    *matA;
   hypre_ParCSRMatrix    *matM;
   hypre_ParVector       *F;
   hypre_ParVector       *U;
   hypre_ParVector       *residual;
   HYPRE_Real            *rel_res_norms;  
   HYPRE_Int             num_iterations;
   HYPRE_Real            *l1_norms;
   HYPRE_Real            final_rel_residual_norm;
   HYPRE_Real            tol;
   HYPRE_Real            operator_complexity;
   
   HYPRE_Int             logging;
   HYPRE_Int             print_level;
   HYPRE_Int             max_iter;
   
   /* common data slots */
   /* droptol[0]: droptol for MR 
    * droptol[1]: droptol for NSH
    */
   HYPRE_Real            *droptol;
   HYPRE_Int             own_droptol_data;
   
   /* temp vectors for solve phase */
   hypre_ParVector       *Utemp;
   hypre_ParVector       *Ftemp;
   
   /* data slots for local MR */
   HYPRE_Int             mr_max_iter;
   HYPRE_Real            mr_tol;
   HYPRE_Int             mr_max_row_nnz;
   HYPRE_Int             mr_col_version;/* global version or column version MR */
   
   /* data slots for global NSH */
   HYPRE_Int             nsh_max_iter;
   HYPRE_Real            nsh_tol;
   HYPRE_Int             nsh_max_row_nnz;
}hypre_ParNSHData;

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

#define DIVIDE_TOL 1e-32

#endif
