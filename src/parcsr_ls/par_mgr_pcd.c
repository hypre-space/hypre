/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * MGR PCD (pressure convection-diffusion) coarse grid correction.
 *
 * See hypre_MGRPCDData in par_mgr.h for a description of the method. The data
 * object is created by hypre_MGRCreate and owned by the parent MGR solver;
 * users configure it through the HYPRE_MGRPCD* interface.
 *--------------------------------------------------------------------------*/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_mgr.h"

/*--------------------------------------------------------------------------
 * hypre_MGRPCDCreate (internal)
 *
 * Creates the PCD (pressure convection-diffusion) coarse grid correction
 * data. See hypre_MGRPCDData in par_mgr.h for a description of the method.
 * Allocated by hypre_MGRCreate; the object is owned and destroyed by the
 * parent MGR solver.
 *--------------------------------------------------------------------------*/

void *
hypre_MGRPCDCreate(void)
{
   hypre_MGRPCDData  *pcd_data = hypre_CTAlloc(hypre_MGRPCDData, 1, HYPRE_MEMORY_HOST);

   hypre_SolverSetup(&(pcd_data -> base))   = (HYPRE_PtrToSolverFcn) hypre_MGRPCDSetup;
   hypre_SolverSolve(&(pcd_data -> base))   = (HYPRE_PtrToSolverFcn) hypre_MGRPCDSolve;
   hypre_SolverDestroy(&(pcd_data -> base)) = (HYPRE_PtrToDestroyFcn) hypre_MGRPCDDestroy;
   hypre_SolverResetIsSetup(&(pcd_data -> base));

   (pcd_data -> Fp)              = NULL;
   (pcd_data -> Ap)              = NULL;
   (pcd_data -> Mp)              = NULL;
   (pcd_data -> apply_order)     = 0;
   (pcd_data -> mass_inv_type)   = 0;
   (pcd_data -> print_level)     = 0;
   (pcd_data -> ap_solver)       = NULL;
   (pcd_data -> ap_solver_owned) = 0;
   (pcd_data -> ap_solve)        = NULL;
   (pcd_data -> ap_setup)        = NULL;
   (pcd_data -> mp_solver)       = NULL;
   (pcd_data -> mp_solve)        = NULL;
   (pcd_data -> mp_setup)        = NULL;
   (pcd_data -> mass_diag)       = NULL;
   (pcd_data -> r_work)          = NULL;
   (pcd_data -> z_work)          = NULL;
   (pcd_data -> w_work)          = NULL;
   (pcd_data -> ap_setup_mat)    = NULL;
   (pcd_data -> mp_setup_mat)    = NULL;

   return (void *) pcd_data;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDDestroyInternals
 *
 * Frees data created during setup (work vectors, mass diagonal, and the
 * internally owned Ap hierarchy). Injected solvers are never destroyed.
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_MGRPCDDestroyInternals(hypre_MGRPCDData *pcd_data)
{
   if ((pcd_data -> ap_solver) && (pcd_data -> ap_solver_owned))
   {
      hypre_BoomerAMGDestroy((void *) (pcd_data -> ap_solver));
   }
   if (pcd_data -> ap_solver_owned)
   {
      (pcd_data -> ap_solver) = NULL;
      (pcd_data -> ap_solver_owned) = 0;
      (pcd_data -> ap_solve) = NULL;
      (pcd_data -> ap_setup) = NULL;
   }
   if (pcd_data -> mass_diag)
   {
      hypre_ParVectorDestroy(pcd_data -> mass_diag);
      (pcd_data -> mass_diag) = NULL;
   }
   if (pcd_data -> r_work)
   {
      hypre_ParVectorDestroy(pcd_data -> r_work);
      (pcd_data -> r_work) = NULL;
   }
   if (pcd_data -> z_work)
   {
      hypre_ParVectorDestroy(pcd_data -> z_work);
      (pcd_data -> z_work) = NULL;
   }
   if (pcd_data -> w_work)
   {
      hypre_ParVectorDestroy(pcd_data -> w_work);
      (pcd_data -> w_work) = NULL;
   }
   (pcd_data -> ap_setup_mat) = NULL;
   (pcd_data -> mp_setup_mat) = NULL;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDDestroy (internal)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRPCDDestroy(void *pcd_vdata)
{
   hypre_MGRPCDData  *pcd_data = (hypre_MGRPCDData *) pcd_vdata;

   if (pcd_data)
   {
      hypre_MGRPCDDestroyInternals(pcd_data);
      hypre_TFree(pcd_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDSetOperators
 *
 * Sets the pressure-space operators and enables the PCD coarse grid
 * correction. The matrices are not owned and must remain valid through
 * setup and solve; they must follow MGR's coarse grid partitioning.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRPCDSetOperators( void                *mgr_vdata,
                          hypre_ParCSRMatrix  *Fp,
                          hypre_ParCSRMatrix  *Ap,
                          hypre_ParCSRMatrix  *Mp )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData *) mgr_vdata;
   hypre_MGRPCDData  *pcd_data;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   pcd_data = (mgr_data -> pcd_data);
   (pcd_data -> Fp) = Fp;
   (pcd_data -> Ap) = Ap;
   (pcd_data -> Mp) = Mp;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDSetApSolver
 *
 * Injects a caller-owned solver for the pressure Laplacian (Ap) solves.
 * The solver is set up on Ap during MGR setup unless its base hypre_Solver
 * setup-reuse flag is raised, which lets callers preserve, e.g., an AMG
 * hierarchy across MGR rebuilds. Passing a NULL solver reverts to the
 * internal default (BoomerAMG).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRPCDSetApSolver( void *mgr_vdata, HYPRE_Solver ap_solver )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData *) mgr_vdata;
   hypre_MGRPCDData  *pcd_data;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* The solve/setup entry points are discovered through the base solver
    * struct that hypre solver objects embed as their first member. */
   if (ap_solver && !hypre_SolverSolve((hypre_Solver *) ap_solver))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "PCD Ap solver does not provide a base solve function!");
      return hypre_error_flag;
   }

   pcd_data = (mgr_data -> pcd_data);

   /* Release a previously built internal default */
   if ((pcd_data -> ap_solver) && (pcd_data -> ap_solver_owned))
   {
      hypre_BoomerAMGDestroy((void *) (pcd_data -> ap_solver));
   }

   (pcd_data -> ap_solver)       = ap_solver;
   (pcd_data -> ap_solver_owned) = 0;
   (pcd_data -> ap_solve)        = ap_solver ?
                                   (HYPRE_Int (*)(void*, void*, void*, void*))
                                   hypre_SolverSolve((hypre_Solver *) ap_solver) : NULL;
   (pcd_data -> ap_setup)        = ap_solver ?
                                   (HYPRE_Int (*)(void*, void*, void*, void*))
                                   hypre_SolverSetup((hypre_Solver *) ap_solver) : NULL;
   (pcd_data -> ap_setup_mat)    = NULL;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDSetMpSolver
 *
 * Injects a caller-owned solver for the pressure mass matrix (Mp) solves,
 * with the same setup-reuse semantics as the Ap solver. Passing a NULL
 * solver reverts to the internal default ((lumped) mass diagonal scaling,
 * see hypre_MGRPCDSetMassInvType).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRPCDSetMpSolver( void *mgr_vdata, HYPRE_Solver mp_solver )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData *) mgr_vdata;
   hypre_MGRPCDData  *pcd_data;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* The solve/setup entry points are discovered through the base solver
    * struct that hypre solver objects embed as their first member. */
   if (mp_solver && !hypre_SolverSolve((hypre_Solver *) mp_solver))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "PCD Mp solver does not provide a base solve function!");
      return hypre_error_flag;
   }

   pcd_data = (mgr_data -> pcd_data);
   (pcd_data -> mp_solver) = mp_solver;
   (pcd_data -> mp_solve)  = mp_solver ?
                             (HYPRE_Int (*)(void*, void*, void*, void*))
                             hypre_SolverSolve((hypre_Solver *) mp_solver) : NULL;
   (pcd_data -> mp_setup)  = mp_solver ?
                             (HYPRE_Int (*)(void*, void*, void*, void*))
                             hypre_SolverSetup((hypre_Solver *) mp_solver) : NULL;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDSetApplyOrder
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRPCDSetApplyOrder( void *mgr_vdata, HYPRE_Int apply_order )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData *) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   (mgr_data -> pcd_data -> apply_order) = apply_order;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDSetMassInvType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRPCDSetMassInvType( void *mgr_vdata, HYPRE_Int mass_inv_type )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData *) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   (mgr_data -> pcd_data -> mass_inv_type) = mass_inv_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDSetup (internal)
 *
 * Sets up the PCD coarse grid correction: validates the operators against
 * the coarse grid matrix A (MGR's RAP), prepares the Ap and Mp solvers, and
 * computes the (lumped) mass diagonal when no Mp solver is injected.
 *
 * Injected solvers are set up on their operator unless their setup-reuse
 * flag is raised. Internal components are rebuilt only when their operator
 * pointer changed (Fp typically changes every nonlinear iteration while Ap
 * and Mp are constant, and Fp needs no setup work at all).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRPCDSetup( void                *pcd_vdata,
                   hypre_ParCSRMatrix  *A,
                   hypre_ParVector     *f,
                   hypre_ParVector     *u )
{
   hypre_MGRPCDData     *pcd_data = (hypre_MGRPCDData *) pcd_vdata;

   hypre_ParCSRMatrix   *Fp, *Ap, *Mp;
   HYPRE_MemoryLocation  memory_location;
   HYPRE_Solver          amg_solver;
   MPI_Comm              comm;
   HYPRE_Int             my_id;

   HYPRE_UNUSED_VAR(f);
   HYPRE_UNUSED_VAR(u);

   if (!pcd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   Fp = (pcd_data -> Fp);
   Ap = (pcd_data -> Ap);
   Mp = (pcd_data -> Mp);

   if (!Fp || !Ap || !Mp)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "MGR PCD coarse solver requires Fp, Ap, and Mp operators!");
      return hypre_error_flag;
   }

   /* Validate partitioning against the MGR coarse grid matrix */
   if (A)
   {
      if (hypre_ParCSRMatrixGlobalNumRows(A) != hypre_ParCSRMatrixGlobalNumRows(Ap) ||
          hypre_ParCSRMatrixFirstRowIndex(A) != hypre_ParCSRMatrixFirstRowIndex(Ap) ||
          hypre_ParCSRMatrixNumRows(A)       != hypre_ParCSRMatrixNumRows(Ap))
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "MGR PCD operators do not match the coarse grid partitioning!");
         return hypre_error_flag;
      }
   }

   comm            = hypre_ParCSRMatrixComm(Ap);
   memory_location = hypre_ParCSRMatrixMemoryLocation(Ap);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Work vectors (and internal components) follow Ap's partitioning */
   HYPRE_Int rebuild_vec = (!(pcd_data -> r_work) || (pcd_data -> ap_setup_mat) != Ap);
   HYPRE_Int rebuild_ap  = rebuild_vec && (pcd_data -> ap_solver_owned ||
                                           !(pcd_data -> ap_solver));
   HYPRE_Int rebuild_mp  = (!(pcd_data -> mass_diag) || (pcd_data -> mp_setup_mat) != Mp);

   if (rebuild_vec)
   {
      if ((pcd_data -> ap_solver) && (pcd_data -> ap_solver_owned))
      {
         hypre_BoomerAMGDestroy((void *) (pcd_data -> ap_solver));
         (pcd_data -> ap_solver)       = NULL;
         (pcd_data -> ap_solver_owned) = 0;
      }
      if (pcd_data -> mass_diag) { hypre_ParVectorDestroy(pcd_data -> mass_diag); }
      if (pcd_data -> r_work)    { hypre_ParVectorDestroy(pcd_data -> r_work); }
      if (pcd_data -> z_work)    { hypre_ParVectorDestroy(pcd_data -> z_work); }
      if (pcd_data -> w_work)    { hypre_ParVectorDestroy(pcd_data -> w_work); }
      rebuild_mp = 1;

      (pcd_data -> mass_diag) = hypre_ParVectorCreate(comm,
                                                      hypre_ParCSRMatrixGlobalNumRows(Ap),
                                                      hypre_ParCSRMatrixRowStarts(Ap));
      (pcd_data -> r_work)    = hypre_ParVectorCreate(comm,
                                                      hypre_ParCSRMatrixGlobalNumRows(Ap),
                                                      hypre_ParCSRMatrixRowStarts(Ap));
      (pcd_data -> z_work)    = hypre_ParVectorCreate(comm,
                                                      hypre_ParCSRMatrixGlobalNumRows(Ap),
                                                      hypre_ParCSRMatrixRowStarts(Ap));
      (pcd_data -> w_work)    = hypre_ParVectorCreate(comm,
                                                      hypre_ParCSRMatrixGlobalNumRows(Ap),
                                                      hypre_ParCSRMatrixRowStarts(Ap));
      hypre_ParVectorInitialize_v2(pcd_data -> mass_diag, memory_location);
      hypre_ParVectorInitialize_v2(pcd_data -> r_work, memory_location);
      hypre_ParVectorInitialize_v2(pcd_data -> z_work, memory_location);
      hypre_ParVectorInitialize_v2(pcd_data -> w_work, memory_location);
   }

   /* Mp solver: injected (honoring setup reuse) or (lumped) mass diagonal.
    * mass_diag is always kept current so that reverting to diagonal scaling via
    * SetMpSolver(NULL) between setups does not produce stale zeros. */
   if (pcd_data -> mp_solver)
   {
      if ((pcd_data -> mp_setup) &&
          !hypre_SolverSetupReuseRequested((hypre_Solver *) (pcd_data -> mp_solver)))
      {
         (pcd_data -> mp_setup)((void *) (pcd_data -> mp_solver), Mp,
                                pcd_data -> r_work, pcd_data -> z_work);
      }
   }
   if (rebuild_mp)
   {
      if ((pcd_data -> mass_inv_type) == 1)
      {
         /* Diagonal of Mp */
         hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(Mp),
                                        hypre_VectorData(hypre_ParVectorLocalVector(
                                                            pcd_data -> mass_diag)), 0);
      }
      else
      {
         /* Lumped mass: Mp * ones */
         hypre_ParVectorSetConstantValues(pcd_data -> z_work, 1.0);
         hypre_ParCSRMatrixMatvec(1.0, Mp, pcd_data -> z_work, 0.0,
                                  pcd_data -> mass_diag);
      }
      (pcd_data -> mp_setup_mat) = Mp;
   }

   /* Ap solver: injected (honoring setup reuse) or internal default AMG */
   if ((pcd_data -> ap_solver) && !(pcd_data -> ap_solver_owned))
   {
      if ((pcd_data -> ap_setup) &&
          !hypre_SolverSetupReuseRequested((hypre_Solver *) (pcd_data -> ap_solver)))
      {
         (pcd_data -> ap_setup)((void *) (pcd_data -> ap_solver), Ap,
                                pcd_data -> r_work, pcd_data -> z_work);
      }
   }
   else if (rebuild_ap)
   {
      amg_solver = (HYPRE_Solver) hypre_BoomerAMGCreate();
      hypre_BoomerAMGSetMaxIter(amg_solver, 1);
      hypre_BoomerAMGSetTol(amg_solver, 0.0);
      hypre_BoomerAMGSetPrintLevel(amg_solver, 0);
      hypre_BoomerAMGSetStrongThreshold(amg_solver, 0.25);
      hypre_BoomerAMGSetCoarsenType(amg_solver, 10);  /* HMIS */
      hypre_BoomerAMGSetInterpType(amg_solver, 6);    /* extended+i */
      hypre_BoomerAMGSetAggNumLevels(amg_solver, 1);
      hypre_BoomerAMGSetup((void *) amg_solver, Ap, pcd_data -> r_work,
                           pcd_data -> z_work);
      (pcd_data -> ap_solver)       = amg_solver;
      (pcd_data -> ap_solver_owned) = 1;
      (pcd_data -> ap_solve) =
         (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSolve;
      (pcd_data -> ap_setup) =
         (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSetup;
   }
   (pcd_data -> ap_setup_mat) = Ap;

   if ((pcd_data -> print_level) > 0 && my_id == 0)
   {
      hypre_printf("MGR PCD coarse solver: n = %b, apply_order = %d, "
                   "ap solver %s, mp solver %s\n",
                   hypre_ParCSRMatrixGlobalNumRows(Ap),
                   (pcd_data -> apply_order),
                   (pcd_data -> ap_solver_owned) ? "internal" : "injected",
                   (pcd_data -> mp_solver) ? "injected" :
                   ((pcd_data -> mass_inv_type) == 1 ? "diag(Mp)" : "lumped"));
   }

   hypre_SolverSetIsSetup(&(pcd_data -> base));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRPCDSolve (internal)
 *
 * Applies the PCD approximation in residual-correction form:
 *
 *    u <- u + Mp^{-1} Fp Ap^{-1} (f - A u)    (apply_order = 0)
 *    u <- u + Ap^{-1} Fp Mp^{-1} (f - A u)    (apply_order = 1)
 *
 * A is MGR's coarse grid matrix; when A is NULL, u is assumed zero and
 * the raw right-hand side is used as residual.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRPCDSolve( void                *pcd_vdata,
                   hypre_ParCSRMatrix  *A,
                   hypre_ParVector     *f,
                   hypre_ParVector     *u )
{
   hypre_MGRPCDData     *pcd_data = (hypre_MGRPCDData *) pcd_vdata;

   hypre_ParCSRMatrix   *Fp, *Ap, *Mp;
   hypre_ParVector      *r_work, *z_work, *w_work, *mass_diag;

   if (!pcd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   Fp        = (pcd_data -> Fp);
   Ap        = (pcd_data -> Ap);
   Mp        = (pcd_data -> Mp);
   r_work    = (pcd_data -> r_work);
   z_work    = (pcd_data -> z_work);
   w_work    = (pcd_data -> w_work);
   mass_diag = (pcd_data -> mass_diag);

   if (!(pcd_data -> ap_solver) || !(pcd_data -> ap_solve) || !r_work)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "MGR PCD solver has not been set up!");
      return hypre_error_flag;
   }

   /* r = f - A u */
   if (A)
   {
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u, 1.0, f, r_work);
   }
   else
   {
      hypre_ParVectorCopy(f, r_work);
   }

   if ((pcd_data -> apply_order) == 0)
   {
      /* u += Mp^{-1} Fp Ap^{-1} r */
      hypre_ParVectorSetZeros(z_work);
      (pcd_data -> ap_solve)((void *) (pcd_data -> ap_solver), Ap, r_work, z_work);
      hypre_ParCSRMatrixMatvec(1.0, Fp, z_work, 0.0, w_work);
      if (pcd_data -> mp_solver)
      {
         /* r_work is free at this point; reuse it for the Mp solve */
         hypre_ParVectorSetZeros(r_work);
         (pcd_data -> mp_solve)((void *) (pcd_data -> mp_solver), Mp, w_work, r_work);
         hypre_ParVectorAxpy(1.0, r_work, u);
      }
      else
      {
         hypre_ParVectorPointwiseDivpy(w_work, mass_diag, u);
      }
   }
   else
   {
      /* u += Ap^{-1} Fp Mp^{-1} r */
      hypre_ParVectorSetZeros(w_work);
      if (pcd_data -> mp_solver)
      {
         (pcd_data -> mp_solve)((void *) (pcd_data -> mp_solver), Mp, r_work, w_work);
      }
      else
      {
         hypre_ParVectorPointwiseDivpy(r_work, mass_diag, w_work);
      }
      hypre_ParCSRMatrixMatvec(1.0, Fp, w_work, 0.0, z_work);
      hypre_ParVectorSetZeros(r_work);
      (pcd_data -> ap_solve)((void *) (pcd_data -> ap_solver), Ap, z_work, r_work);
      hypre_ParVectorAxpy(1.0, r_work, u);
   }

   return hypre_error_flag;
}
