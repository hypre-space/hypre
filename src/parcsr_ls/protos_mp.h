/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Mixed precision function protos */
/* parcsr_ls_mp.c */

#ifdef HYPRE_MIXED_PRECISION

/* par_cycle_mp.c */
HYPRE_Int hypre_MPAMGCycle_mp( void *amg_vdata, hypre_ParVector **F_array, hypre_ParVector **U_array );

/* par_mpamg_mp.c */
void * hypre_MPAMGCreate_mp( void );
HYPRE_Int hypre_MPAMGDestroy_mp( void *data );
HYPRE_Int hypre_MPAMGSetPrecisionArray_mp( void *data, HYPRE_Precision *precision_array);

/* par_mpamg_setup_mp.c */
HYPRE_Int hypre_MPAMGSetup_mp( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u );

/* par_mpamg_solve_mp.c */
HYPRE_Int hypre_MPAMGSolve_mp( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u );

/* par_stats_mp.c */
HYPRE_Int hypre_MPAMGSetupStats_mp( void *amg_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_MPAMGWriteSolverParams_mp(void* data);

#endif
