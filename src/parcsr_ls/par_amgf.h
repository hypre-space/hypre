/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParAMGF_DATA_HEADER
#define hypre_ParAMGF_DATA_HEADER

typedef struct 
{
  MPI_Comm comm;

  HYPRE_Solver * coarse_solver;
  HYPRE_Int (*coarse_solve_setup)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
  HYPRE_Int (*coarse_solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector); 
  HYPRE_Int set_coarse_solver;
  
  HYPRE_Solver * amg;
  HYPRE_Int (*amg_solve_setup)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
  HYPRE_Int (*amg_solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector); 
  HYPRE_Int set_amg_solver;

  HYPRE_ParCSRMatrix Ac;
  HYPRE_IJMatrix Pij;
  HYPRE_ParCSRMatrix P;
  HYPRE_IJMatrix Rij;
  HYPRE_ParCSRMatrix R;
  
  HYPRE_BigInt ilower, iupper, cilower, ciupper;
  HYPRE_BigInt local_size, clocal_size;

  HYPRE_BigInt * ipartition;
  HYPRE_BigInt * cpartition;

  HYPRE_BigInt global_n;
  HYPRE_BigInt global_m;
  
  /* work vectors */
  HYPRE_ParVector r;
  HYPRE_ParVector rc;
  HYPRE_ParVector ec;
  HYPRE_ParVector e;

  HYPRE_Int * constraint_mask;
  HYPRE_Int set_mask;

} hypre_ParAMGFData;

void * hypre_AMGFCreate();


#endif
