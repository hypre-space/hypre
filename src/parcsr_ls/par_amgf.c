/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMGF functions
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amgf.h" // can this be removed by updating the _hypre_parcsr_ls.h file??
/*--------------------------------------------------------------------------
 * hypre_AMGFCreate
 *--------------------------------------------------------------------------*/

//HYPRE_Int 
void * 
hypre_AMGFCreate()//MPI_Comm comm, HYPRE_Solver * solver)
{
   hypre_ParAMGFData * amgf_data;
   hypre_Solver      *base;

   /*-----------------------------------------------------------------------
    * Create the hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amgf_data = hypre_CTAlloc(hypre_ParAMGFData, 1, HYPRE_MEMORY_HOST);
   
   amgf_data->set_mask = 0;   
   amgf_data->set_coarse_solver = 0;
   amgf_data->set_amg_solver = 0;
   
   
   base     = (hypre_Solver*) amgf_data;

   /* Set base solver function pointers */
   //hypre_SolverSetup(base)   = (HYPRE_PtrToSolverFcn)  HYPRE_BoomerAMGSetup;
   //hypre_SolverSolve(base)   = (HYPRE_PtrToSolverFcn)  HYPRE_BoomerAMGSolve;
   //hypre_SolverDestroy(base) = (HYPRE_PtrToDestroyFcn) HYPRE_BoomerAMGDestroy;

   return amgf_data;
};
