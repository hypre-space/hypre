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
#include "_hypre_parcsr_ls.h"
#include "par_mgr.h"
#include "par_amg.h"

/* Setup MGR data */
HYPRE_Int
hypre_MGRSetup( void               *mgr_vdata,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{
	MPI_Comm 	         comm = hypre_ParCSRMatrixComm(A);
	hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;

	HYPRE_Int       j, final_coarse_size, block_size, idx, row, size, *cols = NULL, *block_cf_marker, *additional_coarse_indices;
	HYPRE_Int	   lev, num_coarsening_levs, last_level, num_c_levels, num_threads, gnumrows;
	HYPRE_Int	   debug_flag = 0, old_coarse_size, coarse_size_diff;
	HYPRE_Int      ierr;

	hypre_ParCSRMatrix  *RT = NULL;
	hypre_ParCSRMatrix  *P = NULL;
	hypre_ParCSRMatrix  *S = NULL;
	hypre_ParCSRMatrix  *ST = NULL;
	hypre_ParCSRMatrix  *AT = NULL;
  hypre_ParCSRMatrix  *RT_f = NULL;
  hypre_ParCSRMatrix  *P_f = NULL;

	HYPRE_Int * col_offd_S_to_A = NULL;
	HYPRE_Int * col_offd_ST_to_AT = NULL;
	HYPRE_Int * dof_func_buff = NULL;
	HYPRE_Int * coarse_pnts_global = NULL;
	HYPRE_Real         **l1_norms = NULL;

	hypre_ParVector     *Ztemp;
	hypre_ParVector     *Vtemp;
	hypre_ParVector     *Utemp;
	hypre_ParVector     *Ftemp;

	/* pointers to mgr data */
	HYPRE_Int  use_default_cgrid_solver = (mgr_data -> use_default_cgrid_solver);
	HYPRE_Int  logging = (mgr_data -> logging);
	HYPRE_Int  print_level = (mgr_data -> print_level);
	HYPRE_Int  relax_type = (mgr_data -> relax_type);
	HYPRE_Int  relax_order = (mgr_data -> relax_order);
	HYPRE_Int  interp_type = (mgr_data -> interp_type);
  HYPRE_Int  restrict_type = (mgr_data -> restrict_type);
	HYPRE_Int num_interp_sweeps = (mgr_data -> num_interp_sweeps);
	HYPRE_Int num_restrict_sweeps = (mgr_data -> num_interp_sweeps);
	HYPRE_Int	max_elmts = (mgr_data -> P_max_elmts);
	HYPRE_Real   max_row_sum = (mgr_data -> max_row_sum);
	HYPRE_Real   strong_threshold = (mgr_data -> strong_threshold);
	HYPRE_Real   trunc_factor = (mgr_data -> trunc_factor);
	HYPRE_Real   S_commpkg_switch = (mgr_data -> S_commpkg_switch);
	HYPRE_Int  old_num_coarse_levels = (mgr_data -> num_coarse_levels);
	HYPRE_Int  max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
	HYPRE_Int * final_coarse_indexes = (mgr_data -> final_coarse_indexes);
	HYPRE_Int ** CF_marker_array = (mgr_data -> CF_marker_array);
	hypre_ParCSRMatrix  **A_array = (mgr_data -> A_array);
  hypre_ParCSRMatrix  **A_ff_array = (mgr_data -> A_ff_array);
	hypre_ParCSRMatrix  **P_array = (mgr_data -> P_array);
  hypre_ParCSRMatrix  **P_f_array = (mgr_data -> P_f_array);
	hypre_ParCSRMatrix  **RT_array = (mgr_data -> RT_array);
	hypre_ParCSRMatrix  *RAP_ptr = NULL;
  hypre_ParCSRMatrix  *A_ff_ptr = NULL;
  HYPRE_Solver *aff_solver = NULL;

	hypre_ParVector    **F_array = (mgr_data -> F_array);
	hypre_ParVector    **U_array = (mgr_data -> U_array);
  hypre_ParVector    **F_fine_array = (mgr_data -> F_fine_array);
  hypre_ParVector    **U_fine_array = (mgr_data -> U_fine_array);
	hypre_ParVector    *residual = (mgr_data -> residual);
	HYPRE_Real    *rel_res_norms = (mgr_data -> rel_res_norms);

	HYPRE_Solver    	*default_cg_solver;
   HYPRE_Int   (*fine_grid_solver_setup)(void*,void*,void*,void*);
   HYPRE_Int   (*fine_grid_solver_solve)(void*,void*,void*,void*);
	HYPRE_Int	(*coarse_grid_solver_setup)(void*,void*,void*,void*) = (HYPRE_Int (*)(void*, void*, void*, void*)) (mgr_data -> coarse_grid_solver_setup);
	HYPRE_Int	(*coarse_grid_solver_solve)(void*,void*,void*,void*) = (HYPRE_Int (*)(void*, void*, void*, void*)) (mgr_data -> coarse_grid_solver_solve);

	HYPRE_Int    global_smooth      =  (mgr_data -> global_smooth);
	HYPRE_Int    global_smooth_type =  (mgr_data -> global_smooth_type);

	HYPRE_Int    reserved_coarse_size = (mgr_data -> reserved_coarse_size);

	HYPRE_Real          *diaginv = (mgr_data -> diaginv);
	HYPRE_Int		   num_procs,  my_id;
	hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
	HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);
	HYPRE_Int    blk_size  = (mgr_data -> block_size);
  HYPRE_Int    *num_coarse_points = (mgr_data -> tmp_num_coarse_points);
  HYPRE_Int num_additional_coarse_indices = (mgr_data -> num_additional_coarse_indices);
  HYPRE_Int splitting_strategy = (mgr_data -> splitting_strategy);
  HYPRE_Int coarse_size = 0;


	/* ----- begin -----*/

	num_threads = hypre_NumThreads();

  block_size = (mgr_data -> block_size);
  block_cf_marker = (mgr_data -> block_cf_marker);
  HYPRE_Int **tmp_block_cf_marker = (mgr_data -> tmp_block_cf_marker);
  additional_coarse_indices = (mgr_data -> additional_coarse_indices);
  HYPRE_Int **coarse_indices_lvls = (mgr_data -> coarse_indices_lvls);

   if (print_level > 0)
   {
      hypre_printf("Solver info: \n");
      hypre_printf("Relax type: %d\n", relax_type);
      hypre_printf("Splitting Strategy: %d\n", splitting_strategy);
      hypre_printf("Number of relax sweeps: %d\n", (mgr_data -> num_relax_sweeps));
      hypre_printf("Interpolation type: %d\n", interp_type);
      hypre_printf("Number of interpolation sweeps: %d\n", num_interp_sweeps);
      hypre_printf("Restriction type: %d\n", restrict_type);
      hypre_printf("Max number of iterations: %d\n", (mgr_data -> max_iter));
      hypre_printf("Number of coarse levels: %d\n", (mgr_data -> num_coarse_levels));
      hypre_printf("Max number of coarse levels: %d\n", (mgr_data -> max_num_coarse_levels));
      hypre_printf("Tolerance: %e\n", (mgr_data -> conv_tol));
   }

	gnumrows = hypre_ParCSRMatrixGlobalNumRows(A);
	if(((gnumrows-reserved_coarse_size) % block_size) != 0)
	{
		hypre_printf("ERROR: Global number of rows minus reserved_coarse_grid_size must be a multiple of block_size ... n = %d, reserved_coarse_size = %d, block_size = %d \n", gnumrows,reserved_coarse_size, block_size);
		hypre_MPI_Abort(comm, -1);
	}
        
        /* Trivial case: simply solve the coarse level problem */
	if( block_size < 2 || (mgr_data -> max_num_coarse_levels) < 1)
	{
		hypre_printf("Warning: Block size is < 2 or number of coarse levels is < 1. \n");
		hypre_printf("Solving scalar problem on fine grid using coarse level solver \n");

		if(use_default_cgrid_solver)
		{
			hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
			/* create and set default solver parameters here */
			/* create and initialize default_cg_solver */
			default_cg_solver = (HYPRE_Solver *) hypre_BoomerAMGCreate();
			hypre_BoomerAMGSetMaxIter ( default_cg_solver, (mgr_data -> max_iter) );

			hypre_BoomerAMGSetRelaxOrder( default_cg_solver, 0);
			hypre_BoomerAMGSetPrintLevel(default_cg_solver, 0);
			/* set setup and solve functions */
			coarse_grid_solver_setup = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSetup;
			coarse_grid_solver_solve = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSolve;
			(mgr_data -> coarse_grid_solver_setup) = coarse_grid_solver_setup;
			(mgr_data -> coarse_grid_solver_solve) = coarse_grid_solver_solve;
			(mgr_data -> coarse_grid_solver) = default_cg_solver;
		}
		/* setup coarse grid solver */
		coarse_grid_solver_setup((mgr_data -> coarse_grid_solver), A, f, u);
		(mgr_data -> max_num_coarse_levels) = 0;

		return hypre_error_flag;
	}

	/* setup default block data if not set. Use first index as C-point */
	if((mgr_data -> block_cf_marker)==NULL)
	{
		(mgr_data -> block_cf_marker) = hypre_CTAlloc(HYPRE_Int,block_size);
		memset((mgr_data -> block_cf_marker), FMRK, block_size*sizeof(HYPRE_Int));
		(mgr_data -> block_cf_marker)[0] = CMRK;
	}

  /* Initialize local indexes of coarse sets at different levels */
  HYPRE_Int num_coarse_levels = (mgr_data -> num_coarse_levels);
  if ((mgr_data -> coarse_indices_lvls) != NULL) {
    for (j = 0; j < num_coarse_levels; j++)
      if ((mgr_data -> coarse_indices_lvls)[j])
        hypre_TFree((mgr_data -> coarse_indices_lvls)[j]);
    hypre_TFree((mgr_data -> coarse_indices_lvls));
  }

	HYPRE_Int nloc =  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
	HYPRE_Int ilower =  hypre_ParCSRMatrixFirstRowIndex(A);
	HYPRE_Int iupper =  hypre_ParCSRMatrixLastRowIndex(A);

	/* Initialize local indexes of final coarse set -- to be passed on to subsequent levels */
  if (splitting_strategy == 0) {
	if((mgr_data -> final_coarse_indexes) != NULL)
		hypre_TFree((mgr_data -> final_coarse_indexes));
	(mgr_data -> final_coarse_indexes) = hypre_CTAlloc(HYPRE_Int, nloc);
	final_coarse_indexes = (mgr_data -> final_coarse_indexes);
	final_coarse_size = 0;
	for( row = ilower; row <= iupper; row++)
	{
		idx = row % block_size;
		if(block_cf_marker[idx] == CMRK)
		{
			final_coarse_indexes[final_coarse_size++] = row - ilower;
		}
		if (row>=gnumrows-reserved_coarse_size)
		{
			final_coarse_indexes[final_coarse_size++] = row - ilower;
		}
	}
  }
  else if (splitting_strategy == 1) {
  if((mgr_data -> final_coarse_indexes) != NULL)
    hypre_TFree((mgr_data -> final_coarse_indexes));
  (mgr_data -> final_coarse_indexes) = hypre_CTAlloc(HYPRE_Int, nloc);
  final_coarse_indexes = (mgr_data -> final_coarse_indexes);
  if ((mgr_data -> coarse_indices_lvls) == NULL)
    (mgr_data -> coarse_indices_lvls) = hypre_CTAlloc(HYPRE_Int*, num_coarse_levels);
  if (coarse_indices_lvls == NULL)
    coarse_indices_lvls = hypre_CTAlloc(HYPRE_Int*, nloc);
  (mgr_data -> coarse_indices_lvls) = coarse_indices_lvls;
  if (coarse_indices_lvls[0] == NULL)
    coarse_indices_lvls[0] = hypre_CTAlloc(HYPRE_Int,nloc);
  final_coarse_size = 0;
  for( row = ilower; row <= iupper; row++)
  {
    idx = row % block_size;
    if(tmp_block_cf_marker[0][idx] == CMRK)
    {
      final_coarse_indexes[final_coarse_size] = row - ilower;
      (mgr_data -> coarse_indices_lvls)[0][final_coarse_size] = row - ilower;
      final_coarse_size++;
    }
    if (row>=gnumrows-reserved_coarse_size)
    {
      final_coarse_indexes[final_coarse_size++] = row - ilower;
    }
  }
  HYPRE_Int i;
  if ((mgr_data -> additional_coarse_indices) != NULL)
  for (i = 0; i < num_additional_coarse_indices; i++) {
    row = additional_coarse_indices[i] - 1 + ilower;
    idx = row % block_size;
    if (tmp_block_cf_marker[0][idx] != CMRK) {
      final_coarse_indexes[final_coarse_size] = additional_coarse_indices[i] - 1 + ilower;
      (mgr_data -> coarse_indices_lvls)[0][final_coarse_size] = additional_coarse_indices[i] - 1 - ilower;
      final_coarse_size++;
    }
  }
  }
  coarse_size = final_coarse_size;

  /* Free Previously allocated data, if any not destroyed */
  if (A_array || P_array || P_f_array || RT_array || CF_marker_array || A_ff_array)
  {
    for (j = 1; j < (old_num_coarse_levels); j++)
    {
      if (A_array[j])
      {
            hypre_ParCSRMatrixDestroy(A_array[j]);
            A_array[j] = NULL;
      }
    }

    for (j = 0; j < old_num_coarse_levels; j++)
    {
         if (P_array[j])
         {
            hypre_ParCSRMatrixDestroy(P_array[j]);
            P_array[j] = NULL;
         }

         if (P_f_array[j])
         {
            hypre_ParCSRMatrixDestroy(P_f_array[j]);
            P_f_array[j] = NULL;
         }

         if (RT_array[j])
         {
       hypre_ParCSRMatrixDestroy(RT_array[j]);
       RT_array[j] = NULL;
         }

         if (CF_marker_array[j])
         {
       hypre_TFree(CF_marker_array[j]);
       CF_marker_array[j] = NULL;
         }
    }
  }

  /* destroy final A_ff matrix, if not previously destroyed */
  if((mgr_data -> A_ff))
  {
    hypre_ParCSRMatrixDestroy((mgr_data -> A_ff));
    (mgr_data -> A_ff) = NULL;
  }

  /* destroy final coarse grid matrix, if not previously destroyed */
  if((mgr_data -> RAP))
  {
    hypre_ParCSRMatrixDestroy((mgr_data -> RAP));
    (mgr_data -> RAP) = NULL;
  }

	/* Setup for global block smoothers*/

	hypre_MPI_Comm_size(comm,&num_procs);
	hypre_MPI_Comm_rank(comm,&my_id);
	if (my_id == num_procs)
	{
		mgr_data -> n_block   = (n - reserved_coarse_size) / blk_size;
		mgr_data -> left_size = n - blk_size*(mgr_data -> n_block);
	}
	else
	{
		mgr_data -> n_block = n / blk_size;
		mgr_data -> left_size = n - blk_size*(mgr_data -> n_block);
	}
	if (global_smooth_type == 0)
	{
		hypre_blockRelax_setup(A,blk_size,reserved_coarse_size,&(mgr_data -> diaginv));
	}
	else if (global_smooth_type == 3)
	{
		ierr = HYPRE_EuclidCreate(comm, &(mgr_data -> global_smoother));
		HYPRE_EuclidSetLevel(mgr_data -> global_smoother, 0);
		HYPRE_EuclidSetBJ(mgr_data -> global_smoother, 1);
		HYPRE_EuclidSetup(mgr_data -> global_smoother, A, f, u);
	}


	/* clear old l1_norm data, if created */
	if((mgr_data -> l1_norms))
	{
		for (j = 0; j < (old_num_coarse_levels); j++)
		{
			if ((mgr_data -> l1_norms)[j])
			{
				hypre_TFree((mgr_data -> l1_norms)[j]);
				(mgr_data -> l1_norms)[j] = NULL;
			}
		}
		hypre_TFree((mgr_data -> l1_norms));
	}

	/* setup temporary storage */
	if ((mgr_data -> Ztemp))
	{
		hypre_ParVectorDestroy((mgr_data -> Ztemp));
		(mgr_data -> Ztemp) = NULL;
	}
	if ((mgr_data -> Vtemp))
	{
		hypre_ParVectorDestroy((mgr_data -> Vtemp));
		(mgr_data -> Vtemp) = NULL;
	}
	if ((mgr_data -> Utemp))
	{
		hypre_ParVectorDestroy((mgr_data -> Utemp));
		(mgr_data -> Utemp) = NULL;
	}
	if ((mgr_data -> Ftemp))
	{
		hypre_ParVectorDestroy((mgr_data -> Ftemp));
		(mgr_data -> Ftemp) = NULL;
	}
	if ((mgr_data -> residual))
	{
		hypre_ParVectorDestroy((mgr_data -> residual));
		(mgr_data -> residual) = NULL;
	}
	if ((mgr_data -> rel_res_norms))
	{
		hypre_TFree((mgr_data -> rel_res_norms));
		(mgr_data -> rel_res_norms) = NULL;
	}

	Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
								  hypre_ParCSRMatrixGlobalNumRows(A),
								  hypre_ParCSRMatrixRowStarts(A));
	hypre_ParVectorInitialize(Vtemp);
	hypre_ParVectorSetPartitioningOwner(Vtemp,0);
	(mgr_data ->Vtemp) = Vtemp;

	Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
								  hypre_ParCSRMatrixGlobalNumRows(A),
								  hypre_ParCSRMatrixRowStarts(A));
	hypre_ParVectorInitialize(Ztemp);
	hypre_ParVectorSetPartitioningOwner(Ztemp,0);
	(mgr_data -> Ztemp) = Ztemp;

	Utemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
								  hypre_ParCSRMatrixGlobalNumRows(A),
								  hypre_ParCSRMatrixRowStarts(A));
	hypre_ParVectorInitialize(Utemp);
	hypre_ParVectorSetPartitioningOwner(Utemp,0);
	(mgr_data ->Utemp) = Utemp;

	Ftemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
								  hypre_ParCSRMatrixGlobalNumRows(A),
								  hypre_ParCSRMatrixRowStarts(A));
	hypre_ParVectorInitialize(Ftemp);
	hypre_ParVectorSetPartitioningOwner(Ftemp,0);
	(mgr_data ->Ftemp) = Ftemp;

  /* Allocate memory for level structure */
  if (A_array == NULL)
    A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels);
  if (A_ff_array == NULL)
    A_ff_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels);
  if (P_array == NULL && max_num_coarse_levels > 0)
    P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels);
  if (P_f_array == NULL && max_num_coarse_levels > 0)
    P_f_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels);
  if (RT_array == NULL && max_num_coarse_levels > 0)
    RT_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels);
  if (CF_marker_array == NULL)
    CF_marker_array = hypre_CTAlloc(HYPRE_Int*, max_num_coarse_levels);
  if (aff_solver == NULL)
    aff_solver = hypre_CTAlloc(HYPRE_Solver, max_num_coarse_levels);

  /* set pointers to mgr data */
  (mgr_data -> A_array) = A_array;
  (mgr_data -> A_ff_array) = A_ff_array;
  (mgr_data -> P_array) = P_array;
  (mgr_data -> P_f_array) = P_f_array;
  (mgr_data -> RT_array) = RT_array;
  (mgr_data -> CF_marker_array) = CF_marker_array;
  (mgr_data -> aff_solver) = aff_solver;

  /* Set up solution and rhs arrays */
  if (F_array != NULL || U_array != NULL || F_fine_array != NULL || U_fine_array != NULL)
  {
    for (j = 1; j < old_num_coarse_levels+1; j++)
    {
      if (F_array[j] != NULL)
      {
        hypre_ParVectorDestroy(F_array[j]);
        F_array[j] = NULL;
      }
      if (U_array[j] != NULL)
      {
        hypre_ParVectorDestroy(U_array[j]);
        U_array[j] = NULL;
      }
      if (F_fine_array[j] != NULL)
      {
        hypre_ParVectorDestroy(F_fine_array[j]);
        F_fine_array[j] = NULL;
      }
      if (U_fine_array[j] != NULL)
      {
        hypre_ParVectorDestroy(U_fine_array[j]);
        U_fine_array[j] = NULL;
      }
    }
  }

  if (F_array == NULL)
    F_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels+1);
  if (U_array == NULL)
    U_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels+1);
  if (F_fine_array == NULL)
    F_fine_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels+1);
  if (U_fine_array == NULL)
    U_fine_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels+1);

	/* set solution and rhs pointers */
	F_array[0] = f;
	U_array[0] = u;

	(mgr_data -> F_array) = F_array;
	(mgr_data -> U_array) = U_array;
  (mgr_data -> F_fine_array) = F_fine_array;
  (mgr_data -> U_fine_array) = U_fine_array;

	/* begin coarsening loop */
	num_coarsening_levs = max_num_coarse_levels;
	/* initialize level data matrix here */
	RAP_ptr = A;
	old_coarse_size = hypre_ParCSRMatrixGlobalNumRows(RAP_ptr);
	coarse_size_diff = old_coarse_size;
	/* loop over levels of coarsening */
	for(lev = 0; lev < num_coarsening_levs; lev++)
	{
		/* check if this is the last level */
		last_level = ((lev == num_coarsening_levs-1) || (coarse_size_diff < 10));

		/* initialize A_array */
		A_array[lev] = RAP_ptr;
    nloc = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[lev]));

		/* Compute strength matrix for interpolation operator - use default parameters, to be modified later */
		hypre_BoomerAMGCreateS(A_array[lev], strong_threshold, max_row_sum, 1, NULL, &S);

		/* use appropriate communication package for Strength matrix */
		if (strong_threshold > S_commpkg_switch)
			hypre_BoomerAMGCreateSCommPkg(A_array[lev],S,&col_offd_S_to_A);
		
		/* Coarsen: Build CF_marker array based on rows of A */
                if (splitting_strategy == 0) {
                    hypre_MGRCoarsen(S, A_array[lev], final_coarse_size, final_coarse_indexes,debug_flag, &CF_marker_array[lev], last_level);
                } else if (splitting_strategy == 1) {
                    hypre_MGRCoarsen(S, A_array[lev], coarse_size, coarse_indices_lvls[lev], debug_flag, &CF_marker_array[lev], 1);
                }

		/* Get global coarse sizes. Note that we assume num_functions = 1
		 * so dof_func arrays are NULL */
		hypre_BoomerAMGCoarseParms(comm, nloc, 1, NULL, CF_marker_array[lev], &dof_func_buff,&coarse_pnts_global);
		/* Compute Petrov-Galerkin operators */
		/* Interpolation operator */
		num_interp_sweeps = (mgr_data -> num_interp_sweeps);

		hypre_MGRBuildInterp(A_array[lev], CF_marker_array[lev], S, coarse_pnts_global, 1, dof_func_buff, 
                                       debug_flag, trunc_factor, max_elmts, col_offd_S_to_A, &P, 1, interp_type, num_interp_sweeps);
		//hypre_MGRBuildP( A_array[lev],CF_marker_array[lev],coarse_pnts_global,2,debug_flag,&P);
		
//    hypre_ParCSRMatrixPrintIJ(P, 1, 1, "P.mat");

		P_array[lev] = P;

      /* Build AT (transpose A) */
      hypre_ParCSRMatrixTranspose(A_array[lev], &AT, 1);

      /* Build new strength matrix */
      hypre_BoomerAMGCreateS(AT, strong_threshold, max_row_sum, 1, NULL, &ST);
      /* use appropriate communication package for Strength matrix */
      if (strong_threshold > S_commpkg_switch)
         hypre_BoomerAMGCreateSCommPkg(AT, ST, &col_offd_ST_to_AT);

      num_restrict_sweeps = 0; /* do injection for restriction */
      hypre_MGRBuildInterp(AT, CF_marker_array[lev], ST, coarse_pnts_global, 1, dof_func_buff,
                         	debug_flag, trunc_factor, max_elmts, col_offd_ST_to_AT, &RT, last_level, 0, num_restrict_sweeps);
	  //hypre_MGRBuildP(AT,CF_marker_array[lev],coarse_pnts_global,2,debug_flag,&RT);
	  //hypre_MGRBuildP(A_array[lev],CF_marker_array[lev],coarse_pnts_global,0,debug_flag,&RT);
      RT_array[lev] = RT;

      /* Compute RAP for next level */
      hypre_BoomerAMGBuildCoarseOperator(RT, A_array[lev], P, &RAP_ptr);

//    hypre_ParCSRMatrixPrintIJ(RAP_ptr, 1, 1, "RAP.mat");

    if (splitting_strategy == 1) {
    HYPRE_Int nloc_next = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(RAP_ptr));
    ilower =  hypre_ParCSRMatrixFirstRowIndex(A_array[lev]);
    iupper =  hypre_ParCSRMatrixLastRowIndex(A_array[lev]);
    if (lev < num_coarsening_levs - 1) {
      coarse_indices_lvls[lev+1] = hypre_CTAlloc(HYPRE_Int, nloc_next);
      HYPRE_Int i;
      for (i = 0; i < num_additional_coarse_indices; i++) {
        CF_marker_array[lev][additional_coarse_indices[i]-1-ilower] = S_CMRK;
      }
      HYPRE_Int nc = 0;
      HYPRE_Int index_i = 0;
      HYPRE_Int add_cnt = 0;
      HYPRE_Int incr = 0;
      for (row = 0; row < nloc; row++) {
        if (CF_marker_array[lev][row] == CMRK) {
          idx = (nc - incr) % num_coarse_points[lev];
          if (tmp_block_cf_marker[lev+1][idx] == CMRK) {
            coarse_indices_lvls[lev+1][index_i++] = nc;
          }
          nc++;
        }
        else if (CF_marker_array[lev][row] == S_CMRK) {
          additional_coarse_indices[add_cnt++] = nc+1;
          //hypre_printf("constraint idx at level %d: %d\n", lev+1, nc);
          idx = row % (lev == 0 ? blk_size : num_coarse_points[lev]);
          if (tmp_block_cf_marker[lev][idx] != CMRK || lev >= 1) incr++;
          //incr++;
          if (lev != num_coarsening_levs - 2) {
            coarse_indices_lvls[lev+1][index_i++] = nc;
          }
          nc++;
          CF_marker_array[lev][row] = CMRK;
        }
        //hypre_printf("row %d CF_marker %d index_i %d\n", row, CF_marker_array[lev][row], index_i);
      }
      coarse_size = index_i;
      //hypre_printf("Number of coarse points at lev %d: %d\n", lev + 1, nc);
      //hypre_printf("Number of coarse points at lev %d: %d\n", lev + 2, coarse_size);
    }

    if (mgr_data -> build_aff) {
      hypre_MGRBuildAff(comm, nloc, 1, NULL, CF_marker_array[lev], &dof_func_buff, &coarse_pnts_global,
        A_array[lev], debug_flag, &P_f, &A_ff_ptr);
      //hypre_ParCSRMatrixPrintIJ(P_f, 1, 1, "P_f.mat");
      //hypre_ParCSRMatrixPrintIJ(A_ff_ptr, 1, 1, "A_ff.mat");
      F_fine_array[lev+1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_ff_ptr),
                               hypre_ParCSRMatrixGlobalNumRows(A_ff_ptr),
                               hypre_ParCSRMatrixRowStarts(A_ff_ptr));
      hypre_ParVectorInitialize(F_fine_array[lev+1]);
      hypre_ParVectorSetPartitioningOwner(F_fine_array[lev+1],0);

      U_fine_array[lev+1] =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_ff_ptr),
                               hypre_ParCSRMatrixGlobalNumRows(A_ff_ptr),
                               hypre_ParCSRMatrixRowStarts(A_ff_ptr));
      hypre_ParVectorInitialize(U_fine_array[lev+1]);
      hypre_ParVectorSetPartitioningOwner(U_fine_array[lev+1],0);
      A_ff_array[lev] = A_ff_ptr;
      P_f_array[lev] = P_f;
      ierr = HYPRE_EuclidCreate(comm, &(aff_solver[lev]));
      HYPRE_EuclidSetLevel((mgr_data -> aff_solver)[lev], 0);
      HYPRE_EuclidSetup((mgr_data -> aff_solver)[lev], A_ff_ptr, F_fine_array[lev+1], U_fine_array[lev+1]);
    }
    }

    /* allocate space for solution and rhs arrays */
    F_array[lev+1] =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(RAP_ptr),
                            hypre_ParCSRMatrixGlobalNumRows(RAP_ptr),
                            hypre_ParCSRMatrixRowStarts(RAP_ptr));
    hypre_ParVectorInitialize(F_array[lev+1]);
    hypre_ParVectorSetPartitioningOwner(F_array[lev+1],0);

    U_array[lev+1] =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(RAP_ptr),
                            hypre_ParCSRMatrixGlobalNumRows(RAP_ptr),
                            hypre_ParCSRMatrixRowStarts(RAP_ptr));
    hypre_ParVectorInitialize(U_array[lev+1]);
    hypre_ParVectorSetPartitioningOwner(U_array[lev+1],0);

    /* free memory before starting next level */
    hypre_ParCSRMatrixDestroy(S);
    S = NULL;
    hypre_TFree(col_offd_S_to_A);
    col_offd_S_to_A = NULL;

    hypre_ParCSRMatrixDestroy(AT);
    hypre_ParCSRMatrixDestroy(ST);
    ST = NULL;
    hypre_TFree(col_offd_ST_to_AT);
    col_offd_ST_to_AT = NULL;

    /* check if last level */
    if(last_level) break;

    /* update coarse_size_diff and old_coarse_size */
    int num_c_rows = hypre_ParCSRMatrixGlobalNumRows(RAP_ptr);
    coarse_size_diff = old_coarse_size - num_c_rows;
    old_coarse_size = num_c_rows;
  }

   /* set pointer to last level matrix */
    num_c_levels = lev+1;
   (mgr_data->num_coarse_levels) = num_c_levels;
   (mgr_data->RAP) = RAP_ptr;

   /* setup default coarse grid solver */
   /* default is BoomerAMG */
   if(use_default_cgrid_solver)
   {
      hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
      /* create and set default solver parameters here */
      default_cg_solver = (HYPRE_Solver*) hypre_BoomerAMGCreate();
      hypre_BoomerAMGSetMaxIter ( default_cg_solver, 1 );
      hypre_BoomerAMGSetRelaxOrder( default_cg_solver, 1);
      hypre_BoomerAMGSetPrintLevel(default_cg_solver, 0);
      /* set setup and solve functions */
      coarse_grid_solver_setup =  (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSetup;
      coarse_grid_solver_solve =  (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSolve;
      (mgr_data -> coarse_grid_solver_setup) =   coarse_grid_solver_setup;
      (mgr_data -> coarse_grid_solver_solve) =   coarse_grid_solver_solve;
      (mgr_data -> coarse_grid_solver) = default_cg_solver;
   }
   /* setup coarse grid solver */
   coarse_grid_solver_setup((mgr_data -> coarse_grid_solver), RAP_ptr, F_array[num_c_levels], U_array[num_c_levels]);

   /* Setup smoother for fine grid */
   if (	relax_type == 8 || relax_type == 13 || relax_type == 14 || relax_type == 18 )
   {
      l1_norms = hypre_CTAlloc(HYPRE_Real *, num_c_levels);
      (mgr_data -> l1_norms) = l1_norms;
   }
   for (j = 0; j < num_c_levels; j++)
   {
      if (num_threads == 1)
      {
         if (relax_type == 8 || relax_type == 13 || relax_type == 14)
         {
            if (relax_order)
               hypre_ParCSRComputeL1Norms(A_array[j], 4, CF_marker_array[j], &l1_norms[j]);
            else
               hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norms[j]);
         }
         else if (relax_type == 18)
         {
            if (relax_order)
               hypre_ParCSRComputeL1Norms(A_array[j], 1, CF_marker_array[j], &l1_norms[j]);
            else
               hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norms[j]);
         }
      }
      else
      {
         if (relax_type == 8 || relax_type == 13 || relax_type == 14)
         {
            if (relax_order)
               hypre_ParCSRComputeL1NormsThreads(A_array[j], 4, num_threads, CF_marker_array[j] , &l1_norms[j]);
            else
               hypre_ParCSRComputeL1NormsThreads(A_array[j], 4, num_threads, NULL, &l1_norms[j]);
         }
         else if (relax_type == 18)
         {
            if (relax_order)
               hypre_ParCSRComputeL1NormsThreads(A_array[j], 1, num_threads, CF_marker_array[j] , &l1_norms[j]);
            else
               hypre_ParCSRComputeL1NormsThreads(A_array[j], 1, num_threads, NULL, &l1_norms[j]);
         }
      }
   }

   if ( logging > 1 ) {

      residual =
	hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                              hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                              hypre_ParCSRMatrixRowStarts(A_array[0]) );
      hypre_ParVectorInitialize(residual);
      hypre_ParVectorSetPartitioningOwner(residual,0);
      (mgr_data -> residual) = residual;
   }
   else{
      (mgr_data -> residual) = NULL;
   }
   rel_res_norms = hypre_CTAlloc(HYPRE_Real,(mgr_data -> max_iter));
   (mgr_data -> rel_res_norms) = rel_res_norms;

   return hypre_error_flag;
}

