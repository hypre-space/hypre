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

	HYPRE_Int       cnt,i,j, final_coarse_size, block_size, idx, row, **block_cf_marker;
	HYPRE_Int	   lev, num_coarsening_levs, last_level, num_c_levels, num_threads,nc,index_i,cflag;
	HYPRE_Int	   debug_flag = 0;

	hypre_ParCSRMatrix  *RT = NULL;
	hypre_ParCSRMatrix  *P = NULL;
	hypre_ParCSRMatrix  *S = NULL;
	hypre_ParCSRMatrix  *ST = NULL;
	hypre_ParCSRMatrix  *AT = NULL;

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
	HYPRE_Int * reserved_Cpoint_local_indexes = (mgr_data -> reserved_Cpoint_local_indexes);
	HYPRE_Int ** CF_marker_array = (mgr_data -> CF_marker_array);
	hypre_ParCSRMatrix  **A_array = (mgr_data -> A_array);
	hypre_ParCSRMatrix  **P_array = (mgr_data -> P_array);
	hypre_ParCSRMatrix  **RT_array = (mgr_data -> RT_array);	
	hypre_ParCSRMatrix  *RAP_ptr = NULL;

	hypre_ParVector    **F_array = (mgr_data -> F_array);
	hypre_ParVector    **U_array = (mgr_data -> U_array);
	hypre_ParVector    *residual = (mgr_data -> residual);
	HYPRE_Real    *rel_res_norms = (mgr_data -> rel_res_norms);

	HYPRE_Solver    	default_cg_solver;
	HYPRE_Int	(*coarse_grid_solver_setup)(void*,void*,void*,void*) = (HYPRE_Int (*)(void*, void*, void*, void*)) (mgr_data -> coarse_grid_solver_setup);
	HYPRE_Int	(*coarse_grid_solver_solve)(void*,void*,void*,void*) = (HYPRE_Int (*)(void*, void*, void*, void*)) (mgr_data -> coarse_grid_solver_solve);

	HYPRE_Int    global_smooth_type =  (mgr_data -> global_smooth_type);

	HYPRE_Int    reserved_coarse_size = (mgr_data -> reserved_coarse_size);

	HYPRE_Int		   num_procs,  my_id;
	hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
	HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);
	HYPRE_Int    blk_size  = (mgr_data -> block_size);
  
	hypre_ParAMGData    **FrelaxVcycleData = (mgr_data -> FrelaxVcycleData);  
	HYPRE_Int Frelax_method = (mgr_data -> Frelax_method);

	/* ----- begin -----*/

	num_threads = hypre_NumThreads();

  block_size = (mgr_data -> block_size);
  block_cf_marker = (mgr_data -> block_cf_marker);
  
    HYPRE_Int **level_coarse_indexes = NULL;
    HYPRE_Int *level_coarse_size = NULL;
    HYPRE_Int setNonCpointToF = (mgr_data -> set_non_Cpoints_to_F);
    HYPRE_Int *reserved_coarse_indexes = (mgr_data -> reserved_coarse_indexes);


//  HYPRE_Int num_coarse_levels = (mgr_data -> max_num_coarse_levels);
  
  HYPRE_Int nloc =  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
  HYPRE_Int ilower =  hypre_ParCSRMatrixFirstRowIndex(A);
  HYPRE_Int iupper =  hypre_ParCSRMatrixLastRowIndex(A);

	hypre_MPI_Comm_size(comm,&num_procs);
	hypre_MPI_Comm_rank(comm,&my_id);
       
        /* Trivial case: simply solve the coarse level problem */
	if( block_size < 2 || (mgr_data -> max_num_coarse_levels) < 1)
	{
		if (my_id == 0 && print_level > 0)
		{
		   hypre_printf("Warning: Block size is < 2 or number of coarse levels is < 1. \n");
		   hypre_printf("Solving scalar problem on fine grid using coarse level solver \n");
		}

		if(use_default_cgrid_solver)
		{
			if (my_id == 0 && print_level > 0) hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
			/* create and set default solver parameters here */
			/* create and initialize default_cg_solver */
			default_cg_solver = (HYPRE_Solver) hypre_BoomerAMGCreate();
			hypre_BoomerAMGSetMaxIter ( default_cg_solver, (mgr_data -> max_iter) );

			hypre_BoomerAMGSetRelaxOrder( default_cg_solver, 1);
			hypre_BoomerAMGSetPrintLevel(default_cg_solver, 3);
			/* set setup and solve functions */
			coarse_grid_solver_setup = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSetup;
			coarse_grid_solver_solve = (HYPRE_Int (*)(void*, void*, void*, void*)) hypre_BoomerAMGSolve;
			(mgr_data -> coarse_grid_solver_setup) = coarse_grid_solver_setup;
			(mgr_data -> coarse_grid_solver_solve) = coarse_grid_solver_solve;
			(mgr_data -> coarse_grid_solver) = default_cg_solver;
		}

		// keep reserved coarse indexes to coarsest grid
  		if((mgr_data -> reserved_Cpoint_local_indexes) != NULL)
     		 hypre_TFree((mgr_data -> reserved_Cpoint_local_indexes), HYPRE_MEMORY_HOST);
  		if (reserved_coarse_size > 0)
  		{
			(mgr_data -> reserved_Cpoint_local_indexes) = hypre_CTAlloc(HYPRE_Int,  reserved_coarse_size, HYPRE_MEMORY_HOST);
			reserved_Cpoint_local_indexes = (mgr_data -> reserved_Cpoint_local_indexes);
			cnt=0;
			for(i=0; i<reserved_coarse_size; i++)
			{
				row = reserved_coarse_indexes[i];
				reserved_Cpoint_local_indexes[cnt++] = row - ilower;
			}
		        HYPRE_BoomerAMGSetCpointsToKeep((mgr_data ->coarse_grid_solver), 25,reserved_coarse_size,reserved_Cpoint_local_indexes);	
   		}

		/* setup coarse grid solver */
//		hypre_BoomerAMGSetMaxIter ( (mgr_data -> coarse_grid_solver), (mgr_data -> max_iter) );
//		hypre_BoomerAMGSetPrintLevel((mgr_data -> coarse_grid_solver), 3);
		coarse_grid_solver_setup((mgr_data -> coarse_grid_solver), A, f, u);
		(mgr_data -> num_coarse_levels) = 0;

		return hypre_error_flag;
	}
	
/*
  if ((mgr_data -> level_coarse_indexes) != NULL)
  {
     for(i=0; i<max_num_coarse_levels; i++)
     {
        if((mgr_data -> level_coarse_indexes)[i] != NULL)
        {
           hypre_TFree((mgr_data -> level_coarse_indexes)[i], HYPRE_MEMORY_HOST);
        }
     }
     hypre_TFree((mgr_data -> level_coarse_indexes), HYPRE_MEMORY_HOST);
     (mgr_data -> level_coarse_indexes) = NULL;
     hypre_TFree((mgr_data -> num_coarse_per_level), HYPRE_MEMORY_HOST);
     (mgr_data -> num_coarse_per_level) = NULL;     
  }
*/

   /* Initialize local indexes of coarse sets at different levels */      
  level_coarse_indexes = hypre_CTAlloc(HYPRE_Int*,  max_num_coarse_levels, HYPRE_MEMORY_HOST);
  for (i = 0; i < max_num_coarse_levels; i++) 
  {
     level_coarse_indexes[i] = hypre_CTAlloc(HYPRE_Int, nloc, HYPRE_MEMORY_HOST);
  }

  level_coarse_size = hypre_CTAlloc(HYPRE_Int,  max_num_coarse_levels, HYPRE_MEMORY_HOST);

  // loop over levels
  for(i=0; i<max_num_coarse_levels; i++)
  {
     // loop over rows
     final_coarse_size = 0;
     for(row = ilower; row <=iupper; row++)
     {
        idx = row % block_size;
        if(block_cf_marker[i][idx] == CMRK)
        {
           level_coarse_indexes[i][final_coarse_size++] = row - ilower;
        }
     }
     level_coarse_size[i] = final_coarse_size;
  }
  
  // Set reserved coarse indexes to be kept to the coarsest level of the MGR solver
  if((mgr_data -> reserved_Cpoint_local_indexes) != NULL)
     hypre_TFree((mgr_data -> reserved_Cpoint_local_indexes), HYPRE_MEMORY_HOST);
  if (reserved_coarse_size > 0)
  {
	(mgr_data -> reserved_Cpoint_local_indexes) = hypre_CTAlloc(HYPRE_Int,  reserved_coarse_size, HYPRE_MEMORY_HOST);
	reserved_Cpoint_local_indexes = (mgr_data -> reserved_Cpoint_local_indexes);
//	cnt=0;
	for(i=0; i<reserved_coarse_size; i++)
	{
		row = reserved_coarse_indexes[i];
		idx = row % block_size;
		for(j=0; j<max_num_coarse_levels; j++)
		{
		   if(block_cf_marker[j][idx] != CMRK)
		   {
		      level_coarse_indexes[j][level_coarse_size[j]++] = row - ilower;
		   }
		}
		reserved_Cpoint_local_indexes[i] = row - ilower;
	}
   }   
    
  (mgr_data -> level_coarse_indexes) = level_coarse_indexes;
  
  (mgr_data -> num_coarse_per_level) = level_coarse_size;

  /* Free Previously allocated data, if any not destroyed */
  if (A_array || P_array || RT_array || CF_marker_array)
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

         if (RT_array[j])
         {
       hypre_ParCSRMatrixDestroy(RT_array[j]);
       RT_array[j] = NULL;
         }

         if (CF_marker_array[j])
         {
       hypre_TFree(CF_marker_array[j], HYPRE_MEMORY_HOST);
       CF_marker_array[j] = NULL;
         }
    }
    hypre_TFree(P_array, HYPRE_MEMORY_HOST);
    P_array = NULL;
    hypre_TFree(RT_array, HYPRE_MEMORY_HOST);
    RT_array = NULL;
    hypre_TFree(CF_marker_array, HYPRE_MEMORY_HOST);
    CF_marker_array = NULL;   
  }

   /* Free previously allocated FrelaxVcycleData if not destroyed
   */
   if(FrelaxVcycleData)
   {
      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if (FrelaxVcycleData[j])
         {
            hypre_MGRDestroyFrelaxVcycleData(FrelaxVcycleData[j]);
            FrelaxVcycleData[j] = NULL;
         }
      }
      hypre_TFree(FrelaxVcycleData, HYPRE_MEMORY_HOST);
      FrelaxVcycleData = NULL;   
   }
   // reset pointer to NULL
   (mgr_data -> FrelaxVcycleData) = FrelaxVcycleData;

  /* destroy final coarse grid matrix, if not previously destroyed */
  if((mgr_data -> RAP))
  {
    hypre_ParCSRMatrixDestroy((mgr_data -> RAP));
    (mgr_data -> RAP) = NULL;
  }

	/* Setup for global block smoothers*/

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
	else if (global_smooth_type == 8)
	{
		HYPRE_EuclidCreate(comm, &(mgr_data -> global_smoother));
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
			 hypre_TFree((mgr_data -> l1_norms)[j], HYPRE_MEMORY_HOST);
				(mgr_data -> l1_norms)[j] = NULL;
			}
		}
	 hypre_TFree((mgr_data -> l1_norms), HYPRE_MEMORY_HOST);
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
	 hypre_TFree((mgr_data -> rel_res_norms), HYPRE_MEMORY_HOST);
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
    A_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  max_num_coarse_levels, HYPRE_MEMORY_HOST);
  if (P_array == NULL && max_num_coarse_levels > 0)
    P_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  max_num_coarse_levels, HYPRE_MEMORY_HOST);
  if (RT_array == NULL && max_num_coarse_levels > 0)
    RT_array = hypre_CTAlloc(hypre_ParCSRMatrix*,  max_num_coarse_levels, HYPRE_MEMORY_HOST);
  if (CF_marker_array == NULL)
    CF_marker_array = hypre_CTAlloc(HYPRE_Int*,  max_num_coarse_levels, HYPRE_MEMORY_HOST);

  /* set pointers to mgr data */
  (mgr_data -> A_array) = A_array;
  (mgr_data -> P_array) = P_array;
  (mgr_data -> RT_array) = RT_array;
  (mgr_data -> CF_marker_array) = CF_marker_array;

  /* Set up solution and rhs arrays */
  if (F_array != NULL || U_array != NULL)
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
    }
  }

  if (F_array == NULL)
    F_array = hypre_CTAlloc(hypre_ParVector*,  max_num_coarse_levels+1, HYPRE_MEMORY_HOST);
  if (U_array == NULL)
    U_array = hypre_CTAlloc(hypre_ParVector*,  max_num_coarse_levels+1, HYPRE_MEMORY_HOST);

  /* set solution and rhs pointers */
  F_array[0] = f;
  U_array[0] = u;

  (mgr_data -> F_array) = F_array;
  (mgr_data -> U_array) = U_array;

  /* begin coarsening loop */
	num_coarsening_levs = max_num_coarse_levels;
	/* initialize level data matrix here */
	RAP_ptr = A;
	/* loop over levels of coarsening */
	for(lev = 0; lev < num_coarsening_levs; lev++)
	{
		/* check if this is the last level */
		last_level = ((lev == num_coarsening_levs-1));

		/* initialize A_array */
		A_array[lev] = RAP_ptr;
    		nloc = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[lev]));

		/* Compute strength matrix for interpolation operator - use default parameters, to be modified later */
		hypre_BoomerAMGCreateS(A_array[lev], strong_threshold, max_row_sum, 1, NULL, &S);

		/* use appropriate communication package for Strength matrix */
		if (strong_threshold > S_commpkg_switch)
			hypre_BoomerAMGCreateSCommPkg(A_array[lev],S,&col_offd_S_to_A);
		
		/* Coarsen: Build CF_marker array based on rows of A */
                cflag = ((last_level || setNonCpointToF));
                hypre_MGRCoarsen(S, A_array[lev], level_coarse_size[lev], level_coarse_indexes[lev],debug_flag, &CF_marker_array[lev], cflag);

		/* Get global coarse sizes. Note that we assume num_functions = 1
		 * so dof_func arrays are NULL */
		hypre_BoomerAMGCoarseParms(comm, nloc, 1, NULL, CF_marker_array[lev], &dof_func_buff,&coarse_pnts_global);
		/* Compute Petrov-Galerkin operators */
		/* Interpolation operator */
		num_interp_sweeps = (mgr_data -> num_interp_sweeps);

		hypre_MGRBuildInterp(A_array[lev], CF_marker_array[lev], S, coarse_pnts_global, 1, dof_func_buff, 
                                       debug_flag, trunc_factor, max_elmts, col_offd_S_to_A, &P, 1, interp_type, num_interp_sweeps);
		
		P_array[lev] = P;

      		/* Build AT (transpose A) */
      		hypre_ParCSRMatrixTranspose(A_array[lev], &AT, 1);

      		/* Build new strength matrix */
      		hypre_BoomerAMGCreateS(AT, strong_threshold, max_row_sum, 1, NULL, &ST);
      		/* use appropriate communication package for Strength matrix */
      		if (strong_threshold > S_commpkg_switch)
         		hypre_BoomerAMGCreateSCommPkg(AT, ST, &col_offd_ST_to_AT);

      		num_restrict_sweeps = (mgr_data -> num_restrict_sweeps); /* restriction */
      		hypre_MGRBuildInterp(AT, CF_marker_array[lev], ST, coarse_pnts_global, 1, dof_func_buff,
                         	debug_flag, trunc_factor, max_elmts, col_offd_ST_to_AT, &RT, last_level, restrict_type, num_restrict_sweeps);
                         	
      		RT_array[lev] = RT;

      		/* Compute RAP for next level */
      		hypre_BoomerAMGBuildCoarseOperator(RT, A_array[lev], P, &RAP_ptr);
      		
      		/* Update coarse level indexes for next levels */
      		if (lev < num_coarsening_levs - 1) 
      		{
      		   // first mark indexes to be updated
      		   for(i=0; i<level_coarse_size[lev+1]; i++)
      		   {
      		      CF_marker_array[lev][level_coarse_indexes[lev+1][i]] = S_CMRK;
      		   }
      		   // next: loop over levels to update indexes
      		   for(i=lev+1; i<max_num_coarse_levels; i++)
      		   {
      		      nc = 0;
       		      index_i = 0;
      		      for(j=0; j<nloc; j++)
      		      {
      		         if(CF_marker_array[lev][j] == CMRK) nc++;
      		         if(CF_marker_array[lev][j] == S_CMRK)
      		         {
      		            level_coarse_indexes[i][index_i++] = nc++;
      		         }
      		         if(index_i == level_coarse_size[i]) break;
      		      }
      		   }
      		   // then: reset previously marked indexes
      		   for(i=0; i<level_coarse_size[lev]; i++)
      		   {
      		      CF_marker_array[lev][level_coarse_indexes[lev][i]] = CMRK;
      		   }
      		}
//      		printf("#### nloc = %d, level_coarse_index = %d \n",nloc, level_coarse_size[lev]);
      		// update reserved coarse indexes to be kept to coarsest level
      		// first mark indexes to be updated
      		for(i=0; i<reserved_coarse_size; i++)
      		{
      		   CF_marker_array[lev][reserved_Cpoint_local_indexes[i]] = S_CMRK;
      		}   
      		// loop to update reserved Cpoints
      		nc = 0;
      		index_i = 0;
      		for(i=0; i<nloc; i++)
      		{
      		   if(CF_marker_array[lev][i] == CMRK) nc++;
      		   if(CF_marker_array[lev][i] == S_CMRK)
      		   {
      		      reserved_Cpoint_local_indexes[index_i++] = nc++;
      		      // reset modified CF marker array indexes
      		      CF_marker_array[lev][i] = CMRK;
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
    hypre_TFree(col_offd_S_to_A, HYPRE_MEMORY_HOST);
    col_offd_S_to_A = NULL;

    hypre_ParCSRMatrixDestroy(AT);
    hypre_ParCSRMatrixDestroy(ST);
    ST = NULL;
    hypre_TFree(col_offd_ST_to_AT, HYPRE_MEMORY_HOST);
    col_offd_ST_to_AT = NULL;

    /* check if last level */
    if(last_level) break;

  }

   /* set pointer to last level matrix */
   num_c_levels = lev+1;
   (mgr_data->num_coarse_levels) = num_c_levels;
   (mgr_data->RAP) = RAP_ptr;

   /* setup default coarse grid solver */
   /* default is BoomerAMG */
   if(use_default_cgrid_solver)
   {
      if (my_id==0) hypre_fprintf(stderr,"No coarse grid solver provided. Using default AMG solver ... \n");
      /* create and set default solver parameters here */
      default_cg_solver = (HYPRE_Solver) hypre_BoomerAMGCreate();
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
   // keep reserved coarse indexes to coarsest grid
   if(reserved_coarse_size > 0)
      HYPRE_BoomerAMGSetCpointsToKeep((mgr_data ->coarse_grid_solver), 25,reserved_coarse_size,reserved_Cpoint_local_indexes);

   /* setup coarse grid solver */
   coarse_grid_solver_setup((mgr_data -> coarse_grid_solver), RAP_ptr, F_array[num_c_levels], U_array[num_c_levels]);

   /* Setup smoother for fine grid */
   if (	relax_type == 8 || relax_type == 13 || relax_type == 14 || relax_type == 18 )
   {
      l1_norms = hypre_CTAlloc(HYPRE_Real *,  num_c_levels, HYPRE_MEMORY_HOST);
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
   
   /* Setup Vcycle data for Frelax_method > 0 */
   if(Frelax_method == 1)
   {
      /* allocate memory and set pointer to (mgr_data -> FrelaxVcycleData) */
      if(FrelaxVcycleData == NULL)
         FrelaxVcycleData = hypre_CTAlloc(hypre_ParAMGData*,  max_num_coarse_levels, HYPRE_MEMORY_HOST);
      (mgr_data -> FrelaxVcycleData) = FrelaxVcycleData;  
      /* loop over levels */
      for(i=0; i<(mgr_data->num_coarse_levels); i++)
      { 
         FrelaxVcycleData[i] = (hypre_ParAMGData*) hypre_MGRCreateFrelaxVcycleData();
         (FrelaxVcycleData[i] -> Vtemp) = Vtemp;
         (FrelaxVcycleData[i] -> Ztemp) = Ztemp;
      
         // setup variables for the V-cycle in the F-relaxation step //
         hypre_MGRSetupFrelaxVcycleData(mgr_data, A_array[i], F_array[i], U_array[i], i);
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
   rel_res_norms = hypre_CTAlloc(HYPRE_Real, (mgr_data -> max_iter), HYPRE_MEMORY_HOST);
   (mgr_data -> rel_res_norms) = rel_res_norms;

   /* free level_coarse_indexes data */
   if ( level_coarse_indexes != NULL)
   {
      for(i=0; i<max_num_coarse_levels; i++)
      {
         hypre_TFree(level_coarse_indexes[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree( level_coarse_indexes, HYPRE_MEMORY_HOST);
      level_coarse_indexes = NULL;
      hypre_TFree(level_coarse_size, HYPRE_MEMORY_HOST);
      level_coarse_size = NULL;       
   }   

   return hypre_error_flag;
}

/* Setup data for Frelax V-cycle */
HYPRE_Int
hypre_MGRSetupFrelaxVcycleData( void *mgr_vdata,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u,
                  HYPRE_Int	     lev )
{
  MPI_Comm           comm = hypre_ParCSRMatrixComm(A);
  hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
  hypre_ParAMGData    **FrelaxVcycleData = mgr_data -> FrelaxVcycleData;
  
  HYPRE_Int i, j, num_procs, my_id;
  
  HYPRE_Int max_local_lvls = (mgr_data -> max_local_lvls);  
  HYPRE_Int lev_local;
  HYPRE_Int not_finished;
//  HYPRE_Int min_local_coarse_size = 0;
  HYPRE_Int max_local_coarse_size = 2;
  HYPRE_Int ge_relax_type = 9;
  HYPRE_Int            **CF_marker_array = (mgr_data -> CF_marker_array);
  HYPRE_Int local_size;
  HYPRE_Int local_coarse_size;

  HYPRE_Int *coarse_pnts_global_lvl = NULL;
  HYPRE_Int *coarse_dof_func_lvl = NULL;

  hypre_ParCSRMatrix *RAP_local = NULL;
  hypre_ParCSRMatrix *P_local = NULL;
  hypre_ParCSRMatrix *S_local = NULL;
  
  HYPRE_Int 		smrk_local = -1;
  
  HYPRE_Int	      old_num_levels = (FrelaxVcycleData[lev] -> num_levels);
  HYPRE_Int            **CF_marker_array_local = (FrelaxVcycleData[lev] -> CF_marker_array);
  HYPRE_Int            *CF_marker_local = NULL;
  hypre_ParCSRMatrix   **A_array_local = (FrelaxVcycleData[lev] -> A_array);
  hypre_ParCSRMatrix   **P_array_local = (FrelaxVcycleData[lev] -> P_array);
  hypre_ParVector      **F_array_local = (FrelaxVcycleData[lev] -> F_array);
  hypre_ParVector      **U_array_local = (FrelaxVcycleData[lev] -> U_array);

  hypre_MPI_Comm_size(comm, &num_procs);
  hypre_MPI_Comm_rank(comm,&my_id);
  
  local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

  /* Free any local data not previously destroyed */
  if (A_array_local || P_array_local || CF_marker_array_local)
  {
    for (j = 1; j < old_num_levels; j++)
    {
      if (A_array_local[j])
      {
         hypre_ParCSRMatrixDestroy(A_array_local[j]);
         A_array_local[j] = NULL;
      }
    }

    for (j = 0; j < old_num_levels-1; j++)
    {
       if (P_array_local[j])
       {
          hypre_ParCSRMatrixDestroy(P_array_local[j]);
          P_array_local[j] = NULL;
       }
    }

    for (j = 0; j < old_num_levels-1; j++)
    {
      if (CF_marker_array_local[j]) 
      {
         hypre_TFree(CF_marker_array_local[j], HYPRE_MEMORY_HOST);
         CF_marker_array_local[j] = NULL;
      } 
    }
    hypre_TFree(A_array_local, HYPRE_MEMORY_HOST);
    A_array_local = NULL;
    hypre_TFree(P_array_local, HYPRE_MEMORY_HOST);
    P_array_local = NULL;
    hypre_TFree(CF_marker_array_local, HYPRE_MEMORY_HOST);
    CF_marker_array_local = NULL;
  }
  /* free solution arrays not previously destroyed */
  if (F_array_local != NULL || U_array_local != NULL)
  {
    for (j = 1; j < old_num_levels; j++)
    {
       if (F_array_local[j] != NULL)
       {
          hypre_ParVectorDestroy(F_array_local[j]);
          F_array_local[j] = NULL;
       }
       if (U_array_local[j] != NULL)
       {
          hypre_ParVectorDestroy(U_array_local[j]);
          U_array_local[j] = NULL;
       }
    }
    hypre_TFree(F_array_local, HYPRE_MEMORY_HOST);
    F_array_local = NULL;
    hypre_TFree(U_array_local, HYPRE_MEMORY_HOST);
    U_array_local = NULL;    
 }

  /* Initialize some variables and allocate memory */ 
  not_finished = 1;
  lev_local = 0;
  if(A_array_local == NULL)
     A_array_local = hypre_CTAlloc(hypre_ParCSRMatrix*,  max_local_lvls, HYPRE_MEMORY_HOST);
  if(P_array_local == NULL && max_local_lvls > 1)
    P_array_local = hypre_CTAlloc(hypre_ParCSRMatrix*,  max_local_lvls-1, HYPRE_MEMORY_HOST);
  if(F_array_local == NULL)
    F_array_local = hypre_CTAlloc(hypre_ParVector*,  max_local_lvls, HYPRE_MEMORY_HOST);
  if(U_array_local == NULL)
    U_array_local = hypre_CTAlloc(hypre_ParVector*,  max_local_lvls, HYPRE_MEMORY_HOST);
  if(CF_marker_array_local == NULL)
    CF_marker_array_local = hypre_CTAlloc(HYPRE_Int*,  max_local_lvls, HYPRE_MEMORY_HOST);

  A_array_local[0] = A;
  F_array_local[0] = f;
  U_array_local[0] = u;
  
  /* Special case max_local_lvls == 1 */
  if (max_local_lvls == 1)
  {
     CF_marker_local = hypre_CTAlloc(HYPRE_Int,  local_size , HYPRE_MEMORY_HOST);
     for (i=0; i < local_size ; i++)
        CF_marker_local[i] = 1;
     CF_marker_array_local[0] = CF_marker_local;
     lev_local = max_local_lvls;
     not_finished = 0;
  }

  while (not_finished) 
  {
    local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_local[lev_local]));

    if (lev_local == 0) {
      /* use the CF_marker from the outer MGR cycle to create the strength connection matrix */
      hypre_BoomerAMGCreateSFromCFMarker(A_array_local[lev_local], 0.25, 0.9, CF_marker_array[lev], smrk_local, &S_local);
    } else if (lev_local > 0) {
      hypre_BoomerAMGCreateS(A_array_local[lev_local], 0.25, 0.9, 1, NULL, &S_local);
    }

//    hypre_BoomerAMGCoarsenFalgout(S_local, A_array_local[lev_local], 0, 0, &CF_marker_local); 
    hypre_BoomerAMGCoarsen(S_local, A_array_local[lev_local], 0, 0, &CF_marker_local);     
    /* For the lev_local=0, the coarsening routine is called on the fine-grid (the whole matrix) 
     * thus, some C-points of the outer MGR level may have been set to F-points in the coarsening 
     * routine. We need to reset these back to C-points (before building the interpolation operator.
    */
    if (lev_local == 0) {
      for (i = 0; i < local_size; i++) {
        if (CF_marker_array[lev][i] == 1) {
          CF_marker_local[i] = 1;
        }
      }
    }

    hypre_BoomerAMGCoarseParms(comm, local_size,
                                   1, NULL, CF_marker_local,
                                   &coarse_dof_func_lvl, &coarse_pnts_global_lvl);
    hypre_BoomerAMGBuildInterp(A_array_local[lev_local], CF_marker_local, 
                                   S_local, coarse_pnts_global_lvl, 1, NULL, 
                                   0, 0.0, 0, NULL, &P_local);

    /* save the CF_marker and interpolation matrix pointers */
    CF_marker_array_local[lev_local] = CF_marker_local;
    P_array_local[lev_local] = P_local;

    /* build the coarse grid */
    hypre_BoomerAMGBuildCoarseOperatorKT(P_local, A_array_local[lev_local], 
                                    P_local, 0, &RAP_local);
//    hypre_printf("Coarse size lev %d = %d\n", lev_local+1, hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(RAP_local)));

#ifdef HYPRE_NO_GLOBAL_PARTITION
        if (my_id == (num_procs -1)) local_coarse_size = coarse_pnts_global_lvl[1];
        hypre_MPI_Bcast(&local_coarse_size, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
        local_coarse_size = coarse_pnts_global_lvl[num_procs];
#endif

    lev_local++;

    if (S_local) hypre_ParCSRMatrixDestroy(S_local);
    S_local = NULL;
    if ( (lev_local == max_local_lvls-1) || (local_coarse_size <= max_local_coarse_size) )
    {
      not_finished = 0;
    }

    A_array_local[lev_local] = RAP_local;
    F_array_local[lev_local] = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(RAP_local),
                                          hypre_ParCSRMatrixGlobalNumRows(RAP_local),
                                          hypre_ParCSRMatrixRowStarts(RAP_local));
    hypre_ParVectorInitialize(F_array_local[lev_local]);
    hypre_ParVectorSetPartitioningOwner(F_array_local[lev_local], 0);

    U_array_local[lev_local] = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(RAP_local),
                                          hypre_ParCSRMatrixGlobalNumRows(RAP_local),
                                          hypre_ParCSRMatrixRowStarts(RAP_local));
    hypre_ParVectorInitialize(U_array_local[lev_local]);
    hypre_ParVectorSetPartitioningOwner(U_array_local[lev_local], 0);        
  
  } // end while loop
  
  // setup Vcycle data
  (FrelaxVcycleData[lev] -> A_array) = A_array_local;
  (FrelaxVcycleData[lev] -> P_array) = P_array_local;
  (FrelaxVcycleData[lev] -> F_array) = F_array_local;
  (FrelaxVcycleData[lev] -> U_array) = U_array_local;
  (FrelaxVcycleData[lev] -> CF_marker_array) = CF_marker_array_local;
  (FrelaxVcycleData[lev] -> num_levels) = lev_local+1;

  if(lev_local > 1)
    hypre_GaussElimSetup(FrelaxVcycleData[lev], lev_local, ge_relax_type);

  return hypre_error_flag;
}
