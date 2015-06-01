/*BHEADER**********************************************************************
 * Copyright (c) 2015,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Two-grid system solver
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include <assert.h>

#define FMRK  -1
#define CMRK  1
#define UMRK  0 
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

typedef struct
{
   // block data
   HYPRE_Int  block_size;
   HYPRE_Int  num_coarse_indexes;
   HYPRE_Int  *block_cf_marker;
   
   //general data
   HYPRE_Int num_coarse_levels;
   HYPRE_Int max_num_coarse_levels;
   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **P_array;
   hypre_ParCSRMatrix **RT_array;
   hypre_ParCSRMatrix *RAP;   
   HYPRE_Int **CF_marker_array;
   HYPRE_Int *final_coarse_indexes;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParVector    *residual;
   HYPRE_Real    *rel_res_norms;

   HYPRE_Real   max_row_sum;
   HYPRE_Real	num_interp_sweeps;
   HYPRE_Real   strong_threshold;   
   HYPRE_Real   trunc_factor;   
   HYPRE_Real   S_commpkg_switch;
   HYPRE_Int	P_max_elmts;      
   HYPRE_Int  	num_iterations;

   HYPRE_Real   **l1_norms;      
   HYPRE_Real	final_rel_residual_norm;
   HYPRE_Real	conv_tol;
   HYPRE_Real	relax_weight;
   HYPRE_Int	relax_type;
   HYPRE_Int	logging;
   HYPRE_Int	print_level;
   HYPRE_Int	max_iter;
   HYPRE_Int	relax_order;
   HYPRE_Int	num_relax_sweeps;
      
   HYPRE_Solver *coarse_grid_solver;
   HYPRE_Int	(*coarse_grid_solver_setup)();
   HYPRE_Int	(*coarse_grid_solver_solve)();   
   HYPRE_Int	use_default_cgrid_solver;
   HYPRE_Real	omega;
   
   /* temp vectors for solve phase */
   hypre_ParVector   *Vtemp;
   hypre_ParVector   *Ztemp;

	HYPRE_Int num_wells;

} hypre_ParSysTGData;

/* Create */
void *
hypre_SysTGCreate()
{
   hypre_ParSysTGData  *systg_data;

   systg_data = hypre_CTAlloc(hypre_ParSysTGData, 1);
   
   /* block data */
   (systg_data -> block_size) = 1;
   (systg_data -> num_coarse_indexes) = 1;   
   (systg_data -> block_cf_marker) = NULL;   
   
   /* general data */
   (systg_data -> max_num_coarse_levels) = 10;   
   (systg_data -> A_array) = NULL;
   (systg_data -> P_array) = NULL;
   (systg_data -> RT_array) = NULL;
   (systg_data -> RAP) = NULL;   
   (systg_data -> CF_marker_array) = NULL;         
   (systg_data -> final_coarse_indexes) = NULL; 

   (systg_data -> F_array) = NULL;
   (systg_data -> U_array) = NULL;
   (systg_data -> residual) = NULL;   
   (systg_data -> rel_res_norms) = NULL;   
   (systg_data -> Vtemp) = NULL;   
   (systg_data -> Ztemp) = NULL; 
     
   (systg_data -> num_iterations) = 0;
   (systg_data -> num_interp_sweeps) = 1;
   (systg_data -> trunc_factor) = 0.0;
   (systg_data -> max_row_sum) = 0.9;
   (systg_data -> strong_threshold) = 0.25;
   (systg_data -> S_commpkg_switch) = 1.0;      
   (systg_data -> P_max_elmts) = 0;

   (systg_data -> coarse_grid_solver) = NULL;  
   (systg_data -> coarse_grid_solver_setup) = NULL; 
   (systg_data -> coarse_grid_solver_solve) = NULL;    
   (systg_data -> use_default_cgrid_solver) = 1;
   (systg_data -> omega) = 1.;
   (systg_data -> max_iter) = 20;
   (systg_data -> conv_tol) = 1.0e-7;
   (systg_data -> relax_type) = 0;     
   (systg_data -> relax_order) = 1;
   (systg_data -> num_relax_sweeps) = 1;   
   (systg_data -> relax_weight) = 1.0; 
   
   (systg_data -> logging) = 0;
   (systg_data -> print_level) = 0;
   
   (systg_data -> l1_norms) = NULL;          

   (systg_data -> num_wells) = 0;
   
   return (void *) systg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* Destroy */
HYPRE_Int
hypre_SysTGDestroy( void *data )
{
   hypre_ParSysTGData * systg_data = data;
   
   HYPRE_Int i;
   HYPRE_Int num_coarse_levels = (systg_data -> num_coarse_levels);
   if((systg_data -> block_cf_marker)) 
   {
      hypre_TFree (systg_data -> block_cf_marker);
      (systg_data -> block_cf_marker) = NULL;
   }
   /* final coarse indexes */
   if((systg_data -> final_coarse_indexes)) 
   {
      hypre_TFree ((systg_data -> final_coarse_indexes));
      (systg_data -> final_coarse_indexes) = NULL;
   }   
   /* final residual vector */
   if((systg_data -> residual))
   {
      hypre_ParVectorDestroy( (systg_data -> residual) );
      (systg_data -> residual) = NULL; 
   }
   if((systg_data -> rel_res_norms))
   {
      hypre_TFree( (systg_data -> rel_res_norms) );
      (systg_data -> rel_res_norms) = NULL; 
   }   
   /* temp vectors for solve phase */
   if((systg_data -> Vtemp))
   {
      hypre_ParVectorDestroy( (systg_data -> Vtemp) );
      (systg_data -> Vtemp) = NULL; 
   }
   if((systg_data -> Ztemp))
   {
      hypre_ParVectorDestroy( (systg_data -> Ztemp) );
      (systg_data -> Ztemp) = NULL; 
   }      
   /* coarse grid solver */
   if((systg_data -> use_default_cgrid_solver))
   {
      if((systg_data -> coarse_grid_solver))
         hypre_BoomerAMGDestroy( (systg_data -> coarse_grid_solver) );
      
      (systg_data -> coarse_grid_solver) = NULL; 
   }
   /* l1_norms */
   if ((systg_data -> l1_norms))
   {
      for (i=0; i < (num_coarse_levels); i++)
         if ((systg_data -> l1_norms)[i])
           hypre_TFree((systg_data -> l1_norms)[i]);
      hypre_TFree((systg_data -> l1_norms));
   }   

   /* linear system and cf marker array */
   if(systg_data -> A_array || systg_data -> P_array || systg_data -> RT_array || systg_data -> CF_marker_array)
   {
   	for (i=1; i < num_coarse_levels+1; i++)
  	{
		hypre_ParVectorDestroy((systg_data -> F_array)[i]);
		hypre_ParVectorDestroy((systg_data -> U_array)[i]);

        	if ((systg_data -> P_array)[i-1])
           		hypre_ParCSRMatrixDestroy((systg_data -> P_array)[i-1]);

        	if ((systg_data -> RT_array)[i-1])
           		hypre_ParCSRMatrixDestroy((systg_data -> RT_array)[i-1]);

		hypre_TFree((systg_data -> CF_marker_array)[i-1]);
   	}
   	for (i=1; i < (num_coarse_levels); i++)
   	{
        	if ((systg_data -> A_array)[i])
           	hypre_ParCSRMatrixDestroy((systg_data -> A_array)[i]);
   	} 
   }  

   if((systg_data -> F_array))
   {
   	hypre_TFree((systg_data -> F_array));
   	(systg_data -> F_array) = NULL;
   }
   if((systg_data -> U_array))
   {
   	hypre_TFree((systg_data -> U_array));
   	(systg_data -> U_array) = NULL;
   }
   if((systg_data -> A_array))
   {
   	hypre_TFree((systg_data -> A_array));
   	(systg_data -> A_array) = NULL;
   }
   if((systg_data -> P_array))
   {
   	hypre_TFree((systg_data -> P_array));
   	(systg_data -> P_array) = NULL;
   }
   if((systg_data -> RT_array))
   {
   	hypre_TFree((systg_data -> RT_array)); 
   	(systg_data -> RT_array) = NULL;
   }  
   if((systg_data -> CF_marker_array))
   {
   	hypre_TFree((systg_data -> CF_marker_array));
   	(systg_data -> CF_marker_array) = NULL;
   }
   
   /* coarse level matrix - RAP */
   if ((systg_data -> RAP))
      hypre_ParCSRMatrixDestroy((systg_data -> RAP));
   /* systg data */
   hypre_TFree(systg_data);
   
   return hypre_error_flag;
}

/* Initialize/ set block data information */
HYPRE_Int
hypre_SysTGSetBlockData( void      *systg_vdata,
                         HYPRE_Int  block_size, 
                         HYPRE_Int num_coarse_points, 
                         HYPRE_Int  *block_coarse_indexes)
{
   HYPRE_Int  i;
   HYPRE_Int  *indexes;
   
   hypre_ParSysTGData   *systg_data = systg_vdata;
   
   indexes = (systg_data -> block_cf_marker);
   if(indexes != NULL)
   {
      hypre_TFree (systg_data -> block_cf_marker);
   }
   (systg_data -> block_cf_marker) = hypre_CTAlloc(HYPRE_Int,block_size);        
   indexes = (systg_data -> block_cf_marker);   

   memset(indexes, FMRK, block_size*sizeof(HYPRE_Int));
   
   for(i=0; i<num_coarse_points; i++)
   {
      indexes[block_coarse_indexes[i]] = CMRK;
   }

   (systg_data -> block_size) = block_size;
   (systg_data -> num_coarse_indexes) = num_coarse_points;
   
   return hypre_error_flag;
} 

/* Set the number of wells in the system */
HYPRE_Int
hypre_SysTGSetNumWells(void      *systg_vdata,
					   HYPRE_Int num_wells)
{
	hypre_ParSysTGData   *systg_data = systg_vdata;

	if (!systg_data)
	{
		hypre_printf("Warning! SysTG object empty!\n");
		hypre_error_in_arg(1);
		return hypre_error_flag;
	}
	
	(systg_data -> num_wells) = num_wells;
	
	return hypre_error_flag;
}

/* Set CF marker array */
HYPRE_Int
hypre_SysTGCoarsen(hypre_ParCSRMatrix *S, 
                        hypre_ParCSRMatrix *A,
			HYPRE_Int final_coarse_size,
			HYPRE_Int *final_coarse_indexes,
			HYPRE_Int debug_flag,
			HYPRE_Int **CF_marker,
			HYPRE_Int *last_level)
{
   HYPRE_Int *cf_marker, i, row, nc, index_i;
   HYPRE_Int *cindexes = final_coarse_indexes;

   HYPRE_Int nloc =  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));  

   /* If this is the last level, coarsen onto final coarse set */
   if(*last_level)
   {
      if(*CF_marker != NULL)
      {
         hypre_TFree (*CF_marker);   
      }
      cf_marker = hypre_CTAlloc(HYPRE_Int,nloc);        
      memset(cf_marker, FMRK, nloc*sizeof(HYPRE_Int));
      
      /* first mark final coarse set */
      nc = final_coarse_size;
      for(i = 0; i < nc; i++)
      {
         cf_marker[cindexes[i]] = CMRK;
      }  
   }
   else
   {
   
      /* First coarsen to get initial CF splitting.
       * This is then followed by updating the CF marker to pass 
       * coarse information to the next levels. NOTE: It may be 
       * convenient to implement this way (allows the use of multiple
       * coarsening strategies without changing too much code), 
       * but not necessarily the best option, compared to initializing 
       * CF_marker first and then coarsening on subgraph which excludes 
       * the initialized coarse nodes.
      */
      hypre_BoomerAMGCoarsen(S, A, 0, debug_flag, &cf_marker);   
   
      /* Update CF_marker to ensure final coarse nodes are tansferred to next level. */
      nc = final_coarse_size;
      for(i = 0; i < nc; i++)
      {
         cf_marker[cindexes[i]] = S_CMRK;
      }
      /* IMPORTANT: Update coarse_indexes array to define the positions of the final coarse points 
       * in the next level.
       */ 
      nc = 0;
      index_i = 0;
      for (row = 0; row <nloc; row++)
      {
         /* loop through new c-points */
         if(cf_marker[row] == CMRK) nc++;
         else if(cf_marker[row] == S_CMRK)
         {
            /* previously marked c-point is part of final coarse set. Track its current local index */
            cindexes[index_i++] = nc;
            /* reset c-point from S_CMRK to CMRK */
            cf_marker[row] = CMRK;
            nc++;
         }
         /* set F-points to FMRK. This is necessary since the different coarsening schemes differentiate
          * between type of F-points (example Ruge coarsening). We do not need that distinction here.
          */
         else 
         {
            cf_marker[row] = FMRK;
         }  
      }
      /* check if this should be last level */
      if( nc == final_coarse_size) 
         *last_level = 1;
//      printf(" nc = %d and final coarse size = %d \n", nc, final_coarse_size);
   }
   /* set CF_marker */
   *CF_marker = cf_marker;
   
   return hypre_error_flag;
}

/* Interpolation for MGR - Adapted from BoomerAMGBuildInterp */
HYPRE_Int
hypre_SysTGBuildP( hypre_ParCSRMatrix   *A,
				   HYPRE_Int            *CF_marker,
				   HYPRE_Int            *num_cpts_global,
				   HYPRE_Int             method,
				   HYPRE_Int             debug_flag,
				   hypre_ParCSRMatrix  **P_ptr)
{
	MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   
	hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
	hypre_ParCSRCommHandle  *comm_handle;
	
	hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
	HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
	HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
	HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);
	
	hypre_CSRMatrix *A_offd         = hypre_ParCSRMatrixOffd(A);   
	HYPRE_Real      *A_offd_data    = hypre_CSRMatrixData(A_offd);
	HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
	HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
	HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
	HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
	HYPRE_Real       *a_diag;
	
	hypre_ParCSRMatrix    *P;
	HYPRE_Int		      *col_map_offd_P;
	
	HYPRE_Int             *CF_marker_offd = NULL;
	
	hypre_CSRMatrix    *P_diag;
	hypre_CSRMatrix    *P_offd;   
	
	HYPRE_Real      *P_diag_data;
	HYPRE_Int             *P_diag_i;
	HYPRE_Int             *P_diag_j;
	HYPRE_Real      *P_offd_data;
	HYPRE_Int             *P_offd_i;
	HYPRE_Int             *P_offd_j;
	
	HYPRE_Int              P_diag_size, P_offd_size;
	
	HYPRE_Int             *P_marker, *P_marker_offd;
	
	HYPRE_Int              jj_counter,jj_counter_offd;
	HYPRE_Int             *jj_count, *jj_count_offd;
	HYPRE_Int              jj_begin_row,jj_begin_row_offd;
	HYPRE_Int              jj_end_row,jj_end_row_offd;
	
	HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */
	
	HYPRE_Int              n_fine = hypre_CSRMatrixNumRows(A_diag);
	
	HYPRE_Int             *fine_to_coarse;
	HYPRE_Int             *fine_to_coarse_offd;
	HYPRE_Int             *coarse_counter;
	HYPRE_Int              coarse_shift;
	HYPRE_Int              total_global_cpts;
	HYPRE_Int              num_cols_P_offd,my_first_cpt;
	
	HYPRE_Int              i,i1;
	HYPRE_Int              j,jl,jj;
	HYPRE_Int              k,kc;
	HYPRE_Int              start;
	
	HYPRE_Real       zero = 0.0;
	HYPRE_Real       one  = 1.0;
	
	HYPRE_Int              my_id;
	HYPRE_Int              num_procs;
	HYPRE_Int              num_threads;
	HYPRE_Int              num_sends;
	HYPRE_Int              index;
	HYPRE_Int              ns, ne, size, rest;
	HYPRE_Int              print_level = 0;
	HYPRE_Int             *int_buf_data;
	
	HYPRE_Int col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
	HYPRE_Int local_numrows = hypre_CSRMatrixNumRows(A_diag);
	HYPRE_Int col_n = col_1 + local_numrows;
	
	HYPRE_Real       wall_time;  /* for debugging instrumentation  */

	hypre_MPI_Comm_size(comm, &num_procs);   
	hypre_MPI_Comm_rank(comm,&my_id);
	num_threads = hypre_NumThreads();

#ifdef HYPRE_NO_GLOBAL_PARTITION
	my_first_cpt = num_cpts_global[0];
	if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
	hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
	my_first_cpt = num_cpts_global[my_id];
	total_global_cpts = num_cpts_global[num_procs];
#endif

	/*-------------------------------------------------------------------
	 * Get the CF_marker data for the off-processor columns
	 *-------------------------------------------------------------------*/

	if (debug_flag < 0)
	{
		debug_flag = -debug_flag;
		print_level = 1;
	}
	
	if (debug_flag==4) wall_time = time_getWallclockSeconds();

	if (num_cols_A_offd) CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);

	if (!comm_pkg)
	{
		hypre_MatvecCommPkgCreate(A);
		comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
	}

	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
	int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
																			num_sends));

	index = 0;
	for (i = 0; i < num_sends; i++)
	{
		start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
		for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
			int_buf_data[index++] 
				= CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
	}

	comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
												CF_marker_offd);
	hypre_ParCSRCommHandleDestroy(comm_handle);
	
	if (debug_flag==4)
	{
		wall_time = time_getWallclockSeconds() - wall_time;
		hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
					 my_id, wall_time);
		fflush(NULL);
	}

	/*-----------------------------------------------------------------------
	 *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
	 *-----------------------------------------------------------------------*/

	/*-----------------------------------------------------------------------
	 *  Intialize counters and allocate mapping vector.
	 *-----------------------------------------------------------------------*/

	coarse_counter = hypre_CTAlloc(HYPRE_Int, num_threads);
	jj_count = hypre_CTAlloc(HYPRE_Int, num_threads);
	jj_count_offd = hypre_CTAlloc(HYPRE_Int, num_threads);

	fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
	for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;
	
	jj_counter = start_indexing;
	jj_counter_offd = start_indexing;

	   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

/* RDF: this looks a little tricky, but doable */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) HYPRE_SMP_SCHEDULE
#endif
	for (j = 0; j < num_threads; j++)
	{
		size = n_fine/num_threads;
		rest = n_fine - size*num_threads;

		if (j < rest)
		{
			ns = j*size+j;
			ne = (j+1)*size+j+1;
		}
		else
		{
			ns = j*size+rest;
			ne = (j+1)*size+rest;
		}
		for (i = ns; i < ne; i++)
		{
			/*--------------------------------------------------------------------
			 *  If i is a C-point, interpolation is the identity. Also set up
			 *  mapping vector.
			 *--------------------------------------------------------------------*/
			
			if (CF_marker[i] >= 0)
			{
				jj_count[j]++;
				fine_to_coarse[i] = coarse_counter[j];
				coarse_counter[j]++;
			}
			/*--------------------------------------------------------------------
			 *  If i is an F-point, interpolation is the approximation of A_{ff}^{-1}A_{fc}
			 *--------------------------------------------------------------------*/
			 else
			 {
				 for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
				 {
					 i1 = A_diag_j[jj];           
					 if (CF_marker[i1] >= 0)
					 {
						 jj_count[j]++;
					 }
				 }

				 if (num_procs > 1)
				 {
					 for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
					 {
						 i1 = A_offd_j[jj];           
						 if (CF_marker_offd[i1] >= 0)
						 {
							 jj_count_offd[j]++;
						 }
					 }
				 }
			 }			
		}
	}

	/*-----------------------------------------------------------------------
	 *  Allocate  arrays.
	 *-----------------------------------------------------------------------*/
	for (i=0; i < num_threads-1; i++)
	{
		coarse_counter[i+1] += coarse_counter[i];
		jj_count[i+1] += jj_count[i];
		jj_count_offd[i+1] += jj_count_offd[i];
	}
	i = num_threads-1;
	jj_counter = jj_count[i];
	jj_counter_offd = jj_count_offd[i];
	
	P_diag_size = jj_counter;
	
	P_diag_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
	P_diag_j    = hypre_CTAlloc(HYPRE_Int, P_diag_size);
	P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size);
	
	P_diag_i[n_fine] = jj_counter; 
	
	
	P_offd_size = jj_counter_offd;
	
	P_offd_i    = hypre_CTAlloc(HYPRE_Int, n_fine+1);
	P_offd_j    = hypre_CTAlloc(HYPRE_Int, P_offd_size);
	P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size);

	/*-----------------------------------------------------------------------
	 *  Intialize some stuff.
	 *-----------------------------------------------------------------------*/
	
	jj_counter = start_indexing;
	jj_counter_offd = start_indexing;
	
	if (debug_flag==4)
	{
		wall_time = time_getWallclockSeconds() - wall_time;
		hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
					 my_id, wall_time);
		fflush(NULL);
	}

	/*-----------------------------------------------------------------------
	 *  Send and receive fine_to_coarse info.
	 *-----------------------------------------------------------------------*/

	if (debug_flag==4) wall_time = time_getWallclockSeconds();
	
	fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd); 
	
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) HYPRE_SMP_SCHEDULE
#endif
	for (j = 0; j < num_threads; j++)
	{
		coarse_shift = 0;
		if (j > 0) coarse_shift = coarse_counter[j-1];
		size = n_fine/num_threads;
		rest = n_fine - size*num_threads;
		if (j < rest)
		{
			ns = j*size+j;
			ne = (j+1)*size+j+1;
		}
		else
		{
			ns = j*size+rest;
			ne = (j+1)*size+rest;
		}
		for (i = ns; i < ne; i++)
			fine_to_coarse[i] += my_first_cpt+coarse_shift;
	}

	index = 0;
	for (i = 0; i < num_sends; i++)
	{
		start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
		for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
			int_buf_data[index++] 
				= fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
	}
	
	comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
												fine_to_coarse_offd);  
	
	hypre_ParCSRCommHandleDestroy(comm_handle);   

	if (debug_flag==4)
	{
		wall_time = time_getWallclockSeconds() - wall_time;
		hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
					 my_id, wall_time);
		fflush(NULL);
	}
	
	if (debug_flag==4) wall_time = time_getWallclockSeconds();
	
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
	for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

	/*-----------------------------------------------------------------------
	 *  Loop over fine grid points.
	 *-----------------------------------------------------------------------*/
	a_diag = hypre_CTAlloc(HYPRE_Real, n_fine);
	for (i = 0; i < n_fine; i++)
	{
		for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
		{
			i1 = A_diag_j[jj];
			if ( i==i1 )  /* diagonal of A only */
			{
				a_diag[i] = 1.0/A_diag_data[jj];
			}
		}
	}
	
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,P_marker,P_marker_offd,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
#endif
	for (jl = 0; jl < num_threads; jl++)
	{
		size = n_fine/num_threads;
		rest = n_fine - size*num_threads;
		if (jl < rest)
		{
			ns = jl*size+jl;
			ne = (jl+1)*size+jl+1;
		}
		else
		{
			ns = jl*size+rest;
			ne = (jl+1)*size+rest;
		}
		jj_counter = 0;
		if (jl > 0) jj_counter = jj_count[jl-1];
		jj_counter_offd = 0;
		if (jl > 0) jj_counter_offd = jj_count_offd[jl-1];
		P_marker = hypre_CTAlloc(HYPRE_Int, n_fine);
		if (num_cols_A_offd)
			P_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);
		else
			P_marker_offd = NULL;

		for (i = 0; i < n_fine; i++)
		{      
			P_marker[i] = -1;
		}
		for (i = 0; i < num_cols_A_offd; i++)
		{      
			P_marker_offd[i] = -1;
		}
		for (i = ns; i < ne; i++)
		{
			/*--------------------------------------------------------------------
			 *  If i is a c-point, interpolation is the identity.
			 *--------------------------------------------------------------------*/
			if (CF_marker[i] >= 0)
			{
				P_diag_i[i] = jj_counter;
				P_diag_j[jj_counter]    = fine_to_coarse[i];
				P_diag_data[jj_counter] = one;
				jj_counter++;
			}
			/*--------------------------------------------------------------------
			 *  If i is an F-point, build interpolation.
			 *--------------------------------------------------------------------*/
			else
			{			
				/* Diagonal part of P */
				P_diag_i[i] = jj_counter;
				jj_begin_row = jj_counter;
				for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
				{
					i1 = A_diag_j[jj];   
					
					/*--------------------------------------------------------------
					 * If neighbor i1 is a C-point, set column number in P_diag_j
					 * and initialize interpolation weight to zero.
					 *--------------------------------------------------------------*/
					
					if (CF_marker[i1] >= 0)
					{
						P_marker[i1] = jj_counter;
						P_diag_j[jj_counter]    = fine_to_coarse[i1];
						if(method == 0)
						{
							P_diag_data[jj_counter] = 0.0;
						}
						else if (method == 1)
						{
							P_diag_data[jj_counter] = - A_diag_data[jj];
						}
						else if (method == 2)
						{
								P_diag_data[jj_counter] = - A_diag_data[jj]*a_diag[i];
							
						}
						jj_counter++;
					}          
				}
				jj_end_row = jj_counter;

				/* Off-Diagonal part of P */
				P_offd_i[i] = jj_counter_offd;
				jj_begin_row_offd = jj_counter_offd;

				if (num_procs > 1)
				{           
					for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
					{
						i1 = A_offd_j[jj];   
						
						/*-----------------------------------------------------------
						 * If neighbor i1 is a C-point, set column number in P_offd_j
						 * and initialize interpolation weight to zero.
						 *-----------------------------------------------------------*/
						
						if (CF_marker_offd[i1] >= 0)
						{
							P_marker_offd[i1] = jj_counter_offd;
							/*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
							P_offd_j[jj_counter_offd]  = i1;
							if(method == 0)
						{
						    P_offd_data[jj_counter_offd] = 0.0;
						}
						else if (method == 1)
						{
							P_offd_data[jj_counter_offd] = - A_offd_data[jj];
						}
						else if (method == 2)
						{
							P_offd_data[jj_counter_offd] = - A_offd_data[jj]*a_diag[i];
						}
							
							jj_counter_offd++;
						}           
					}
				}
				jj_end_row_offd = jj_counter_offd;
			}
			P_offd_i[i+1] = jj_counter_offd;
		}
		hypre_TFree(P_marker);
		hypre_TFree(P_marker_offd);
	}
	hypre_TFree(a_diag);
	P = hypre_ParCSRMatrixCreate(comm,
								 hypre_ParCSRMatrixGlobalNumRows(A),
								 total_global_cpts,
								 hypre_ParCSRMatrixColStarts(A),
								 num_cpts_global,
								 0,
								 P_diag_i[n_fine],
								 P_offd_i[n_fine]);

	P_diag = hypre_ParCSRMatrixDiag(P);
	hypre_CSRMatrixData(P_diag) = P_diag_data;
	hypre_CSRMatrixI(P_diag) = P_diag_i;
	hypre_CSRMatrixJ(P_diag) = P_diag_j;
	P_offd = hypre_ParCSRMatrixOffd(P);
	hypre_CSRMatrixData(P_offd) = P_offd_data;
	hypre_CSRMatrixI(P_offd) = P_offd_i;
	hypre_CSRMatrixJ(P_offd) = P_offd_j;
	hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

	num_cols_P_offd = 0;

	if (P_offd_size)
	{
		P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
		for (i=0; i < num_cols_A_offd; i++)
			P_marker[i] = 0;
		num_cols_P_offd = 0;
		for (i=0; i < P_offd_size; i++)
		{
			index = P_offd_j[i];
			if (!P_marker[index])
			{
				num_cols_P_offd++;
				P_marker[index] = 1;
			}
		}

		col_map_offd_P = hypre_CTAlloc(HYPRE_Int,num_cols_P_offd);
		index = 0;
		for (i=0; i < num_cols_P_offd; i++)
		{
			while (P_marker[index]==0) index++;
			col_map_offd_P[i] = index++;
		}

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
		for (i=0; i < P_offd_size; i++)
			P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
											 P_offd_j[i],
											 num_cols_P_offd);
		hypre_TFree(P_marker); 
	}

	for (i=0; i < n_fine; i++)
		if (CF_marker[i] == -3) CF_marker[i] = -1;
	if (num_cols_P_offd)
	{ 
		hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
		hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
	}
	hypre_GetCommPkgRTFromCommPkgA(P,A, fine_to_coarse_offd);
	
	*P_ptr = P;
	
	hypre_TFree(CF_marker_offd);
	hypre_TFree(int_buf_data);
	hypre_TFree(fine_to_coarse);
	hypre_TFree(fine_to_coarse_offd);
	hypre_TFree(coarse_counter);
	hypre_TFree(jj_count);
	hypre_TFree(jj_count_offd);
	
	return(0);
}


/* Setup interpolation operator. This code uses Jacobi relaxation
 * (diagonal scaling) for interpolation at the last level
*/
HYPRE_Int 
hypre_sysTGBuildInterp(hypre_ParCSRMatrix     *A, 
			HYPRE_Int             *CF_marker, 
                         hypre_ParCSRMatrix   *S,
                         HYPRE_Int            *num_cpts_global,
                         HYPRE_Int            num_functions,
                         HYPRE_Int            *dof_func,
                         HYPRE_Int            debug_flag,
                         HYPRE_Real           trunc_factor,
                         HYPRE_Int	      max_elmts,
                         HYPRE_Int 	      *col_offd_S_to_A,			
			hypre_ParCSRMatrix    **P,
			HYPRE_Int	      last_level,
			HYPRE_Int	      level,
			HYPRE_Int	      numsweeps)
{
   HYPRE_Int i;
   hypre_ParCSRMatrix    *P_ptr;
   HYPRE_Real		 jac_trunc_threshold = trunc_factor;
   HYPRE_Real		 jac_trunc_threshold_minus = 0.5*jac_trunc_threshold;
   
   /* Build interpolation operator using (hypre default) */
   if(!last_level) 
   {
	   //hypre_BoomerAMGBuildInterp(A, CF_marker, S, num_cpts_global,1, NULL,debug_flag,
	   //	        trunc_factor, max_elmts, col_offd_S_to_A, &P_ptr);
		hypre_SysTGBuildP( A,CF_marker,num_cpts_global,0,debug_flag,&P_ptr);
   }
   /* Do Jacobi interpolation for last level */
   else
   {
        /* clone or copy nonzero pattern of A to B */
        //hypre_ParCSRMatrix	*B;
        //hypre_ParCSRMatrixClone( A, &B, 0 );
        /* Build interp with B to initialize P as injection [0 I] operator */
        //hypre_BoomerAMGBuildInterp(B, CF_marker, S, num_cpts_global,1, NULL,debug_flag,
		//	        trunc_factor, max_elmts, col_offd_S_to_A, &P_ptr);        
        //hypre_ParCSRMatrixDestroy(B);
		hypre_SysTGBuildP( A,CF_marker,num_cpts_global,2,debug_flag,&P_ptr);
		
        /* Do k steps of Jacobi build W for P = [-W I]. 
         * Note that BoomerAMGJacobiInterp assumes you have some initial P, 
         * hence we need to initialize P as above, before calling this routine.
         * If numsweeps = 0, the following step is skipped and P is returned as the 
         * injection operator.
         * Looping here is equivalent to improving P by Jacobi interpolation 
        */                       
        //for(i=0; i<numsweeps; i++)  
        //   hypre_BoomerAMGJacobiInterp(A, &P_ptr, S,1, NULL, CF_marker,
        //                              level, jac_trunc_threshold, 
        //                              jac_trunc_threshold_minus );               
   }
   /* set pointer to P */
   *P = P_ptr;
   
   return hypre_error_flag;
}
/* Setup sysTG data */
HYPRE_Int
hypre_SysTGSetup( void               *systg_vdata,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{
   MPI_Comm 	         comm = hypre_ParCSRMatrixComm(A); 
   hypre_ParSysTGData   *systg_data = systg_vdata;
   
   HYPRE_Int       j, final_coarse_size, block_size, idx, row, size, *cols = NULL, *block_cf_marker;
   HYPRE_Int	   lev, num_coarsening_levs, last_level, num_c_levels, num_threads, gnumrows;
   HYPRE_Int	   debug_flag = 0, old_coarse_size, coarse_size_diff;
   
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
   
   /* pointers to systg data */
   HYPRE_Int  use_default_cgrid_solver = (systg_data -> use_default_cgrid_solver);
   HYPRE_Int  logging = (systg_data -> logging);
   HYPRE_Int  print_level = (systg_data -> print_level);   
   HYPRE_Int  relax_type = (systg_data -> relax_type);   
   HYPRE_Int  relax_order = (systg_data -> relax_order);
   HYPRE_Int num_interp_sweeps = (systg_data -> num_interp_sweeps);
   HYPRE_Int num_restrict_sweeps = (systg_data -> num_interp_sweeps);               
   HYPRE_Int	max_elmts = (systg_data -> P_max_elmts);
   HYPRE_Real   max_row_sum = (systg_data -> max_row_sum);      
   HYPRE_Real   strong_threshold = (systg_data -> strong_threshold);   
   HYPRE_Real   trunc_factor = (systg_data -> trunc_factor);   
   HYPRE_Real   S_commpkg_switch = (systg_data -> S_commpkg_switch);
   HYPRE_Int  old_num_coarse_levels = (systg_data -> num_coarse_levels);
   HYPRE_Int  max_num_coarse_levels = (systg_data -> max_num_coarse_levels);
   HYPRE_Int * final_coarse_indexes = (systg_data -> final_coarse_indexes);
   HYPRE_Int ** CF_marker_array = (systg_data -> CF_marker_array);
   hypre_ParCSRMatrix  **A_array = (systg_data -> A_array);
   hypre_ParCSRMatrix  **P_array = (systg_data -> P_array);
   hypre_ParCSRMatrix  **RT_array = (systg_data -> RT_array);      
   hypre_ParCSRMatrix  *RAP_ptr = NULL;

   hypre_ParVector    **F_array = (systg_data -> F_array);
   hypre_ParVector    **U_array = (systg_data -> U_array);
   hypre_ParVector    *residual = (systg_data -> residual);
   HYPRE_Real    *rel_res_norms = (systg_data -> rel_res_norms);   

   HYPRE_Solver    	*default_cg_solver;     
   HYPRE_Int	(*coarse_grid_solver_setup)() = (systg_data -> coarse_grid_solver_setup);
   HYPRE_Int	(*coarse_grid_solver_solve)() = (systg_data -> coarse_grid_solver_solve);
   HYPRE_Int    num_wells = (systg_data -> num_wells);
   
   /* ----- begin -----*/
   
   num_threads = hypre_NumThreads();

   block_size = (systg_data -> block_size);
   block_cf_marker = (systg_data -> block_cf_marker);
   
   
   gnumrows = hypre_ParCSRMatrixGlobalNumRows(A);
   if(((gnumrows-num_wells) % block_size) != 0)
   {
	   hypre_printf("ERROR: Global number of rows minus wells is not a multiple of block_size ... n = %d, num_wells = %d, block_size = %d \n", gnumrows,num_wells, block_size);
	   hypre_MPI_Abort(comm, -1);
   }
   
   if( block_size < 2 || (systg_data -> max_num_coarse_levels) < 1)
   {
      hypre_printf("Warning: Block size is < 2 or number of coarse levels is < 1. \n");
      hypre_printf("Solving scalar problem on fine grid using coarse level solver \n");      
      /* Trivial case: simply solve the coarse level problem */
      if(use_default_cgrid_solver)
      {
         hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
         /* create and set default solver parameters here */
         /* create and initialize default_cg_solver */
         default_cg_solver = hypre_BoomerAMGCreate();
         hypre_BoomerAMGSetMaxIter ( default_cg_solver, (systg_data -> max_iter) );
//         hypre_BoomerAMGSetMaxIter ( default_cg_solver, 1 );
         hypre_BoomerAMGSetRelaxOrder( default_cg_solver, 1);
         hypre_BoomerAMGSetPrintLevel(default_cg_solver, 1);
         /* set setup and solve functions */
         coarse_grid_solver_setup = hypre_BoomerAMGSetup;
         coarse_grid_solver_solve = hypre_BoomerAMGSolve;
         (systg_data -> coarse_grid_solver_setup) = coarse_grid_solver_setup;
         (systg_data -> coarse_grid_solver_solve) = coarse_grid_solver_solve;
         (systg_data -> coarse_grid_solver) = default_cg_solver;
      }   
      /* setup coarse grid solver */
      coarse_grid_solver_setup((systg_data -> coarse_grid_solver), A, f, u);
      (systg_data -> max_num_coarse_levels) = 0;
      
      return hypre_error_flag;
   } 
   
   /* setup default block data if not set. Use first index as C-point */
   if((systg_data -> block_cf_marker)==NULL)
   {
      (systg_data -> block_cf_marker) = hypre_CTAlloc(HYPRE_Int,block_size);           
      memset((systg_data -> block_cf_marker), FMRK, block_size*sizeof(HYPRE_Int));
      (systg_data -> block_cf_marker)[0] = CMRK;
   }

   HYPRE_Int nloc =  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int ilower =  hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int iupper =  hypre_ParCSRMatrixLastRowIndex(A);

   /* Initialize local indexes of final coarse set -- to be passed on to subsequent levels */
   if((systg_data -> final_coarse_indexes) != NULL)
   	hypre_TFree((systg_data -> final_coarse_indexes));
   (systg_data -> final_coarse_indexes) = hypre_CTAlloc(HYPRE_Int, nloc);
   final_coarse_indexes = (systg_data -> final_coarse_indexes);
   final_coarse_size = 0;
   for( row = ilower; row <= iupper; row++)
   {
      idx = row % block_size;
      if(block_cf_marker[idx] == CMRK)
      {
		  final_coarse_indexes[final_coarse_size++] = row - ilower;/*Lu: WHY minus ilower needed?*/
      }
	  if (row>=gnumrows-num_wells)
	  {
		  final_coarse_indexes[final_coarse_size++] = row - ilower;
      }	  
   }
  
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
            hypre_TFree(CF_marker_array[j]);
            CF_marker_array[j] = NULL;
         }                  
      }
      /* destroy final coarse grid matrix, if not previously destroyed */
      if((systg_data -> RAP))
      {
         hypre_ParCSRMatrixDestroy((systg_data -> RAP));
         (systg_data -> RAP) = NULL;
      }
   }
   /* clear old l1_norm data, if created */
   if((systg_data -> l1_norms))
   {
      for (j = 0; j < (old_num_coarse_levels); j++)
      {
         if ((systg_data -> l1_norms)[j])
         {
            hypre_TFree((systg_data -> l1_norms)[j]);
            (systg_data -> l1_norms)[j] = NULL;
         }
      }
      hypre_TFree((systg_data -> l1_norms));
   }   
   
   /* setup temporary storage */
   if ((systg_data -> Ztemp))
   {
      hypre_ParVectorDestroy((systg_data -> Ztemp));
      (systg_data -> Ztemp) = NULL;
   }
   if ((systg_data -> Vtemp))
   {
      hypre_ParVectorDestroy((systg_data -> Vtemp));
      (systg_data -> Vtemp) = NULL;
   }
   if ((systg_data -> residual))
   {
      hypre_ParVectorDestroy((systg_data -> residual));
      (systg_data -> residual) = NULL;
   }
   if ((systg_data -> rel_res_norms))
   {
      hypre_TFree((systg_data -> rel_res_norms));
      (systg_data -> rel_res_norms) = NULL;
   }   

   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Vtemp);
   hypre_ParVectorSetPartitioningOwner(Vtemp,0);
   (systg_data ->Vtemp) = Vtemp;

   Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ztemp);
   hypre_ParVectorSetPartitioningOwner(Ztemp,0);
   (systg_data -> Ztemp) = Ztemp;   
            
   /* Allocate memory for level structure */
   if (A_array == NULL)
      A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels);
   if (P_array == NULL && max_num_coarse_levels > 0)
      P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels);
   if (RT_array == NULL && max_num_coarse_levels > 0)
      RT_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_num_coarse_levels);  
   if (CF_marker_array == NULL)
      CF_marker_array = hypre_CTAlloc(HYPRE_Int*, max_num_coarse_levels);

   /* set pointers to systg data */
   (systg_data -> A_array) = A_array;
   (systg_data -> P_array) = P_array;
   (systg_data -> RT_array) = RT_array;
   (systg_data -> CF_marker_array) = CF_marker_array;

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
      F_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels+1);
   if (U_array == NULL)
      U_array = hypre_CTAlloc(hypre_ParVector*, max_num_coarse_levels+1);
   
   /* set solution and rhs pointers */
   F_array[0] = f;
   U_array[0] = u;
   
   (systg_data -> F_array) = F_array;
   (systg_data -> U_array) = U_array;   
   
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

      /* Compute strength matrix for interpolation operator - use default parameters, to be modified later */
      hypre_BoomerAMGCreateS(A_array[lev], strong_threshold, max_row_sum, 1, NULL, &S);   
   
      /* use appropriate communication package for Strength matrix */
      if (strong_threshold > S_commpkg_switch)
         hypre_BoomerAMGCreateSCommPkg(A_array[lev],S,&col_offd_S_to_A);      
      /* Coarsen: Build CF_marker array based on rows of A */
      hypre_SysTGCoarsen(S, A_array[lev], final_coarse_size, final_coarse_indexes,debug_flag, &CF_marker_array[lev], &last_level);
      /* Get global coarse sizes. Note that we assume num_functions = 1
       * so dof_func arrays are NULL */
      hypre_BoomerAMGCoarseParms(comm, nloc, 1, NULL, CF_marker_array[lev], &dof_func_buff,&coarse_pnts_global);         
      /* Compute Petrov-Galerkin operators */
      /* Interpolation operator */
      num_interp_sweeps = (systg_data -> num_interp_sweeps);   
      hypre_sysTGBuildInterp(A_array[lev], CF_marker_array[lev], S, coarse_pnts_global, 1, dof_func_buff,
                         	debug_flag, trunc_factor, max_elmts, col_offd_S_to_A, &P, last_level, lev, num_interp_sweeps);
      P_array[lev] = P;                   	      

      /* Build AT (transpose A) */
      hypre_ParCSRMatrixTranspose(A_array[lev], &AT, 1);

      /* Build new strength matrix */
      hypre_BoomerAMGCreateS(AT, strong_threshold, max_row_sum, 1, NULL, &ST);
      /* use appropriate communication package for Strength matrix */
      if (strong_threshold > S_commpkg_switch)
         hypre_BoomerAMGCreateSCommPkg(AT, ST, &col_offd_ST_to_AT);         
      
      num_restrict_sweeps = 0; /* do injection for restriction */
      hypre_sysTGBuildInterp(AT, CF_marker_array[lev], ST, coarse_pnts_global, 1, dof_func_buff,
                         	debug_flag, trunc_factor, max_elmts, col_offd_ST_to_AT, &RT, last_level, lev, num_restrict_sweeps);      

      RT_array[lev] = RT;

      /* Compute RAP for next level */
      hypre_BoomerAMGBuildCoarseOperator(RT, A_array[lev], P, &RAP_ptr);   
//      (systg_data -> RAP) = RAP_ptr;

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
   (systg_data->num_coarse_levels) = num_c_levels;
   (systg_data->RAP) = RAP_ptr;

   /* setup default coarse grid solver */
   /* default is BoomerAMG */
   if(use_default_cgrid_solver)
   {
      hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
      /* create and set default solver parameters here */
      default_cg_solver = hypre_BoomerAMGCreate();
      hypre_BoomerAMGSetMaxIter ( default_cg_solver, 1 );
      hypre_BoomerAMGSetRelaxOrder( default_cg_solver, 1);
      hypre_BoomerAMGSetPrintLevel(default_cg_solver, 0);
      /* set setup and solve functions */
      coarse_grid_solver_setup = hypre_BoomerAMGSetup;
      coarse_grid_solver_solve = hypre_BoomerAMGSolve;
      (systg_data -> coarse_grid_solver_setup) = coarse_grid_solver_setup;
      (systg_data -> coarse_grid_solver_solve) = coarse_grid_solver_solve;
      (systg_data -> coarse_grid_solver) = default_cg_solver;
   }
   /* setup coarse grid solver */
   coarse_grid_solver_setup((systg_data -> coarse_grid_solver), RAP_ptr, F_array[num_c_levels], U_array[num_c_levels]);

   /* Setup smoother for fine grid */
   if (	relax_type == 8 || relax_type == 13 || relax_type == 14 || relax_type == 18 )
   {
      l1_norms = hypre_CTAlloc(HYPRE_Real *, num_c_levels);
      (systg_data -> l1_norms) = l1_norms;
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
      (systg_data -> residual) = residual;      
   }
   else{
      (systg_data -> residual) = NULL;
   }
   rel_res_norms = hypre_CTAlloc(HYPRE_Real,(systg_data -> max_iter));
   (systg_data -> rel_res_norms) = rel_res_norms;  
 
   return hypre_error_flag;
}     

/* Solve */
HYPRE_Int
hypre_SysTGSolve( void               *systg_vdata,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{

   MPI_Comm 	         comm = hypre_ParCSRMatrixComm(A);   
   hypre_ParSysTGData   *systg_data = systg_vdata;

   hypre_ParCSRMatrix  **A_array = (systg_data -> A_array);
   hypre_ParVector    **F_array = (systg_data -> F_array);
   hypre_ParVector    **U_array = (systg_data -> U_array);   

   HYPRE_Real		tol = (systg_data -> conv_tol);
   HYPRE_Int		logging = (systg_data -> logging);
   HYPRE_Int		print_level = (systg_data -> print_level);
   HYPRE_Int		max_iter = (systg_data -> max_iter);
   HYPRE_Real		*norms = (systg_data -> rel_res_norms);
   hypre_ParVector     	*Vtemp = (systg_data -> Vtemp);
   hypre_ParVector     	*residual;   
   
   HYPRE_Real           alpha = -1;
   HYPRE_Real           beta = 1;
   HYPRE_Real           conv_factor = 0.0;
   HYPRE_Real   	resnorm = 1.0;
   HYPRE_Real   	init_resnorm = 0.0;
   HYPRE_Real   	rel_resnorm;   
   HYPRE_Real   	res_resnorm;
   HYPRE_Real   	rhs_norm = 0.0;
   HYPRE_Real   	old_resnorm;
   HYPRE_Real   	ieee_check = 0.;   
   
   HYPRE_Int		iter, num_procs, my_id;
   HYPRE_Int		Solve_err_flag;

   HYPRE_Real   total_coeffs;
   HYPRE_Real   total_variables;
   HYPRE_Real   operat_cmplxty;
   HYPRE_Real   grid_cmplxty;

   HYPRE_Solver    	*cg_solver = (systg_data -> coarse_grid_solver);     
   HYPRE_Int		(*coarse_grid_solver_solve)() = (systg_data -> coarse_grid_solver_solve); 
   
   if(logging > 1)
   {
      residual = (systg_data -> residual);
   }

   (systg_data -> num_iterations) = 0;
   
   if((systg_data -> max_num_coarse_levels) == 0)
   {
      /* Do standard AMG solve when only one level */
      coarse_grid_solver_solve(cg_solver, A, f, u);     
      return hypre_error_flag;
   }    

   A_array[0] = A;
   U_array[0] = u;
   F_array[0] = f;

   hypre_MPI_Comm_size(comm, &num_procs);   
   hypre_MPI_Comm_rank(comm,&my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/
//   if (my_id == 0 && print_level > 1)
//      hypre_SysTGWriteSolverParams(systg_data); 

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag and assorted bookkeeping variables
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;

   total_coeffs = 0;
   total_variables = 0;
   operat_cmplxty = 0;
   grid_cmplxty = 0;

   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && tol > 0.)
     hypre_printf("\n\nTWO-GRID SOLVER SOLUTION INFO:\n");


   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print 
    *-----------------------------------------------------------------------*/
   if (print_level > 1 || logging > 1 || tol > 0.)
   {  
     if ( logging > 1 ) {
        hypre_ParVectorCopy(F_array[0], residual );
        if (tol > 0)
	   hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, residual );
           resnorm = sqrt(hypre_ParVectorInnerProd( residual, residual ));
     }
     else {
        hypre_ParVectorCopy(F_array[0], Vtemp);
        if (tol > 0)
           hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
        resnorm = sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
     }

     /* Since it is does not diminish performance, attempt to return an error flag
        and notify users when they supply bad input. */
     if (resnorm != 0.) ieee_check = resnorm/resnorm; /* INF -> NaN conversion */
     if (ieee_check != ieee_check)
     {
        /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
           for ieee_check self-equality works on all IEEE-compliant compilers/
           machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
           by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
           found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
        if (print_level > 0)
        {
          hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
          hypre_printf("ERROR -- hypre_StsTGSolve: INFs and/or NaNs detected in input.\n");
          hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
          hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
        }
        hypre_error(HYPRE_ERROR_GENERIC);
        return hypre_error_flag;
     }

     init_resnorm = resnorm;
     rhs_norm = sqrt(hypre_ParVectorInnerProd(f, f));
     if (rhs_norm)
     {
       rel_resnorm = init_resnorm / rhs_norm;
     }
     else
     {
       /* rhs is zero, return a zero solution */
       hypre_ParVectorSetConstantValues(U_array[0], 0.0);
       if(logging > 0)
       {
          rel_resnorm = 0.0;
          (systg_data -> final_rel_residual_norm) = rel_resnorm;
       }
       return hypre_error_flag;
     }
   }
   else
   {
     rel_resnorm = 1.;
   }

   if (my_id == 0 && print_level > 1)
   {     
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n",init_resnorm,
              rel_resnorm);
   }
   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;
   while ((rel_resnorm >= tol || iter < 1)
          && iter < max_iter)
   {
      /* Do one cycle of reduction solve */      
      /**** Write SysTGCycle() or solver -- until convergence is reached in this loop as in par_amg_solve*/
      /* Search for matvecT in par_cycle for ideas */      
      hypre_SysTGCycle(systg_data, F_array, U_array);

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.)
      {
        old_resnorm = resnorm;

        if ( logging > 1 ) {
           hypre_ParVectorCopy(F_array[0], residual);
           hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, residual );
           resnorm = sqrt(hypre_ParVectorInnerProd( residual, residual ));
        }
        else {
           hypre_ParVectorCopy(F_array[0], Vtemp);
           hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
           resnorm = sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
        }

        if (old_resnorm) conv_factor = resnorm / old_resnorm;
        else conv_factor = resnorm;
        if (rhs_norm)
        {
           rel_resnorm = resnorm / rhs_norm;
        }
        else
        {
           rel_resnorm = resnorm;
        }

        norms[iter] = rel_resnorm;
      }

      ++iter;
      (systg_data -> num_iterations) = iter;
      (systg_data -> final_rel_residual_norm) = rel_resnorm;

      if (my_id == 0 && print_level > 1)
      { 
         hypre_printf("    Cycle %2d   %e    %f     %e \n", iter,
                 resnorm, conv_factor, rel_resnorm);
      }     
   }

   /* check convergence within max_iter */
   if (iter == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      hypre_error(HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Print closing statistics
    *	 Add operator and grid complexity stats
    *-----------------------------------------------------------------------*/

   if (iter > 0 && init_resnorm) 
     conv_factor = pow((resnorm/init_resnorm),(1.0/(HYPRE_Real) iter));
   else
     conv_factor = 1.;

   if (print_level > 1) 
   {
      /*** compute operator and grid complexities here ?? ***/
      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            hypre_printf("\n\n==============================================");
            hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            hypre_printf("      within the allowed %d iterations\n",max_iter);
            hypre_printf("==============================================");
         }
         hypre_printf("\n\n Average Convergence Factor = %f \n",conv_factor);
         hypre_printf(" Number of coarse levels = %d \n",(systg_data -> num_coarse_levels));
//         hypre_printf("\n\n     Complexity:    grid = %f\n",grid_cmplxty);
//         hypre_printf("                operator = %f\n",operat_cmplxty);
//         hypre_printf("                   cycle = %f\n\n\n\n",cycle_cmplxty);
      }
   }
   
   return hypre_error_flag;
}

HYPRE_Int
hypre_SysTGCycle( void               *systg_vdata,
                  hypre_ParVector    **F_array,
                  hypre_ParVector    **U_array )
{
   MPI_Comm 	         comm;   
   hypre_ParSysTGData   *systg_data = systg_vdata;
   
   HYPRE_Int       Solve_err_flag;   
   HYPRE_Int       level;   
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       Not_Finished;
   HYPRE_Int	   cycle_type;

   hypre_ParCSRMatrix  	**A_array = (systg_data -> A_array); 
   hypre_ParCSRMatrix  	**RT_array  = (systg_data -> RT_array);
   hypre_ParCSRMatrix  	**P_array   = (systg_data -> P_array);
   hypre_ParCSRMatrix  	*RAP = (systg_data -> RAP);
   HYPRE_Solver    	*cg_solver = (systg_data -> coarse_grid_solver);     
   HYPRE_Int		(*coarse_grid_solver_solve)() = (systg_data -> coarse_grid_solver_solve);    
   HYPRE_Int		(*coarse_grid_solver_setup)() = (systg_data -> coarse_grid_solver_setup); 

   HYPRE_Int           	**CF_marker = (systg_data -> CF_marker_array);
   HYPRE_Int            nsweeps = (systg_data -> num_relax_sweeps);   
   HYPRE_Int            relax_type = (systg_data -> relax_type);
   HYPRE_Real           relax_weight = (systg_data -> relax_weight);
   HYPRE_Real           relax_order = (systg_data -> relax_order);
   HYPRE_Real           omega = (systg_data -> omega);
   HYPRE_Real          	**relax_l1_norms = (systg_data -> l1_norms);
   hypre_ParVector     	*Vtemp = (systg_data -> Vtemp);
   hypre_ParVector     	*Ztemp = (systg_data -> Ztemp);    
   hypre_ParVector    	*Aux_U;
   hypre_ParVector    	*Aux_F;

   HYPRE_Int            i, relax_points; 
   HYPRE_Int           	num_coarse_levels = (systg_data -> num_coarse_levels);  
      
   HYPRE_Real    alpha;    
   HYPRE_Real    beta;   
   
   /* Initialize */
   comm = hypre_ParCSRMatrixComm(A_array[0]);
   Solve_err_flag = 0;  
   Not_Finished = 1;
   cycle_type = 1;
   level = 0;
  
   /***** Main loop ******/
   while (Not_Finished)
   {
      /* Do coarse grid correction solve */
      if(cycle_type == 3)
      {
         /* call coarse grid solver here *
         /* default is BoomerAMG */
         coarse_grid_solver_solve(cg_solver, RAP, F_array[level], U_array[level]);
         /**** cycle up ***/
         cycle_type = 2;
      }
      /* restrict */
      else if(cycle_type == 1)
      {

         fine_grid = level;
         coarse_grid = level + 1;

         hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0); 
          
         hypre_ParVectorCopy(F_array[fine_grid],Vtemp);
         alpha = -1.0;
         beta = 1.0;

         hypre_ParCSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
                                     beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;

         hypre_ParCSRMatrixMatvecT(alpha,RT_array[fine_grid],Vtemp,
                                      beta,F_array[coarse_grid]);

         ++level;
         cycle_type = 1;
         if (level == num_coarse_levels) cycle_type = 3;         
      }
      else if(level != 0)
      {
         /* Interpolate */

         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

         hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid], 
                                     U_array[coarse_grid],
                                     beta, U_array[fine_grid]);            
         
         /* Relax solution - F-relaxation */
         relax_points = -1;
         if (relax_type == 18)
         {   /* L1 - Jacobi*/
             hypre_ParCSRRelax_L1_Jacobi(A_array[fine_grid], F_array[fine_grid], CF_marker[fine_grid],
                                            relax_points, relax_weight, relax_l1_norms[fine_grid], 
                                            U_array[fine_grid], Vtemp);
         }
         else if(relax_type == 8 || relax_type == 13 || relax_type == 14)
         {
             hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid], CF_marker[fine_grid], 
      	 			                   relax_type, relax_points, relax_weight,
      	 			                   omega, relax_l1_norms[fine_grid], U_array[fine_grid], Vtemp, Ztemp);             
         }
         else
         {
            for(i=0; i<nsweeps; i++)
              Solve_err_flag = hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid], CF_marker[fine_grid], 
      	 			                   relax_type, relax_points, relax_weight,
      	 			                   omega, NULL, U_array[fine_grid], Vtemp, Ztemp);

/*
	    hypre_BoomerAMGRelax_FCFJacobi(A_array[fine_grid], 
                                              F_array[fine_grid],
                                              CF_marker[fine_grid],
                                              relax_weight,
                                              U_array[fine_grid],
                                              Vtemp);
*/
      	 }

         if (Solve_err_flag != 0)
             return(Solve_err_flag);
             
         --level;
         cycle_type = 2;
      }
      else
      {
         Not_Finished = 0;
      }            
   }
      
   return Solve_err_flag;
}

/* set coarse grid solver */
HYPRE_Int
hypre_SysTGSetCoarseSolver( void  *systg_vdata,
                       HYPRE_Int  (*coarse_grid_solver_solve)(),
                       HYPRE_Int  (*coarse_grid_solver_setup)(),
                       void  *coarse_grid_solver )
{
   hypre_ParSysTGData *systg_data = systg_vdata;

   if (!systg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
 
   (systg_data -> coarse_grid_solver_solve) = coarse_grid_solver_solve;
   (systg_data -> coarse_grid_solver_setup) = coarse_grid_solver_setup;
   (systg_data -> coarse_grid_solver)       = coarse_grid_solver;
   
   (systg_data -> use_default_cgrid_solver) = 0;
  
   return hypre_error_flag;
}

/* Set the maximum number of coarse levels. 
 * maxcoarselevs = 1 yields the default 2-grid scheme.
*/
HYPRE_Int
hypre_SysTGSetMaxCoarseLevels( void *systg_vdata, HYPRE_Int maxcoarselevs )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> max_num_coarse_levels) = maxcoarselevs;
   return hypre_error_flag;
}
/* Set the system block size */
HYPRE_Int
hypre_SysTGSetBlockSize( void *systg_vdata, HYPRE_Int bsize )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> block_size) = bsize;
   return hypre_error_flag;
}
/* Set the relaxation type for the fine levels of the reduction. 
 * Currently supports the following flavors of relaxation types 
 * as described in the documentation:
 * relax_types 0 - 8, 13, 14, 18, 19, 98.
 * See par_relax.c and par_relax_more.c for more details.
 *
*/
HYPRE_Int
hypre_SysTGSetRelaxType( void *systg_vdata, HYPRE_Int relax_type )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> relax_type) = relax_type;
   return hypre_error_flag;
}
/* Set the number of relaxation sweeps */
HYPRE_Int
hypre_SysTGSetNumRelaxSweeps( void *systg_vdata, HYPRE_Int nsweeps )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> num_relax_sweeps) = nsweeps;
   return hypre_error_flag;
}
/* Set the number of Jacobi interpolation iterations 
 * for computing interpolation operator
*/
HYPRE_Int
hypre_SysTGSetNumInterpSweeps( void *systg_vdata, HYPRE_Int nsweeps )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> num_interp_sweeps) = nsweeps;
   return hypre_error_flag;
}
/* Set print level for systg solver */
HYPRE_Int
hypre_SysTGSetPrintLevel( void *systg_vdata, HYPRE_Int print_level )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> print_level) = print_level;
   return hypre_error_flag;
}
/* Set print level for systg solver */
HYPRE_Int
hypre_SysTGSetLogging( void *systg_vdata, HYPRE_Int logging )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> logging) = logging;
   return hypre_error_flag;
}
/* Set max number of iterations for systg solver */
HYPRE_Int
hypre_SysTGSetMaxIters( void *systg_vdata, HYPRE_Int max_iter )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> max_iter) = max_iter;
   return hypre_error_flag;
}
/* Set convergence tolerance for systg solver */
HYPRE_Int
hypre_SysTGSetConvTol( void *systg_vdata, HYPRE_Real conv_tol )
{
   hypre_ParSysTGData   *systg_data = systg_vdata;
   (systg_data -> conv_tol) = conv_tol;
   return hypre_error_flag;
}

