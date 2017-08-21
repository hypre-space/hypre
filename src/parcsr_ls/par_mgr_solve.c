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





/******************************************************************************
 *
 * MGR solve routine
 *
 *****************************************************************************/
#include "_hypre_parcsr_ls.h"
#include "par_mgr.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * hypre_MGRSolve
 *--------------------------------------------------------------------*/
HYPRE_Int
hypre_MGRSolve( void               *mgr_vdata,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{

   MPI_Comm 	         comm = hypre_ParCSRMatrixComm(A);
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   hypre_ParCSRMatrix  **A_array = (mgr_data -> A_array);
   hypre_ParVector    **F_array = (mgr_data -> F_array);
   hypre_ParVector    **U_array = (mgr_data -> U_array);

   HYPRE_Real		tol = (mgr_data -> conv_tol);
   HYPRE_Int		logging = (mgr_data -> logging);
   HYPRE_Int		print_level = (mgr_data -> print_level);
   HYPRE_Int		max_iter = (mgr_data -> max_iter);
   HYPRE_Real		*norms = (mgr_data -> rel_res_norms);
   hypre_ParVector     	*Vtemp = (mgr_data -> Vtemp);
   hypre_ParVector     	*Ztemp = (mgr_data -> Ztemp);
   hypre_ParVector     	*Utemp = (mgr_data -> Utemp);
   hypre_ParVector     	*Ftemp = (mgr_data -> Ftemp);
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

   HYPRE_Solver    	*cg_solver = (mgr_data -> coarse_grid_solver);
   HYPRE_Int		(*coarse_grid_solver_solve)(void*,void*,void*,void*) = (mgr_data -> coarse_grid_solver_solve);

   HYPRE_Int    reserved_coarse_size = (mgr_data -> reserved_coarse_size);
   HYPRE_Int    blk_size  = (mgr_data -> block_size);
   HYPRE_Real    *diaginv = (mgr_data -> diaginv);
   HYPRE_Int      n_block = (mgr_data -> n_block);
   HYPRE_Int    left_size = (mgr_data -> left_size);

   HYPRE_Int    global_smooth      =  (mgr_data -> global_smooth);
   HYPRE_Int    global_smooth_type =  (mgr_data -> global_smooth_type);
   HYPRE_Int    splitting_strategy =  (mgr_data -> splitting_strategy);

   int i,j,k;

   if(logging > 1)
   {
      residual = (mgr_data -> residual);
   }

   (mgr_data -> num_iterations) = 0;

   if((mgr_data -> max_num_coarse_levels) == 0)
   {
      /* Do standard AMG solve when only one level */
      coarse_grid_solver_solve(cg_solver, A, f, u);
      return hypre_error_flag;
   }

   U_array[0] = u;
   F_array[0] = f;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/
//   if (my_id == 0 && print_level > 1)
//      hypre_MGRWriteSolverParams(mgr_data);

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
          hypre_printf("ERROR -- hypre_MGRSolve: INFs and/or NaNs detected in input.\n");
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
          (mgr_data -> final_rel_residual_norm) = rel_resnorm;
       }
       return hypre_error_flag;
     }
   }
   else
   {
     rel_resnorm = 1.;
   }

   //hypre_ParVectorSetConstantValues(U_array[0], 0.0);

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
    if (splitting_strategy == 0) {
	   // for (i = 0;i < 10;i ++)
	   //   hypre_BoomerAMGRelax(A_array[0], F_array[0], NULL, 0, 0, 1.0, 0.0, NULL, U_array[0], Vtemp, NULL);
	   //hypre_blockRelax(A_array[0], F_array[0],U_array[0], blk_size,reserved_coarse_size,Vtemp,NULL);
	   if (global_smooth_type == 0)//block Jacobi smoother
	   {
	   	   for (i = 0;i < global_smooth;i ++)
	  		   hypre_block_jacobi(A_array[0],F_array[0],U_array[0],blk_size,n_block,left_size,diaginv,Vtemp);
	   }
	   else if (global_smooth_type == 1 ||global_smooth_type == 6)
	   {

		   for (i = 0;i < global_smooth;i ++)
			   hypre_BoomerAMGRelax(A_array[0], F_array[0], NULL, global_smooth_type-1, 0, 1.0, 0.0, NULL, U_array[0], Vtemp, NULL);
	   }
	   else if (global_smooth_type == 3)//ILU smoother
	   {
		   for (i = 0;i < global_smooth;i ++)
			   HYPRE_EuclidSolve( (mgr_data -> global_smoother),A_array[0],F_array[0],U_array[0]);

	   }
    }


      /* compute residual and reset pointers for MGR cycle */
//      hypre_ParVectorCopy(F_array[0], Ftemp);
//      hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Ftemp);
      // set pointer to residual
//      F_array[0] = Ftemp;
      // initial guess/ solution for error
//      hypre_ParVectorSetConstantValues(Utemp, 0.0);
      // pointer to initial guess/ solution
//      U_array[0] = Utemp;

      /* Do one cycle of reduction solve on Ae=r */
      hypre_MGRCycle(mgr_data, F_array, U_array);

      /* Done with MGR cycle. Update solution and Reset pointers to solution and rhs.
       * Note: Utemp = U_array[0] holds the error update
       */
      // set pointers to problem rhs and initial guess/solution
//      F_array[0] = f;
//      U_array[0] = u;
      // update solution with computed error correction
//      hypre_ParVectorAxpy(beta, Utemp, U_array[0]);

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
      (mgr_data -> num_iterations) = iter;
      (mgr_data -> final_rel_residual_norm) = rel_resnorm;

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
         hypre_printf(" Number of coarse levels = %d \n",(mgr_data -> num_coarse_levels));
//         hypre_printf("\n\n     Complexity:    grid = %f\n",grid_cmplxty);
//         hypre_printf("                operator = %f\n",operat_cmplxty);
//         hypre_printf("                   cycle = %f\n\n\n\n",cycle_cmplxty);
      }
   }

   return hypre_error_flag;
}


HYPRE_Int
hypre_MGRCycle( void               *mgr_vdata,
                  hypre_ParVector    **F_array,
                  hypre_ParVector    **U_array )
{
   MPI_Comm 	         comm;
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   HYPRE_Int       Solve_err_flag;
   HYPRE_Int       level;
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       Not_Finished;
   HYPRE_Int	   cycle_type;

   hypre_ParCSRMatrix  	**A_array = (mgr_data -> A_array);
   hypre_ParCSRMatrix  	**RT_array  = (mgr_data -> RT_array);
   hypre_ParCSRMatrix  	**P_array   = (mgr_data -> P_array);
  hypre_ParCSRMatrix   **P_f_array = (mgr_data -> P_f_array);
   hypre_ParCSRMatrix  	*RAP = (mgr_data -> RAP);
  hypre_ParCSRMatrix   **A_ff_array = (mgr_data -> A_ff_array);
   HYPRE_Solver    	*cg_solver = (mgr_data -> coarse_grid_solver);
   HYPRE_Int		(*coarse_grid_solver_solve)(void*, void*, void*, void*) = (mgr_data -> coarse_grid_solver_solve);
   HYPRE_Int		(*coarse_grid_solver_setup)(void*, void*, void*, void*) = (mgr_data -> coarse_grid_solver_setup);

  HYPRE_Solver      *fg_solver = (mgr_data -> fine_grid_solver);
  HYPRE_Int      (*fine_grid_solver_setup)(void*, void*, void*, void*) = (mgr_data -> fine_grid_solver_setup);
  HYPRE_Int      (*fine_grid_solver_solve)(void*, void*, void*, void*) = (mgr_data -> fine_grid_solver_solve);

   HYPRE_Int           	**CF_marker = (mgr_data -> CF_marker_array);
   HYPRE_Int            nsweeps = (mgr_data -> num_relax_sweeps);
   HYPRE_Int            relax_type = (mgr_data -> relax_type);
  HYPRE_Int            relax_method = (mgr_data -> relax_method);
   HYPRE_Real           relax_weight = (mgr_data -> relax_weight);
   HYPRE_Real           relax_order = (mgr_data -> relax_order);
   HYPRE_Real           omega = (mgr_data -> omega);
   HYPRE_Real          	**relax_l1_norms = (mgr_data -> l1_norms);
   hypre_ParVector     	*Vtemp = (mgr_data -> Vtemp);
   hypre_ParVector     	*Ztemp = (mgr_data -> Ztemp);
   hypre_ParVector     	*Utemp = (mgr_data -> Utemp);
  hypre_ParVector      **U_fine_array = (mgr_data -> U_fine_array);
  hypre_ParVector      **F_fine_array = (mgr_data -> F_fine_array);
   hypre_ParVector    	*Aux_U;
   hypre_ParVector    	*Aux_F;

   HYPRE_Int            i, relax_points;
   HYPRE_Int           	num_coarse_levels = (mgr_data -> num_coarse_levels);

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
		   /* call coarse grid solver here */
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
		   /* Relax solution - F-relaxation */
		   relax_points = -1;
      //hypre_ParVectorPrintIJ(F_array[fine_grid], 0, "F_array");
      //hypre_ParVectorPrintIJ(U_array[fine_grid], 0, "U_array");
      //hypre_ParCSRMatrixPrintIJ(P_f, 1, 1, "P_f.mat");

      Aux_F = hypre_ParVectorCreate(comm,
                      hypre_ParCSRMatrixGlobalNumRows(A_array[fine_grid]),
                      hypre_ParCSRMatrixRowStarts(A_array[fine_grid]));
      hypre_ParVectorInitialize(Aux_F);
      hypre_ParVectorSetPartitioningOwner(Aux_F, 0);
      hypre_ParVectorSetConstantValues(Aux_F, 0.0);

        if (relax_method == 0) { /* Original MGR relaxation for A_ff */
		   if (relax_type == 18)
		   {
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
		     }
      }
      else if (relax_method == 1) { /* Do a solve based on AMG or ILU for the A_ff part */
        /*
        // solve the A_ff part instead of doing a relaxation
        hypre_ParVectorCopy(F_array[fine_grid], Aux_F);
        hypre_ParCSRMatrixMatvec(-1.0, A_array[fine_grid], U_array[fine_grid],
                  1.0, Aux_F);
        hypre_ParCSRMatrixMatvecT(1.0, P_f, Aux_F, 0.0, F_fine);
        //hypre_ParCSRMatrixMatvecT(1.0, P_f, U_array[fine_grid], 0.0, U_fine);
        hypre_ParVectorSetConstantValues(U_fine, 0.0);
        hypre_ParVectorPrintIJ(F_fine, 0, "F_fine");
        //fine_grid_solver_solve(fg_solver, A_ff, F_fine, U_fine); // solve using AMG
        HYPRE_EuclidSolve( (mgr_data -> aff_solver), A_ff, F_fine, U_fine);
        hypre_ParVectorPrintIJ(U_fine, 0, "U_fine_euclid");
        hypre_ParCSRMatrixMatvec(1.0, P_f, U_fine, 1.0, U_array[fine_grid]);
        hypre_ParVectorPrintIJ(U_array[fine_grid], 0, "U_array_after_euclid");
        */
      }
      else if (relax_method == 99) {
      /* Experimental method for multilevel reduction with 
       * user-defined input data for choosing coarse grid */
        if (level == 0 || level == 2) {
          /* solve the A_ff part instead of doing a relaxation
           * for the first and next to last level. This correspond
           * to grouping all the constraints together and eliminating
           * the constraints that have zero diagonal from the final 
           * coarse grid. */
          hypre_ParVectorCopy(F_array[fine_grid], Aux_F);
          hypre_ParCSRMatrixMatvec(-1.0, A_array[fine_grid], U_array[fine_grid],
                    1.0, Aux_F);
          hypre_ParCSRMatrixMatvecT(1.0, P_f_array[fine_grid], Aux_F, 0.0, F_fine_array[coarse_grid]);
          //hypre_ParCSRMatrixMatvecT(1.0, P_f, U_array[fine_grid], 0.0, U_fine);
          hypre_ParVectorSetConstantValues(U_fine_array[coarse_grid], 0.0);
          //fine_grid_solver_solve(fg_solver, A_ff, F_fine, U_fine); // solve using AMG
          HYPRE_EuclidSolve( (mgr_data -> aff_solver)[fine_grid], A_ff_array[fine_grid], F_fine_array[coarse_grid], U_fine_array[coarse_grid]);
          hypre_ParCSRMatrixMatvec(1.0, P_f_array[fine_grid], U_fine_array[coarse_grid], 1.0, U_array[fine_grid]);
        } else {
          for (i=0; i<nsweeps; i++)
            //hypre_ParVectorPrintIJ(U_array[fine_grid], 0, "U_array_before_relax");
            Solve_err_flag = hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid], CF_marker[fine_grid],
                            relax_type, relax_points, relax_weight,
                            omega, NULL, U_array[fine_grid], Vtemp, Ztemp);
            //hypre_ParVectorPrintIJ(U_array[fine_grid], 0, "U_array_after_relax");
          }
      }


		   // Update residual and compute coarse-grid rhs
		   hypre_ParVectorCopy(F_array[fine_grid],Vtemp);
		   alpha = -1.0;
		   beta = 1.0;

		   hypre_ParCSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
									beta, Vtemp);

		   alpha = 1.0;
		   beta = 0.0;

		   hypre_ParCSRMatrixMatvecT(alpha,RT_array[fine_grid],Vtemp,
									 beta,F_array[coarse_grid]);

		   // initialize coarse grid solution array
		   hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0);

		   ++level;

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


		   if (Solve_err_flag != 0)
			   return(Solve_err_flag);

		   --level;

	   }
	   else
	   {
		   Not_Finished = 0;
	   }
   }
  //if (Aux_U && relax_method == 0) hypre_ParVectorDestroy(Aux_U);
  //if (Aux_F) hypre_ParVectorDestroy(Aux_F);

   return Solve_err_flag;
}
