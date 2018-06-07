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
 * ILU solve routine
 *
 *****************************************************************************/
#include "_hypre_parcsr_ls.h"
#include "par_ilu.h"

/*--------------------------------------------------------------------
 * hypre_ILUSolve
 *--------------------------------------------------------------------*/
HYPRE_Int
hypre_ILUSolve( void               *ilu_vdata,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{
   MPI_Comm 	         comm = hypre_ParCSRMatrixComm(A);
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

//   HYPRE_Int  ilu_type = (ilu_data -> ilu_type);
   HYPRE_Int * perm = (ilu_data -> perm);
   hypre_ParCSRMatrix  *matA = (ilu_data -> matA);
   hypre_ParCSRMatrix  *matL = (ilu_data -> matL);
   HYPRE_Real  *matD = (ilu_data -> matD);	
   hypre_ParCSRMatrix  *matU = (ilu_data -> matU);
	
   HYPRE_Int		   iter, num_procs,  my_id;

   hypre_ParVector    *F_array = (ilu_data -> F);
   hypre_ParVector    *U_array = (ilu_data -> U);


   HYPRE_Real		tol = (ilu_data -> tol);
   HYPRE_Int		logging = (ilu_data -> logging);
   HYPRE_Int		print_level = (ilu_data -> print_level);
   HYPRE_Int		max_iter = (ilu_data -> max_iter);
   HYPRE_Real		*norms = (ilu_data -> rel_res_norms);
   hypre_ParVector     	*Ftemp = (ilu_data -> Ftemp);
   hypre_ParVector     	*Utemp = (ilu_data -> Utemp);
   hypre_ParVector     	*residual;

   HYPRE_Real           alpha = -1;
   HYPRE_Real           beta = 1;
   HYPRE_Real           conv_factor = 0.0;
   HYPRE_Real   	resnorm = 1.0;
   HYPRE_Real   	init_resnorm = 0.0;
   HYPRE_Real   	rel_resnorm;
   HYPRE_Real   	rhs_norm = 0.0;
   HYPRE_Real   	old_resnorm;
   HYPRE_Real   	ieee_check = 0.;
   HYPRE_Real		operat_cmplxty = (ilu_data -> operator_complexity);

   HYPRE_Int		Solve_err_flag;
   
//   HYPRE_Int            n = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int            nLU = (ilu_data -> nLU);   
   
   /* begin */
      
   if(logging > 1)
   {
      residual = (ilu_data -> residual);
   }

   (ilu_data -> num_iterations) = 0;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/
   if (my_id == 0 && print_level > 1)
      hypre_ILUWriteSolverParams(ilu_data);

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;   
   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && tol > 0.)
     hypre_printf("\n\n ILU SOLVER SOLUTION INFO:\n");


   /*-----------------------------------------------------------------------
    *    Compute initial residual and print
    *-----------------------------------------------------------------------*/
   if (print_level > 1 || logging > 1 || tol > 0.)
   {
     if ( logging > 1 ) {
        hypre_ParVectorCopy(f, residual );
        if (tol > 0.0)
	   hypre_ParCSRMatrixMatvec(alpha, A, u, beta, residual );
           resnorm = sqrt(hypre_ParVectorInnerProd( residual, residual ));
     }
     else {
        hypre_ParVectorCopy(f, Ftemp);
        if (tol > 0.0)
           hypre_ParCSRMatrixMatvec(alpha, A, u, beta, Ftemp);
        resnorm = sqrt(hypre_ParVectorInnerProd(Ftemp, Ftemp));
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
          hypre_printf("ERROR -- hypre_ILUSolve: INFs and/or NaNs detected in input.\n");
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
       hypre_ParVectorSetConstantValues(U_array, 0.0);
       if(logging > 0)
       {
          rel_resnorm = 0.0;
          (ilu_data -> final_rel_residual_norm) = rel_resnorm;
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

   matA = A;
   U_array = u;
   F_array = f;

   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;
   while ((rel_resnorm >= tol || iter < 1)
          && iter < max_iter)
   {

      /* Do one solve on LUe=r */
      hypre_ILUSolveLU(matA, f, u, perm, nLU, matL, matD, matU, Utemp, Ftemp);

      /*---------------------------------------------------------------
       *    Compute residual and residual norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.)
      {
        old_resnorm = resnorm;

        if ( logging > 1 ) {
           hypre_ParVectorCopy(F_array, residual);
           hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, residual );
           resnorm = sqrt(hypre_ParVectorInnerProd( residual, residual ));
        }
        else {
           hypre_ParVectorCopy(F_array, Ftemp);
           hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, Ftemp);
           resnorm = sqrt(hypre_ParVectorInnerProd(Ftemp, Ftemp));
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
      (ilu_data -> num_iterations) = iter;
      (ilu_data -> final_rel_residual_norm) = rel_resnorm;

      if (my_id == 0 && print_level > 1)
      {
         hypre_printf("    ILUSolve %2d   %e    %f     %e \n", iter,
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
      /*** compute operator and grid complexity (fill factor) here ?? ***/
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
         hypre_printf("                operator = %f\n",operat_cmplxty);
      }
   }

   return hypre_error_flag;
}

/* Incomplete LU solve
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the 
 * L and U factors are local.
*/

HYPRE_Int
hypre_ILUSolveLU(hypre_ParCSRMatrix *A, hypre_ParVector    *f,
                  hypre_ParVector    *u, HYPRE_Int *perm, 
                  HYPRE_Int nLU, hypre_ParCSRMatrix *L, 
                  HYPRE_Real* D, hypre_ParCSRMatrix *U,
                  hypre_ParVector *ftemp, hypre_ParVector *utemp)
{
   hypre_CSRMatrix *L_diag = hypre_ParCSRMatrixDiag(L);
   HYPRE_Real   *L_diag_data = hypre_CSRMatrixData(L_diag);
   HYPRE_Int    *L_diag_i = hypre_CSRMatrixI(L_diag);
   HYPRE_Int    *L_diag_j = hypre_CSRMatrixJ(L_diag);

   hypre_CSRMatrix *U_diag = hypre_ParCSRMatrixDiag(U);
   HYPRE_Real   *U_diag_data = hypre_CSRMatrixData(U_diag);
   HYPRE_Int    *U_diag_i = hypre_CSRMatrixI(U_diag);
   HYPRE_Int    *U_diag_j = hypre_CSRMatrixJ(U_diag);
   
   hypre_Vector   *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real     *utemp_data  = hypre_VectorData(utemp_local);
   
   hypre_Vector   *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real     *ftemp_data  = hypre_VectorData(ftemp_local);      

   HYPRE_Real    alpha;
   HYPRE_Real    beta;   
   HYPRE_Int     i, j, k1, k2;

   /* begin */
   alpha = -1.0;
   beta = 1.0;

   /* Initialize Utemp to zero. 
    * This is necessary for correctness, when we use optimized 
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
   */
   hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);
   
   /* L solve - Forward solve */
   /* copy rhs to account for diagonal of L (which is identity) */
   for( i = 0; i < nLU; i++ )
   {
      utemp_data[perm[i]] = ftemp_data[perm[i]];  
   } 
   /* update with remaining (off-diagonal) entries of L */     
   for( i = 0; i < nLU; i++ ) {
      k1 = L_diag_i[i] ; k2 = L_diag_i[i+1];
      for(j=k1; j <k2; j++) {
	   utemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
      }
    }
    /*-------------------- U solve - Backward substitution */    
    for( i = nLU-1; i >= 0; i-- ) {
        /* first update with the remaining (off-diagonal) entries of U */
	k1 = U_diag_i[i] ; k2 = U_diag_i[i+1];
	for(j=k1; j <k2; j++) {
           utemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[U_diag_j[j]]];
        }
        /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
        utemp_data[perm[i]] *= D[i];        
    }   
    /* Update solution */
    hypre_ParVectorAxpy(beta, utemp, u); 
             
   return hypre_error_flag;
}
