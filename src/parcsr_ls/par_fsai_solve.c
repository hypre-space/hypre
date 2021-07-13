/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *    
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

/******************************************************************************
 *  
 * FSAI solve routine
 * 
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_fsai.h"

/*--------------------------------------------------------------------
 * hypre_FSAISolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISolve( void   *fsai_vdata,
                        hypre_ParCSRMatrix *A,
                        hypre_ParVector    *f,
                        hypre_ParVector    *u )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   hypre_ParFSAIData    *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   hypre_ParCSRMatrix   *Amat                = hypre_ParFSAIDataAmat(fsai_data); 
   hypre_ParCSRMatrix   *Gmat                = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParVector      *residual            = hypre_ParFSAIResidual(fsai_data);
   HYPRE_Int            tol                  = hypre_ParFSAITolerance(fsai_data);
   HYPRE_Int            max_iter             = hypre_ParFSAIMaxIterations(fsai_data);
   HYPRE_Int            print_level          = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int            logging              = hypre_ParFSAIDataLogging(fsai_data);
   hypre_ParVector      *F_array             = hypre_ParFSAIDataF(fsai_data);
   hypre_ParVector      *U_array             = hypre_ParFSAIDataU(fsai_data);
   hypre_ParVector      *Ftemp               = hypre_ParFSAIDataFTemp(fsai_data);
   hypre_ParVector      *Utemp               = hypre_ParFSAIDataUTemp(fsai_data);
   hypre_ParVector      *Xtemp               = hypre_ParFSAIDataXTemp(fsai_data);
   hypre_ParVector      *Ytemp               = hypre_ParFSAIDataYTemp(fsai_data);
   HYPRE_Real           *fext                = hypre_ParFSAIDataFExt(fsai_data);
   HYPRE_Real           *uext                = hypre_ParFSAIDataUExt(fsai_data);
   HYPRE_Real           *rel_res_norms       = hypre_ParFSAIRelativeResidualNorms(fsai_data);

   /* Local variables */

   HYPRE_Int            iter, num_procs, my_id;
   HYPRE_Int            n = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   HYPRE_Int            Solve_err_flag = 0;
   HYPRE_Real           alpha          = -1.0;
   HYPRE_Real           beta           = 1.0;
   HYPRE_Real           conv_factor    = 0.0;
   HYPRE_Real           resnorm        = 1.0;
   HYPRE_Real           init_resnorm   = 1.0;
   HYPRE_Real           rel_resnorm;
   HYPRE_Real           rhs_norm       = 0.0;
   HYPRE_Real           old_resnorm;
   HYPRE_Real           ieee_check     = 0.0;


   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_ParFSAINumIterations(fsai_data) = 0;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*----------------------------------------------
    * Initial Residual Info
    *----------------------------------------------*/

   if(my_id == 0 && print_level > 1 && tol > 0.0)
      hypre_printf("\n\n FSAI SOLVER SOLUTION INFO: \n");

   if(print_level > 1 || logging > 1 || tol > 0.0)
   {
      if(logging > 1)
      {
         hypre_ParVectorCopy(f, residual);
         if(tol > 0.0)
            hypre_ParCSRMatrixMatvec(alpha, A, u, beta, residual);
         resnorm = sqrt(hypre_ParVectorInnerProd(residual, residual));
      }
      else
      {
         hypre_ParVectorCopy(f, Ftemp);
         if(tol > 0.0)
            hypre_ParCSRMatrixMatvec(alpha, A, u, beta, Ftemp);
         resnorm = sqrt(hypre_ParVectorInnerProd(Ftemp, Ftemp));
      }
      
      /* When users supply bad input, return an error flag and notify them */
      if(resnorm != 0.0)
         ieee_check = resnorm/resnorm; /* INF -> NaN Conversion */

      if(ieee_check != ieee_check)
      {
         /* Copied from par_ilu_solve.c */
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
          * for ieee_check self-equality works on all IEEE-compliant compilers/
          * machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
          * by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
          * found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if(print_level > 1)
         {
            hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
            hypre_printf("ERROR -- hypre_FSAISolve: INFs and/or NaNs detected in input.\n");
            hypre_printf("User probably placed non-numerics in supplied A, x, or b.\n");
            hypre_printf("ERROR detected by Hypre ... END\n\n\n");
         }
         hypre_error(HYPRE_ERROR_GENERIC);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }  

      init_resnorm = resnorm;
      rhs_norm = sqrt(hypre_ParVectorInnerProd(f, f));

      if(rhs > HYPRE_REAL_EPSILON)
         rel_resnorm = init_resnorm/rhs_norm;
      else
      {
         /* Return zero solution if rhs is zero */
         hypre_ParVectorSetConstantValues(U_array, 0.0);
         if(logging > 0)
         {
            rel_resnorm = 0.0;
            hypre_ParFSAIFinalRelResidualNorm(fsai_data) = rel_resnorm; 
         }
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }
   }
   else
      rel_resnorm = 1.0;

   if(my_id == 0 && print_level > 1)
   {
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n", init_resnorm, rel_resnorm);
   }

   Amat     = A;
   U_array  = U;
   F_array  = F;

   /* Main solver loop - Do 1 iteration at least */
   iter = 0;

   do
   {

   }while(rel_resnorm >= tol && iter < max_iter);
 
   HYPRE_ANNOTATE_FUNC_END;

}
