/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/





#include "headers.h"
#include "amg.h"


#define DIAG 0

/*****************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

HYPRE_Int
hypre_AMGSetup( void            *amg_vdata,
                hypre_CSRMatrix *A,
                hypre_Vector    *f,
                hypre_Vector    *u         )
{
   hypre_AMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_CSRMatrix **A_array;
   hypre_BCSRMatrix **B_array;
   hypre_Vector    **F_array;
   hypre_Vector    **U_array;
   hypre_CSRMatrix **P_array;
   hypre_BCSRMatrix **PB_array;

   HYPRE_Int             **dof_func_array;
   HYPRE_Int              *dof_func;
   HYPRE_Int              *coarse_dof_func;   HYPRE_Int              *domain_i;
   HYPRE_Int              *domain_j;

   HYPRE_Int             **CF_marker_array;   
   double           *relax_weight;
   double           *unit_vector;
   double            strong_threshold;
   double            A_trunc_factor;
   double            P_trunc_factor;

   HYPRE_Int      num_variables;
   HYPRE_Int      max_levels; 
   HYPRE_Int      A_max_elmts;
   HYPRE_Int      P_max_elmts;
   HYPRE_Int      amg_ioutdat;
   HYPRE_Int      interp_type;
   HYPRE_Int      num_functions;
   HYPRE_Int      agg_levels;
   HYPRE_Int      agg_coarsen_type;
   HYPRE_Int      agg_interp_type;
   HYPRE_Int      num_jacs;
   HYPRE_Int      main_coarsen_type;
   HYPRE_Int      main_interp_type;
   HYPRE_Int use_block_flag;
 
   /* Local variables */
   HYPRE_Int              *CF_marker;
   HYPRE_Int              *new_CF_marker;
   hypre_CSRMatrix  *S;
   hypre_CSRMatrix  *S2;
/*   hypre_CSRMatrix  *S3;*/
   hypre_CSRMatrix  *P;
   hypre_CSRMatrix  *A_H;
   hypre_CSRMatrix  *A_tilde;
   hypre_BCSRMatrix *B;
   hypre_BCSRMatrix *PB;
/*   double *S2_data; */
/*   HYPRE_Int       num_nz; */

   HYPRE_Int       num_levels;
   HYPRE_Int       level;
   HYPRE_Int       coarse_size;
   HYPRE_Int       first_coarse_size;
   HYPRE_Int       fine_size;
   HYPRE_Int       not_finished_coarsening = 1;
   HYPRE_Int       Setup_err_flag;
   HYPRE_Int       coarse_threshold = 9;
   HYPRE_Int       i, j;
   HYPRE_Int	     coarsen_type;
   HYPRE_Int	    *grid_relax_type;
   HYPRE_Int	     relax_type;
   HYPRE_Int	     num_relax_steps;
   HYPRE_Int	    *schwarz_option;
   HYPRE_Int       mode, S_mode;
   HYPRE_Int       num_domains;
   HYPRE_Int      *i_domain_dof;
   HYPRE_Int      *j_domain_dof;
   double   *domain_matrixinverse;

   HYPRE_Int* fake_dof_func;

/*   char f_name[256];
     FILE* f_out; */

   mode = hypre_AMGDataMode(amg_data);
   S_mode = 0;
   relax_weight = hypre_AMGDataRelaxWeight(amg_data);
   num_relax_steps = hypre_AMGDataNumRelaxSteps(amg_data);
   grid_relax_type = hypre_AMGDataGridRelaxType(amg_data);
   max_levels = hypre_AMGDataMaxLevels(amg_data);
   amg_ioutdat = hypre_AMGDataIOutDat(amg_data);
   main_interp_type = hypre_AMGDataInterpType(amg_data);
   num_functions = hypre_AMGDataNumFunctions(amg_data);
   relax_type = grid_relax_type[0];
   schwarz_option = hypre_AMGDataSchwarzOption(amg_data);
   A_trunc_factor = hypre_AMGDataATruncFactor(amg_data);
   P_trunc_factor = hypre_AMGDataPTruncFactor(amg_data);
   P_max_elmts = hypre_AMGDataPMaxElmts(amg_data);
   A_max_elmts = hypre_AMGDataAMaxElmts(amg_data);
   use_block_flag = hypre_AMGDataUseBlockFlag(amg_data);
 
   dof_func = hypre_AMGDataDofFunc(amg_data);

   A_array = hypre_CTAlloc(hypre_CSRMatrix*, max_levels);
   B_array = hypre_CTAlloc(hypre_BCSRMatrix*, max_levels);
   P_array = hypre_CTAlloc(hypre_CSRMatrix*, max_levels-1);
   PB_array = hypre_CTAlloc(hypre_BCSRMatrix*, max_levels-1);
   CF_marker_array = hypre_CTAlloc(HYPRE_Int*, max_levels);
   dof_func_array = hypre_CTAlloc(HYPRE_Int*, max_levels);
   coarse_dof_func = NULL;

   if (schwarz_option[0] > -1)
   {
      hypre_AMGDataNumDomains(amg_data) = hypre_CTAlloc(HYPRE_Int, max_levels);
      hypre_AMGDataIDomainDof(amg_data) = hypre_CTAlloc(HYPRE_Int*, max_levels);
      hypre_AMGDataJDomainDof(amg_data) = hypre_CTAlloc(HYPRE_Int*, max_levels);
      hypre_AMGDataDomainMatrixInverse(amg_data) = 
				hypre_CTAlloc(double*, max_levels);
      for (i=0; i < max_levels; i++)
      {
         hypre_AMGDataIDomainDof(amg_data)[i] = NULL;
         hypre_AMGDataJDomainDof(amg_data)[i] = NULL;
         hypre_AMGDataDomainMatrixInverse(amg_data)[i] = NULL;
      }
   }

   if (num_functions > 0) dof_func_array[0] = dof_func;

   A_array[0] = A;

   use_block_flag = use_block_flag && (num_functions > 1);
   hypre_AMGDataUseBlockFlag(amg_data) = use_block_flag;
   if(use_block_flag) {
     A_tilde = hypre_CSRMatrixDeleteZeros(A, 0.0);
     if(A_tilde) {
       B = hypre_BCSRMatrixFromCSRMatrix(A_tilde, num_functions,
					 num_functions);
       hypre_CSRMatrixDestroy(A_tilde);
     }
     else {
       B = hypre_BCSRMatrixFromCSRMatrix(A, num_functions,
					 num_functions);
     }
     B_array[0] = B;
   }

   /*----------------------------------------------------------
    * Initialize hypre_AMGData
    *----------------------------------------------------------*/

   num_variables = hypre_CSRMatrixNumRows(A);
   unit_vector = hypre_CTAlloc(double, num_variables);

   for (i=0; i < num_variables; i++)
      unit_vector[i] = 1;

   hypre_AMGDataNumVariables(amg_data) = num_variables;

   not_finished_coarsening = 1;
   level = 0;
  
   strong_threshold = hypre_AMGDataStrongThreshold(amg_data);

   main_coarsen_type = hypre_AMGDataCoarsenType(amg_data);
   agg_levels = hypre_AMGDataAggLevels(amg_data);
   agg_coarsen_type = hypre_AMGDataAggCoarsenType(amg_data);
   agg_interp_type = hypre_AMGDataAggInterpType(amg_data);
   num_jacs = hypre_AMGDataNumJacs(amg_data);

   /*-----------------------------------------------------
    *  Enter Coarsening Loop
    *-----------------------------------------------------*/

   while (not_finished_coarsening)
   {
     /*****************************************************************/

      if (level < agg_levels)
      {
	 coarsen_type = agg_coarsen_type;
	 interp_type = agg_interp_type;
      }
      else
      {
	 coarsen_type = main_coarsen_type;
	 interp_type = main_interp_type;
      }

      if(use_block_flag) {
	A_tilde = hypre_BCSRMatrixCompress(B_array[level]);
	fine_size = hypre_CSRMatrixNumRows(A_tilde);
	fake_dof_func = hypre_CTAlloc(HYPRE_Int, fine_size); 
	hypre_AMGCreateS(A_tilde, strong_threshold, S_mode, fake_dof_func, &S);
	/* hypre_AMGCoarsen(A_tilde, strong_threshold, A_tilde,*/
	hypre_AMGCoarsen(A_tilde, strong_threshold, S,
			 &CF_marker, &coarse_size);
	hypre_TFree(fake_dof_func);
	hypre_CSRMatrixDestroy(A_tilde);

	CF_marker_array[level] = CF_marker;

#if DIAG
	PB = hypre_BCSRMatrixBuildInterpD(B_array[level], CF_marker,
					  S, coarse_size);
#else
	PB = hypre_BCSRMatrixBuildInterp(B_array[level], CF_marker,
					  S, coarse_size);
#endif

	P_array[level] = hypre_BCSRMatrixToCSRMatrix(PB);

#if 0 /* for debugging */
        {  
           char new_file[80];
           hypre_sprintf(new_file,"%s.level.%d","P.out" ,level);
           hypre_CSRMatrixPrint(P_array[level],  new_file);
        }
#endif

	if(P_trunc_factor > 0 || P_max_elmts > 0) {
	  hypre_AMGTruncation(P_array[level], P_trunc_factor, P_max_elmts);
	}

	hypre_BCSRMatrixDestroy(PB);
	PB_array[level] = hypre_BCSRMatrixFromCSRMatrix(P_array[level],
							num_functions,
							num_functions); 


	hypre_AMGBuildCoarseOperator(P_array[level], A_array[level],
				     P_array[level], &A_array[level + 1]);


#if 0 /* for debugging - to compare with parallel */
        {
           hypre_CSRMatrix *A_no_zeros = NULL;
           char new_file[80];
           hypre_sprintf(new_file,"%s.level.%d","RAP.out" ,level+1);
           A_no_zeros = hypre_CSRMatrixDeleteZeros(A_array[level+1], 1e-14);
           if (A_no_zeros)
           {
              hypre_CSRMatrixPrint(A_no_zeros,  new_file);
              hypre_CSRMatrixDestroy(A_no_zeros);
           }
           
           else
           {
               hypre_CSRMatrixPrint(A_array[level+1],  new_file);
           }
        }
        
#endif


	if(A_trunc_factor > 0 || A_max_elmts > 0) {
	  hypre_AMGOpTruncation(A_array[level + 1], 
				A_trunc_factor, A_max_elmts);
	}


	B_array[level + 1] = hypre_BCSRMatrixFromCSRMatrix(A_array[level + 1],
							   num_functions,
							   num_functions);

	level++;

	if (level+1 >= max_levels ||
	    coarse_size == fine_size || 
	    coarse_size <= coarse_threshold) {
	  not_finished_coarsening = 0;
	}

	continue;
      }
      /*****************************************************************/

      fine_size = hypre_CSRMatrixNumRows(A_array[level]);

      /*-------------------------------------------------------------
       * Select coarse-grid points on 'level' : returns CF_marker
       * for the level.  Returns strength matrix, S  
       *--------------------------------------------------------------*/

      if (relax_weight[level] == 0.0)
      {
	 hypre_CSRMatrixScaledNorm(A_array[level], &relax_weight[level]);
	 if (relax_weight[level] != 0.0)
            relax_weight[level] = (4.0/3.0)/relax_weight[level];
         else
           hypre_printf (" Warning ! Matrix norm is zero !!!");
      }

      if (schwarz_option[level] > -1)
      {
         /* if (level == 0) 
      	    hypre_AMGNodalSchwarzSmoother (A_array[level], dof_func_array[level],
				    	num_functions,
					schwarz_option[level],
					&i_domain_dof, &j_domain_dof,
					&domain_matrixinverse,
					&num_domains);
            else */
      	 hypre_AMGCreateDomainDof (A_array[level],
				&i_domain_dof, &j_domain_dof,
				&domain_matrixinverse,
				&num_domains); 
         hypre_AMGDataIDomainDof(amg_data)[level] = i_domain_dof;
         hypre_AMGDataJDomainDof(amg_data)[level] = j_domain_dof;
         hypre_AMGDataNumDomains(amg_data)[level] = num_domains;
         hypre_AMGDataDomainMatrixInverse(amg_data)[level] = 
			domain_matrixinverse;
      }

      if (coarsen_type == 1)
      {
	 hypre_AMGCreateS(A_array[level], strong_threshold, S_mode, 
	fake_dof_func, &S);
	 hypre_AMGCoarsenRuge(A_array[level], strong_threshold,
                       S, &CF_marker, &coarse_size); 
         /*              A_array[level], &CF_marker, &coarse_size); */
      }
      else if (coarsen_type == 2)
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
	 hypre_AMGCoarsenRugeLoL(A_array[level], strong_threshold,
				 S, &CF_marker, &coarse_size); 
			  /* A_array[level], &CF_marker, &coarse_size); */ 
      }
      /* begin HANS added */
      else if (coarsen_type == 4)
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
         hypre_AMGCoarsenwLJP(A_array[level], strong_threshold,
                       S, &CF_marker, &coarse_size); 
      }
      else if (coarsen_type == 5)
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
	 hypre_AMGCoarsenRugeOnePass(A_array[level], strong_threshold,
                       S, &CF_marker, &coarse_size); 
      }
      /* end HANS added */
      else if (coarsen_type == 8)
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
         hypre_AMGCoarsen(A_array[level], strong_threshold,
                       S, &CF_marker, &first_coarse_size);  
	 hypre_AMGCreate2ndS(S, first_coarse_size, CF_marker, 2, &S2);
         hypre_AMGCoarsen(S2, strong_threshold,
                       S2, &new_CF_marker, &coarse_size); 
         hypre_CSRMatrixDestroy(S2);
         hypre_AMGCorrectCFMarker(CF_marker, fine_size, new_CF_marker); 
	 hypre_TFree(new_CF_marker);
	 /* if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S2);
	 S3 = hypre_CSRMatrixMultiply(S2,S2);
	 hypre_AMGCompressS(S3, 2);
	 S = hypre_CSRMatrixAdd(S2,S3);
         hypre_CSRMatrixDestroy(S2);
         hypre_CSRMatrixDestroy(S3);
         hypre_AMGCoarsen(A_array[level], strong_threshold,
                       S, &CF_marker, &coarse_size);  */
      }
      else if (coarsen_type == 10)
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
         hypre_AMGCoarsen(A_array[level], strong_threshold,
                       S, &CF_marker, &first_coarse_size); 
	 hypre_AMGCreate2ndS(S, first_coarse_size, CF_marker, 1, &S2);
         hypre_AMGCoarsen(S2, strong_threshold,
                       S2, &new_CF_marker, &coarse_size); 
         hypre_CSRMatrixDestroy(S2);
         hypre_AMGCorrectCFMarker(CF_marker, fine_size, new_CF_marker); 
	 hypre_TFree(new_CF_marker);
	 /*if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S2);
	 S3 = hypre_CSRMatrixMultiply(S2,S2);
	 hypre_AMGCompressS(S3, 1);
	 S = hypre_CSRMatrixAdd(S2,S3);
         hypre_CSRMatrixDestroy(S2);
         hypre_CSRMatrixDestroy(S3);
         hypre_AMGCoarsen(A_array[level], strong_threshold,
                       S, &CF_marker, &coarse_size); */
			  /* A_array[level], &CF_marker, &coarse_size); */ 
      }
      else if (coarsen_type == 9)
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
	 hypre_AMGCoarsenRugeOnePass(A_array[level], strong_threshold,
                       S, &CF_marker, &first_coarse_size); 
	 hypre_AMGCreate2ndS(S, first_coarse_size, CF_marker, 2, &S2);
	 hypre_AMGCoarsenRugeOnePass(S2, strong_threshold,
                       S2, &new_CF_marker, &coarse_size); 
         hypre_CSRMatrixDestroy(S2);
         hypre_AMGCorrectCFMarker(CF_marker, fine_size, new_CF_marker); 
	 hypre_TFree(new_CF_marker);
	 /* if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S2);
         S2_data = hypre_CSRMatrixData(S2);
	 num_nz = hypre_CSRMatrixI(S2)[hypre_CSRMatrixNumRows(S2)];
         for (i=0; i < num_nz; i++)
	   S2_data[i] = -S2_data[i];
	 S3 = hypre_CSRMatrixMultiply(S2,S2);
	 hypre_AMGCompressS(S3, 2);
	 S = hypre_CSRMatrixAdd(S2,S3);
         hypre_CSRMatrixDestroy(S2);
         hypre_CSRMatrixDestroy(S3);
	 hypre_AMGCoarsenRugeLoL(A_array[level], -strong_threshold,
				 S, &CF_marker, &coarse_size); */
			  /* A_array[level], &CF_marker, &coarse_size); */ 
      }
      else if (coarsen_type == 11)
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
	 hypre_AMGCoarsenRugeOnePass(A_array[level], strong_threshold,
                       S, &CF_marker, &first_coarse_size); 
	 hypre_AMGCreate2ndS(S, first_coarse_size, CF_marker, 1, &S2);
	 hypre_AMGCoarsenRugeOnePass(S2, strong_threshold,
                       S2, &new_CF_marker, &coarse_size); 
         hypre_CSRMatrixDestroy(S2);
         hypre_AMGCorrectCFMarker(CF_marker, fine_size, new_CF_marker); 
	 hypre_TFree(new_CF_marker);
	 /* if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S2);
         S2_data = hypre_CSRMatrixData(S2);
	 num_nz = hypre_CSRMatrixI(S2)[hypre_CSRMatrixNumRows(S2)];
         for (i=0; i < num_nz; i++)
	   S2_data[i] = -S2_data[i];
	 S3 = hypre_CSRMatrixMultiply(S2,S2);
	 hypre_AMGCompressS(S3, 1);
	 S = hypre_CSRMatrixAdd(S2,S3);
         hypre_CSRMatrixDestroy(S2);
         hypre_CSRMatrixDestroy(S3);
	 hypre_AMGCoarsenRugeLoL(A_array[level], -strong_threshold,
				 S, &CF_marker, &coarse_size); */
			  /* A_array[level], &CF_marker, &coarse_size); */ 
      }
      else if (coarsen_type == 12)
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
         hypre_AMGCoarsenwLJP(A_array[level], strong_threshold,
                       S, &CF_marker, &first_coarse_size); 
	 hypre_AMGCreate2ndS(S, first_coarse_size, CF_marker, 1, &S2);
         hypre_AMGCoarsenwLJP(S2, strong_threshold,
                       S2, &new_CF_marker, &coarse_size); 
         hypre_CSRMatrixDestroy(S2);
         hypre_AMGCorrectCFMarker(CF_marker, fine_size, new_CF_marker); 
	 hypre_TFree(new_CF_marker);
      }
      else if (coarsen_type == 3)
      {
         hypre_AMGCoarsenCR(A_array[level], strong_threshold,
			relax_weight[level], relax_type, 
			num_relax_steps, &CF_marker, &coarse_size); 
      }
      else
      {
	 if (mode == 1 || mode == 2) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
         hypre_AMGCoarsen(A_array[level], strong_threshold,
                       S, &CF_marker, &coarse_size); 
                       /* A_array[level], &CF_marker, &coarse_size); */
      }
      /* if no coarse-grid, stop coarsening */
      if (coarse_size == 0)
         break;

      CF_marker_array[level] = CF_marker;
      
      /*-------------------------------------------------------------
       * Build prolongation matrix, P, and place in P_array[level] 
       *--------------------------------------------------------------*/

      if (interp_type == 2)
      {
          hypre_AMGBuildCRInterp(A_array[level], 
                                 CF_marker_array[level], 
                                 coarse_size,
                                 num_relax_steps,
                                 relax_type,
                                 relax_weight[level],
                                 &P);
      }
      else if (interp_type == 1)
      {
	  if (coarsen_type == 3)
             hypre_AMGBuildRBMInterp(A_array[level], 

                                  CF_marker_array[level], 
                                  A_array[level], 
                                  dof_func_array[level],
                                  num_functions,
                                  &coarse_dof_func,
                                  &P);
          else
             hypre_AMGBuildRBMInterp(A_array[level], 
                                  CF_marker_array[level], 
                                  S, 
                                  dof_func_array[level],
                                  num_functions,
                                  &coarse_dof_func,
                                  &P);
          /* this will need some cleanup, to make sure we do the right thing 
             when it is a scalar function */ 
      }
      else if (coarsen_type == 3)
      {
          hypre_AMGBuildInterp(A_array[level], CF_marker_array[level], 
					A_array[level], dof_func_array[level],
                                        &coarse_dof_func, &P);
      }
      else if (interp_type == 3)
      {
	 hypre_CreateDomain(CF_marker_array[level], A_array[level], coarse_size,
                             dof_func_array[level], &coarse_dof_func,
                             &domain_i, &domain_j);
         hypre_InexactPartitionOfUnityInterpolation(&P, 
				hypre_CSRMatrixI(A_array[level]), 
				hypre_CSRMatrixJ(A_array[level]), 
				hypre_CSRMatrixData(A_array[level]), 
				unit_vector, domain_i, domain_j,
				coarse_size, fine_size);
	 hypre_TFree(domain_i);
	 hypre_TFree(domain_j);
      }
      else if (interp_type == 5)
      {
          hypre_AMGBuildMultipass(A_array[level], 
                                 CF_marker_array[level], 
                                 S,
                             dof_func_array[level], &coarse_dof_func, &P);
      }
      else if (interp_type == 6)
      {
          hypre_AMGBuildMultipass(A_array[level], CF_marker_array[level], S,
                             dof_func_array[level], &coarse_dof_func, &P);
        for(i=0;i<num_jacs;i++)
	  hypre_AMGJacobiIterate(A_array[level], CF_marker_array[level], S,
				 dof_func_array[level], &coarse_dof_func, &P);
      }
      else 
      {
	 if (S) hypre_CSRMatrixDestroy(S);
	 S_mode = 0;
	 if (mode == 1) S_mode = 1;
	 hypre_AMGCreateS(A_array[level], strong_threshold, 
			S_mode, dof_func_array[level], &S);
	hypre_AMGBuildInterp(A_array[level], CF_marker_array[level], S,
                             dof_func_array[level], &coarse_dof_func, &P);
      }

      if (P_trunc_factor > 0 || P_max_elmts > 0)
         hypre_AMGTruncation(P,P_trunc_factor,P_max_elmts);

      /*hypre_printf("END computing level %d interpolation matrix; =======\n", level);
      */

      dof_func_array[level+1] = coarse_dof_func;
      P_array[level] = P; 
      
      if (amg_ioutdat == 5 && level == 0)
      {
         hypre_CSRMatrixPrint(S,"S_mat");
      }
      if (coarsen_type != 3 ) hypre_CSRMatrixDestroy(S);
 
      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      hypre_AMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                   P_array[level], &A_H);
      if (A_trunc_factor > 0 || A_max_elmts > 0)
         hypre_AMGOpTruncation(A_H,A_trunc_factor,A_max_elmts);

      ++level;
      A_array[level] = A_H;

      if (level+1 >= max_levels || 
          coarse_size == fine_size || 
          coarse_size <= coarse_threshold)
         not_finished_coarsening = 0;
   } 
   
   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   num_levels = level+1;
   hypre_AMGDataNumLevels(amg_data) = num_levels;
   hypre_AMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_AMGDataAArray(amg_data) = A_array;
   hypre_AMGDataPArray(amg_data) = P_array;
   hypre_AMGDataBArray(amg_data) = B_array;
   hypre_AMGDataPBArray(amg_data) = PB_array;

   hypre_AMGDataDofFuncArray(amg_data) = dof_func_array;
   hypre_AMGDataNumFunctions(amg_data) = num_functions;	
   /*-----------------------------------------------------------------------
    * Setup F and U arrays
    *-----------------------------------------------------------------------*/

   F_array = hypre_CTAlloc(hypre_Vector*, num_levels);
   U_array = hypre_CTAlloc(hypre_Vector*, num_levels);

   F_array[0] = f;
   U_array[0] = u;

   for (j = 1; j < num_levels; j++)
   {
     F_array[j] = hypre_SeqVectorCreate(hypre_CSRMatrixNumRows(A_array[j]));
     hypre_SeqVectorInitialize(F_array[j]);

     U_array[j] = hypre_SeqVectorCreate(hypre_CSRMatrixNumRows(A_array[j]));
     hypre_SeqVectorInitialize(U_array[j]);
   }

   hypre_AMGDataFArray(amg_data) = F_array;
   hypre_AMGDataUArray(amg_data) = U_array;


#if 0

 /*-----------------------------------------------------------------------
  * Debugging
  *-----------------------------------------------------------------------*/
   {
      hypre_Vector *y1, *y2, *x;
      HYPRE_Int row_size, col_size;
      

      row_size = hypre_CSRMatrixNumRows(P_array[0]);
      col_size = hypre_CSRMatrixNumCols(P_array[0]);

      y1 = hypre_SeqVectorCreate(row_size);
      y2 =  hypre_SeqVectorCreate(row_size);
      x =  hypre_SeqVectorCreate(col_size);


      hypre_SeqVectorInitialize(x);
      hypre_SeqVectorInitialize(y1);
      hypre_SeqVectorInitialize(y2);


      hypre_SeqVectorSetRandomValues( x, 1);
      

      hypre_CSRMatrixMatvec( 1.0, P_array[0], x, 1.0, y1);
      
      hypre_SeqVectorPrint( y1, "vector.out");
      
      hypre_SeqVectorDestroy(x);
      hypre_SeqVectorDestroy(y1);
      hypre_SeqVectorDestroy(y2);
      
   }
   
#endif

   /*-----------------------------------------------------------------------
    * Print some stuff
    *-----------------------------------------------------------------------*/

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
	hypre_AMGSetupStats(amg_data);

   if (amg_ioutdat == -3)
   {  
      char     fnam[255];

      HYPRE_Int j;

      for (j = 0; j < level+1; j++)
      {
         hypre_sprintf(fnam,"SP_A_%d.ysmp",j);
         hypre_CSRMatrixPrint(A_array[j],fnam);

      }                         

      for (j = 0; j < level; j++)
      { 
         hypre_sprintf(fnam,"SP_P_%d.ysmp",j);
         hypre_CSRMatrixPrint(P_array[j],fnam);
      }   
   } 

   hypre_TFree(unit_vector);
   Setup_err_flag = 0;
   return(Setup_err_flag);
}  
