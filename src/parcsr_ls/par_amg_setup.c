/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.95 $
 ***********************************************************************EHEADER*/





#include "headers.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"	


#define DEBUG 0
#define PRINT_CF 0

/*****************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

/*****************************************************************************
 * hypre_BoomerAMGSetup
 *****************************************************************************/

HYPRE_Int
hypre_BoomerAMGSetup( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u         )
{
   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A); 
   hypre_ParAMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParVector     *Vtemp;
   hypre_ParVector     *Rtemp;
   hypre_ParVector     *Ptemp;
   hypre_ParVector     *Ztemp;
   hypre_ParCSRMatrix **P_array;
   hypre_ParVector    *Residual_array;
   HYPRE_Int                **CF_marker_array;   
   HYPRE_Int                **dof_func_array;   
   HYPRE_Int                 *dof_func;
   HYPRE_Int                 *col_offd_S_to_A;
   HYPRE_Int                 *col_offd_SN_to_AN;
   double              *relax_weight;
   double              *omega;
   double               schwarz_relax_wt = 1;
   double               strong_threshold;
   double               CR_strong_th;
   double               max_row_sum;
   double               trunc_factor, jacobi_trunc_threshold;
   double               agg_trunc_factor, agg_P12_trunc_factor;
   double               S_commpkg_switch;
   double      		CR_rate;
   HYPRE_Int       relax_order;
   HYPRE_Int      max_levels; 
   HYPRE_Int      amg_logging;
   HYPRE_Int      amg_print_level;
   HYPRE_Int      debug_flag;
   HYPRE_Int      dbg_flg;
   HYPRE_Int      local_num_vars;
   HYPRE_Int      P_max_elmts;
   HYPRE_Int      agg_P_max_elmts;
   HYPRE_Int      agg_P12_max_elmts;
   HYPRE_Int      IS_type;
   HYPRE_Int      num_CR_relax_steps;
   HYPRE_Int      CR_use_CG; 
   HYPRE_Int      cgc_its; /* BM Aug 25, 2006 */

   hypre_ParCSRBlockMatrix **A_block_array, **P_block_array;
 
   /* Local variables */
   HYPRE_Int                 *CF_marker;
   HYPRE_Int                 *CFN_marker;
   HYPRE_Int                 *CF2_marker;
   hypre_ParCSRMatrix  *S = NULL;
   hypre_ParCSRMatrix  *S2;
   hypre_ParCSRMatrix  *SN;
   hypre_ParCSRMatrix  *SCR;
   hypre_ParCSRMatrix  *P = NULL;
   hypre_ParCSRMatrix  *A_H;
   hypre_ParCSRMatrix  *AN;
   hypre_ParCSRMatrix  *P1;
   hypre_ParCSRMatrix  *P2;
   double              *SmoothVecs = NULL;
   double             **l1_norms = NULL;

   HYPRE_Int       old_num_levels, num_levels;
   HYPRE_Int       level;
   HYPRE_Int       local_size, i;
   HYPRE_Int       first_local_row;
   HYPRE_Int       coarse_size;
   HYPRE_Int       coarsen_type;
   HYPRE_Int       measure_type;
   HYPRE_Int       setup_type;
   HYPRE_Int       fine_size;
   HYPRE_Int       rest, tms, indx;
   double    size;
   HYPRE_Int       not_finished_coarsening = 1;
   HYPRE_Int       Setup_err_flag = 0;
   HYPRE_Int       coarse_threshold = hypre_ParAMGDataMaxCoarseSize(amg_data);
   HYPRE_Int       seq_threshold = hypre_ParAMGDataSeqThreshold(amg_data);
   HYPRE_Int       j, k;
   HYPRE_Int       num_procs,my_id,num_threads;
   HYPRE_Int      *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   HYPRE_Int       num_functions = hypre_ParAMGDataNumFunctions(amg_data);
   HYPRE_Int       nodal = hypre_ParAMGDataNodal(amg_data);
   HYPRE_Int       nodal_levels = hypre_ParAMGDataNodalLevels(amg_data);
   HYPRE_Int       nodal_diag = hypre_ParAMGDataNodalDiag(amg_data);
   HYPRE_Int       num_paths = hypre_ParAMGDataNumPaths(amg_data);
   HYPRE_Int       agg_num_levels = hypre_ParAMGDataAggNumLevels(amg_data);
   HYPRE_Int       agg_interp_type = hypre_ParAMGDataAggInterpType(amg_data);
   HYPRE_Int       sep_weight = hypre_ParAMGDataSepWeight(amg_data); 
   HYPRE_Int	    *coarse_dof_func = NULL;
   HYPRE_Int	    *coarse_pnts_global;
   HYPRE_Int	    *coarse_pnts_global1;
   HYPRE_Int       num_cg_sweeps;

   double *max_eig_est = NULL;
   double *min_eig_est = NULL;

   HYPRE_Solver *smoother = NULL;
   HYPRE_Int       smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   HYPRE_Int       smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   HYPRE_Int	     sym;
   HYPRE_Int	     nlevel;
   double    thresh;
   double    filter;
   double    drop_tol;
   HYPRE_Int	     max_nz_per_row;
   char     *euclidfile;
   HYPRE_Int	     eu_level;
   HYPRE_Int	     eu_bj;
   double    eu_sparse_A;

   HYPRE_Int interp_type;
   HYPRE_Int post_interp_type;  /* what to do after computing the interpolation matrix
                             0 for nothing, 1 for a Jacobi step */


   /*for fittting interp vectors */
   /*HYPRE_Int                smooth_interp_vectors= hypre_ParAMGSmoothInterpVectors(amg_data); */
   double             abs_q_trunc= hypre_ParAMGInterpVecAbsQTrunc(amg_data);
   HYPRE_Int                q_max = hypre_ParAMGInterpVecQMax(amg_data);
   HYPRE_Int                num_interp_vectors= hypre_ParAMGNumInterpVectors(amg_data);
   HYPRE_Int                num_levels_interp_vectors = hypre_ParAMGNumLevelsInterpVectors(amg_data);
   hypre_ParVector  **interp_vectors = hypre_ParAMGInterpVectors(amg_data);
   hypre_ParVector ***interp_vectors_array= hypre_ParAMGInterpVectorsArray(amg_data);
   HYPRE_Int                interp_vec_variant= hypre_ParAMGInterpVecVariant(amg_data);
   HYPRE_Int                interp_refine= hypre_ParAMGInterpRefine(amg_data);
   HYPRE_Int                interp_vec_first_level= hypre_ParAMGInterpVecFirstLevel(amg_data);
   double            *expandp_weights =  hypre_ParAMGDataExpandPWeights(amg_data);


   hypre_ParCSRBlockMatrix *A_H_block;

   HYPRE_Int block_mode = 0;

   double    wall_time;   /* for debugging instrumentation */

   
   hypre_MPI_Comm_size(comm, &num_procs);   
   hypre_MPI_Comm_rank(comm,&my_id);

   num_threads = hypre_NumThreads();

   
   old_num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   amg_logging = hypre_ParAMGDataLogging(amg_data);
   amg_print_level = hypre_ParAMGDataPrintLevel(amg_data);
   coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);
   measure_type = hypre_ParAMGDataMeasureType(amg_data);
   setup_type = hypre_ParAMGDataSetupType(amg_data);
   debug_flag = hypre_ParAMGDataDebugFlag(amg_data);
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
   omega = hypre_ParAMGDataOmega(amg_data);
   dof_func = hypre_ParAMGDataDofFunc(amg_data);
   sym = hypre_ParAMGDataSym(amg_data);
   nlevel = hypre_ParAMGDataLevel(amg_data);
   filter = hypre_ParAMGDataFilter(amg_data);
   thresh = hypre_ParAMGDataThreshold(amg_data);
   drop_tol = hypre_ParAMGDataDropTol(amg_data);
   max_nz_per_row = hypre_ParAMGDataMaxNzPerRow(amg_data);
   euclidfile = hypre_ParAMGDataEuclidFile(amg_data);
   eu_level = hypre_ParAMGDataEuLevel(amg_data);
   eu_sparse_A = hypre_ParAMGDataEuSparseA(amg_data);
   eu_bj = hypre_ParAMGDataEuBJ(amg_data);
   interp_type = hypre_ParAMGDataInterpType(amg_data);
   post_interp_type = hypre_ParAMGDataPostInterpType(amg_data);
   IS_type = hypre_ParAMGDataISType(amg_data);
   num_CR_relax_steps = hypre_ParAMGDataNumCRRelaxSteps(amg_data);
   CR_rate = hypre_ParAMGDataCRRate(amg_data);
   CR_use_CG = hypre_ParAMGDataCRUseCG(amg_data);
   cgc_its = hypre_ParAMGDataCGCIts(amg_data);

   relax_order         = hypre_ParAMGDataRelaxOrder(amg_data);

   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParCSRMatrixSetDNumNonzeros(A);
   hypre_ParAMGDataNumVariables(amg_data) = hypre_ParCSRMatrixNumRows(A);

   if (num_procs == 1) seq_threshold = 0;
   if (setup_type == 0) return Setup_err_flag;

   S = NULL;

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);
   dof_func_array = hypre_ParAMGDataDofFuncArray(amg_data);
   local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

 
   A_block_array = hypre_ParAMGDataABlockArray(amg_data);
   P_block_array = hypre_ParAMGDataPBlockArray(amg_data);

   grid_relax_type[3] = hypre_ParAMGDataUserCoarseRelaxType(amg_data); 

   /* change in definition of standard and multipass interpolation, by
      eliminating interp_type 9 and 5 and setting sep_weight instead
      when using separation of weights option */
   if (interp_type == 9)
   {
      interp_type = 8;
      sep_weight = 1;
   }
   else if (interp_type == 5)
   {
      interp_type = 4;
      sep_weight = 1;
   }


   /* Verify that if the user has selected the interp_vec_variant > 0
      (so GM or LN interpolation) then they have nodal coarsening
      selected also */
   if (interp_vec_variant > 0 && nodal < 1)
   {
      nodal = 1;
      if (my_id == 0)
         hypre_printf("WARNING: Changing to node-based coarsening because LN of GM interpolation has been specified via HYPRE_BoomerAMGSetInterpVecVariant.\n");
   }

   /* Verify that settings are correct for solving systmes */
   /* If the user has specified either a block interpolation or a block relaxation then 
      we need to make sure the other has been choosen as well  - so we can be 
      in "block mode" - storing only block matrices on the coarse levels*/
   /* Furthermore, if we are using systems and nodal = 0, then 
      we will change nodal to 1 */
   /* probably should disable stuff like smooth num levels at some point */


   if (grid_relax_type[0] >= 20) /* block relaxation choosen */
   {

      if (!(interp_type >= 20 || interp_type == 11 || interp_type == 10 ) )
      {
         hypre_ParAMGDataInterpType(amg_data) = 20;
         interp_type = hypre_ParAMGDataInterpType(amg_data) ;
      }
      
      for (i=1; i < 3; i++)
      {
         if (grid_relax_type[i] < 20)
         {
            grid_relax_type[i] = 23;
         }
         
      }
      if (grid_relax_type[3] < 20) grid_relax_type[3] = 29;  /* GE */
 
      block_mode = 1;
   }

   if (interp_type >= 20 || interp_type == 11 || interp_type == 10 ) /* block interp choosen */
   {
      if (!(nodal)) 
      {
         hypre_ParAMGDataNodal(amg_data) = 1;
         nodal = hypre_ParAMGDataNodal(amg_data);
      }
      for (i=0; i < 3; i++)
      {
         if (grid_relax_type[i] < 20)      
            grid_relax_type[i] = 23;
      }
             
      if (grid_relax_type[3] < 20) grid_relax_type[3] = 29; /* GE */

      block_mode = 1;      

   }

   hypre_ParAMGDataBlockMode(amg_data) = block_mode;


   /* end of systems checks */



   if (A_array || A_block_array || P_array || P_block_array || CF_marker_array || dof_func_array)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (A_array[j])
         {
            hypre_ParCSRMatrixDestroy(A_array[j]);
            A_array[j] = NULL;
         }

         if (A_block_array[j])
         {
            hypre_ParCSRBlockMatrixDestroy(A_block_array[j]);
            A_block_array[j] = NULL;
         }
        


         if (dof_func_array[j])
         {
            hypre_TFree(dof_func_array[j]);
            dof_func_array[j] = NULL;
         }
      }

      for (j = 0; j < old_num_levels-1; j++)
      {
         if (P_array[j])
         {
            hypre_ParCSRMatrixDestroy(P_array[j]);
            P_array[j] = NULL;
         }

         if (P_block_array[j])
         {
            hypre_ParCSRBlockMatrixDestroy(P_block_array[j]);
            P_array[j] = NULL;
         }

      }

/* Special case use of CF_marker_array when old_num_levels == 1
   requires us to attempt this deallocation every time */
      if (CF_marker_array[0])
      {
        hypre_TFree(CF_marker_array[0]);
        CF_marker_array[0] = NULL;
      }

      for (j = 1; j < old_num_levels-1; j++)
      {
         if (CF_marker_array[j])
         {
            hypre_TFree(CF_marker_array[j]);
            CF_marker_array[j] = NULL;
         }
      }
   }

   if (A_array == NULL)
      A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels);
   if (A_block_array == NULL)
      A_block_array = hypre_CTAlloc(hypre_ParCSRBlockMatrix*, max_levels);


   if (P_array == NULL && max_levels > 1)
      P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels-1);
   if (P_block_array == NULL && max_levels > 1)
      P_block_array = hypre_CTAlloc(hypre_ParCSRBlockMatrix*, max_levels-1);


   if (CF_marker_array == NULL)
      CF_marker_array = hypre_CTAlloc(HYPRE_Int*, max_levels);
   if (dof_func_array == NULL)
      dof_func_array = hypre_CTAlloc(HYPRE_Int*, max_levels);
   if (num_functions > 1 && dof_func == NULL)
   {
      first_local_row = hypre_ParCSRMatrixFirstRowIndex(A);
      dof_func = hypre_CTAlloc(HYPRE_Int,local_size);
      rest = first_local_row-((first_local_row/num_functions)*num_functions);
      indx = num_functions-rest;
      if (rest == 0) indx = 0;
      k = num_functions - 1;
      for (j = indx-1; j > -1; j--)
         dof_func[j] = k--;
      tms = local_size/num_functions;
      if (tms*num_functions+indx > local_size) tms--;
      for (j=0; j < tms; j++)
      {
         for (k=0; k < num_functions; k++)
            dof_func[indx++] = k;
      }
      k = 0;
      while (indx < local_size)
         dof_func[indx++] = k++;
      hypre_ParAMGDataDofFunc(amg_data) = dof_func;
   }

   A_array[0] = A;
                                                                             
                                                                             
   /* interp vectors setup */
   if (interp_vec_variant == 1)
   {
      num_levels_interp_vectors = interp_vec_first_level + 1;
      hypre_ParAMGNumLevelsInterpVectors(amg_data) = num_levels_interp_vectors;
   }
   if ( interp_vec_variant > 0 &&  num_interp_vectors > 0)
   {
      interp_vectors_array =  hypre_CTAlloc(hypre_ParVector**, num_levels_interp_vectors);
      interp_vectors_array[0] = interp_vectors;
      hypre_ParAMGInterpVectorsArray(amg_data)= interp_vectors_array;
   }

   

   if (block_mode)
   {
      A_block_array[0] =  hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(
         A_array[0], num_functions);
      hypre_ParCSRBlockMatrixSetNumNonzeros(A_block_array[0]);
      hypre_ParCSRBlockMatrixSetDNumNonzeros(A_block_array[0]);
   }
   

   dof_func_array[0] = dof_func;
   hypre_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_ParAMGDataDofFuncArray(amg_data) = dof_func_array;
   hypre_ParAMGDataAArray(amg_data) = A_array;
   hypre_ParAMGDataPArray(amg_data) = P_array;
   hypre_ParAMGDataRArray(amg_data) = P_array;

   hypre_ParAMGDataABlockArray(amg_data) = A_block_array;
   hypre_ParAMGDataPBlockArray(amg_data) = P_block_array;
   hypre_ParAMGDataRBlockArray(amg_data) = P_block_array;

   Vtemp = hypre_ParAMGDataVtemp(amg_data);

   if (Vtemp != NULL)
   {
      hypre_ParVectorDestroy(Vtemp);
      Vtemp = NULL;
   }

   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
   hypre_ParVectorInitialize(Vtemp);
   hypre_ParVectorSetPartitioningOwner(Vtemp,0);
   hypre_ParAMGDataVtemp(amg_data) = Vtemp;

   if ((smooth_num_levels > 0 && smooth_type > 9) 
		|| relax_weight[0] < 0 || omega[0] < 0 ||
                hypre_ParAMGDataSchwarzRlxWeight(amg_data) < 0)
   {
      Ptemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Ptemp);
      hypre_ParVectorSetPartitioningOwner(Ptemp,0);
      hypre_ParAMGDataPtemp(amg_data) = Ptemp;
      Rtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Rtemp);
      hypre_ParVectorSetPartitioningOwner(Rtemp,0);
      hypre_ParAMGDataRtemp(amg_data) = Rtemp;
   }
  
   /* See if we need the Ztemp vector */
   if ((smooth_num_levels > 0 && smooth_type > 6)
		 || relax_weight[0] < 0 || omega[0] < 0 ||
                 hypre_ParAMGDataSchwarzRlxWeight(amg_data) < 0)
   {
      Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                    hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                    hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Ztemp);
      hypre_ParVectorSetPartitioningOwner(Ztemp,0);
      hypre_ParAMGDataZtemp(amg_data) = Ztemp;
   }
   else if (grid_relax_type[0] == 16 || grid_relax_type[1] == 16 || grid_relax_type[2] == 16 || grid_relax_type[3] == 16)
   {
      /* Chebyshev */
       Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                     hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                     hypre_ParCSRMatrixRowStarts(A_array[0]));
       hypre_ParVectorInitialize(Ztemp);
       hypre_ParVectorSetPartitioningOwner(Ztemp,0);
       hypre_ParAMGDataZtemp(amg_data) = Ztemp;
      
   }
   else if (num_threads > 1)
   {
      /* we need the temp Z vector for relaxation 3 and 6 now if we are
       * using threading */
      for (j = 0; j < 4; j++)
      {
         if (grid_relax_type[j] ==3 || grid_relax_type[j] == 6)
         {
            Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                          hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                          hypre_ParCSRMatrixRowStarts(A_array[0]));
            hypre_ParVectorInitialize(Ztemp);
            hypre_ParVectorSetPartitioningOwner(Ztemp,0);
            hypre_ParAMGDataZtemp(amg_data) = Ztemp;
            break;
         }
      }
   }
   


   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);

   if (F_array != NULL || U_array != NULL)
   {
      for (j = 1; j < old_num_levels; j++)
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
      F_array = hypre_CTAlloc(hypre_ParVector*, max_levels);
   if (U_array == NULL)
      U_array = hypre_CTAlloc(hypre_ParVector*, max_levels);

   F_array[0] = f;
   U_array[0] = u;

   hypre_ParAMGDataFArray(amg_data) = F_array;
   hypre_ParAMGDataUArray(amg_data) = U_array;

   /*----------------------------------------------------------
    * Initialize hypre_ParAMGData
    *----------------------------------------------------------*/

   not_finished_coarsening = 1;
   level = 0;
  
   strong_threshold = hypre_ParAMGDataStrongThreshold(amg_data);
   CR_strong_th = hypre_ParAMGDataCRStrongTh(amg_data);
   max_row_sum = hypre_ParAMGDataMaxRowSum(amg_data);
   trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);
   agg_trunc_factor = hypre_ParAMGDataAggTruncFactor(amg_data);
   agg_P12_trunc_factor = hypre_ParAMGDataAggP12TruncFactor(amg_data);
   P_max_elmts = hypre_ParAMGDataPMaxElmts(amg_data);
   agg_P_max_elmts = hypre_ParAMGDataAggPMaxElmts(amg_data);
   agg_P12_max_elmts = hypre_ParAMGDataAggP12MaxElmts(amg_data);
   jacobi_trunc_threshold = hypre_ParAMGDataJacobiTruncThreshold(amg_data);
   S_commpkg_switch = hypre_ParAMGDataSCommPkgSwitch(amg_data);
   if (smooth_num_levels > level)
   {
      smoother = hypre_CTAlloc(HYPRE_Solver, smooth_num_levels);
      hypre_ParAMGDataSmoother(amg_data) = smoother;
   }

   /*-----------------------------------------------------
    *  Enter Coarsening Loop
    *-----------------------------------------------------*/

   while (not_finished_coarsening)
   {


      /* only do nodal coarsening on a fixed number of levels */
      if (level >= nodal_levels)
      {
         nodal = 0;
      }

      if (block_mode)
      {
         fine_size =    hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[level]);
      }
      else 
      {
         fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
      }
      



      if (level > 0)
      {   

         if (block_mode)
         {
            F_array[level] =
               hypre_ParVectorCreateFromBlock(hypre_ParCSRBlockMatrixComm(A_block_array[level]),
                                              hypre_ParCSRMatrixGlobalNumRows(A_block_array[level]),
                                              hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]),
                                              hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]));
            hypre_ParVectorInitialize(F_array[level]);
            
            U_array[level] =  
               hypre_ParVectorCreateFromBlock(hypre_ParCSRBlockMatrixComm(A_block_array[level]),
                                              hypre_ParCSRMatrixGlobalNumRows(A_block_array[level]),
                                              hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]),
                                              hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]));

            hypre_ParVectorInitialize(U_array[level]);
         }
         else 
         {
            F_array[level] =
               hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                     hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                     hypre_ParCSRMatrixRowStarts(A_array[level]));
            hypre_ParVectorInitialize(F_array[level]);
            hypre_ParVectorSetPartitioningOwner(F_array[level],0);
            
            U_array[level] =
               hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                     hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                     hypre_ParCSRMatrixRowStarts(A_array[level]));
            hypre_ParVectorInitialize(U_array[level]);
            hypre_ParVectorSetPartitioningOwner(U_array[level],0);
         }
         
      }

      /*-------------------------------------------------------------
       * Select coarse-grid points on 'level' : returns CF_marker
       * for the level.  Returns strength matrix, S  
       *--------------------------------------------------------------*/
     
      if (debug_flag==1) wall_time = time_getWallclockSeconds();
      if (debug_flag==3)
      {
          hypre_printf("\n ===== Proc = %d     Level = %d  =====\n",
                        my_id, level);
          fflush(NULL);
      }

      if ( max_levels == 1)
      {
	 S = NULL;
	 coarse_pnts_global = NULL;
         CF_marker = hypre_CTAlloc(HYPRE_Int, local_size );
	 for (i=0; i < local_size ; i++)
	    CF_marker[i] = 1;
         /* AB removed below - already allocated */
         /* CF_marker_array = hypre_CTAlloc(HYPRE_Int*, 1);*/
	 CF_marker_array[level] = CF_marker;
	 coarse_size = fine_size;
      }
      else /* max_levels > 1 */
      {
         if (block_mode)
         {
            local_num_vars =
               hypre_CSRBlockMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[level]));
         }
         else 
         {
            local_num_vars =
               hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
         }
	 if (hypre_ParAMGDataGSMG(amg_data) || 
             hypre_ParAMGDataInterpType(amg_data) == 1)
         {
	    hypre_BoomerAMGCreateSmoothVecs(amg_data, A_array[level],
	       hypre_ParAMGDataNumGridSweeps(amg_data)[1],
               level, &SmoothVecs);
         }


         /**** Get the Strength Matrix ****/        

         if (hypre_ParAMGDataGSMG(amg_data) == 0)
	 {
	    if (nodal) /* if we are solving systems and 
                          not using the unknown approach then we need to 
                          convert A to a nodal matrix - values that represent the
                          blocks  - before getting the strength matrix*/
	    {

               if (block_mode)
               {
                  hypre_BoomerAMGBlockCreateNodalA( A_block_array[level], abs(nodal), nodal_diag, &AN);
               }
               else
               {
                  hypre_BoomerAMGCreateNodalA(A_array[level],num_functions,
                                              dof_func_array[level], abs(nodal), nodal_diag, &AN);
               }

               /* dof array not needed for creating S because we pass in that 
                  the number of functions is 1 */
               /* creat s two different ways - depending on if any entries in AN are negative: */

               /* first: positive and negative entries */  
               if (nodal == 3 || nodal == 6 || nodal_diag > 0)  
                  hypre_BoomerAMGCreateS(AN, strong_threshold, max_row_sum,
                                   1, NULL,&SN);
               else /* all entries are positive */
		  hypre_BoomerAMGCreateSabs(AN, strong_threshold, max_row_sum,
                                   1, NULL,&SN);
            

               col_offd_S_to_A = NULL;
	       col_offd_SN_to_AN = NULL;
	       if (strong_threshold > S_commpkg_switch)
                  hypre_BoomerAMGCreateSCommPkg(AN,SN,&col_offd_SN_to_AN);
	    }
	    else /* standard AMG or unknown approach */
	    {
	       hypre_BoomerAMGCreateS(A_array[level], 
				   strong_threshold, max_row_sum, 
				   num_functions, dof_func_array[level],&S);
	       col_offd_S_to_A = NULL;
	       if (strong_threshold > S_commpkg_switch)
                  hypre_BoomerAMGCreateSCommPkg(A_array[level],S,
				&col_offd_S_to_A);
	    }
	 }
	 else
	 {
	    hypre_BoomerAMGCreateSmoothDirs(amg_data, A_array[level],
	       SmoothVecs, strong_threshold, 
               num_functions, dof_func_array[level], &S);
	 }


         /**** Do the appropriate coarsening ****/ 

         if (nodal == 0) /* no nodal coarsening */
         { 
           if (coarsen_type == 6)
               hypre_BoomerAMGCoarsenFalgout(S, A_array[level], measure_type,
                                             debug_flag, &CF_marker);
           else if (coarsen_type == 7)
               hypre_BoomerAMGCoarsen(S, A_array[level], 2,
                                      debug_flag, &CF_marker);
           else if (coarsen_type == 8)
               hypre_BoomerAMGCoarsenPMIS(S, A_array[level], 0,
                                          debug_flag, &CF_marker);
           else if (coarsen_type == 9)
               hypre_BoomerAMGCoarsenPMIS(S, A_array[level], 2,
                                          debug_flag, &CF_marker);
           else if (coarsen_type == 10)
               hypre_BoomerAMGCoarsenHMIS(S, A_array[level], measure_type,
                                          debug_flag, &CF_marker);
           else if (coarsen_type == 21 || coarsen_type == 22)
               hypre_BoomerAMGCoarsenCGCb(S, A_array[level], measure_type,
                           coarsen_type, cgc_its, debug_flag, &CF_marker);
           else if (coarsen_type == 98)
               hypre_BoomerAMGCoarsenCR1(A_array[level], &CF_marker,
                        &coarse_size,
                        num_CR_relax_steps, IS_type, 0);
           else if (coarsen_type == 99)
           {
                  hypre_BoomerAMGCreateS(A_array[level],
                        CR_strong_th, 1,
                        num_functions, dof_func_array[level],&SCR);
                  hypre_BoomerAMGCoarsenCR(A_array[level], &CF_marker,
                        &coarse_size,
                        num_CR_relax_steps, IS_type, 1, grid_relax_type[0],
                        relax_weight[level], omega[level], CR_rate,
                        NULL,NULL,CR_use_CG,SCR);
                  hypre_ParCSRMatrixDestroy(SCR);
           }
           else if (coarsen_type)
                  hypre_BoomerAMGCoarsenRuge(S, A_array[level],
                        measure_type, coarsen_type, debug_flag, &CF_marker);
           else
                  hypre_BoomerAMGCoarsen(S, A_array[level], 0,
                                      debug_flag, &CF_marker);
           if (level < agg_num_levels)
           {
               hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        1, dof_func_array[level], CF_marker,
                        &coarse_dof_func,&coarse_pnts_global1);
               hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                       coarse_pnts_global1, &S2);
               if (coarsen_type == 10)
                  hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type+3,
                                     debug_flag, &CFN_marker);
               else if (coarsen_type == 8)
                   hypre_BoomerAMGCoarsenPMIS(S2, S2, 3,
                                     debug_flag, &CFN_marker);
               else if (coarsen_type == 9)
                   hypre_BoomerAMGCoarsenPMIS(S2, S2, 4,
                                     debug_flag, &CFN_marker);
               else if (coarsen_type == 6)
                  hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type,
                                       debug_flag, &CFN_marker);
               else if (coarsen_type == 21 || coarsen_type == 22)
                  hypre_BoomerAMGCoarsenCGCb(S2, S2, measure_type,
                           coarsen_type, cgc_its, debug_flag, &CFN_marker);
               else if (coarsen_type == 7)
                  hypre_BoomerAMGCoarsen(S2, S2, 2, debug_flag, &CFN_marker);
               else if (coarsen_type)
                  hypre_BoomerAMGCoarsenRuge(S2, S2, measure_type, coarsen_type, 
				debug_flag, &CFN_marker);
               else 
                  hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CFN_marker);
               hypre_ParCSRMatrixDestroy(S2);
           }
         }
         else if (block_mode)
         {
           if (coarsen_type == 6)
               hypre_BoomerAMGCoarsenFalgout(SN, SN, measure_type,
                                             debug_flag, &CF_marker);
           else if (coarsen_type == 7)
               hypre_BoomerAMGCoarsen(SN, SN, 2,
                                      debug_flag, &CF_marker);
           else if (coarsen_type == 8)
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 0,
                                          debug_flag, &CF_marker);
           else if (coarsen_type == 9)
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 2,
                                          debug_flag, &CF_marker);
           else if (coarsen_type == 10)
               hypre_BoomerAMGCoarsenHMIS(SN, SN, measure_type,
                                          debug_flag, &CF_marker);
           else if (coarsen_type == 21 || coarsen_type == 22)
               hypre_BoomerAMGCoarsenCGCb(SN, SN, measure_type,
                           coarsen_type, cgc_its, debug_flag, &CF_marker);
           else if (coarsen_type)
                  hypre_BoomerAMGCoarsenRuge(SN, SN,
                        measure_type, coarsen_type, debug_flag, &CF_marker);
           else
                  hypre_BoomerAMGCoarsen(SN, SN, 0,
                                      debug_flag, &CF_marker);
         }
         else if (nodal > 0)
         {
           if (coarsen_type == 6)
               hypre_BoomerAMGCoarsenFalgout(SN, SN, measure_type,
                                             debug_flag, &CFN_marker);
           else if (coarsen_type == 7)
               hypre_BoomerAMGCoarsen(SN, SN, 2, debug_flag, &CFN_marker);
           else if (coarsen_type == 8)
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 0, debug_flag, &CFN_marker);
           else if (coarsen_type == 9)
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 2, debug_flag, &CFN_marker);
           else if (coarsen_type == 10)
               hypre_BoomerAMGCoarsenHMIS(SN, SN, measure_type,
                                          debug_flag, &CFN_marker);
           else if (coarsen_type == 21 || coarsen_type == 22)
               hypre_BoomerAMGCoarsenCGCb(SN, SN, measure_type,
                           coarsen_type, cgc_its, debug_flag, &CFN_marker);
           else if (coarsen_type)
                  hypre_BoomerAMGCoarsenRuge(SN, SN,
                        measure_type, coarsen_type, debug_flag, &CFN_marker);
           else
                  hypre_BoomerAMGCoarsen(SN, SN, 0,
                                      debug_flag, &CFN_marker);
           if (level < agg_num_levels)
           {
               hypre_BoomerAMGCoarseParms(comm, local_num_vars/num_functions,
                        1, dof_func_array[level], CFN_marker,
                        &coarse_dof_func,&coarse_pnts_global1);
               hypre_BoomerAMGCreate2ndS (SN, CFN_marker, num_paths,
                                       coarse_pnts_global1, &S2);
               if (coarsen_type == 10)
                  hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type+3,
                                     debug_flag, &CF2_marker);
               else if (coarsen_type == 8)
                   hypre_BoomerAMGCoarsenPMIS(S2, S2, 3,
                                     debug_flag, &CF2_marker);
               else if (coarsen_type == 9)
                   hypre_BoomerAMGCoarsenPMIS(S2, S2, 4,
                                     debug_flag, &CF2_marker);
               else if (coarsen_type == 6)
                  hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type,
                                       debug_flag, &CF2_marker);
               else if (coarsen_type == 21 || coarsen_type == 22)
                  hypre_BoomerAMGCoarsenCGCb(S2, S2, measure_type,
                           coarsen_type, cgc_its, debug_flag, &CF2_marker);
               else if (coarsen_type == 7)
                  hypre_BoomerAMGCoarsen(S2, S2, 2, debug_flag, &CF2_marker);
               else if (coarsen_type)
                  hypre_BoomerAMGCoarsenRuge(S2, S2, measure_type, coarsen_type, 
				debug_flag, &CF2_marker);
               else 
                  hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CF2_marker);
               hypre_ParCSRMatrixDestroy(S2);
           }
           else
           {
              col_offd_S_to_A = NULL;
              /* hypre_BoomerAMGCreateScalarCFS(A_array[level],
              			SN, CFN_marker, col_offd_SN_to_AN,
                                num_functions, nodal, 0, NULL, &CF_marker, 
                                &col_offd_S_to_A, &S); */
              
              hypre_BoomerAMGCreateScalarCFS(
              			SN, CFN_marker, col_offd_SN_to_AN,
                                num_functions, nodal, 0, NULL, &CF_marker, 
                                &col_offd_S_to_A, &S);
              if (col_offd_SN_to_AN == NULL)
              	col_offd_S_to_A = NULL;
              hypre_TFree(CFN_marker);
              hypre_TFree(col_offd_SN_to_AN);
              hypre_ParCSRMatrixDestroy(SN);
              hypre_ParCSRMatrixDestroy(AN);
           }
         }

         if (level < agg_num_levels)
         {
            if (nodal == 0)
            {
	       if (agg_interp_type == 1)
          	  hypre_BoomerAMGBuildExtPIInterp(A_array[level], 
		        CF_marker, S, coarse_pnts_global1, 
			num_functions, dof_func_array[level], debug_flag, 
			agg_P12_trunc_factor, agg_P12_max_elmts, col_offd_S_to_A, &P1);
	       else if (agg_interp_type == 2)
                  hypre_BoomerAMGBuildStdInterp(A_array[level], 
		        CF_marker, S, coarse_pnts_global1, 
			num_functions, dof_func_array[level], debug_flag, 
			agg_P12_trunc_factor, agg_P12_max_elmts, 0, col_offd_S_to_A, &P1);
	       else if (agg_interp_type == 3)
                  hypre_BoomerAMGBuildExtInterp(A_array[level], 
		        CF_marker, S, coarse_pnts_global1, 
			num_functions, dof_func_array[level], debug_flag, 
			agg_P12_trunc_factor, agg_P12_max_elmts, col_offd_S_to_A, &P1);
               if (agg_interp_type == 4)
               {
                  hypre_BoomerAMGCorrectCFMarker (CF_marker, local_num_vars, 
			CFN_marker);
                  hypre_TFree(coarse_pnts_global1);
                  /*hypre_TFree(coarse_dof_func);
                  coarse_dof_func = NULL;*/
                  hypre_TFree(CFN_marker);
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        num_functions, dof_func_array[level], CF_marker,
                        &coarse_dof_func,&coarse_pnts_global);
                  hypre_BoomerAMGBuildMultipass(A_array[level], 
			CF_marker, S, coarse_pnts_global, 
			num_functions, dof_func_array[level], debug_flag, 
			agg_trunc_factor, agg_P_max_elmts, sep_weight, 
			col_offd_S_to_A, &P);
               }
               else
               {
                  hypre_BoomerAMGCorrectCFMarker2 (CF_marker, local_num_vars, 
			CFN_marker);
                  hypre_TFree(CFN_marker);
                  /*hypre_TFree(coarse_dof_func);
                  coarse_dof_func = NULL;*/
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        num_functions, dof_func_array[level], CF_marker,
                        &coarse_dof_func,&coarse_pnts_global);
                  /*if (num_functions > 1 && nodal > -1 && (!block_mode) )
                     dof_func_array[level+1] = coarse_dof_func;*/
	          hypre_TFree(col_offd_S_to_A);
                  if (agg_interp_type == 1)
		     hypre_BoomerAMGBuildPartialExtPIInterp(A_array[level], 
		       	CF_marker, S, coarse_pnts_global, 
			coarse_pnts_global1, num_functions, 
			dof_func_array[level], debug_flag, agg_P12_trunc_factor, 
			agg_P12_max_elmts, col_offd_S_to_A, &P2);
                  else if (agg_interp_type == 2)
		     hypre_BoomerAMGBuildPartialStdInterp(A_array[level], 
		       	CF_marker, S, coarse_pnts_global, 
			coarse_pnts_global1, num_functions, 
			dof_func_array[level], debug_flag, agg_P12_trunc_factor, 
			agg_P12_max_elmts, sep_weight, col_offd_S_to_A, &P2);
                  else if (agg_interp_type == 3)
		     hypre_BoomerAMGBuildPartialExtInterp(A_array[level], 
		       	CF_marker, S, coarse_pnts_global, 
			coarse_pnts_global1, num_functions, 
			dof_func_array[level], debug_flag, agg_P12_trunc_factor, 
			agg_P12_max_elmts, col_offd_S_to_A, &P2);
                  P = hypre_ParMatmul(P1,P2);
                  hypre_BoomerAMGInterpTruncation(P, agg_trunc_factor, 
			agg_P_max_elmts);          
	          hypre_MatvecCommPkgCreate(P);
                  hypre_ParCSRMatrixDestroy(P1);
                  hypre_ParCSRMatrixOwnsColStarts(P2) = 0;
                  hypre_ParCSRMatrixDestroy(P2);
                  hypre_ParCSRMatrixOwnsColStarts(P) = 1;
               }
            }
            else if (nodal > 0)
            {
               if (agg_interp_type == 4)
               {
		  hypre_BoomerAMGCorrectCFMarker (CFN_marker, 
			local_num_vars/num_functions, CF2_marker);
		  hypre_TFree (CF2_marker);
                  hypre_TFree(coarse_pnts_global1);
                  col_offd_S_to_A = NULL;
/*                  hypre_BoomerAMGCreateScalarCFS(A_array[level],SN, CFN_marker, 
			col_offd_SN_to_AN, num_functions, nodal, 0, NULL, 
			&CF_marker, &col_offd_S_to_A, &S); */

                  hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, 
			col_offd_SN_to_AN, num_functions, nodal, 0, NULL, 
			&CF_marker, &col_offd_S_to_A, &S);
                  if (col_offd_SN_to_AN == NULL)
              	     col_offd_S_to_A = NULL;
                  hypre_TFree(CFN_marker);
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        num_functions, dof_func_array[level], CF_marker,
                        &coarse_dof_func,&coarse_pnts_global);
                  hypre_BoomerAMGBuildMultipass(A_array[level], 
			CF_marker, S, coarse_pnts_global, 
			num_functions, dof_func_array[level], debug_flag, 
			agg_trunc_factor, agg_P_max_elmts, sep_weight, 
			col_offd_S_to_A, &P);
               }
	       else
	       {
                  col_offd_S_to_A = NULL;
/*                  hypre_BoomerAMGCreateScalarCFS(A_array[level],SN, CFN_marker, 
			col_offd_SN_to_AN, num_functions, nodal, 0, NULL, 
			&CF_marker, &col_offd_S_to_A, &S);*/
                  hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, 
			col_offd_SN_to_AN, num_functions, nodal, 0, NULL, 
                                 &CF_marker, &col_offd_S_to_A, &S);
#ifdef HYPRE_NO_GLOBAL_PARTITION 
                  for (i=0; i < 2; i++)
		      coarse_pnts_global1[i] *= num_functions;
#else
                  for (i=1; i < num_procs+1; i++)
		      coarse_pnts_global1[i] *= num_functions;
#endif
                  if (col_offd_SN_to_AN == NULL)
              	     col_offd_S_to_A = NULL;
	          if (agg_interp_type == 1)
          	     hypre_BoomerAMGBuildExtPIInterp(A_array[level], 
		        CF_marker, S, coarse_pnts_global1, 
			num_functions, dof_func_array[level], debug_flag, 
			agg_P12_trunc_factor, agg_P12_max_elmts, col_offd_S_to_A, &P1);
	          else if (agg_interp_type == 2)
                     hypre_BoomerAMGBuildStdInterp(A_array[level], 
		        CF_marker, S, coarse_pnts_global1, 
			num_functions, dof_func_array[level], debug_flag, 
			agg_P12_trunc_factor, agg_P12_max_elmts, 0, col_offd_S_to_A, &P1);
	          else if (agg_interp_type == 3)
                     hypre_BoomerAMGBuildExtInterp(A_array[level], 
		        CF_marker, S, coarse_pnts_global1, 
			num_functions, dof_func_array[level], debug_flag, 
			agg_P12_trunc_factor, agg_P12_max_elmts, col_offd_S_to_A, &P1);
		  hypre_BoomerAMGCorrectCFMarker2 (CFN_marker, 
			local_num_vars/num_functions, CF2_marker);
                  hypre_TFree(CF2_marker);
                  hypre_TFree(CF_marker);
                  hypre_TFree(col_offd_S_to_A);
                  col_offd_S_to_A = NULL;
                  CF_marker = NULL;
                  hypre_ParCSRMatrixDestroy(S);
                  /* hypre_BoomerAMGCreateScalarCFS(A_array[level],SN, CFN_marker, 
			col_offd_SN_to_AN, num_functions, nodal, 0, NULL, 
			&CF_marker, &col_offd_S_to_A, &S); */
                  hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, 
			col_offd_SN_to_AN, num_functions, nodal, 0, NULL, 
			&CF_marker, &col_offd_S_to_A, &S);

                  if (col_offd_SN_to_AN == NULL)
              	     col_offd_S_to_A = NULL;
                  hypre_TFree(CFN_marker);
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        num_functions, dof_func_array[level], CF_marker,
                        &coarse_dof_func,&coarse_pnts_global);
                  /*if (num_functions > 1 && nodal > -1 && (!block_mode) )
                     dof_func_array[level+1] = coarse_dof_func;*/
                  if (agg_interp_type == 1)
		     hypre_BoomerAMGBuildPartialExtPIInterp(A_array[level], 
		       	CF_marker, S, coarse_pnts_global, 
			coarse_pnts_global1, num_functions, 
			dof_func_array[level], debug_flag, agg_P12_trunc_factor, 
			agg_P12_max_elmts, col_offd_S_to_A, &P2);
                  else if (agg_interp_type == 2)
		     hypre_BoomerAMGBuildPartialStdInterp(A_array[level], 
		       	CF_marker, S, coarse_pnts_global, 
			coarse_pnts_global1, num_functions, 
			dof_func_array[level], debug_flag, agg_P12_trunc_factor, 
			agg_P12_max_elmts, sep_weight, col_offd_S_to_A, &P2);
                  else if (agg_interp_type == 3)
		     hypre_BoomerAMGBuildPartialExtInterp(A_array[level], 
		       	CF_marker, S, coarse_pnts_global, 
			coarse_pnts_global1, num_functions, 
			dof_func_array[level], debug_flag, agg_P12_trunc_factor, 
			agg_P12_max_elmts, col_offd_S_to_A, &P2);
                  P = hypre_ParMatmul(P1,P2);
                  hypre_BoomerAMGInterpTruncation(P, agg_trunc_factor, 
			agg_P_max_elmts);          
	          hypre_MatvecCommPkgCreate(P);
                  hypre_ParCSRMatrixDestroy(P1);
                  hypre_ParCSRMatrixOwnsColStarts(P2) = 0;
                  hypre_ParCSRMatrixDestroy(P2);
                  hypre_ParCSRMatrixOwnsColStarts(P) = 1;
               }
               hypre_ParCSRMatrixDestroy(SN);
               hypre_ParCSRMatrixDestroy(AN);
            }
#ifdef HYPRE_NO_GLOBAL_PARTITION
            if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global[1];
            hypre_MPI_Bcast(&coarse_size, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
            coarse_size = coarse_pnts_global[num_procs];
#endif
         }
         else /* no aggressive coarsening */
         {
            /**** Get the coarse parameters ****/
            if (block_mode )
            {
               /* here we will determine interpolation using a nodal matrix */
               hypre_BoomerAMGCoarseParms(comm,
                                          hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AN)),
                                          1, NULL, CF_marker, NULL, &coarse_pnts_global);
            }
            else
            {
               hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                          num_functions, dof_func_array[level], CF_marker,
                                          &coarse_dof_func,&coarse_pnts_global);
            }
#ifdef HYPRE_NO_GLOBAL_PARTITION
            if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global[1];
            hypre_MPI_Bcast(&coarse_size, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
            coarse_size = coarse_pnts_global[num_procs];
#endif

            if (debug_flag==1) wall_time = time_getWallclockSeconds();

            if (interp_type == 4) 
            {
               hypre_BoomerAMGBuildMultipass(A_array[level], CF_marker, 
                                             S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                             debug_flag, trunc_factor, P_max_elmts, sep_weight, col_offd_S_to_A, &P);
	       hypre_TFree(col_offd_S_to_A);
            }
            else if (interp_type == 1)
            {
               hypre_BoomerAMGNormalizeVecs(
                  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level])),
                  hypre_ParAMGDataNumSamples(amg_data), SmoothVecs);
               
               hypre_BoomerAMGBuildInterpLS(NULL, CF_marker, S,
                                            coarse_pnts_global, num_functions, dof_func_array[level], 
                                            debug_flag, trunc_factor, 
                                            hypre_ParAMGDataNumSamples(amg_data), SmoothVecs, &P);
            }
            else if (interp_type == 2)
            {
               hypre_BoomerAMGBuildInterpHE(A_array[level], CF_marker, 
                                            S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                            debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
	       hypre_TFree(col_offd_S_to_A);
            }
            else if (interp_type == 3)
            {
               hypre_BoomerAMGBuildDirInterp(A_array[level], CF_marker, 
                                             S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                             debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
	       hypre_TFree(col_offd_S_to_A);
            }
            else if (interp_type == 6) /*Extended+i classical interpolation */
            {
               hypre_BoomerAMGBuildExtPIInterp(A_array[level], CF_marker, 
                                               S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                               debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
               hypre_TFree(col_offd_S_to_A);
            }
            else if (interp_type == 14) /*Extended classical interpolation */
            {
               hypre_BoomerAMGBuildExtInterp(A_array[level], CF_marker, 
                                             S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                             debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
               hypre_TFree(col_offd_S_to_A);
            }
            else if (interp_type == 7) /*Extended+i (if no common C) interpolation */
            {
               hypre_BoomerAMGBuildExtPICCInterp(A_array[level], CF_marker, 
                                                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                                 debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
               hypre_TFree(col_offd_S_to_A);
            }
            else if (interp_type == 12) /*FF interpolation */
            {
               hypre_BoomerAMGBuildFFInterp(A_array[level], CF_marker, 
                                            S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                            debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
               hypre_TFree(col_offd_S_to_A);
            }
            else if (interp_type == 13) /*FF1 interpolation */
            {
               hypre_BoomerAMGBuildFF1Interp(A_array[level], CF_marker, 
                                             S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                             debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
               hypre_TFree(col_offd_S_to_A);
           }
           else if (interp_type == 8) /*Standard interpolation */
           {
              hypre_BoomerAMGBuildStdInterp(A_array[level], CF_marker, 
                                            S, coarse_pnts_global, num_functions, dof_func_array[level], 
                                            debug_flag, trunc_factor, P_max_elmts, sep_weight, col_offd_S_to_A, &P);
	      hypre_TFree(col_offd_S_to_A);
           }
           else if (hypre_ParAMGDataGSMG(amg_data) == 0) /* none of above choosen and not GMSMG */
           {
              if (block_mode) /* nodal interpolation */
              {

                 /* convert A to a block matrix if there isn't already a block
                    matrix - there should be one already*/
                 if (!(A_block_array[level]))
                 {
                    A_block_array[level] =  hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(
                       A_array[level], num_functions);
                 }
                 
                 /* note that the current CF_marker is nodal */
                 if (interp_type == 11)
                 {
                    hypre_BoomerAMGBuildBlockInterpDiag( A_block_array[level], CF_marker, 
                                                         SN,
                                                         coarse_pnts_global, 1,
                                                         NULL,
                                                         debug_flag,
                                                         trunc_factor, P_max_elmts,1,
                                                         col_offd_S_to_A,
                                                         &P_block_array[level]);
                    

                 }
                 else if (interp_type == 22)
                 {
                    hypre_BoomerAMGBuildBlockInterpRV( A_block_array[level], CF_marker, 
                                                       SN,
                                                       coarse_pnts_global, 1,
                                                       NULL,
                                                       debug_flag,
                                                       trunc_factor, P_max_elmts,
                                                       col_offd_S_to_A,
                                                       &P_block_array[level]);
                 }
                 else if (interp_type == 23)
                 {
                    hypre_BoomerAMGBuildBlockInterpRV( A_block_array[level], CF_marker, 
                                                       SN,
                                                       coarse_pnts_global, 1,
                                                       NULL,
                                                       debug_flag,
                                                       trunc_factor, P_max_elmts,
                                                       col_offd_S_to_A,
                                                       &P_block_array[level]);
                 }
                 else if (interp_type == 20)
                 {
                    hypre_BoomerAMGBuildBlockInterp( A_block_array[level], CF_marker, 
                                                     SN,
                                                     coarse_pnts_global, 1,
                                                     NULL,
                                                     debug_flag,
                                                     trunc_factor, P_max_elmts, 0,
                                                     col_offd_S_to_A,
                                                     &P_block_array[level]);
                    
                 }
                 else if (interp_type == 21)
                 {
                    hypre_BoomerAMGBuildBlockInterpDiag( A_block_array[level], CF_marker, 
                                                         SN,
                                                         coarse_pnts_global, 1,
                                                         NULL,
                                                         debug_flag,
                                                         trunc_factor, P_max_elmts, 0,
                                                         col_offd_S_to_A,
                                                         &P_block_array[level]);
                 }
                 else if (interp_type == 24)
                 {
                    hypre_BoomerAMGBuildBlockDirInterp( A_block_array[level], CF_marker, 
                                                        SN,
                                                        coarse_pnts_global, 1,
                                                        NULL,
                                                        debug_flag,
                                                        trunc_factor, P_max_elmts,
                                                        col_offd_S_to_A,
                                                        &P_block_array[level]);
                 }

                 else /* interp_type ==10 */
                 {
                    
                    hypre_BoomerAMGBuildBlockInterp( A_block_array[level], CF_marker, 
                                                     SN,
                                                     coarse_pnts_global, 1,
                                                     NULL,
                                                     debug_flag,
                                                     trunc_factor, P_max_elmts, 1,
                                                     col_offd_S_to_A,
                                                     &P_block_array[level]);
                    
                 }
            
#ifdef HYPRE_NO_GLOBAL_PARTITION 
                 /* we need to set the global number of cols in P, as this was 
                    not done in the interp
                    (which calls the matrix create) since we didn't 
                    have the global partition */
                 /*  this has to be done before converting from block to non-block*/
                 hypre_ParCSRBlockMatrixGlobalNumCols(P_block_array[level]) = coarse_size;
#endif
                 
                 /* if we don't do nodal relaxation, we need a CF_array that is 
                    not nodal - right now we don't allow this to happen though*/
                 /*
                   if (grid_relax_type[0] < 20  )
                   {
                   hypre_BoomerAMGCreateScalarCF(CFN_marker, num_functions,
                   hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AN)),
                   &dof_func1, &CF_marker);
               
                   dof_func_array[level+1] = dof_func1;
                   hypre_TFree (CFN_marker);
                   CF_marker_array[level] = CF_marker;
                   }
                 */
                 
                 /* clean up other things */
                 hypre_ParCSRMatrixDestroy(AN);
                 hypre_ParCSRMatrixDestroy(SN);
                 
              }
              else /* not block mode - use default interp (interp_type = 0) */
              {
                 if (nodal > -1) /* non-systems, or systems with unknown approach interpolation*/
                 {
                    /* if systems, do we want to use an interp. that uses the full strength matrix?*/
                    
                    if ( (num_functions > 1) && (interp_type == 19 || interp_type == 18 || interp_type == 17 || interp_type == 16))   
                    {
                       /* so create a second strength matrix and build interp with with num_functions = 1 */
                       hypre_BoomerAMGCreateS(A_array[level], 
                                              strong_threshold, max_row_sum, 
                                              1, dof_func_array[level],&S2);
                       col_offd_S_to_A = NULL;
                       switch (interp_type) 
                       {
                          
                          case 19:
                             dbg_flg = debug_flag;
                             if (amg_print_level) dbg_flg = -debug_flag;
                             hypre_BoomerAMGBuildInterp(A_array[level], CF_marker, 
                                                        S2, coarse_pnts_global, 1, 
                                                        dof_func_array[level], 
                                                        dbg_flg, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
                             break;
                             
                          case 18:
                             hypre_BoomerAMGBuildStdInterp(A_array[level], CF_marker, 
                                                           S2, coarse_pnts_global, 1, dof_func_array[level], 
                                                           debug_flag, trunc_factor, P_max_elmts, 0, col_offd_S_to_A, &P);
                             
                             break;
                             
                          case 17:
                             hypre_BoomerAMGBuildExtPIInterp(A_array[level], CF_marker, 
                                                             S2, coarse_pnts_global, 1, dof_func_array[level], 
                                                             debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
                             break;
                          case 16:
                             dbg_flg = debug_flag;
                             if (amg_print_level) dbg_flg = -debug_flag;
                             hypre_BoomerAMGBuildInterpModUnk(A_array[level], CF_marker, 
                                                              S2, coarse_pnts_global, num_functions, dof_func_array[level], 
                                                              dbg_flg, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
                             break;
                             
                       }
                  

                       hypre_ParCSRMatrixDestroy(S2);
             
                    }
                    else /* one function only or unknown-based interpolation- */
                    {
                       dbg_flg = debug_flag;
                       if (amg_print_level) dbg_flg = -debug_flag;
                       
                       hypre_BoomerAMGBuildInterp(A_array[level], CF_marker, 
                                                  S, coarse_pnts_global, num_functions, 
                                                  dof_func_array[level], 
                                                  dbg_flg, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
                       
                       
                    }
               
                    hypre_TFree(col_offd_S_to_A);
                 }
              } 
           }
           else
           {
              hypre_BoomerAMGBuildInterpGSMG(NULL, CF_marker, S,
                                             coarse_pnts_global, num_functions, dof_func_array[level], 
                                             debug_flag, trunc_factor, &P);
              
              
           }
            
            
         } /* end of no aggressive coarsening */

         /*dof_func_array[level+1] = NULL;
         if (num_functions > 1 && nodal > -1 && (!block_mode) )
            dof_func_array[level+1] = coarse_dof_func;*/


         /* store the CF array */
         CF_marker_array[level] = CF_marker;
         

         if (debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            hypre_printf("Proc = %d    Level = %d    Coarsen Time = %f\n",
                       my_id,level, wall_time); 
	    fflush(NULL);
         }

         dof_func_array[level+1] = NULL;
         if (num_functions > 1 && nodal > -1 && (!block_mode) )
	    dof_func_array[level+1] = coarse_dof_func;
         
      
      } /* end of if max_levels > 1 */

      /* if no coarse-grid, stop coarsening, and set the
       * coarsest solve to be a single sweep of Jacobi */
      if ((coarse_size == 0) ||
          (coarse_size == fine_size))
      {
         HYPRE_Int     *num_grid_sweeps =
            hypre_ParAMGDataNumGridSweeps(amg_data);
         HYPRE_Int    **grid_relax_points =
            hypre_ParAMGDataGridRelaxPoints(amg_data);
         if (grid_relax_type[3] == 9)
	 {
	    grid_relax_type[3] = grid_relax_type[0];
	    num_grid_sweeps[3] = 1;
	    if (grid_relax_points) grid_relax_points[3][0] = 0; 
	 }
	 if (S)
            hypre_ParCSRMatrixDestroy(S);
	 if (P)
            hypre_ParCSRMatrixDestroy(P);
         if (level > 0)
         {
            /* note special case treatment of CF_marker is necessary
             * to do CF relaxation correctly when num_levels = 1 */
            hypre_TFree(CF_marker_array[level]);
            hypre_ParVectorDestroy(F_array[level]);
            hypre_ParVectorDestroy(U_array[level]);
         }

         break; 
      }

      /*-------------------------------------------------------------
       * Build prolongation matrix, P, and place in P_array[level] 
       *--------------------------------------------------------------*/

      if (interp_refine > 0 )
      {
         for (k = 0; k < interp_refine; k++)
            hypre_BoomerAMGRefineInterp(A_array[level],
                                        &P,
                                        coarse_pnts_global,
                                        &num_functions, 
                                        dof_func_array[level], 
                                        CF_marker_array[level], level);
      }


      /*  Post processing of interpolation operators to incorporate
          smooth vectors NOTE: must pick nodal coarsening !!!
          (nodal is changed above to 1 if it is 0)  */
      if (interp_vec_variant && nodal && num_interp_vectors)
      {
         /* TO DO: add option of smoothing the vectors at
          * coarser levels?*/

         if (level < interp_vec_first_level)
         {

            /* coarsen the smooth vecs */
            hypre_BoomerAMGCoarsenInterpVectors( P, 
                                                 num_interp_vectors,
                                                 interp_vectors_array[level],
                                                 CF_marker_array[level], 
                                                 &interp_vectors_array[level+1], 
                                                 0, num_functions);
            
         }
         /* do  GM 2 and LN (3) at all levels and GM 1 only on first level */
         if (( interp_vec_variant > 1  && level >= interp_vec_first_level) || 
             (interp_vec_variant == 1 && interp_vec_first_level == level))

         {
            if (interp_vec_variant < 3) /* GM */
            {
               hypre_BoomerAMG_GMExpandInterp( A_array[level],
                                               &P,
                                               num_interp_vectors,
                                               interp_vectors_array[level],
                                               &num_functions, 
                                               dof_func_array[level], 
                                               &dof_func_array[level+1],
                                               interp_vec_variant, level, 
                                               abs_q_trunc, 
                                               expandp_weights, 
                                               q_max,
                                               CF_marker_array[level], interp_vec_first_level);
            }
            else /* LN */
            {
               hypre_BoomerAMG_LNExpandInterp( A_array[level],
                                               &P, 
                                               coarse_pnts_global,
                                               &num_functions, 
                                               dof_func_array[level], 
                                               &dof_func_array[level+1],
                                               CF_marker_array[level],
                                               level,
                                               expandp_weights, 
                                               num_interp_vectors,
                                               interp_vectors_array[level],
                                               abs_q_trunc, 
                                               q_max,
                                               interp_vec_first_level);
            }
            
            if (level == interp_vec_first_level)
            {
               /* check to see if we made A bigger - this can happen
                * in 3D with certain coarsenings   - if so, need to fix vtemp*/
               
               if (hypre_ParCSRMatrixGlobalNumRows(A_array[0]) < hypre_ParCSRMatrixGlobalNumCols(P))
               {
                  
                  hypre_ParVectorDestroy(Vtemp);
                  Vtemp = NULL;
                  
                  Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(P),
                                                hypre_ParCSRMatrixGlobalNumCols(P),
                                                hypre_ParCSRMatrixColStarts(P));
                  hypre_ParVectorInitialize(Vtemp);
                  hypre_ParVectorSetPartitioningOwner(Vtemp,0);
                  hypre_ParAMGDataVtemp(amg_data) = Vtemp;
               }
            }
            /* at the first level we have to add space for the new
             * unknowns in the smooth vectors */
            if (interp_vec_variant > 1 && level < max_levels)
            {
               HYPRE_Int expand_level = 0;
               
               if (level == interp_vec_first_level)
                  expand_level = 1;
               
               hypre_BoomerAMGCoarsenInterpVectors( P, 
                                                    num_interp_vectors,
                                                    interp_vectors_array[level],
                                                    CF_marker_array[level], 
                                                    &interp_vectors_array[level+1], 
                                                    expand_level, num_functions);
            }
         } /* end apply variant */
      }/* end interp_vec_variant > 0 */
      
      for (i=0; i < post_interp_type; i++)
         /* Improve on P with Jacobi interpolation */
         hypre_BoomerAMGJacobiInterp( A_array[level], &P, S,
                                      num_functions, dof_func,
                                      CF_marker_array[level],
                                      level, jacobi_trunc_threshold, 0.5*jacobi_trunc_threshold );


      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         hypre_printf("Proc = %d    Level = %d    Build Interp Time = %f\n",
                     my_id,level, wall_time);
         fflush(NULL);
      }

      if (!block_mode)
      {
         P_array[level] = P; 
      }
      
      if (S) hypre_ParCSRMatrixDestroy(S);
      S = NULL;

      hypre_TFree(SmoothVecs);
      SmoothVecs = NULL;

      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      if (debug_flag==1) wall_time = time_getWallclockSeconds();

      if (block_mode)
      {

         hypre_ParCSRBlockMatrixRAP(P_block_array[level],
                                    A_block_array[level],
                                    P_block_array[level], &A_H_block);
         
         hypre_ParCSRBlockMatrixSetNumNonzeros(A_H_block);
         hypre_ParCSRBlockMatrixSetDNumNonzeros(A_H_block);
         A_block_array[level+1] = A_H_block;

      }
      else
      {
         
         hypre_BoomerAMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                            P_array[level], &A_H);
      }
 
      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         hypre_printf("Proc = %d    Level = %d    Build Coarse Operator Time = %f\n",
                       my_id,level, wall_time);
	 fflush(NULL);
      }

      ++level;

      if (!block_mode)
      {
         hypre_ParCSRMatrixSetNumNonzeros(A_H);
         hypre_ParCSRMatrixSetDNumNonzeros(A_H);
         A_array[level] = A_H;
      }
      
      size = ((double) fine_size )*.75;
      if (coarsen_type > 0 && coarse_size >= (HYPRE_Int) size)
      {
	coarsen_type = 0;      
      }


      {
	 HYPRE_Int max_thresh = hypre_max(coarse_threshold, seq_threshold);
         if ( (level == max_levels-1) || (coarse_size <= max_thresh) )
         {
            not_finished_coarsening = 0;
         }
      }
   } 

   /* redundant coarse grid solve */
   if (  (seq_threshold >= coarse_threshold) && (coarse_size > coarse_threshold) && (level != max_levels-1))
   {
      hypre_seqAMGSetup( amg_data, level, coarse_threshold);
   }


   if (level > 0)
   {
      if (block_mode)
      {
         F_array[level] =
            hypre_ParVectorCreateFromBlock(hypre_ParCSRBlockMatrixComm(A_block_array[level]),
                                           hypre_ParCSRMatrixGlobalNumRows(A_block_array[level]),
                                           hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]),
                                           hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]));
         hypre_ParVectorInitialize(F_array[level]);
         
         U_array[level] =  
            hypre_ParVectorCreateFromBlock(hypre_ParCSRBlockMatrixComm(A_block_array[level]),
                                           hypre_ParCSRMatrixGlobalNumRows(A_block_array[level]),
                                           hypre_ParCSRBlockMatrixRowStarts(A_block_array[level]),
                                           hypre_ParCSRBlockMatrixBlockSize(A_block_array[level]));
         
         hypre_ParVectorInitialize(U_array[level]);
      }
      else 
      {
         F_array[level] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                  hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorInitialize(F_array[level]);
         hypre_ParVectorSetPartitioningOwner(F_array[level],0);
         
         U_array[level] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                  hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorInitialize(U_array[level]);
         hypre_ParVectorSetPartitioningOwner(U_array[level],0);
      }   
   }
   
   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   num_levels = level+1;
   hypre_ParAMGDataNumLevels(amg_data) = num_levels;
   if (hypre_ParAMGDataSmoothNumLevels(amg_data) > num_levels-1)
      hypre_ParAMGDataSmoothNumLevels(amg_data) = num_levels-1;
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   
   /*-----------------------------------------------------------------------
    * Setup of special smoothers when needed
    *-----------------------------------------------------------------------*/

   if (grid_relax_type[1] == 8 )
   {
      l1_norms = hypre_CTAlloc(double *, num_levels);
      hypre_ParAMGDataL1Norms(amg_data) = l1_norms;
   }
   if (grid_relax_type[1] == 18)
   {
      l1_norms = hypre_CTAlloc(double *, num_levels);
      hypre_ParAMGDataL1Norms(amg_data) = l1_norms;
   }
   if (grid_relax_type[0] == 16 ||grid_relax_type[1] == 16 || grid_relax_type[2] == 16 || grid_relax_type[3] == 16)
      /* Chebyshev */
   {
      max_eig_est = hypre_CTAlloc(double, num_levels);
      min_eig_est = hypre_CTAlloc(double, num_levels);
      hypre_ParAMGDataMaxEigEst(amg_data) = max_eig_est;
      hypre_ParAMGDataMinEigEst(amg_data) = min_eig_est;
   }
   if (grid_relax_type[0] == 15 ||grid_relax_type[1] == 15 ||  grid_relax_type[2] == 15 || grid_relax_type[3] == 15)
      /* CG */
   {
      smoother = hypre_CTAlloc(HYPRE_Solver, num_levels);
      hypre_ParAMGDataSmoother(amg_data) = smoother;
   }

   for (j = 0; j < num_levels; j++)
   {
      if (num_threads == 1)
      {
         if (grid_relax_type[1] == 8 && j < num_levels-1)
         {
            if (relax_order)
               hypre_ParCSRComputeL1Norms(A_array[j], 4, CF_marker_array[j], &l1_norms[j]);
            else
               hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norms[j]);
         }
         else if (grid_relax_type[3] == 8 && j == num_levels-1)
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norms[j]);
         }
         if (grid_relax_type[1] == 18 && j < num_levels-1)
         {
            if (relax_order)
               hypre_ParCSRComputeL1Norms(A_array[j], 1, CF_marker_array[j], &l1_norms[j]);
            else
               hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norms[j]);
         }
         else if (grid_relax_type[3] == 18 && j == num_levels-1)
         {
            hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norms[j]);
         }
      }
      else
      {
         if (grid_relax_type[1] == 8 && j < num_levels-1)
         {
            if (relax_order)
               hypre_ParCSRComputeL1NormsThreads(A_array[j], 4, num_threads, CF_marker_array[j] , &l1_norms[j]);
            else
               hypre_ParCSRComputeL1NormsThreads(A_array[j], 4, num_threads, NULL, &l1_norms[j]);
         }
         else if (grid_relax_type[3] == 8 && j == num_levels-1)
         {
            hypre_ParCSRComputeL1NormsThreads(A_array[j], 4, num_threads, NULL, &l1_norms[j]);
         }
         if (grid_relax_type[1] == 18 && j < num_levels-1)
         {
            if (relax_order)
               hypre_ParCSRComputeL1NormsThreads(A_array[j], 1, num_threads, CF_marker_array[j], &l1_norms[j]);
            else
               hypre_ParCSRComputeL1NormsThreads(A_array[j], 1, num_threads, NULL, &l1_norms[j]);
         }
         else if (grid_relax_type[3] == 18 && j == num_levels-1)
         {
            hypre_ParCSRComputeL1NormsThreads(A_array[j], 1, num_threads, NULL, &l1_norms[j]);
         }

      }
      if (grid_relax_type[1] == 16 || grid_relax_type[2] == 16 || (grid_relax_type[3] == 16 && j== (num_levels-1)))
      {
         HYPRE_Int scale = 1;
         double temp_d, temp_d2;
         hypre_ParCSRMaxEigEstimateCG(A_array[j], scale, 10, &temp_d, &temp_d2);
         max_eig_est[j] = temp_d;
         min_eig_est[j] = temp_d2;
      }
     else if (grid_relax_type[1] == 15 || (grid_relax_type[3] == 15 && j == (num_levels-1))  )
     {
        
        HYPRE_ParCSRPCGCreate(comm, &smoother[j]);
        HYPRE_ParCSRPCGSetup(smoother[j],
                             (HYPRE_ParCSRMatrix) A_array[j],
                             (HYPRE_ParVector) F_array[j],
                             (HYPRE_ParVector) U_array[j]);
        
        HYPRE_PCGSetTol(smoother[j], 1e-12); /* make small */
        HYPRE_PCGSetTwoNorm(smoother[j], 1); /* use 2-norm*/
        
        HYPRE_ParCSRPCGSetup(smoother[j], 
                             (HYPRE_ParCSRMatrix) A_array[j], 
                             (HYPRE_ParVector) F_array[j], 
                             (HYPRE_ParVector) U_array[j]);
        
        
     }
     if (relax_weight[j] == 0.0)
     {
        hypre_ParCSRMatrixScaledNorm(A_array[j], &relax_weight[j]);
        if (relax_weight[j] != 0.0)
           relax_weight[j] = 4.0/3.0/relax_weight[j];
        else
           hypre_printf (" Warning ! Matrix norm is zero !!!");
     }
     if ((smooth_type == 6 || smooth_type == 16) && smooth_num_levels > j)
     {
        
        schwarz_relax_wt = hypre_ParAMGDataSchwarzRlxWeight(amg_data);
        
        HYPRE_SchwarzCreate(&smoother[j]);
        HYPRE_SchwarzSetNumFunctions(smoother[j],num_functions);
        HYPRE_SchwarzSetVariant(smoother[j],
                                hypre_ParAMGDataVariant(amg_data));
        HYPRE_SchwarzSetOverlap(smoother[j],
                                hypre_ParAMGDataOverlap(amg_data));
        HYPRE_SchwarzSetDomainType(smoother[j],
                                   hypre_ParAMGDataDomainType(amg_data));
        HYPRE_SchwarzSetNonSymm(smoother[j],
                                hypre_ParAMGDataSchwarzUseNonSymm(amg_data));
        if (schwarz_relax_wt > 0)
           HYPRE_SchwarzSetRelaxWeight(smoother[j],schwarz_relax_wt);
        HYPRE_SchwarzSetup(smoother[j],
                           (HYPRE_ParCSRMatrix) A_array[j],
                           (HYPRE_ParVector) f,
                           (HYPRE_ParVector) u);
        if (schwarz_relax_wt < 0 )
        {
           num_cg_sweeps = (HYPRE_Int) (-schwarz_relax_wt);
           hypre_BoomerAMGCGRelaxWt(amg_data, j, num_cg_sweeps,
                                    &schwarz_relax_wt);
           /*hypre_printf (" schwarz weight %f \n", schwarz_relax_wt);*/
           HYPRE_SchwarzSetRelaxWeight(smoother[j], schwarz_relax_wt);
           if (hypre_ParAMGDataVariant(amg_data) > 0)
           {
              local_size = hypre_CSRMatrixNumRows
                 (hypre_ParCSRMatrixDiag(A_array[j]));
              hypre_SchwarzReScale(smoother[j], local_size, 
                                   schwarz_relax_wt);
           }
           schwarz_relax_wt = 1;
        }
     }
     else if ((smooth_type == 9 || smooth_type == 19) && smooth_num_levels > j)
     {
        HYPRE_EuclidCreate(comm, &smoother[j]);
        if (euclidfile)
           HYPRE_EuclidSetParamsFromFile(smoother[j],euclidfile); 
        HYPRE_EuclidSetLevel(smoother[j],eu_level); 
        if (eu_bj)
           HYPRE_EuclidSetBJ(smoother[j],eu_bj); 
        if (eu_sparse_A)
           HYPRE_EuclidSetSparseA(smoother[j],eu_sparse_A); 
        HYPRE_EuclidSetup(smoother[j],
                          (HYPRE_ParCSRMatrix) A_array[j],
                          (HYPRE_ParVector) F_array[j],
                          (HYPRE_ParVector) U_array[j]); 
     }
     else if ((smooth_type == 8 || smooth_type == 18) && smooth_num_levels > j)
     {
        HYPRE_ParCSRParaSailsCreate(comm, &smoother[j]);
        HYPRE_ParCSRParaSailsSetParams(smoother[j],thresh,nlevel);
        HYPRE_ParCSRParaSailsSetFilter(smoother[j],filter);
        HYPRE_ParCSRParaSailsSetSym(smoother[j],sym);
        HYPRE_ParCSRParaSailsSetup(smoother[j],
                                   (HYPRE_ParCSRMatrix) A_array[j],
                                   (HYPRE_ParVector) F_array[j],
                                   (HYPRE_ParVector) U_array[j]);
     }
     else if ((smooth_type == 7 || smooth_type == 17) && smooth_num_levels > j)
     {
        HYPRE_ParCSRPilutCreate(comm, &smoother[j]);
        HYPRE_ParCSRPilutSetup(smoother[j],
                               (HYPRE_ParCSRMatrix) A_array[j],
                               (HYPRE_ParVector) F_array[j],
                               (HYPRE_ParVector) U_array[j]);
        HYPRE_ParCSRPilutSetDropTolerance(smoother[j],drop_tol);
        HYPRE_ParCSRPilutSetFactorRowSize(smoother[j],max_nz_per_row);
     }
     else if ((j < num_levels-1) || ((j == num_levels-1) && (grid_relax_type[3]!= 9) && coarse_size > 9))
     {
        if (relax_weight[j] < 0 )
        {
           num_cg_sweeps = (HYPRE_Int) (-relax_weight[j]);
           hypre_BoomerAMGCGRelaxWt(amg_data, j, num_cg_sweeps,
                                 &relax_weight[j]);
        }
        if (omega[j] < 0 )
        {
           num_cg_sweeps = (HYPRE_Int) (-omega[j]);
           hypre_BoomerAMGCGRelaxWt(amg_data, j, num_cg_sweeps,
                                 &omega[j]);
        }
     }
   } /* end of levels loop */

   if ( amg_logging > 1 ) {

      Residual_array= 
	hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                              hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                              hypre_ParCSRMatrixRowStarts(A_array[0]) );
      hypre_ParVectorInitialize(Residual_array);
      hypre_ParVectorSetPartitioningOwner(Residual_array,0);
      hypre_ParAMGDataResidual(amg_data) = Residual_array;
   }
   else
      hypre_ParAMGDataResidual(amg_data) = NULL;

   /*-----------------------------------------------------------------------
    * Print some stuff
    *-----------------------------------------------------------------------*/

   if (amg_print_level == 1 || amg_print_level == 3)
      hypre_BoomerAMGSetupStats(amg_data,A);

/* print out CF info to plot grids in matlab (see 'tools/AMGgrids.m') */

   if (hypre_ParAMGDataPlotGrids(amg_data))
   {
     HYPRE_Int *CF, *CFc, *itemp;
     FILE* fp;
     char filename[256];
     HYPRE_Int coorddim = hypre_ParAMGDataCoordDim (amg_data);
     float *coordinates = hypre_ParAMGDataCoordinates (amg_data);
                                                                                
     if (!coordinates) coorddim=0;
                                                                                

     if (block_mode)
        local_size = hypre_CSRMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[0]));
     else      
        local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

     CF = hypre_CTAlloc(HYPRE_Int, local_size);
     CFc = hypre_CTAlloc(HYPRE_Int, local_size);

     for (level = (num_levels - 2); level >= 0; level--)
     {
      /* swap pointers */
        itemp = CFc;
        CFc = CF;
        CF = itemp;
        if (block_mode)
           local_size = hypre_CSRMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[level]));
        else
           local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
      

      for (i = 0, j = 0; i < local_size; i++)
      {
         /* if a C-point */
         CF[i] = 0;
         if (CF_marker_array[level][i] > -1)
         {
            CF[i] = CFc[j] + 1;
            j++;
         }
      }
     }
     if (block_mode)
        local_size = hypre_CSRMatrixNumRows(hypre_ParCSRBlockMatrixDiag(A_block_array[0]));
     else
        local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
     hypre_sprintf (filename,"%s.%05d",hypre_ParAMGDataPlotFileName (amg_data),my_id);     fp = fopen(filename, "w");

     for (i = 0; i < local_size; i++)
     {
       for (j = 0; j < coorddim; j++)
         hypre_fprintf (fp,"%f ",coordinates[coorddim*i+j]);
       hypre_fprintf(fp, "%d\n", CF[i]);
     }
     fclose(fp);

     hypre_TFree(CF);
     hypre_TFree(CFc);
  }

/* print out matrices on all levels  */
#if 0
{
   char  filename[256];

   if (block_mode)
   {
      hypre_ParCSRMatrix *temp_A;

      for (level = 0; level < num_levels; level++)
      {
         hypre_sprintf(filename, "BoomerAMG.out.A_blk.%02d.ij", level);
         temp_A =  hypre_ParCSRBlockMatrixConvertToParCSRMatrix(
            A_block_array[level]);
         hypre_ParCSRMatrixPrintIJ(temp_A, 0, 0, filename);
         hypre_ParCSRMatrixDestroy(temp_A);
      }
      
   }
   else
   {
      
      for (level = 0; level < num_levels; level++)
      {
         hypre_sprintf(filename, "BoomerAMG.out.A.%02d.ij", level);
         hypre_ParCSRMatrixPrintIJ(A_array[level], 0, 0, filename);
      }
   }
}
#endif

/* run compatible relaxation on all levels and print results */
#if 0
{
   hypre_ParVector *u_vec, *f_vec;
   double          *u, rho0, rho1, rho;
   HYPRE_Int              n;

   for (level = 0; level < (num_levels-1); level++)
   {
      u_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                    hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                    hypre_ParCSRMatrixRowStarts(A_array[level]));
      hypre_ParVectorInitialize(u_vec);
      hypre_ParVectorSetPartitioningOwner(u_vec,0);
      f_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                    hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                    hypre_ParCSRMatrixRowStarts(A_array[level]));
      hypre_ParVectorInitialize(f_vec);
      hypre_ParVectorSetPartitioningOwner(f_vec,0);

      hypre_ParVectorSetRandomValues(u_vec, 99);
      hypre_ParVectorSetConstantValues(f_vec, 0.0);

      /* set C-pt values to zero */
      n = hypre_VectorSize(hypre_ParVectorLocalVector(u_vec));
      u = hypre_VectorData(hypre_ParVectorLocalVector(u_vec));
      for (i = 0; i < n; i++)
      {
         if (CF_marker_array[level][i] == 1)
         {
            u[i] = 0.0;
         }
      }

      rho1 = hypre_ParVectorInnerProd(u_vec, u_vec);
      for (i = 0; i < 5; i++)
      {
         rho0 = rho1;
         hypre_BoomerAMGRelax(A_array[level], f_vec, CF_marker_array[level],
                              grid_relax_type[0], -1,
                              relax_weight[level], omega[level], l1_norms[level],
                              u_vec, Vtemp, Ztemp);
         rho1 = hypre_ParVectorInnerProd(u_vec,u_vec);
         rho = sqrt(rho1/rho0);
         if (rho < 0.01)
         {
            break;
         }
      }

      hypre_ParVectorDestroy(u_vec);
      hypre_ParVectorDestroy(f_vec);

      if (my_id == 0)
      {
         hypre_printf("level = %d, rhocr = %f\n", level, rho);
      }
   }
}
#endif

   return(Setup_err_flag);
}  
