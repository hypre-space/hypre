/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "par_amg.h"

#define DEBUG 0

/*****************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

/*****************************************************************************
 * hypre_BoomerAMGSetup
 *****************************************************************************/

int
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
   hypre_ParCSRMatrix **P_array;
   hypre_ParVector    *Residual_array;
   int                **CF_marker_array;   
   int                **dof_func_array;   
   int                 *dof_func;
   double              *relax_weight;
   double              *omega;
   double               schwarz_relax_wt = 1;
   double               strong_threshold;
   double               max_row_sum;
   double               trunc_factor;

   int      max_levels; 
   int      amg_log_level;
   int      amg_print_level;
   int      debug_flag;

 
   /* Local variables */
   int                 *CF_marker;
   hypre_ParCSRMatrix  *S;
   hypre_ParCSRMatrix  *P;
   hypre_ParCSRMatrix  *A_H;
   double              *SmoothVecs = NULL;

   int       old_num_levels, num_levels;
   int       level;
   int       local_size, i;
   int       first_local_row;
   int       coarse_size;
   int       coarsen_type;
   int       measure_type;
   int       setup_type;
   int       fine_size;
   int       rest, tms, indx;
   double    size;
   int       not_finished_coarsening = 1;
   int       Setup_err_flag = 0;
   int       coarse_threshold = 9;
   int       j, k;
   int       num_procs,my_id;
   int      *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   int       num_functions = hypre_ParAMGDataNumFunctions(amg_data);
   int	    *coarse_dof_func;
   int	    *coarse_pnts_global;
   int       num_cg_sweeps;

   HYPRE_Solver *smoother;
   int       smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   int       smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   int	     sym;
   int	     nlevel;
   double    thresh;
   double    filter;
   double    drop_tol;
   int	     max_nz_per_row;
   char     *euclidfile;


   double    wall_time;   /* for debugging instrumentation */

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);

   old_num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   amg_log_level = hypre_ParAMGDataLogLevel(amg_data);
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
   
   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParAMGDataNumVariables(amg_data) = hypre_ParCSRMatrixNumRows(A);

   if (setup_type == 0) return Setup_err_flag;

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);
   dof_func_array = hypre_ParAMGDataDofFuncArray(amg_data);
   local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   grid_relax_type[3] = hypre_ParAMGDataUserCoarseRelaxType(amg_data); 
   if (A_array || P_array || CF_marker_array || dof_func_array)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (A_array[j])
         {
            hypre_ParCSRMatrixDestroy(A_array[j]);
            A_array[j] = NULL;
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
   if (P_array == NULL && max_levels > 1)
      P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels-1);
   if (CF_marker_array == NULL)
      CF_marker_array = hypre_CTAlloc(int*, max_levels);
   if (dof_func_array == NULL)
      dof_func_array = hypre_CTAlloc(int*, max_levels);
   if (dof_func == NULL)
   {
      first_local_row = hypre_ParCSRMatrixFirstRowIndex(A);
      dof_func = hypre_CTAlloc(int,local_size);
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
   }

   A_array[0] = A;
   dof_func_array[0] = dof_func;
   hypre_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_ParAMGDataDofFuncArray(amg_data) = dof_func_array;
   hypre_ParAMGDataAArray(amg_data) = A_array;
   hypre_ParAMGDataPArray(amg_data) = P_array;
   hypre_ParAMGDataRArray(amg_data) = P_array;

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
   max_row_sum = hypre_ParAMGDataMaxRowSum(amg_data);
   trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);
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
      fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
      if (level > 0)
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

      /*-------------------------------------------------------------
       * Select coarse-grid points on 'level' : returns CF_marker
       * for the level.  Returns strength matrix, S  
       *--------------------------------------------------------------*/
     
      if (debug_flag==1) wall_time = time_getWallclockSeconds();
      if (debug_flag==3)
      {
          printf("\n ===== Proc = %d     Level = %d  =====\n",
                        my_id, level);
          fflush(NULL);
      }

      if (smooth_type == 6 && smooth_num_levels > level)
      {
	 schwarz_relax_wt = hypre_ParAMGDataSchwarzRlxWeight(amg_data);
         HYPRE_SchwarzCreate(&smoother[level]);
         HYPRE_SchwarzSetNumFunctions(smoother[level],num_functions);
         HYPRE_SchwarzSetVariant(smoother[level],
		hypre_ParAMGDataVariant(amg_data));
         HYPRE_SchwarzSetOverlap(smoother[level],
		hypre_ParAMGDataOverlap(amg_data));
         HYPRE_SchwarzSetDomainType(smoother[level],
		hypre_ParAMGDataDomainType(amg_data));
	 if (schwarz_relax_wt > 0)
            HYPRE_SchwarzSetRelaxWeight(smoother[level],schwarz_relax_wt);
         HYPRE_SchwarzSetup(smoother[level],
                        (HYPRE_ParCSRMatrix) A_array[level],
                        (HYPRE_ParVector) f,
                        (HYPRE_ParVector) u);
      }
      if (max_levels > 1)
      {
         if (hypre_ParAMGDataGSMG(amg_data) || 
             hypre_ParAMGDataInterpType(amg_data) == 1)
         {
	    hypre_BoomerAMGCreateSmoothVecs(amg_data, A_array[level],
	       hypre_ParAMGDataNumGridSweeps(amg_data)[1],
               level, &SmoothVecs);
         }

         if (hypre_ParAMGDataGSMG(amg_data) == 0)
	 {
	    hypre_BoomerAMGCreateS(A_array[level], 
				   strong_threshold, max_row_sum, 
				   num_functions, dof_func_array[level],&S);
	 }
	 else
	 {
	    hypre_BoomerAMGCreateSmoothDirs(amg_data, A_array[level],
	       SmoothVecs, strong_threshold, 
               num_functions, dof_func_array[level], &S);
	 }

         if (coarsen_type == 6)
         {
	    hypre_BoomerAMGCoarsenFalgout(S, A_array[level], measure_type,
                                    debug_flag, &CF_marker);
         }
         else if (coarsen_type)
         {
	    hypre_BoomerAMGCoarsenRuge(S, A_array[level],
                                 measure_type, coarsen_type, debug_flag,
                                 &CF_marker);
         }
         else
         {
	    hypre_BoomerAMGCoarsen(S, A_array[level], 0,
                             debug_flag, &CF_marker);
         }
         CF_marker_array[level] = CF_marker;
 
         if (relax_weight[level] == 0.0)
         {
	    hypre_ParCSRMatrixScaledNorm(A_array[level], &relax_weight[level]);
	    if (relax_weight[level] != 0.0)
	       relax_weight[level] = 4.0/3.0/relax_weight[level];
	    else
	       printf (" Warning ! Matrix norm is zero !!!");
         }
         if (relax_weight[level] < 0 )
         {
	    num_cg_sweeps = (int) (-relax_weight[level]);
 	    hypre_BoomerAMGCGRelaxWt(amg_data, level, num_cg_sweeps,
			&relax_weight[level]);
         }
         if (omega[level] < 0 )
         {
	    num_cg_sweeps = (int) (-omega[level]);
 	    hypre_BoomerAMGCGRelaxWt(amg_data, level, num_cg_sweeps,
			&omega[level]);
         }
         if (schwarz_relax_wt < 0 )
         {
	    num_cg_sweeps = (int) (-schwarz_relax_wt);
 	    hypre_BoomerAMGCGRelaxWt(amg_data, level, num_cg_sweeps,
			&schwarz_relax_wt);
	    printf (" schwarz weight %f \n", schwarz_relax_wt);
	    HYPRE_SchwarzSetRelaxWeight(smoother[level], schwarz_relax_wt);
 	    hypre_SchwarzReScale(smoother[level], local_size, schwarz_relax_wt);
	    schwarz_relax_wt = 1;
         }
         if (debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    Coarsen Time = %f\n",
                       my_id,level, wall_time); 
	    fflush(NULL);
         }

	 hypre_BoomerAMGCoarseParms(comm,
		hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level])),
			num_functions,dof_func_array[level],
			CF_marker,&coarse_dof_func,&coarse_pnts_global);
         dof_func_array[level+1] = NULL;
         if (num_functions > 1) dof_func_array[level+1] = coarse_dof_func;
	 coarse_size = coarse_pnts_global[num_procs];
      
      }
      else
      {
	 S = NULL;
	 coarse_pnts_global = NULL;
         CF_marker = hypre_CTAlloc(int, local_size );
	 for (i=0; i < local_size ; i++)
	    CF_marker[i] = 1;
         CF_marker_array = hypre_CTAlloc(int*, 1);
	 CF_marker_array[level] = CF_marker;
	 coarse_size = fine_size;
      }

      /* if no coarse-grid, stop coarsening, and set the
       * coarsest solve to be a single sweep of Jacobi */
      if ((coarse_size == 0) ||
          (coarse_size == fine_size))
      {
         int     *num_grid_sweeps =
            hypre_ParAMGDataNumGridSweeps(amg_data);
         int    **grid_relax_points =
            hypre_ParAMGDataGridRelaxPoints(amg_data);
         if (grid_relax_type[3] == 9)
	 {
	    grid_relax_type[3] = grid_relax_type[0];
	    num_grid_sweeps[3] = 1;
	    grid_relax_points[3][0] = 0; 
	 }
	 if (S)
            hypre_ParCSRMatrixDestroy(S);
	 hypre_TFree(coarse_pnts_global);
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

      if (debug_flag==1) wall_time = time_getWallclockSeconds();

      if (hypre_ParAMGDataInterpType(amg_data) == 1)
      {
          hypre_BoomerAMGNormalizeVecs(
		hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level])),
                 hypre_ParAMGDataNumSamples(amg_data), SmoothVecs);

          hypre_BoomerAMGBuildInterpLS(NULL, CF_marker_array[level], S,
                 coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, 
                 hypre_ParAMGDataNumSamples(amg_data), SmoothVecs, &P);
      }
      else if (hypre_ParAMGDataGSMG(amg_data) == 0)
      {
          hypre_BoomerAMGBuildInterp(A_array[level], CF_marker_array[level], S,
                 coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, &P);
      }
      else
      {
          hypre_BoomerAMGBuildInterpGSMG(NULL, CF_marker_array[level], S,
                 coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, &P);
      }

      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    Level = %d    Build Interp Time = %f\n",
                     my_id,level, wall_time);
         fflush(NULL);
      }

      P_array[level] = P; 
      hypre_ParCSRMatrixDestroy(S);
      S = NULL;

      hypre_TFree(SmoothVecs);
      SmoothVecs = NULL;

      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      if (debug_flag==1) wall_time = time_getWallclockSeconds();

      hypre_BoomerAMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                      P_array[level], &A_H);

      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    Level = %d    Build Coarse Operator Time = %f\n",
                       my_id,level, wall_time);
	 fflush(NULL);
      }

      ++level;
      hypre_ParCSRMatrixSetNumNonzeros(A_H);
      A_array[level] = A_H;

      size = ((double) fine_size )*.75;
      if (coarsen_type > 0 && coarse_size >= (int) size)
      {
	coarsen_type = 0;      
      }

      if ( (level == max_levels-1) || 
           (coarse_size <= coarse_threshold) )
      {
         not_finished_coarsening = 0;
      }
   } 
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
   
   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   num_levels = level+1;
   hypre_ParAMGDataNumLevels(amg_data) = num_levels;

   /*-----------------------------------------------------------------------
    * Setup F and U arrays
    *-----------------------------------------------------------------------*/

   for (j = 0; j < num_levels; j++)
   {
      if (smooth_type == 9 && smooth_num_levels > j)
      {
         HYPRE_EuclidCreate(comm, &smoother[j]);
         if (euclidfile)
            HYPRE_EuclidSetParamsFromFile(smoother[j],euclidfile); 
         HYPRE_EuclidSetup(smoother[j],
                        (HYPRE_ParCSRMatrix) A_array[j],
                        (HYPRE_ParVector) F_array[j],
                        (HYPRE_ParVector) U_array[j]); 
      }
      else if (smooth_type == 8 && smooth_num_levels > j)
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
      else if (smooth_type == 7 && smooth_num_levels > j)
      {
         HYPRE_ParCSRPilutCreate(comm, &smoother[j]);
         HYPRE_ParCSRPilutSetup(smoother[j],
                        (HYPRE_ParCSRMatrix) A_array[j],
                        (HYPRE_ParVector) F_array[j],
                        (HYPRE_ParVector) U_array[j]);
         HYPRE_ParCSRPilutSetDropTolerance(smoother[j],drop_tol);
         HYPRE_ParCSRPilutSetFactorRowSize(smoother[j],max_nz_per_row);
      }
   }

   if ( amg_log_level > 2 ) {
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

   return(Setup_err_flag);
}  
