/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "amg.h"

/*****************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

int
hypre_AMGSetup( void            *amg_vdata,
                hypre_CSRMatrix *A,
                hypre_Vector    *f,
                hypre_Vector    *u         )
{
   hypre_AMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_CSRMatrix **A_array;
   hypre_Vector    **F_array;
   hypre_Vector    **U_array;
   hypre_CSRMatrix **P_array;

   int             **dof_func_array;
   int              *dof_func;
   int              *coarse_dof_func;

   int             **CF_marker_array;   
   double           *relax_weight;
   double            strong_threshold;

   int      num_variables;
   int      max_levels; 
   int      amg_ioutdat;
   int      interp_type;
   int      num_functions;
 
   /* Local variables */
   int              *CF_marker;
   hypre_CSRMatrix  *S;
   hypre_CSRMatrix  *P;
   hypre_CSRMatrix  *A_H;

   int       num_levels;
   int       level;
   int       coarse_size;
   int       fine_size;
   int       not_finished_coarsening = 1;
   int       Setup_err_flag;
   int       coarse_threshold = 9;
   int       i, j;
   int	     coarsen_type;
   int	    *grid_relax_type;
   int	     relax_type;
   int	     num_relax_steps;
   int	    *schwarz_option;
   int       num_domains;
   int      *i_domain_dof;
   int      *j_domain_dof;
   double   *domain_matrixinverse;

   relax_weight = hypre_AMGDataRelaxWeight(amg_data);
   num_relax_steps = hypre_AMGDataNumRelaxSteps(amg_data);
   grid_relax_type = hypre_AMGDataGridRelaxType(amg_data);
   max_levels = hypre_AMGDataMaxLevels(amg_data);
   amg_ioutdat = hypre_AMGDataIOutDat(amg_data);
   interp_type = hypre_AMGDataInterpType(amg_data);
   num_functions = hypre_AMGDataNumFunctions(amg_data);
   relax_type = grid_relax_type[0];
   schwarz_option = hypre_AMGDataSchwarzOption(amg_data);
 
   dof_func = hypre_AMGDataDofFunc(amg_data);

   A_array = hypre_CTAlloc(hypre_CSRMatrix*, max_levels);
   P_array = hypre_CTAlloc(hypre_CSRMatrix*, max_levels-1);
   CF_marker_array = hypre_CTAlloc(int*, max_levels-1);
   dof_func_array = hypre_CTAlloc(int*, max_levels);
   coarse_dof_func = NULL;

   if (schwarz_option[0] > -1)
   {
      hypre_AMGDataNumDomains(amg_data) = hypre_CTAlloc(int, max_levels);
      hypre_AMGDataIDomainDof(amg_data) = hypre_CTAlloc(int*, max_levels);
      hypre_AMGDataJDomainDof(amg_data) = hypre_CTAlloc(int*, max_levels);
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

   /*----------------------------------------------------------
    * Initialize hypre_AMGData
    *----------------------------------------------------------*/

   num_variables = hypre_CSRMatrixNumRows(A);


   hypre_AMGDataNumVariables(amg_data) = num_variables;

   not_finished_coarsening = 1;
   level = 0;
  
   strong_threshold = hypre_AMGDataStrongThreshold(amg_data);

   coarsen_type = hypre_AMGDataCoarsenType(amg_data);

   /*-----------------------------------------------------
    *  Enter Coarsening Loop
    *-----------------------------------------------------*/

   while (not_finished_coarsening)
   {
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
           printf (" Warning ! Matrix norm is zero !!!");
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
	 hypre_AMGCoarsenRuge(A_array[level], strong_threshold,
                       &S, &CF_marker, &coarse_size); 
      }
      else if (coarsen_type == 2)
      {
	 hypre_AMGCoarsenRugeLoL(A_array[level], strong_threshold,
				 dof_func_array[level],
				 &S, &CF_marker, &coarse_size); 
      }
      else if (coarsen_type == 3)
      {
         hypre_AMGCoarsenCR(A_array[level], strong_threshold,
			relax_weight[level], relax_type, 
			num_relax_steps, &CF_marker, &coarse_size); 
      }
      else
      {
         hypre_AMGCoarsen(A_array[level], strong_threshold,
                       &S, &CF_marker, &coarse_size); 
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
      else 
      {
	hypre_AMGBuildInterp(A_array[level], CF_marker_array[level], S,
                             dof_func_array[level], &coarse_dof_func, &P);
      }

      printf("END computing level %d interpolation matrix; =======\n", level);

      dof_func_array[level+1] = coarse_dof_func;
      P_array[level] = P; 
      
      if (amg_ioutdat == 5 && level == 0)
      {
         hypre_CSRMatrixPrint(S,"S_mat");
      }
      if (coarsen_type != 3) hypre_CSRMatrixDestroy(S);
 
      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      hypre_AMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                   P_array[level], &A_H);

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
      F_array[j] = hypre_VectorCreate(hypre_CSRMatrixNumRows(A_array[j]));
      hypre_VectorInitialize(F_array[j]);

      U_array[j] = hypre_VectorCreate(hypre_CSRMatrixNumRows(A_array[j]));
      hypre_VectorInitialize(U_array[j]);
   }

   hypre_AMGDataFArray(amg_data) = F_array;
   hypre_AMGDataUArray(amg_data) = U_array;

   /*-----------------------------------------------------------------------
    * Print some stuff
    *-----------------------------------------------------------------------*/

   if (amg_ioutdat == 1 || amg_ioutdat == 3)
      hypre_AMGSetupStats(amg_data);

   if (amg_ioutdat == -3)
   {  
      char     fnam[255];

      int j;

      for (j = 0; j < level+1; j++)
      {
         sprintf(fnam,"SP_A_%d.ysmp",j);
         hypre_CSRMatrixPrint(A_array[j],fnam);

      }                         

      for (j = 0; j < level; j++)
      { 
         sprintf(fnam,"SP_P_%d.ysmp",j);
         hypre_CSRMatrixPrint(P_array[j],fnam);
      }   
   } 

   Setup_err_flag = 0;
   return(Setup_err_flag);
}  


