/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "headers.h"
#include "../HYPRE.h" /* BM Aug 15, 2006 */

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define UNDECIDED 0 


/**************************************************************
 *
 *      CGC Coarsening routine
 *
 **************************************************************/
HYPRE_Int
hypre_BoomerAMGCoarsenCGCb( hypre_ParCSRMatrix    *S,
                            hypre_ParCSRMatrix    *A,
                            HYPRE_Int                    measure_type,
                            HYPRE_Int                    coarsen_type,
			    HYPRE_Int                    cgc_its,
                            HYPRE_Int                    debug_flag,
                            HYPRE_Int                  **CF_marker_ptr)
{
   MPI_Comm         comm          = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg   *comm_pkg      = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle *comm_handle;
   hypre_CSRMatrix *S_diag        = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix *S_offd        = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int             *S_i           = hypre_CSRMatrixI(S_diag);
   HYPRE_Int             *S_j           = hypre_CSRMatrixJ(S_diag);
   HYPRE_Int             *S_offd_i      = hypre_CSRMatrixI(S_offd);
   HYPRE_Int             *S_offd_j;
   HYPRE_Int              num_variables = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int              num_cols_offd = hypre_CSRMatrixNumCols(S_offd);
                  
   hypre_CSRMatrix *S_ext;
   HYPRE_Int             *S_ext_i;
   HYPRE_Int             *S_ext_j;
                 
   hypre_CSRMatrix *ST;
   HYPRE_Int             *ST_i;
   HYPRE_Int             *ST_j;
                 
   HYPRE_Int             *CF_marker;
   HYPRE_Int             *CF_marker_offd=NULL;
   HYPRE_Int              ci_tilde = -1;
   HYPRE_Int              ci_tilde_mark = -1;

   HYPRE_Int             *measure_array;
   HYPRE_Int             *measure_array_master;
   HYPRE_Int             *graph_array;
   HYPRE_Int 	           *int_buf_data=NULL;
   HYPRE_Int 	           *ci_array=NULL;

   HYPRE_Int              i, j, k, l, jS;
   HYPRE_Int		    ji, jj, index;
   HYPRE_Int		    set_empty = 1;
   HYPRE_Int		    C_i_nonempty = 0;
   HYPRE_Int		    num_nonzeros;
   HYPRE_Int		    num_procs, my_id;
   HYPRE_Int		    num_sends = 0;
   HYPRE_Int		    first_col, start;
   HYPRE_Int		    col_0, col_n;

   hypre_LinkList   LoL_head;
   hypre_LinkList   LoL_tail;

   HYPRE_Int             *lists, *where;
   HYPRE_Int              measure, new_meas;
   HYPRE_Int              num_left;
   HYPRE_Int              nabor, nabor_two;

   HYPRE_Int              ierr = 0;
   HYPRE_Int              use_commpkg_A = 0;
   double	    wall_time;

   HYPRE_Int              measure_max; /* BM Aug 30, 2006: maximal measure, needed for CGC */

   if (coarsen_type < 0) coarsen_type = -coarsen_type;

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(HYPRE_Int, num_variables);
   where = hypre_CTAlloc(HYPRE_Int, num_variables);

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        use_commpkg_A = 1;
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   if (!comm_pkg)
   {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_cols_offd) S_offd_j = hypre_CSRMatrixJ(S_offd);

   jS = S_i[num_variables];

   ST = hypre_CSRMatrixCreate(num_variables, num_variables, jS);
   ST_i = hypre_CTAlloc(HYPRE_Int,num_variables+1);
   ST_j = hypre_CTAlloc(HYPRE_Int,jS);
   hypre_CSRMatrixI(ST) = ST_i;
   hypre_CSRMatrixJ(ST) = ST_j;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i=0; i <= num_variables; i++)
      ST_i[i] = 0;
 
   for (i=0; i < jS; i++)
   {
	 ST_i[S_j[i]+1]++;
   }
   for (i=0; i < num_variables; i++)
   {
      ST_i[i+1] += ST_i[i];
   }
   for (i=0; i < num_variables; i++)
   {
      for (j=S_i[i]; j < S_i[i+1]; j++)
      {
	 index = S_j[j];
       	 ST_j[ST_i[index]] = i;
       	 ST_i[index]++;
      }
   }      
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i-1];
   }
   ST_i[0] = 0;

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are given by the row sums of ST.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    * correct actual measures through adding influences from
    * neighbor processors
    *----------------------------------------------------------*/

   measure_array_master = hypre_CTAlloc(HYPRE_Int, num_variables);
   measure_array = hypre_CTAlloc(HYPRE_Int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      measure_array_master[i] = ST_i[i+1]-ST_i[i];
   }

   if ((measure_type || (coarsen_type != 1 && coarsen_type != 11)) 
		&& num_procs > 1)
   {
      if (use_commpkg_A)
         S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      else
         S_ext      = hypre_ParCSRMatrixExtractBExt(S,S,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
      num_nonzeros = S_ext_i[num_cols_offd];
      first_col = hypre_ParCSRMatrixFirstColDiag(S);
      col_0 = first_col-1;
      col_n = col_0+num_variables;
      if (measure_type)
      {
	 for (i=0; i < num_nonzeros; i++)
         {
	    index = S_ext_j[i] - first_col;
	    if (index > -1 && index < num_variables)
		measure_array_master[index]++;
         } 
      } 
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   /* first coarsening phase */

  /*************************************************************
   *
   *   Initialize the lists
   *
   *************************************************************/

   CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);
   
   num_left = 0;
   for (j = 0; j < num_variables; j++)
   {
     if ((S_i[j+1]-S_i[j])== 0 &&
	 (S_offd_i[j+1]-S_offd_i[j]) == 0)
     {
       CF_marker[j] = SF_PT;
       measure_array_master[j] = 0;
     }
     else
     {
       CF_marker[j] = UNDECIDED; 
       /*        num_left++; */ /* BM May 19, 2006: see below*/
     }
   } 

   if (coarsen_type==22) {
     /* BM Sep 8, 2006: allow_emptygrids only if the following holds for all points j: 
        (a) the point has no strong connections at all, OR
        (b) the point has a strong connection across a boundary */
     for (j=0;j<num_variables;j++)
       if (S_i[j+1]>S_i[j] && S_offd_i[j+1] == S_offd_i[j]) {coarsen_type=21;break;}
   }

   for (l = 1; l <= cgc_its; l++)
   {
     LoL_head = NULL;
     LoL_tail = NULL;
     num_left = 0;  /* compute num_left before each RS coarsening loop */
     memcpy (measure_array,measure_array_master,num_variables*sizeof(HYPRE_Int));
     memset (lists,0,sizeof(HYPRE_Int)*num_variables);
     memset (where,0,sizeof(HYPRE_Int)*num_variables);

     for (j = 0; j < num_variables; j++) 
     {    
       measure = measure_array[j];
       if (CF_marker[j] != SF_PT)  
       {
	 if (measure > 0)
	 {
	   enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
	   num_left++; /* compute num_left before each RS coarsening loop */
	 }
	 else if (CF_marker[j] == 0) /* increase weight of strongly coupled neighbors only 
					if j is not conained in a previously constructed coarse grid.
					Reason: these neighbors should start with the same initial weight
					in each CGC iteration.                    BM Aug 30, 2006 */
					
	 {
	   if (measure < 0) hypre_printf("negative measure!\n");
/* 	   CF_marker[j] = f_pnt; */
	   for (k = S_i[j]; k < S_i[j+1]; k++)
	   {
	     nabor = S_j[k];
/*  	     if (CF_marker[nabor] != SF_PT)  */
 	     if (CF_marker[nabor] == 0)  /* BM Aug 30, 2006: don't alter weights of points 
 					    contained in other candidate coarse grids */ 
	     {
	       if (nabor < j)
	       {
		 new_meas = measure_array[nabor];
		 if (new_meas > 0)
		   remove_point(&LoL_head, &LoL_tail, new_meas, 
				nabor, lists, where);
		 else num_left++; /* BM Aug 29, 2006 */
		 
		 new_meas = ++(measure_array[nabor]);
		 enter_on_lists(&LoL_head, &LoL_tail, new_meas,
				nabor, lists, where);
	       }
	       else
	       {
		 new_meas = ++(measure_array[nabor]);
	       }
	     }
	   }
	   /* 	   --num_left; */ /* BM May 19, 2006 */
         }
       }
     }

     /* BM Aug 30, 2006: first iteration: determine maximal weight */
     if (num_left && l==1) measure_max = measure_array[LoL_head->head]; 
     /* BM Aug 30, 2006: break CGC iteration if no suitable 
	starting point is available any more */
     if (!num_left || measure_array[LoL_head->head]<measure_max) {
       while (LoL_head) {
	 hypre_LinkList list_ptr = LoL_head;
	 LoL_head = LoL_head->next_elt;
	 dispose_elt (list_ptr);
       }
       break;
     }

   /****************************************************************
    *
    *  Main loop of Ruge-Stueben first coloring pass.
    *
    *  WHILE there are still points to classify DO:
    *        1) find first point, i,  on list with max_measure
    *           make i a C-point, remove it from the lists
    *        2) For each point, j,  in S_i^T,
    *           a) Set j to be an F-point
    *           b) For each point, k, in S_j
    *                  move k to the list in LoL with measure one
    *                  greater than it occupies (creating new LoL
    *                  entry if necessary)
    *        3) For each point, j,  in S_i,
    *                  move j to the list in LoL with measure one
    *                  smaller than it occupies (creating new LoL
    *                  entry if necessary)
    *
    ****************************************************************/

     while (num_left > 0)
     {
       index = LoL_head -> head;
/*         index = LoL_head -> tail;  */

/*        CF_marker[index] = C_PT; */
       CF_marker[index] = l;  /* BM Aug 18, 2006 */
       measure = measure_array[index];
       measure_array[index] = 0;
       measure_array_master[index] = 0; /* BM May 19: for CGC */
       --num_left;
      
       remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
  
       for (j = ST_i[index]; j < ST_i[index+1]; j++)
       {
         nabor = ST_j[j];
/*          if (CF_marker[nabor] == UNDECIDED) */
	 if (measure_array[nabor]>0) /* undecided point */
	 {
	   /* 	   CF_marker[nabor] = F_PT; */ /* BM Aug 18, 2006 */
	   measure = measure_array[nabor];
	   measure_array[nabor]=0;

	   remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
	   --num_left;
	   
	   for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
           {
	     nabor_two = S_j[k];
/* 	     if (CF_marker[nabor_two] == UNDECIDED) */
	     if (measure_array[nabor_two]>0) /* undecided point */
             {
	       measure = measure_array[nabor_two];
	       remove_point(&LoL_head, &LoL_tail, measure, 
			    nabor_two, lists, where);
	       
	       new_meas = ++(measure_array[nabor_two]);
	       
	       enter_on_lists(&LoL_head, &LoL_tail, new_meas,
			      nabor_two, lists, where);
	     }
	   }
         }
       }
       for (j = S_i[index]; j < S_i[index+1]; j++)
       {
         nabor = S_j[j];
/*          if (CF_marker[nabor] == UNDECIDED) */
	 if (measure_array[nabor]>0) /* undecided point */
         {
	   measure = measure_array[nabor];
	   
	   remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
	   
	   measure_array[nabor] = --measure;
	   
	   if (measure > 0)
	     enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, 
			    lists, where);
	   else
	   {
/* 	     CF_marker[nabor] = F_PT; */ /* BM Aug 18, 2006 */
	     --num_left;

	     for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
             {
	       nabor_two = S_j[k];
/* 	       if (CF_marker[nabor_two] == UNDECIDED) */
	       if (measure_array[nabor_two]>0)
               {
		 new_meas = measure_array[nabor_two];
		 remove_point(&LoL_head, &LoL_tail, new_meas, 
			      nabor_two, lists, where);
		 
		 new_meas = ++(measure_array[nabor_two]);
                 
		 enter_on_lists(&LoL_head, &LoL_tail, new_meas,
				nabor_two, lists, where);
	       }
	     }
	   }
         }
       }
     }
     if (LoL_head) hypre_printf ("Linked list not empty! head: %d\n",LoL_head->head);
   }
   l--; /* BM Aug 15, 2006 */

   hypre_TFree(measure_array);
   hypre_TFree(measure_array_master);
   hypre_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Coarsen 1st pass = %f\n",
                     my_id, wall_time); 
   }

   hypre_TFree(lists);
   hypre_TFree(where);
   
     if (num_procs>1) {
       if (debug_flag == 3)  wall_time = time_getWallclockSeconds();
       hypre_BoomerAMGCoarsenCGC (S,l,coarsen_type,CF_marker);
       
       if (debug_flag == 3)  { 
	 wall_time = time_getWallclockSeconds() - wall_time; 
	 hypre_printf("Proc = %d    Coarsen CGC = %f\n", 
		my_id, wall_time);  
       } 
     }
     else {
       /* the first candiate coarse grid is the coarse grid */ 
       for (j=0;j<num_variables;j++) {
	 if (CF_marker[j]==1) CF_marker[j]=C_PT;
	 else CF_marker[j]=F_PT;
       }
     }

   /* BM May 19, 2006:
      Set all undecided points to be fine grid points. */
   for (j=0;j<num_variables;j++)
     if (!CF_marker[j]) CF_marker[j]=F_PT;

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(HYPRE_Int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      for (i=0; i < num_variables; i++)
      {
	 if (ci_tilde_mark != i) ci_tilde = -1;
         if (CF_marker[i] == -1)
         {
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] > 0)
   	          graph_array[j] = i;
    	    }
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] == -1)
   	       {
   	          set_empty = 1;
   	          for (jj = S_i[j]; jj < S_i[j+1]; jj++)
   	          {
   		     index = S_j[jj];
   		     if (graph_array[index] == i)
   		     {
   		        set_empty = 0;
   		        break;
   		     }
   	          }
   	          if (set_empty)
   	          {
   		     if (C_i_nonempty)
   		     {
   		        CF_marker[i] = 1;
   		        if (ci_tilde > -1)
   		        {
   			   CF_marker[ci_tilde] = -1;
   		           ci_tilde = -1;
   		        }
   	    		C_i_nonempty = 0;
   		        break;
   		     }
   		     else
   		     {
   		        ci_tilde = j;
   		        ci_tilde_mark = i;
   		        CF_marker[j] = 1;
   		        C_i_nonempty = 1;
		        i--;
		        break;
		     }
	          }
	       }
	    }
	 }
      }

   if (debug_flag == 3 && coarsen_type != 2)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Coarsen 2nd pass = %f\n",
                       my_id, wall_time); 
   }

   /* third pass, check boundary fine points for coarse neighbors */

      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
    
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
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
    
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        CF_marker_offd);
    
      hypre_ParCSRCommHandleDestroy(comm_handle);
      }
      AmgCGCBoundaryFix (S,CF_marker,CF_marker_offd);
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         hypre_printf("Proc = %d    CGC boundary fix = %f\n",
                       my_id, wall_time); 
      }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   if (coarsen_type != 1)
   {   
     if (CF_marker_offd) hypre_TFree(CF_marker_offd);  /* BM Aug 21, 2006 */
     if (int_buf_data) hypre_TFree(int_buf_data); /* BM Aug 21, 2006 */
     if (ci_array) hypre_TFree(ci_array); /* BM Aug 21, 2006 */
   }   
   hypre_TFree(graph_array);
   if ((measure_type || (coarsen_type != 1 && coarsen_type != 11)) 
		&& num_procs > 1)
   	hypre_CSRMatrixDestroy(S_ext); 
   
   *CF_marker_ptr   = CF_marker;
   
   return (ierr);
}

/* begin Bram added */

HYPRE_Int hypre_BoomerAMGCoarsenCGC (hypre_ParCSRMatrix    *S,HYPRE_Int numberofgrids,HYPRE_Int coarsen_type,HYPRE_Int *CF_marker)
 /* CGC algorithm
  * ====================================================================================================
  * coupling : the strong couplings
  * numberofgrids : the number of grids
  * coarsen_type : the coarsening type
  * gridpartition : the grid partition
  * =====================================================================================================*/
{
  HYPRE_Int j,/*p,*/mpisize,mpirank,/*rstart,rend,*/choice,*coarse,ierr=0;
  HYPRE_Int *vertexrange,*vertexrange_all;
  HYPRE_Int *CF_marker_offd;
  HYPRE_Int num_variables = hypre_CSRMatrixNumRows (hypre_ParCSRMatrixDiag(S));
/*   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols (hypre_ParCSRMatrixOffd (S)); */
/*   HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd (S); */

/*   double wall_time; */

  HYPRE_IJMatrix ijG;
  hypre_ParCSRMatrix *G;
  hypre_CSRMatrix *Gseq;
  MPI_Comm comm = hypre_ParCSRMatrixComm(S);

  hypre_MPI_Comm_size (comm,&mpisize);
  hypre_MPI_Comm_rank (comm,&mpirank);

#if 0
    if (!mpirank) {
      wall_time = time_getWallclockSeconds();
      hypre_printf ("Starting CGC preparation\n");
    }
#endif
  AmgCGCPrepare (S,numberofgrids,CF_marker,&CF_marker_offd,coarsen_type,&vertexrange);
#if 0 /* debugging */
  if (!mpirank) {
    wall_time = time_getWallclockSeconds() - wall_time;
    hypre_printf ("Finished CGC preparation, wall_time = %f s\n",wall_time);
    wall_time = time_getWallclockSeconds();
    hypre_printf ("Starting CGC matrix assembly\n");
  }
#endif
  AmgCGCGraphAssemble (S,vertexrange,CF_marker,CF_marker_offd,coarsen_type,&ijG);
#if 0
  HYPRE_IJMatrixPrint (ijG,"graph.txt");
#endif
  HYPRE_IJMatrixGetObject (ijG,(void**)&G);
#if 0 /* debugging */
  if (!mpirank) {
    wall_time = time_getWallclockSeconds() - wall_time;
    hypre_printf ("Finished CGC matrix assembly, wall_time = %f s\n",wall_time);
    wall_time = time_getWallclockSeconds();
    hypre_printf ("Starting CGC matrix communication\n");
  }
#endif
#if HYPRE_NO_GLOBAL_PARTITION
 {
  /* classical CGC does not really make sense in combination with HYPRE_NO_GLOBAL_PARTITION,
     but anyway, here it is:
  */
   HYPRE_Int nlocal = vertexrange[1]-vertexrange[0];
   vertexrange_all = hypre_CTAlloc (HYPRE_Int,mpisize+1);
   hypre_MPI_Allgather (&nlocal,1,HYPRE_MPI_INT,vertexrange_all+1,1,HYPRE_MPI_INT,comm);
   vertexrange_all[0]=0;
   for (j=2;j<=mpisize;j++) vertexrange_all[j]+=vertexrange_all[j-1];
 }
#else
  vertexrange_all = vertexrange;
#endif
  Gseq = hypre_ParCSRMatrixToCSRMatrixAll (G);
#if 0 /* debugging */
  if (!mpirank) {
    wall_time = time_getWallclockSeconds() - wall_time;
    hypre_printf ("Finished CGC matrix communication, wall_time = %f s\n",wall_time);
  }
#endif

  if (Gseq) { /* BM Aug 31, 2006: Gseq==NULL if G has no local rows */
#if 0 /* debugging */
    if (!mpirank) {
      wall_time = time_getWallclockSeconds();
      hypre_printf ("Starting CGC election\n");
    }
#endif
    AmgCGCChoose (Gseq,vertexrange_all,mpisize,&coarse);
#if 0 /* debugging */
  if (!mpirank) {
    wall_time = time_getWallclockSeconds() - wall_time;
    hypre_printf ("Finished CGC election, wall_time = %f s\n",wall_time);
  }
#endif

#if 0 /* debugging */
    if (!mpirank) {
      for (j=0;j<mpisize;j++) 
	hypre_printf ("Processor %d, choice = %d of range %d - %d\n",j,coarse[j],vertexrange_all[j]+1,vertexrange_all[j+1]);
    }
    fflush(stdout);
#endif
#if 0 /* debugging */
    if (!mpirank) {
      wall_time = time_getWallclockSeconds();
      hypre_printf ("Starting CGC CF assignment\n");
    }
#endif
    choice = coarse[mpirank];
    for (j=0;j<num_variables;j++) {
      if (CF_marker[j]==choice)
	CF_marker[j] = C_PT;
      else
	CF_marker[j] = F_PT;
    }

    hypre_CSRMatrixDestroy (Gseq);
    hypre_TFree (coarse);
  }
  else
    for (j=0;j<num_variables;j++) CF_marker[j] = F_PT;
#if 0
  if (!mpirank) {
    wall_time = time_getWallclockSeconds() - wall_time;
    hypre_printf ("Finished CGC CF assignment, wall_time = %f s\n",wall_time);
  }
#endif

#if 0 /* debugging */
    if (!mpirank) {
      wall_time = time_getWallclockSeconds();
      hypre_printf ("Starting CGC cleanup\n");
    }
#endif
  HYPRE_IJMatrixDestroy (ijG);
  if (vertexrange) hypre_TFree (vertexrange);
#if HYPRE_NO_GLOBAL_PARTITION
  if (vertexrange_all) hypre_TFree (vertexrange_all);
#endif
  if (CF_marker_offd)  hypre_TFree (CF_marker_offd);
#if 0
  if (!mpirank) {
    wall_time = time_getWallclockSeconds() - wall_time;
    hypre_printf ("Finished CGC cleanup, wall_time = %f s\n",wall_time);
  }
#endif
  return(ierr);
}

HYPRE_Int AmgCGCPrepare (hypre_ParCSRMatrix *S,HYPRE_Int nlocal,HYPRE_Int *CF_marker,HYPRE_Int **CF_marker_offd,HYPRE_Int coarsen_type,HYPRE_Int **vrange)
/* assemble a graph representing the connections between the grids
 * ================================================================================================
 * S : the strength matrix
 * nlocal : the number of locally created coarse grids
 * CF_marker, CF_marker_offd : the coare/fine markers
 * coarsen_type : the coarsening type
 * vrange : the ranges of the vertices representing coarse grids
 * ================================================================================================*/
{
  HYPRE_Int ierr=0;
  HYPRE_Int mpisize,mpirank;
  HYPRE_Int num_sends;
  HYPRE_Int *vertexrange=NULL;
  HYPRE_Int vstart,vend;
  HYPRE_Int *int_buf_data;
  HYPRE_Int start;
  HYPRE_Int i,ii,j;
  HYPRE_Int num_variables = hypre_CSRMatrixNumRows (hypre_ParCSRMatrixDiag(S));
  HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols (hypre_ParCSRMatrixOffd (S));

  MPI_Comm comm = hypre_ParCSRMatrixComm(S);
/*   hypre_MPI_Status status; */

  hypre_ParCSRCommPkg    *comm_pkg    = hypre_ParCSRMatrixCommPkg (S);
  hypre_ParCSRCommHandle *comm_handle;


  hypre_MPI_Comm_size (comm,&mpisize);
  hypre_MPI_Comm_rank (comm,&mpirank);

  if (!comm_pkg) {
    hypre_MatvecCommPkgCreate (S);
    comm_pkg = hypre_ParCSRMatrixCommPkg (S);
  }
  num_sends = hypre_ParCSRCommPkgNumSends (comm_pkg);

  if (coarsen_type % 2 == 0) nlocal++; /* even coarsen_type means allow_emptygrids */
#ifdef HYPRE_NO_GLOBAL_PARTITION
   {
      HYPRE_Int scan_recv;
      
      vertexrange = hypre_CTAlloc(HYPRE_Int,2);
      hypre_MPI_Scan(&nlocal, &scan_recv, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      /* first point in my range */ 
      vertexrange[0] = scan_recv - nlocal;
      /* first point in next proc's range */
      vertexrange[1] = scan_recv;
      vstart = vertexrange[0];
      vend   = vertexrange[1];
   }
#else
  vertexrange = hypre_CTAlloc (HYPRE_Int,mpisize+1);
 
  hypre_MPI_Allgather (&nlocal,1,HYPRE_MPI_INT,vertexrange+1,1,HYPRE_MPI_INT,comm);
  vertexrange[0]=0;
  for (i=2;i<=mpisize;i++) vertexrange[i]+=vertexrange[i-1];
  vstart = vertexrange[mpirank];
  vend   = vertexrange[mpirank+1];
#endif

  /* Note: vstart uses 0-based indexing, while CF_marker uses 1-based indexing */
  if (coarsen_type % 2 == 1) { /* see above */
    for (i=0;i<num_variables;i++)
      if (CF_marker[i]>0)
	CF_marker[i]+=vstart;
  }
  else {
/*      hypre_printf ("processor %d: empty grid allowed\n",mpirank);  */
    for (i=0;i<num_variables;i++) {
      if (CF_marker[i]>0)
	CF_marker[i]+=vstart+1; /* add one because vertexrange[mpirank]+1 denotes the empty grid.
				   Hence, vertexrange[mpirank]+2 is the first coarse grid denoted in
				   global indices, ... */
    }
  }

  /* exchange data */
  *CF_marker_offd = hypre_CTAlloc (HYPRE_Int,num_cols_offd);
  int_buf_data = hypre_CTAlloc (HYPRE_Int,hypre_ParCSRCommPkgSendMapStart (comm_pkg,num_sends));

  for (i=0,ii=0;i<num_sends;i++) {
    start = hypre_ParCSRCommPkgSendMapStart (comm_pkg,i);
    for (j=start;j<hypre_ParCSRCommPkgSendMapStart (comm_pkg,i+1);j++)
      int_buf_data [ii++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
  }

  if (mpisize>1) {
    comm_handle = hypre_ParCSRCommHandleCreate (11,comm_pkg,int_buf_data,*CF_marker_offd);
    hypre_ParCSRCommHandleDestroy (comm_handle);
  }
  hypre_TFree (int_buf_data);
  *vrange=vertexrange;
  return (ierr);
}

#define tag_pointrange 301
#define tag_vertexrange 302

HYPRE_Int AmgCGCGraphAssemble (hypre_ParCSRMatrix *S,HYPRE_Int *vertexrange,HYPRE_Int *CF_marker,HYPRE_Int *CF_marker_offd,HYPRE_Int coarsen_type,
			 HYPRE_IJMatrix *ijG)
/* assemble a graph representing the connections between the grids
 * ================================================================================================
 * S : the strength matrix
 * vertexrange : the parallel layout of the candidate coarse grid vertices
 * CF_marker, CF_marker_offd : the coarse/fine markers 
 * coarsen_type : the coarsening type
 * ijG : the created graph
 * ================================================================================================*/
{
  HYPRE_Int ierr=0;
  HYPRE_Int i,/* ii,*/ip,j,jj,m,n,p;
  HYPRE_Int mpisize,mpirank;

  double weight;

  MPI_Comm comm = hypre_ParCSRMatrixComm(S);
/*   hypre_MPI_Status status; */

  HYPRE_IJMatrix ijmatrix;
  hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag (S);
  hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd (S);
/*   HYPRE_Int *S_i = hypre_CSRMatrixI(S_diag); */
/*   HYPRE_Int *S_j = hypre_CSRMatrixJ(S_diag); */
  HYPRE_Int *S_offd_i = hypre_CSRMatrixI(S_offd);
  HYPRE_Int *S_offd_j;
  HYPRE_Int num_variables = hypre_CSRMatrixNumRows (S_diag);
  HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols (S_offd);
  HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd (S);
  HYPRE_Int pointrange_start,pointrange_end;
  HYPRE_Int *pointrange,*pointrange_nonlocal,*pointrange_strong;
  HYPRE_Int vertexrange_start,vertexrange_end;
  HYPRE_Int *vertexrange_strong,*vertexrange_nonlocal;
  HYPRE_Int num_recvs,num_recvs_strong;
  HYPRE_Int *recv_procs,*recv_procs_strong;
  HYPRE_Int /* *zeros,*rownz,*/*rownz_diag,*rownz_offd;
  HYPRE_Int nz;
  HYPRE_Int nlocal;
  HYPRE_Int one=1;

  hypre_ParCSRCommPkg    *comm_pkg    = hypre_ParCSRMatrixCommPkg (S);
 

  hypre_MPI_Comm_size (comm,&mpisize);
  hypre_MPI_Comm_rank (comm,&mpirank);

  /* determine neighbor processors */
  num_recvs = hypre_ParCSRCommPkgNumRecvs (comm_pkg);
  recv_procs = hypre_ParCSRCommPkgRecvProcs (comm_pkg);
  pointrange = hypre_ParCSRMatrixRowStarts (S);
  pointrange_nonlocal = hypre_CTAlloc  (HYPRE_Int, 2*num_recvs);
  vertexrange_nonlocal = hypre_CTAlloc (HYPRE_Int, 2*num_recvs);
#if HYPRE_NO_GLOBAL_PARTITION
  {
    HYPRE_Int num_sends  =  hypre_ParCSRCommPkgNumSends (comm_pkg);
    HYPRE_Int *send_procs =  hypre_ParCSRCommPkgSendProcs (comm_pkg);
    HYPRE_Int *int_buf_data   = hypre_CTAlloc (HYPRE_Int,4*num_sends);
    HYPRE_Int *int_buf_data2  = int_buf_data + 2*num_sends;
    hypre_MPI_Request *sendrequest,*recvrequest;

    nlocal = vertexrange[1] - vertexrange[0];
    pointrange_start = pointrange[0];
    pointrange_end   = pointrange[1];
    vertexrange_start = vertexrange[0];
    vertexrange_end   = vertexrange[1];
    sendrequest = hypre_CTAlloc (hypre_MPI_Request,2*(num_sends+num_recvs));
    recvrequest = sendrequest+2*num_sends;

    for (i=0;i<num_recvs;i++) {
      hypre_MPI_Irecv (pointrange_nonlocal+2*i,2,HYPRE_MPI_INT,recv_procs[i],tag_pointrange,comm,&recvrequest[2*i]);
      hypre_MPI_Irecv (vertexrange_nonlocal+2*i,2,HYPRE_MPI_INT,recv_procs[i],tag_vertexrange,comm,&recvrequest[2*i+1]);
    }
    for (i=0;i<num_sends;i++) {
      int_buf_data[2*i] = pointrange_start;
      int_buf_data[2*i+1] = pointrange_end;
      int_buf_data2[2*i] = vertexrange_start;
      int_buf_data2[2*i+1] = vertexrange_end;
      hypre_MPI_Isend (int_buf_data+2*i,2,HYPRE_MPI_INT,send_procs[i],tag_pointrange,comm,&sendrequest[2*i]);
      hypre_MPI_Isend (int_buf_data2+2*i,2,HYPRE_MPI_INT,send_procs[i],tag_vertexrange,comm,&sendrequest[2*i+1]);
    }
    hypre_MPI_Waitall (2*(num_sends+num_recvs),sendrequest,hypre_MPI_STATUSES_IGNORE);
    hypre_TFree (int_buf_data);
    hypre_TFree (sendrequest);
  }
#else
  nlocal = vertexrange[mpirank+1] - vertexrange[mpirank];
  pointrange_start = pointrange[mpirank];
  pointrange_end   = pointrange[mpirank+1];
  vertexrange_start = vertexrange[mpirank];
  vertexrange_end   = vertexrange[mpirank+1];
  for (i=0;i<num_recvs;i++) {
    pointrange_nonlocal[2*i] = pointrange[recv_procs[i]];
    pointrange_nonlocal[2*i+1] = pointrange[recv_procs[i]+1];
    vertexrange_nonlocal[2*i] = vertexrange[recv_procs[i]];
    vertexrange_nonlocal[2*i+1] = vertexrange[recv_procs[i]+1];
  }  
#endif
  /* now we have the array recv_procs. However, it may contain too many entries as it is 
     inherited from A. We now have to determine the subset which contains only the
     strongly connected neighbors */
  if (num_cols_offd) {
    S_offd_j = hypre_CSRMatrixJ(S_offd);
  
    recv_procs_strong = hypre_CTAlloc (HYPRE_Int,num_recvs);
    memset (recv_procs_strong,0,num_recvs*sizeof(HYPRE_Int));
    /* don't forget to shorten the pointrange and vertexrange arrays accordingly */
    pointrange_strong = hypre_CTAlloc (HYPRE_Int,2*num_recvs);
    memset (pointrange_strong,0,2*num_recvs*sizeof(HYPRE_Int));
    vertexrange_strong = hypre_CTAlloc (HYPRE_Int,2*num_recvs);
    memset (vertexrange_strong,0,2*num_recvs*sizeof(HYPRE_Int));
    
    for (i=0;i<num_variables;i++)
      for (j=S_offd_i[i];j<S_offd_i[i+1];j++) {
	jj = col_map_offd[S_offd_j[j]];
	for (p=0;p<num_recvs;p++) /* S_offd_j is NOT sorted! */
	  if (jj >= pointrange_nonlocal[2*p] && jj < pointrange_nonlocal[2*p+1]) break;
#if 0
	hypre_printf ("Processor %d, remote point %d on processor %d\n",mpirank,jj,recv_procs[p]);
#endif
	recv_procs_strong [p]=1;
      }
    
    for (p=0,num_recvs_strong=0;p<num_recvs;p++) {
      if (recv_procs_strong[p]) {
	recv_procs_strong[num_recvs_strong]=recv_procs[p];
	pointrange_strong[2*num_recvs_strong] = pointrange_nonlocal[2*p];
	pointrange_strong[2*num_recvs_strong+1] = pointrange_nonlocal[2*p+1];
	vertexrange_strong[2*num_recvs_strong] = vertexrange_nonlocal[2*p];
	vertexrange_strong[2*num_recvs_strong+1] = vertexrange_nonlocal[2*p+1];
	num_recvs_strong++;
      }
    }
  }
  else num_recvs_strong=0;

  hypre_TFree (pointrange_nonlocal);
  hypre_TFree (vertexrange_nonlocal);

  rownz_diag = hypre_CTAlloc (HYPRE_Int,2*nlocal);
  rownz_offd = rownz_diag + nlocal;
  for (p=0,nz=0;p<num_recvs_strong;p++) {
    nz += vertexrange_strong[2*p+1]-vertexrange_strong[2*p];
  }
  for (m=0;m<nlocal;m++) {
    rownz_diag[m]=nlocal-1;
    rownz_offd[m]=nz;
  }
 
  
 
  HYPRE_IJMatrixCreate(comm, vertexrange_start, vertexrange_end-1, vertexrange_start, vertexrange_end-1, &ijmatrix);
  HYPRE_IJMatrixSetObjectType(ijmatrix, HYPRE_PARCSR);
  HYPRE_IJMatrixSetDiagOffdSizes (ijmatrix, rownz_diag, rownz_offd);
  HYPRE_IJMatrixInitialize(ijmatrix);
  hypre_TFree (rownz_diag);

  /* initialize graph */
  weight = -1;
  for (m=vertexrange_start;m<vertexrange_end;m++) {
    for (p=0;p<num_recvs_strong;p++) {
      for (n=vertexrange_strong[2*p];n<vertexrange_strong[2*p+1];n++) {
	ierr = HYPRE_IJMatrixAddToValues (ijmatrix,1,&one,&m,&n,&weight);
#if 0
	if (ierr) hypre_printf ("Processor %d: error %d while initializing graphs at (%d, %d)\n",mpirank,ierr,m,n);
#endif
      }
    }
  }
  
  /* weight graph */
  for (i=0;i<num_variables;i++) {

    for (j=S_offd_i[i];j<S_offd_i[i+1];j++) {
      jj = S_offd_j[j]; /* jj is not a global index!!! */
      /* determine processor */
      for (p=0;p<num_recvs_strong;p++) 
	if (col_map_offd[jj] >= pointrange_strong[2*p] && col_map_offd[jj] < pointrange_strong[2*p+1]) break;
      ip=recv_procs_strong[p];
      /* loop over all coarse grids constructed on this processor domain */
      for (m=vertexrange_start;m<vertexrange_end;m++) {
	/* loop over all coarse grids constructed on neighbor processor domain */
	for (n=vertexrange_strong[2*p];n<vertexrange_strong[2*p+1];n++) {
	  /* coarse grid counting inside gridpartition->local/gridpartition->nonlocal starts with one
	     while counting inside range starts with zero */
	  if (CF_marker[i]-1==m && CF_marker_offd[jj]-1==n)
	    /* C-C-coupling */
	    weight = -1;
	  else if ( (CF_marker[i]-1==m && (CF_marker_offd[jj]==0 || CF_marker_offd[jj]-1!=n) )
		   || ( (CF_marker[i]==0 || CF_marker[i]-1!=m) && CF_marker_offd[jj]-1==n ) )
	    /* C-F-coupling */
	    weight = 0;
	  else weight = -8; /* F-F-coupling */
	  ierr = HYPRE_IJMatrixAddToValues (ijmatrix,1,&one,&m,&n,&weight);
#if 0
	  if (ierr) hypre_printf ("Processor %d: error %d while adding %lf to entry (%d, %d)\n",mpirank,ierr,weight,m,n);
#endif
	}
      }
    }
  }

  /* assemble */
  HYPRE_IJMatrixAssemble (ijmatrix);
  if (num_recvs_strong) {
    hypre_TFree (recv_procs_strong); 
    hypre_TFree (pointrange_strong);
    hypre_TFree (vertexrange_strong);
  }

  *ijG = ijmatrix;
  return (ierr);
}

HYPRE_Int AmgCGCChoose (hypre_CSRMatrix *G,HYPRE_Int *vertexrange,HYPRE_Int mpisize,HYPRE_Int **coarse)
  /* chooses one grid for every processor
   * ============================================================
   * G : the connectivity graph
   * map : the parallel layout
   * mpisize : number of procs
   * coarse : the chosen coarse grids
   * ===========================================================*/
{
  HYPRE_Int i,j,jj,p,choice,*processor,ierr=0;
  HYPRE_Int measure,new_measure;

/*   MPI_Comm comm = hypre_ParCSRMatrixComm(G); */

/*   hypre_ParCSRCommPkg    *comm_pkg    = hypre_ParCSRMatrixCommPkg (G); */
/*   hypre_ParCSRCommHandle *comm_handle; */

  double *G_data = hypre_CSRMatrixData (G);
  double max;
  HYPRE_Int *G_i = hypre_CSRMatrixI(G);
  HYPRE_Int *G_j = hypre_CSRMatrixJ(G);
  hypre_CSRMatrix *H,*HT;
  HYPRE_Int *H_i,*H_j,*HT_i,*HT_j;
  HYPRE_Int jG,jH;
  HYPRE_Int num_vertices = hypre_CSRMatrixNumRows (G);
  HYPRE_Int *measure_array;
  HYPRE_Int *lists,*where;

  hypre_LinkList LoL_head = NULL;
  hypre_LinkList LoL_tail = NULL;

  processor = hypre_CTAlloc (HYPRE_Int,num_vertices);
  *coarse = hypre_CTAlloc (HYPRE_Int,mpisize);
  memset (*coarse,0,sizeof(HYPRE_Int)*mpisize);

  measure_array = hypre_CTAlloc (HYPRE_Int,num_vertices);
  lists = hypre_CTAlloc (HYPRE_Int,num_vertices);
  where = hypre_CTAlloc (HYPRE_Int,num_vertices);

/*   for (p=0;p<mpisize;p++) hypre_printf ("%d: %d-%d\n",p,range[p]+1,range[p+1]); */

  /******************************************************************
   * determine heavy edges
   ******************************************************************/

  jG  = G_i[num_vertices];
  H   = hypre_CSRMatrixCreate (num_vertices,num_vertices,jG);
  H_i = hypre_CTAlloc (HYPRE_Int,num_vertices+1);
  H_j = hypre_CTAlloc (HYPRE_Int,jG);
  hypre_CSRMatrixI(H) = H_i;
  hypre_CSRMatrixJ(H) = H_j;

  for (i=0,p=0;i<num_vertices;i++) {
    while (vertexrange[p+1]<=i) p++;
    processor[i]=p;
  }

  H_i[0]=0;
  for (i=0,jj=0;i<num_vertices;i++) {
#if 0 
    hypre_printf ("neighbors of grid %d:",i); 
#endif
    H_i[i+1]=H_i[i];
    for (j=G_i[i],choice=-1,max=0;j<G_i[i+1];j++) {
#if 0
      if (G_data[j]>=0.0) 
	hypre_printf ("G[%d,%d]=0. G_j(j)=%d, G_data(j)=%f.\n",i,G_j[j],j,G_data[j]);
#endif
      /* G_data is always negative, so this test is sufficient */
      if (choice==-1 || G_data[j]>max) {
	choice = G_j[j];
	max = G_data[j];
      }
      if (j==G_i[i+1]-1 || processor[G_j[j+1]] > processor[choice]) {
	/* we are done for this processor boundary */
	H_j[jj++]=choice;
	H_i[i+1]++;
#if 0
 	hypre_printf (" %d",choice); 
#endif
	choice = -1; max=0;
      }
    }
#if 0
     hypre_printf("\n"); 
#endif
  }

  /******************************************************************
   * compute H^T, the transpose of H
   ******************************************************************/

  jH = H_i[num_vertices];
  HT = hypre_CSRMatrixCreate (num_vertices,num_vertices,jH);
  HT_i = hypre_CTAlloc (HYPRE_Int,num_vertices+1);
  HT_j = hypre_CTAlloc (HYPRE_Int,jH);
  hypre_CSRMatrixI(HT) = HT_i;
  hypre_CSRMatrixJ(HT) = HT_j;

   for (i=0; i <= num_vertices; i++)
      HT_i[i] = 0;
   for (i=0; i < jH; i++) {
     HT_i[H_j[i]+1]++;
   }
   for (i=0; i < num_vertices; i++) {
     HT_i[i+1] += HT_i[i];
   }
   for (i=0; i < num_vertices; i++) {
     for (j=H_i[i]; j < H_i[i+1]; j++) {
       HYPRE_Int myindex = H_j[j];
       HT_j[HT_i[myindex]] = i;
       HT_i[myindex]++;
     }
   }      
   for (i = num_vertices; i > 0; i--) {
     HT_i[i] = HT_i[i-1];
   }
   HT_i[0] = 0;

  /*****************************************************************
   * set initial vertex weights
   *****************************************************************/

  for (i=0;i<num_vertices;i++) {
    measure_array[i] = H_i[i+1] - H_i[i] + HT_i[i+1] - HT_i[i];
    enter_on_lists (&LoL_head,&LoL_tail,measure_array[i],i,lists,where);
  }

  /******************************************************************
   * apply CGC iteration
   ******************************************************************/

  while (LoL_head && measure_array[LoL_head->head]) {


    choice = LoL_head->head;
    measure = measure_array[choice];
#if 0
    hypre_printf ("Choice: %d, measure %d, processor %d\n",choice, measure,processor[choice]);
    fflush(stdout);
#endif

    (*coarse)[processor[choice]] = choice+1;  /* add one because coarsegrid indexing starts with 1, not 0 */
    /* new maximal weight */
    new_measure = measure+1;
    for (i=vertexrange[processor[choice]];i<vertexrange[processor[choice]+1];i++) {
      /* set weights for all remaining vertices on this processor to zero */
      measure = measure_array[i];
      remove_point (&LoL_head,&LoL_tail,measure,i,lists,where);
      measure_array[i]=0;
    }
    for (j=H_i[choice];j<H_i[choice+1];j++){
      jj = H_j[j];
      /* if no vertex is chosen on this proc, set weights of all heavily coupled vertices to max1 */
      if (!(*coarse)[processor[jj]]) {
	measure = measure_array[jj];
	remove_point (&LoL_head,&LoL_tail,measure,jj,lists,where);
	enter_on_lists (&LoL_head,&LoL_tail,new_measure,jj,lists,where);
	measure_array[jj]=new_measure;
      }
    }
    for (j=HT_i[choice];j<HT_i[choice+1];j++) {
      jj = HT_j[j];
      /* if no vertex is chosen on this proc, set weights of all heavily coupled vertices to max1 */
      if (!(*coarse)[processor[jj]]) {
	measure = measure_array[jj];
	remove_point (&LoL_head,&LoL_tail,measure,jj,lists,where);
	enter_on_lists (&LoL_head,&LoL_tail,new_measure,jj,lists,where);
	measure_array[jj]=new_measure;
      }
    }
  }

  /* remove remaining list elements, if they exist. They all should have measure 0 */
  while (LoL_head) {
    i = LoL_head->head;
    measure = measure_array[i];
#if 0
    hypre_assert (measure==0);
#endif
    remove_point (&LoL_head,&LoL_tail,measure,i,lists,where);
  }
    

  for (p=0;p<mpisize;p++)
    /* if the algorithm has not determined a coarse vertex for this proc, simply take the last one 
       Do not take the first one, it might by empty! */
    if (!(*coarse)[p]) {
      (*coarse)[p] = vertexrange[p+1];
/*       hypre_printf ("choice for processor %d: %d\n",p,range[p]+1); */
    }

  /********************************************
   * clean up 
   ********************************************/

  hypre_CSRMatrixDestroy (H);
  hypre_CSRMatrixDestroy (HT);


  hypre_TFree (processor);
  hypre_TFree (measure_array);
  hypre_TFree (lists);
  hypre_TFree (where);
  
  return(ierr);
}

HYPRE_Int AmgCGCBoundaryFix (hypre_ParCSRMatrix *S,HYPRE_Int *CF_marker,HYPRE_Int *CF_marker_offd)
  /* Checks whether an interpolation is possible for a fine grid point with strong couplings.
   * Required after CGC coarsening
   * ========================================================================================
   * S : the strength matrix
   * CF_marker, CF_marker_offd : the coarse/fine markers
   * ========================================================================================*/
{
  HYPRE_Int mpirank,i,j,has_c_pt,ierr=0;
  hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag (S);
  hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd (S);
  HYPRE_Int *S_i = hypre_CSRMatrixI(S_diag);
  HYPRE_Int *S_j = hypre_CSRMatrixJ(S_diag);
  HYPRE_Int *S_offd_i = hypre_CSRMatrixI(S_offd);
  HYPRE_Int *S_offd_j;
  HYPRE_Int num_variables = hypre_CSRMatrixNumRows (S_diag);
  HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols (S_offd);
  HYPRE_Int added_cpts=0;
  MPI_Comm comm = hypre_ParCSRMatrixComm(S);

  hypre_MPI_Comm_rank (comm,&mpirank);
  if (num_cols_offd) {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
  }
  
  for (i=0;i<num_variables;i++) {
    if (S_offd_i[i]==S_offd_i[i+1] || CF_marker[i] == C_PT) continue;
    has_c_pt=0;

    /* fine grid point with strong connections across the boundary */
    for (j=S_i[i];j<S_i[i+1];j++) 
      if (CF_marker[S_j[j]] == C_PT) {has_c_pt=1; break;}
    if (has_c_pt) continue;

    for (j=S_offd_i[i];j<S_offd_i[i+1];j++) 
      if (CF_marker_offd[S_offd_j[j]] == C_PT) {has_c_pt=1; break;}
    if (has_c_pt) continue;

    /* all points i is strongly coupled to are fine: make i C_PT */
    CF_marker[i] = C_PT;
#if 0
    hypre_printf ("Processor %d: added point %d in AmgCGCBoundaryFix\n",mpirank,i);
#endif
    added_cpts++;
  }
#if 0
  if (added_cpts)  hypre_printf ("Processor %d: added %d points in AmgCGCBoundaryFix\n",mpirank,added_cpts);
  fflush(stdout);
#endif
  return(ierr);
}
